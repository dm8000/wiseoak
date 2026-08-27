"""
Indice de recuperacao: SQLite para metadado e texto, numpy para o denso, BM25 para o
lexical.

POR QUE NAO CHROMA. O objetivo deste projeto e MEDIR se chunking hierarquico bate
chunking plano. Chroma indexa com HNSW, que e aproximado: o erro de recall do indice
entraria como variavel de confusao junto com o efeito que se quer medir. Com um livro
(3.403 filhos x 768 float32 = 10 MB) a forca bruta e exata e roda em milissegundos.
Quando o corpus crescer a ponto de doer, troca-se — e ai o custo do ANN sera uma decisao
tomada com numero, nao por default de biblioteca.

Busca hibrida com RRF (Reciprocal Rank Fusion): o denso acha parafrase, o BM25 acha
termo tecnico raro. Num corpus de OCR ruim os dois erram de formas diferentes, o que e
exatamente quando fundir vale a pena.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import clientes

ESQUEMA = """
CREATE TABLE IF NOT EXISTS chunk (
    id            TEXT PRIMARY KEY,
    nivel         TEXT NOT NULL,
    pai_id        TEXT,
    ordem         INTEGER,
    livro         TEXT,
    secao_num     INTEGER,
    secao         TEXT,
    capitulo_num  INTEGER,
    capitulo      TEXT,
    caminho       TEXT,
    pagina_inicial INTEGER,
    pagina_final  INTEGER,
    texto         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_nivel ON chunk(nivel);
CREATE INDEX IF NOT EXISTS ix_pai   ON chunk(pai_id);
CREATE INDEX IF NOT EXISTS ix_cap   ON chunk(capitulo_num);
"""

CAMPOS = ("id", "nivel", "pai_id", "ordem", "livro", "secao_num", "secao",
          "capitulo_num", "capitulo", "caminho", "pagina_inicial", "pagina_final",
          "texto")


def _tokenizar(texto: str) -> list[str]:
    """Minusculas e so o que for alfanumerico. O BM25 do corpus e da consulta tem que
    passar pela MESMA funcao, senao a busca lexical silenciosamente nao casa nada."""
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in texto).split()
            if len(t) > 1]


class Indice:
    """
    Um indice em disco: `base.sqlite` com o texto e `base.npy` com os vetores dos filhos.

    Os pais NAO sao vetorizados: eles nunca sao alvo de busca, so de expansao de contexto
    depois que um filho ganha. Vetorizar os dois dobraria o custo de indexacao sem mudar
    nenhum resultado.
    """

    def __init__(self, caminho: Path | str):
        self.caminho = Path(caminho)
        # Conexao POR THREAD. O servidor do LangGraph Studio executa os nos num pool, e
        # uma conexao sqlite criada numa thread e recusada em outra. Um Indice em cache
        # atravessa threads; a conexao nao pode.
        self._local = threading.local()
        self._trava = threading.Lock()
        self.db.executescript(ESQUEMA)
        self._vetores: np.ndarray | None = None
        self._ids: list[str] | None = None
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] | None = None

    @property
    def db(self) -> sqlite3.Connection:
        conexao = getattr(self._local, "db", None)
        if conexao is None:
            conexao = sqlite3.connect(self.caminho.with_suffix(".sqlite"))
            conexao.row_factory = sqlite3.Row
            self._local.db = conexao
        return conexao

    # ---------------------------------------------------------------- indexacao

    def indexar(self, chunks: list[dict], *, progresso=None) -> int:
        linhas = [tuple(
            json.dumps(c[k], ensure_ascii=False) if k == "caminho" else c.get(k)
            for k in CAMPOS) for c in chunks]
        self.db.executemany(
            f"INSERT OR REPLACE INTO chunk ({','.join(CAMPOS)}) "
            f"VALUES ({','.join('?' * len(CAMPOS))})", linhas)
        self.db.commit()

        # Deduplicar por id antes de vetorizar. O id e hash do conteudo e o INSERT OR
        # REPLACE acima ja colapsa texto identico numa linha so — se a lista de vetores
        # nao colapsar junto, ficam dois vetores com o mesmo id, e o mesmo trecho pode
        # ocupar duas das k vagas de contexto. Nao e hipotetico: o corpus normativo
        # repete texto de praxe literalmente entre resolucoes ("entra em vigor na data
        # de sua publicacao"), e 11 dos 664 trechos saiam duplicados.
        vistos: set[str] = set()
        filhos = []
        for c in chunks:
            if c["nivel"] != "filho" or c["id"] in vistos:
                continue
            vistos.add(c["id"])
            filhos.append(c)
        repetidos = sum(1 for c in chunks if c["nivel"] == "filho") - len(filhos)
        if repetidos:
            print(f"  {repetidos} trechos de texto identico colapsados", file=sys.stderr)

        vetores: list[list[float]] = []
        for i in range(0, len(filhos), clientes.LOTE_EMBED):
            lote = filhos[i:i + clientes.LOTE_EMBED]
            vetores.extend(clientes.embed([c["texto"] for c in lote]))
            if progresso:
                progresso(min(i + len(lote), len(filhos)), len(filhos))

        m = np.asarray(vetores, dtype=np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        np.save(self.caminho.with_suffix(".npy"), m)
        self.caminho.with_suffix(".ids.json").write_text(
            json.dumps([c["id"] for c in filhos]))
        self._vetores = self._ids = self._bm25 = None
        return len(filhos)

    # ------------------------------------------------------------------- carga

    def _carregar_denso(self):
        with self._trava:
            if self._vetores is not None:
                return
            self._vetores = np.load(self.caminho.with_suffix(".npy"))
            self._ids = json.loads(self.caminho.with_suffix(".ids.json").read_text())

    def _carregar_bm25(self):
        with self._trava:
            if self._bm25 is not None:
                return
            linhas = self.db.execute(
                "SELECT id, texto FROM chunk WHERE nivel='filho' ORDER BY id").fetchall()
            self._bm25_ids = [r["id"] for r in linhas]
            self._bm25 = BM25Okapi([_tokenizar(r["texto"]) for r in linhas])

    # ------------------------------------------------------------------ busca

    def buscar_denso(self, pergunta: str, k: int = 20) -> list[tuple[str, float]]:
        self._carregar_denso()
        v = np.asarray(clientes.embed([pergunta])[0], dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-9
        sim = self._vetores @ v
        top = np.argsort(-sim)[:k]
        return [(self._ids[i], float(sim[i])) for i in top]

    def buscar_bm25(self, pergunta: str, k: int = 20) -> list[tuple[str, float]]:
        self._carregar_bm25()
        pontos = self._bm25.get_scores(_tokenizar(pergunta))
        top = np.argsort(-pontos)[:k]
        return [(self._bm25_ids[i], float(pontos[i])) for i in top if pontos[i] > 0]

    def buscar(self, pergunta: str, k: int = 20, *, hibrido: bool = True,
               rrf_k: int = 60) -> list[tuple[str, float]]:
        """
        RRF: score = soma de 1/(rrf_k + posicao) sobre as listas. Funde por POSICAO, nao
        por score — cosseno e BM25 nao vivem na mesma escala, e somar os dois direto faz
        a lista de maior amplitude dominar.
        """
        denso = self.buscar_denso(pergunta, k * 2)
        if not hibrido:
            return denso[:k]
        lexical = self.buscar_bm25(pergunta, k * 2)
        pontos: dict[str, float] = {}
        for lista in (denso, lexical):
            for pos, (cid, _) in enumerate(lista):
                pontos[cid] = pontos.get(cid, 0.0) + 1.0 / (rrf_k + pos + 1)
        return sorted(pontos.items(), key=lambda p: -p[1])[:k]

    def rerankear(self, pergunta: str, ids: list[str], k: int = 5
                  ) -> list[tuple[str, float]]:
        docs = [self.obter(i)["texto"] for i in ids]
        pares = clientes.rerank(pergunta, docs, top_n=k)
        return [(ids[i], s) for i, s in pares][:k]

    # ------------------------------------------------------------- leitura

    def obter(self, chunk_id: str) -> dict:
        r = self.db.execute("SELECT * FROM chunk WHERE id=?", (chunk_id,)).fetchone()
        if r is None:
            raise KeyError(chunk_id)
        d = dict(r)
        d["caminho"] = json.loads(d["caminho"] or "[]")
        return d

    def pai_de(self, chunk_id: str) -> dict | None:
        """O contexto que vai para o modelo: o filho ganha a busca, o pai da o entorno."""
        filho = self.obter(chunk_id)
        return self.obter(filho["pai_id"]) if filho.get("pai_id") else None

    def capitulos(self) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT capitulo_num, capitulo, MIN(pagina_inicial) ini, "
            "MAX(pagina_final) fim, COUNT(*) n FROM chunk WHERE nivel='filho' "
            "AND capitulo_num IS NOT NULL GROUP BY capitulo_num ORDER BY capitulo_num")]

    def quantos(self, nivel: str = "filho") -> int:
        return self.db.execute("SELECT COUNT(*) c FROM chunk WHERE nivel=?",
                               (nivel,)).fetchone()["c"]
