#!/usr/bin/env python3
"""
Se a resposta esta no livro e nao veio, ONDE ela esta?

Localiza o trecho portador e mede a distancia entre ele e o que a busca entregou. E o
insumo para iterar a recuperacao com informacao em vez de palpite.

COMO LOCALIZA, e por que assim. Julgar "este paragrafo em ingles responde esta assertiva em
portugues?" ja falhou duas vezes neste projeto: criterio lexical deu 33% de precisao contra
rotulagem manual, e juiz LLM deu 64%. Entao aqui o modelo NAO julga relevancia — ele so faz
a tarefa facil de TRADUZIR os termos-chave da assertiva para o ingles medico. A localizacao
em si e lexical e programatica: chunk que contem o termo traduzido E a ancora numerica, com
os dois na mesma janela de texto.

Traducao de termo e verificavel por leitura em uma linha; juizo de relevancia nao e. E a
diferenca que torna este diagnostico confiavel onde os anteriores nao eram.

    ./onde.py --amostra errou --n 20
    ./onde.py --amostra acertou --n 20 --indice-b dados/indice/m10p
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import statistics
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "bench"))
ANALISES = RAIZ / "eval" / "analises"

from wiseoak import clientes  # noqa: E402
from wiseoak.store import Indice  # noqa: E402
import recuperacao as R  # noqa: E402

SIS_TERMO = (
    "Voce traduz terminologia medica do portugues para o ingles. Devolva de 2 a 5 termos "
    "em INGLES que apareceriam num livro-texto de anestesiologia ao tratar do assunto da "
    "afirmacao. Use a forma que o livro usaria (substantivo tecnico), nao frase. "
    "Nao explique, nao julgue se a afirmacao e verdadeira."
)
SCH_TERMO = {"type": "object",
             "properties": {"termos": {"type": "array", "items": {"type": "string"}}},
             "required": ["termos"]}

JANELA = 200


def termos_en(assertiva: str, modelo: str) -> list[str]:
    try:
        r = clientes.chat(
            [{"role": "system", "content": SIS_TERMO},
             {"role": "user", "content": assertiva[:900]}],
            modelo=modelo, think=False, max_tokens=120, temp=0.0, schema=SCH_TERMO)
        t = json.loads(r["content"]).get("termos") or []
        return [x.strip() for x in t if 3 <= len(x.strip()) <= 60][:5]
    except Exception:
        return []


def localizar(db: sqlite3.Connection, termos: list[str], pats: list[str]) -> list[str]:
    """
    Chunks que contem UM termo traduzido E uma ancora numerica, na mesma janela.
    Varredura do corpus inteiro, nao do top-k — a pergunta e onde o fato ESTA.
    """
    achados = []
    for cid, texto in db.execute("SELECT id, texto FROM chunk WHERE nivel='filho'"):
        for m in (mm for p in pats for mm in re.finditer(p, texto, re.I)):
            jan = texto[max(0, m.start() - JANELA):m.end() + JANELA].lower()
            if any(t.lower() in jan for t in termos):
                achados.append(cid)
                break
    return achados


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amostra", choices=("errou", "acertou"), required=True)
    ap.add_argument("--bloco", default="COTA")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--semente", type=int, default=3)
    ap.add_argument("--indice-a", default="dados/indice/m10")
    ap.add_argument("--indice-b", default=None)
    ap.add_argument("--modelo", default="gemma4-plan")
    ap.add_argument("--k-max", type=int, default=200)
    ap.add_argument("--saida", type=Path, default=None)
    a = ap.parse_args()

    itens = {i["id"]: i for i in R.carregar("dev", "vf")}
    db = sqlite3.connect(RAIZ / "eval" / "resultados.sqlite")
    res: dict[str, bool] = {}
    for (j,) in db.execute("SELECT itens FROM resultado WHERE bloco=?", (a.bloco,)):
        res.update(json.loads(j))
    querido = (a.amostra == "acertou")
    ids = [i for i, ok in res.items() if ok == querido and i in itens]
    random.seed(a.semente)
    ids = random.sample(ids, min(a.n, len(ids)))
    print(f"  {len(ids)} questões que o {a.bloco} {a.amostra}", file=sys.stderr)

    ixs = [("A", Indice(a.indice_a))]
    if a.indice_b and Path(a.indice_b + ".npy").exists():
        ixs.append(("B", Indice(a.indice_b)))

    linhas, resumo = [], {n: [] for n, _ in ixs}
    achou_algum = 0
    for k, iid in enumerate(ids, 1):
        it = itens[iid]
        asr = it.get("assertiva") or it["enunciado"]
        pats = [p for n, u in it["_ancoras"] for p in R.padroes(n, u)]
        termos = termos_en(asr, a.modelo)
        linhas.append("=" * 92)
        linhas.append(f"[{k}] {' '.join(asr.split())[:170]}")
        linhas.append(f"    âncoras={it['_ancoras']}  termos_en={termos}")
        if not termos:
            linhas.append("    (modelo não devolveu termos)")
            continue
        for nome, ix in ixs:
            portadores = set(localizar(ix.db, termos, pats))
            q = it["enunciado"] + "\n\nAssertiva: " + (it.get("assertiva") or "")
            ranking = [c for c, _ in ix.buscar(q, a.k_max, hibrido=False)]
            pos = next((i for i, c in enumerate(ranking, 1) if c in portadores), None)
            resumo[nome].append(pos)
            achou_algum += (nome == ixs[0][0] and bool(portadores))
            linhas.append(f"    [{nome}] portadores no corpus: {len(portadores)} · "
                          f"melhor posição na busca: {pos or '>'+str(a.k_max)}")
            if portadores:
                cid = next(iter(portadores))
                tx = " ".join(ix.obter(cid)["texto"].split())
                m = next((re.search(r".{0,90}" + p + r".{0,90}", tx, re.I)
                          for p in pats if re.search(p, tx, re.I)), None)
                linhas.append(f"         …{(m.group(0) if m else tx[:170])[:190]}…")

    cab = [f"ONDE A RESPOSTA ESTÁ · amostra '{a.amostra}' do bloco {a.bloco} · n={len(ids)}",
           "",
           "Localização LEXICAL: termo traduzido pelo modelo + âncora numérica, na mesma",
           "janela. O modelo só traduz termo — não julga relevância, porque julgar já",
           "falhou aqui (33% lexical, 64% juiz LLM contra rotulagem manual).",
           ""]
    for nome, _ in ixs:
        v = resumo[nome]
        ach = [p for p in v if p]
        cab.append(f"  índice {nome}: portador localizado em {len(ach)}/{len(v)} · "
                   + " · ".join(f"@{k}: {sum(1 for p in ach if p <= k)}" for k in (4, 8, 20, 50))
                   + (f" · mediana {statistics.median(ach):.0f}" if ach else ""))
    cab.append("")
    saida = a.saida or (ANALISES / f"onde-{a.amostra}.txt")
    saida.write_text("\n".join(cab + linhas) + "\n")
    print("\n".join(cab))
    print(f"detalhe em {saida}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
