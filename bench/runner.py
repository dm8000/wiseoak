#!/usr/bin/env python3
"""
Harness de medicao do WiseOak. Uma celula = (grafo x modelo x raciocinio x indice).

Escreve no MESMO schema de Cloud_assistant/eval/resultados.sqlite, de proposito: as
convencoes de leitura daquele relatorio.py valem aqui sem traducao, e o `itens`
(id -> bool) e o que habilita McNemar PAREADO entre duas celulas.

Convencoes herdadas da bancada, todas aprendidas errando la:

  - resposta vazia por TRUNCAGEM conta como falha, e a taxa vai em COLUNA SEPARADA.
    Misturar as duas convencoes ja valeu 22,3 pp de erro naquele projeto.
  - `n` e a amostra DEPOIS de excluir item invalidado.
  - verificacao e programatica, nunca juiz LLM.
  - agrupa por modelo: um swap de 18 GB por modelo, nao por pergunta.

    ./runner.py --grafos v0,v2 --split dev --n 60 --bench vf
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from wiseoak.grafos.comum import verificar_citacoes  # noqa: E402
from wiseoak.grafos.variantes import construir, estado_inicial  # noqa: E402
from wiseoak.store import Indice  # noqa: E402

ESQUEMA = """
CREATE TABLE IF NOT EXISTS resultado (
  id INTEGER PRIMARY KEY,
  modelo TEXT NOT NULL, setup TEXT NOT NULL, bench TEXT NOT NULL,
  quando TEXT NOT NULL, n INTEGER, metricas TEXT NOT NULL, itens TEXT,
  termica TEXT, segundos REAL, bloco TEXT
);
CREATE INDEX IF NOT EXISTS ix_cel ON resultado(modelo, setup, bench);
"""


def temperatura() -> float | None:
    """GPU em C, ou None. Rodada de 14h nao pode cozinhar a placa."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def esperar_esfriar(teto: float = 80.0, retoma: float = 70.0, maximo: int = 600) -> None:
    """
    Mesma convencao da bancada deste projeto: pausa acima de 80 C, retoma abaixo de 70.
    Sem isso, uma rodada longa pode entrar em throttling e a queda de velocidade vira
    ruido na coluna de latencia sem ninguem perceber.
    """
    t = temperatura()
    if t is None or t < teto:
        return
    esperou = 0
    while esperou < maximo:
        time.sleep(15)
        esperou += 15
        t = temperatura()
        if t is None or t <= retoma:
            print(f"\n  retomando a {t} C apos {esperou}s", file=sys.stderr)
            return
    print(f"\n  AVISO: ainda a {t} C apos {maximo}s; seguindo assim mesmo",
          file=sys.stderr)


def carregar(bench: str, split: str, n: int | None, semente: int,
             estratificar: bool = False) -> list[dict]:
    """
    Amostra ALEATORIA quando n limita. Pegar os n primeiros enviesa: na bancada, as 60
    primeiras questoes do HumanEval+ sao mais dificeis que a media (67,5% contra 84,8%)
    e isso contaminou uma celula inteira.
    """
    arquivo = {"vf": "questoes_sba", "mcq": "questoes_sba",
               "healthqa": "questoes_healthqa"}[bench]
    caminho = RAIZ / "dados" / f"{arquivo}.{split}.jsonl"
    itens = [json.loads(l) for l in caminho.read_text().splitlines() if l.strip()]
    if bench == "vf":
        itens = [i for i in itens if i["tipo"] == "vf"]
    elif bench == "mcq":
        itens = [i for i in itens if i["tipo"] == "mcq"]
    if not n or n >= len(itens):
        return itens
    if not estratificar:
        return random.Random(semente).sample(itens, n)

    # ESTRATIFICADO POR CLASSE. Sorteio simples faz a classe rara sumir: 'juridico-
    # normativo' e 2% do banco, entao em n=80 entram 2 itens e nenhuma diferenca ali e
    # detectavel. Fixar n por classe da poder proprio a cada estrato — que e o unico
    # jeito de responder "onde o RAG ajuda" em vez de "ajuda em media".
    rnd = random.Random(semente)
    por_classe: dict[str, list] = {}
    for it in itens:
        por_classe.setdefault(it.get("classe", "?"), []).append(it)
    cota = max(1, n // max(len(por_classe), 1))
    saida = []
    for classe in sorted(por_classe):
        pool = por_classe[classe]
        saida.extend(rnd.sample(pool, min(cota, len(pool))))
    # sobra vai para as classes maiores, para nao desperdicar orcamento
    if len(saida) < n:
        resto = [i for i in itens if i not in saida]
        saida.extend(rnd.sample(resto, min(n - len(saida), len(resto))))
    print("  estratos: " + ", ".join(
        f"{c}={sum(1 for i in saida if i.get('classe') == c)}"
        for c in sorted(por_classe)), file=sys.stderr)
    return saida


def montar_pergunta(item: dict) -> tuple[str, str]:
    """Devolve (texto_da_pergunta, modo)."""
    if item["tipo"] == "vf":
        return (f"{item['enunciado']}\n\nAssertiva: {item['assertiva']}", "vf")
    alts = item.get("alternativas") or {}
    if alts:
        corpo = "\n".join(f"{k}) {v}" for k, v in sorted(alts.items()))
        return (f"{item['enunciado']}\n\n{corpo}", "mcq")
    return (item["enunciado"], "mcq")  # healthqa ja traz as alternativas no enunciado


def rodar_celula(grafo_nome: str, itens: list[dict], *, modelo: str, raciocinio: str,
                 indice: Indice | None, k_busca: int, k_contexto: int,
                 ancoragem: str = "estrita") -> dict:
    grafo = construir(grafo_nome)
    acertos: dict[str, bool] = {}
    truncadas = fid_num = fid_den = pag_ok = 0
    # A ressalva e contada, nao lida. O que interessa e a taxa e o acerto CONDICIONAL:
    # se o modelo acerta menos quando ressalva, a ressalva esta identificando de fato os
    # casos em que o livro nao decide — e isso e informacao util para o usuario final.
    com_ressalva: list[bool] = []
    sem_ressalva: list[bool] = []
    # Tempo POR NO. Ja existia no trace e era descartado; sem ele, decompor a latencia
    # so por subtracao entre grafos, o que assume que o no comum custa o mesmo nos dois.
    por_no: dict[str, list[float]] = {}
    rotas: dict[str, str] = {}
    latencias: list[float] = []
    erros = 0

    for it in itens:
        pergunta, modo = montar_pergunta(it)
        t0 = time.time()
        try:
            r = grafo.invoke(estado_inicial(
                pergunta, modo=modo, modelo=modelo, raciocinio=raciocinio,
                indice=indice, k_busca=k_busca, k_contexto=k_contexto,
                ancoragem=ancoragem))
        except Exception as e:  # nunca derruba o bloco por falha de uma pergunta
            print(f"\n  ERRO em {it['id']}: {type(e).__name__}: {e}", file=sys.stderr)
            acertos[it["id"]] = False
            erros += 1
            continue
        latencias.append(time.time() - t0)
        for passo in (r.get("trace") or []):
            por_no.setdefault(passo["no"], []).append(passo["segundos"])
            # Para onde a pergunta foi roteada. Sem guardar isto da para dizer o placar
            # do braco roteado, mas nao da para dizer se o roteador ACERTOU o destino —
            # e um placar que nao melhora pode ser corpus ruim OU roteamento errado, que
            # sao problemas opostos.
            if passo["no"] == "rotear":
                rotas[it["id"]] = passo.get("indice") or "?"

        dada = (r.get("resposta") or "").strip().upper()[:1]
        certo = dada == item_gabarito(it)
        acertos[it["id"]] = certo
        (com_ressalva if (r.get("ressalva") or "").strip() else sem_ressalva).append(certo)
        if r.get("truncou"):
            truncadas += 1
        v = verificar_citacoes(r)
        fid_num += v["fieis"]
        fid_den += v["citacoes"]
        pag_ok += v["pagina_ok"]
        sys.stderr.write(f"\r  {len(latencias) + erros}/{len(itens)} "
                         f"acerto={sum(acertos.values())}")
        if len(latencias) % 25 == 0:
            esperar_esfriar()
    sys.stderr.write("\n")

    n = len(itens)
    lat = sorted(latencias) or [0.0]
    return {
        "metricas": {
            "acerto": sum(acertos.values()) / n if n else 0.0,
            "n": n, "erros": erros,
            "truncou": truncadas / n if n else 0.0,
            "fidelidade": fid_num / fid_den if fid_den else 0.0,
            "pagina_ok": pag_ok / fid_den if fid_den else 0.0,
            "citacoes_por_resposta": fid_den / n if n else 0.0,
            "seg_por_no": {k: round(statistics.median(v), 2)
                           for k, v in sorted(por_no.items())},
            # rota POR ITEM, nao so o agregado: permite cruzar com a classe real do
            # item depois e separar "corpus nao tinha" de "roteador errou o destino"
            "rotas": rotas,
            "seg_p50": statistics.median(lat),
            "seg_p95": lat[min(int(len(lat) * 0.95), len(lat) - 1)],
            "taxa_ressalva": len(com_ressalva) / n if n else 0.0,
            "acerto_com_ressalva": (sum(com_ressalva) / len(com_ressalva)
                                    if com_ressalva else None),
            "acerto_sem_ressalva": (sum(sem_ressalva) / len(sem_ressalva)
                                    if sem_ressalva else None),
        },
        "itens": acertos,
        "classes": {it["id"]: it.get("classe", "?") for it in itens},
    }


def item_gabarito(item: dict) -> str:
    return (item["resposta"] or "").strip().upper()[:1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grafos", default="v0,v2", help="lista separada por virgula")
    ap.add_argument("--modelos", default="medgemma-clinical")
    ap.add_argument("--ancoragem", default="estrita",
                    choices=("estrita", "com_ressalva", "confiante", "analista", "analista_leve", "bib_so", "falsificacao"),
                    help="estrita = so o contexto; com_ressalva = responde sempre, "
                         "registrando a lacuna quando o contexto nao decide")
    ap.add_argument("--raciocinio", default="nenhum",
                    help="nenhum | prompt | nativo (nativo so no gemma4-plan)")
    ap.add_argument("--bench", default="vf", choices=("vf", "mcq", "healthqa"))
    ap.add_argument("--split", default="dev", choices=("dev", "teste"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--semente", type=int, default=1)
    ap.add_argument("--estratificar", action="store_true",
                    help="cota igual por classe, em vez de sorteio simples")
    # m10 = Miller's Anesthesia 10e, o corpus que mede 81,1%. O padrao era h512
    # (Bases da Anestesia, 3.406 trechos) — o livro REFUTADO nesta bancada: manter
    # aquele padrao fazia uma corrida sem --indice medir o corpus errado em
    # silencio, e so o rotulo ix= no setup denunciava.
    ap.add_argument("--indice", type=Path, default=RAIZ / "dados" / "indice" / "m10")
    ap.add_argument("--k-busca", type=int, default=20)
    ap.add_argument("--k-contexto", type=int, default=5)
    ap.add_argument("--banco", type=Path, default=RAIZ / "eval" / "resultados.sqlite")
    ap.add_argument("--bloco", default="")
    ap.add_argument("--repeticao", type=int, default=1,
                    help="rode 2x a MESMA celula para um A/A antes de acreditar em A/B")
    a = ap.parse_args()

    if a.split == "teste":
        print("ATENCAO: rodando no conjunto de TESTE. So depois de escolher a "
              "configuracao no dev.", file=sys.stderr)

    itens = carregar(a.bench, a.split, a.n, a.semente, a.estratificar)
    if not itens:
        print("nenhum item carregado", file=sys.stderr)
        return 1

    a.banco.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(a.banco)
    db.executescript(ESQUEMA)

    # agrupado por MODELO no laco externo: um swap de 18 GB por modelo, nao por celula
    for modelo in a.modelos.split(","):
        for grafo in a.grafos.split(","):
            ix = None if grafo == "v0" else Indice(a.indice)
            if ix is not None:
                # Dizer em voz alta QUAL corpus vai ser medido. O rotulo ix= no setup ja
                # registrava isso, mas so depois da corrida — e uma corrida contra o
                # corpus errado leva 100 min para revelar que nao media nada.
                livros = [r[0] for r in ix.db.execute(
                    "SELECT DISTINCT livro FROM chunk WHERE livro IS NOT NULL LIMIT 5")]
                print(f"  corpus: {a.indice.name} · {ix.quantos('filho')} trechos · "
                      f"{livros}", file=sys.stderr)
            for rep in range(1, a.repeticao + 1):
                setup = (f"{grafo}|anc={a.ancoragem}|rac={a.raciocinio}|ix={a.indice.name}"
                         f"|k={a.k_busca}/{a.k_contexto}"
                         + (f"|rep={rep}" if a.repeticao > 1 else ""))
                print(f"\n[{modelo}] {setup} · {a.bench}/{a.split} n={len(itens)}",
                      file=sys.stderr)
                t0 = time.time()
                res = rodar_celula(grafo, itens, modelo=modelo,
                                   raciocinio=a.raciocinio, indice=ix,
                                   k_busca=a.k_busca, k_contexto=a.k_contexto,
                                   ancoragem=a.ancoragem)
                db.execute(
                    "INSERT INTO resultado (modelo,setup,bench,quando,n,metricas,itens,"
                    "termica,segundos,bloco) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (modelo, setup, f"{a.bench}-{a.split}",
                     datetime.now().isoformat(timespec="seconds"),
                     res["metricas"]["n"], json.dumps(res["metricas"]),
                     json.dumps(res["itens"]), None, time.time() - t0, a.bloco))
                db.commit()
                m = res["metricas"]
                if m.get("taxa_ressalva"):
                    # v0 nao tem contexto, entao ressalva em 100% dos itens e o grupo
                    # "sem ressalva" fica VAZIO — formatar None estourava aqui, depois
                    # do commit, matando o resto do bloco.
                    pct = lambda x: "—" if x is None else f"{x:.1%}"
                    print(f"  ressalva {m['taxa_ressalva']:.1%} | acerto com "
                          f"{pct(m['acerto_com_ressalva'])} sem "
                          f"{pct(m['acerto_sem_ressalva'])}", file=sys.stderr)
                print(f"  acerto {m['acerto']:.1%} | truncou {m['truncou']:.1%} | "
                      f"fidelidade {m['fidelidade']:.1%} | p50 {m['seg_p50']:.1f}s",
                      file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
