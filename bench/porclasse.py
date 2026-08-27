#!/usr/bin/env python3
"""
Compara DOIS braços classe a classe, pareado, com o destino de roteamento de cada uma.

Escrito depois de rodar a mesma analise a mao duas vezes. Serve porque neste banco o
total nao decide nada: `juridico-normativo` e `gestao` somam 96 de 998 itens, e um ganho
grande neles vale ~1 pp no agregado — abaixo do piso de ruido medido (3,9% dos itens
trocam de resposta entre execucoes identicas, por `temp=0.3`).

AVISO que o proprio relatorio imprime: uma classe roteada ao LIVRO em ambos os bracos
executa pipeline identico, entao qualquer diferenca ali e ruido — inclusive quando o p
sai pequeno. Foi assim que `tecnica` deu p=0,022 sem existir efeito nenhum.

    ./porclasse.py --a COMPLETO-m10:v1 --b COTA:v10
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sqlite3
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def carregar(db, spec: str) -> tuple[dict, dict]:
    bloco, _, grafo = spec.partition(":")
    itens, rotas = {}, {}
    for r in db.execute("SELECT setup, metricas, itens FROM resultado WHERE bloco=?",
                        (bloco,)):
        if grafo and r["setup"].split("|")[0] != grafo:
            continue
        itens.update(json.loads(r["itens"]))
        rotas.update(json.loads(r["metricas"]).get("rotas") or {})
    return itens, rotas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="COMPLETO-m10:v1", help="bloco:grafo de referência")
    ap.add_argument("--b", required=True, help="bloco:grafo a comparar")
    args = ap.parse_args()

    dev = {i["id"]: i for i in (json.loads(l) for l in
           open(RAIZ / "dados" / "questoes_sba.dev.jsonl"))}
    db = sqlite3.connect(RAIZ / "eval" / "resultados.sqlite")
    db.row_factory = sqlite3.Row
    A, _ = carregar(db, args.a)
    B, rotas = carregar(db, args.b)
    com = [i for i in B if i in A and i in dev]
    if not com:
        print("sem itens em comum — a corrida terminou?", file=sys.stderr)
        return 1

    print(f"  {args.a}  ->  {args.b}   ·  {len(com)} itens pareados")
    ta, tb = sum(A[i] for i in com) / len(com), sum(B[i] for i in com) / len(com)
    print(f"  TOTAL  {ta:.1%} -> {tb:.1%}  ({(tb-ta)*100:+.1f} pp)\n")
    print(f"  {'classe':22s} {'destino':8s} {'n':>4s} {'A':>7s} {'B':>7s} {'dif':>8s} "
          f"{'só B':>5s} {'só A':>5s} {'p':>7s}")
    g = collections.defaultdict(list)
    for i in com:
        g[dev[i]["classe"]].append(i)
    aviso = False
    for c, ids in sorted(g.items(), key=lambda kv: -len(kv[1])):
        dest = collections.Counter(rotas.get(i, "?") for i in ids).most_common(1)[0][0]
        b = sum(1 for i in ids if B[i] and not A[i])
        cc = sum(1 for i in ids if A[i] and not B[i])
        pa, pb = sum(A[i] for i in ids) / len(ids), sum(B[i] for i in ids) / len(ids)
        p = mcnemar(b, cc)
        marca = ""
        if dest == "livro" and p < 0.05:
            marca = "  <- RUÍDO (pipeline idêntico)"
            aviso = True
        print(f"  {c:22s} {dest:8s} {len(ids):4d} {pa:6.1%} {pb:6.1%} "
              f"{(pb-pa)*100:+7.1f}pp {b:5d} {cc:5d} {p:7.4f}{marca}")
    if aviso:
        print("\n  Uma classe roteada ao livro em ambos os braços executa o MESMO "
              "pipeline.\n  p pequeno ali é falso positivo de amostragem, não efeito.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
