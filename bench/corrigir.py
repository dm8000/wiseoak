#!/usr/bin/env python3
"""
Corrige as respostas devolvidas por outro modelo e compara com os nossos numeros.

Le um arquivo de texto solto com linhas `q0001: V`. Tolerante de proposito: aceita
maiusculas ou minusculas, com ou sem espaco, com ponto ou tracco no lugar dos dois
pontos, e ignora qualquer prosa em volta. O que NAO faz e adivinhar item faltante —
questao sem resposta e reportada como ausente, nao como erro, para nao confundir
"errou" com "nao respondeu".

    ./corrigir.py respostas.txt
    ./corrigir.py respostas.txt --nome "Opus 5"
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# q0001: V   ·   q0001 - F   ·   Q0001. C   ·   q1: v
_LINHA = re.compile(r"\bq0*(\d{1,4})\s*[:.\-–]\s*([VFABCDvfabcd])\b")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("respostas", type=Path)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--nome", default="modelo externo")
    a = ap.parse_args()

    gab = json.loads((RAIZ / "dados" / "prova" / f"gabarito-{a.split}.json").read_text())
    esperado, tipo, classe = gab["gabarito"], gab["tipo"], gab["classe"]

    dadas: dict[str, str] = {}
    for m in _LINHA.finditer(a.respostas.read_text()):
        dadas[f"q{int(m.group(1)):04d}"] = m.group(2).upper()

    faltando = [k for k in esperado if k not in dadas]
    extras = [k for k in dadas if k not in esperado]
    invalidas = [k for k, v in dadas.items()
                 if k in tipo and ((tipo[k] == "vf" and v not in "VF")
                                   or (tipo[k] == "mcq" and v not in "ABCD"))]

    print(f"=== {a.nome} · split {a.split} ===")
    print(f"  esperadas {len(esperado)} · respondidas {len(dadas)} · "
          f"faltando {len(faltando)} · fora do gabarito {len(extras)} · "
          f"formato invalido {len(invalidas)}")
    if faltando[:6]:
        print(f"  ausentes (primeiras): {faltando[:6]}")

    por_tipo = collections.defaultdict(lambda: [0, 0])
    por_classe = collections.defaultdict(lambda: [0, 0])
    for k, certo in esperado.items():
        if k not in dadas:
            continue
        ok = dadas[k] == certo
        for d, chave in ((por_tipo, tipo[k]), (por_classe, classe.get(k) or "?")):
            d[chave][0] += ok
            d[chave][1] += 1

    print(f"\n  {'formato':10s} {'n':>5s} {'acerto':>8s}  IC 95%")
    for t, (ok, n) in sorted(por_tipo.items()):
        lo, hi = wilson(ok, n)
        print(f"  {t:10s} {n:5d} {ok/n:7.1%}   {lo:.1%}–{hi:.1%}")
    tot_ok = sum(v[0] for v in por_tipo.values())
    tot_n = sum(v[1] for v in por_tipo.values())
    lo, hi = wilson(tot_ok, tot_n)
    print(f"  {'TOTAL':10s} {tot_n:5d} {tot_ok/tot_n:7.1%}   {lo:.1%}–{hi:.1%}")

    print(f"\n  {'classe':22s} {'n':>5s} {'acerto':>8s}")
    for c, (ok, n) in sorted(por_classe.items(), key=lambda kv: -kv[1][1]):
        print(f"  {c:22s} {n:5d} {ok/n:7.1%}")

    # comparacao com o que ja medimos
    db = sqlite3.connect(RAIZ / "eval" / "resultados.sqlite")
    db.row_factory = sqlite3.Row
    nossos = {}
    for r in db.execute("SELECT setup,bench,metricas FROM resultado "
                        "WHERE bloco='COMPLETO-m10'"):
        nossos[(r["bench"], r["setup"].split("|")[0])] = json.loads(r["metricas"])
    print(f"\n  contra o WiseOak (gemma4-plan, corpus Miller 10e):")
    for bench, rotulo in (("vf-dev", "vf"), ("mcq-dev", "mcq")):
        ext = por_tipo.get(rotulo)
        if not ext:
            continue
        v0 = nossos.get((bench, "v0"))
        v1 = nossos.get((bench, "v1"))
        linha = f"  {rotulo:6s} {a.nome[:16]:16s} {ext[0]/ext[1]:6.1%}"
        if v0:
            linha += f"   sem RAG {v0['acerto']:6.1%}"
        if v1:
            linha += f"   com RAG {v1['acerto']:6.1%}"
        print(linha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
