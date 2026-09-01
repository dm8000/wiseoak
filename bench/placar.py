#!/usr/bin/env python3
"""
Placar unico: todo sistema medido neste projeto, no MESMO universo de questoes.

Junta duas origens que ate agora viviam separadas:
  - os bracos do WiseOak, em `eval/resultados.sqlite` (item -> acertou)
  - os modelos externos, em `dados/prova/*.txt` (respostas soltas, conferidas no gabarito)

O universo e a INTERSECAO dos itens que todos responderam. Nao e detalhe: comparar 85%
de mil questoes com 80% de outras novecentas nao compara nada, e McNemar so existe sobre
pares. Item que um sistema deixou em branco sai do universo de todos, e o n cai junto —
"nao respondeu" nao vira "errou".

    ./placar.py                          # todos os sistemas conhecidos
    ./placar.py --bloco ROTEADO          # inclui um bloco novo da bancada
    ./placar.py --saida eval/analises/placar.txt
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
ANALISES = RAIZ / "eval" / "analises"

_LINHA = re.compile(r"\bq0*(\d{1,4})\s*[:.\-–]\s*([VFABCDvfabcd])\b")

# Modelos externos: arquivo em dados/prova -> como aparece no placar.
EXTERNOS = {
    "opus5_no_search.txt":                  "Opus 5 (effort low, sem busca)",
    "DeepSeek-V3-0324_thinking_search.txt": "DeepSeek-V3 (thinking+search)",
    "DeepSeek-V3-0324_thinking.txt":        "DeepSeek-V3 (thinking)",
}

# Bracos do WiseOak: (bloco, prefixo do setup) -> nome no placar.
INTERNOS = {
    ("COMPLETO-m10", "v0"): "WiseOak v0 (Gemma4 31B sem RAG)",
    ("COMPLETO-m10", "v1"): "WiseOak v1 (+ RAG Miller 10e)",
    ("ROTEADO",      "v8"): "WiseOak v2 (+ RAG normas, roteado)",
    ("COTA",         "v10"): "WiseOak v3 (cota: normas + livro)",
    ("V4",           "v10"): "WiseOak v4 (cota + falsificação + k=8)",
    ("V5",           "v13"): "WiseOak v5 (consulta focada + trecho pequeno)",
    ("V7",           "v15"): "WiseOak v7 (modelo escreve a própria consulta)",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def mcnemar(b: int, c: int) -> float:
    """
    Exato (binomial bicaudal) sobre os DISCORDANTES. Item que os dois acertaram ou os
    dois erraram nao carrega informacao sobre qual e melhor, e entrar com ele so encolhe
    o p artificialmente.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cauda = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * cauda)


def carregar_gabarito() -> dict:
    return json.loads((RAIZ / "dados" / "prova" / "gabarito-dev.json").read_text())


def acertos_externos(gab: dict) -> dict[str, dict[str, bool]]:
    fora = {}
    for arquivo, nome in EXTERNOS.items():
        p = RAIZ / "dados" / "prova" / arquivo
        if not p.exists():
            continue
        dadas = {f"q{int(m.group(1)):04d}": m.group(2).upper()
                 for m in _LINHA.finditer(p.read_text())}
        fora[nome] = {gab["chave"][rot]: (v == gab["gabarito"][rot])
                      for rot, v in dadas.items() if rot in gab["chave"]}
    return fora


def acertos_internos(blocos: list[str]) -> dict[str, dict[str, bool]]:
    db = sqlite3.connect(RAIZ / "eval" / "resultados.sqlite")
    db.row_factory = sqlite3.Row
    dentro: dict[str, dict[str, bool]] = {}
    for r in db.execute("SELECT bloco, setup, itens FROM resultado"):
        chave = (r["bloco"], r["setup"].split("|")[0])
        nome = INTERNOS.get(chave)
        if nome is None or r["bloco"] not in blocos:
            continue
        # vf e mcq sao duas celulas do mesmo braco: juntam-se num mapa so
        dentro.setdefault(nome, {}).update(json.loads(r["itens"]))
    return dentro


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bloco", action="append", default=None,
                    help="blocos da bancada a incluir (repetivel)")
    ap.add_argument("--saida", type=Path, default=ANALISES / "placar.txt")
    a = ap.parse_args()
    blocos = a.bloco or ["COMPLETO-m10", "ROTEADO", "COTA", "V4", "V5", "V7"]

    gab = carregar_gabarito()
    sistemas = {**acertos_internos(blocos), **acertos_externos(gab)}
    if len(sistemas) < 2:
        print("preciso de pelo menos dois sistemas medidos", file=sys.stderr)
        return 1

    universo = set.intersection(*(set(v) for v in sistemas.values()))
    porid_tipo = {gab["chave"][r]: t for r, t in gab["tipo"].items() if r in gab["chave"]}
    porid_classe = {gab["chave"][r]: c for r, c in gab["classe"].items() if r in gab["chave"]}

    ordem = sorted(sistemas,
                   key=lambda s: -sum(sistemas[s][i] for i in universo))

    def pct(x: float) -> str:
        return f"{x:.1%}".replace(".", ",")

    def pval(x: float) -> str:
        return ("<0,000001" if x < 1e-6 else
                ("1,000" if x >= 0.9995 else f"{x:.6f}".rstrip("0").rstrip(".").replace(".", ",")))

    L: list[str] = []
    L.append(f"Comparação final, {len(universo)} questões, mesmo universo:")
    L.append("")
    L.append("| sistema | total | V/F | ME | IC 95% |")
    L.append("|---|---:|---:|---:|---:|")
    vf = [i for i in universo if porid_tipo.get(i) == "vf"]
    me = [i for i in universo if porid_tipo.get(i) == "mcq"]
    for s in ordem:
        m = sistemas[s]
        ok = sum(m[i] for i in universo)
        lo, hi = wilson(ok, len(universo))
        pvf = sum(m[i] for i in vf) / len(vf) if vf else 0
        pme = sum(m[i] for i in me) / len(me) if me else 0
        L.append(f"| {s} | {pct(ok/len(universo))} | {pct(pvf)} | {pct(pme)} | "
                 f"{pct(lo)}–{pct(hi)} |")

    # ---- pareado
    pares = []
    for i, x in enumerate(ordem):
        for y in ordem[i + 1:]:
            b = sum(1 for k in universo if sistemas[x][k] and not sistemas[y][k])
            c = sum(1 for k in universo if sistemas[y][k] and not sistemas[x][k])
            pares.append([x, y, (b - c) / len(universo), mcnemar(b, c), b, c])
    pares.sort(key=lambda r: r[3])
    m = len(pares)
    anterior = 0.0
    for j, r in enumerate(pares):
        holm = min(1.0, max(anterior, (m - j) * r[3]))   # Holm e monotono
        anterior = holm
        r.append(holm)
    passam = sum(1 for r in pares if r[6] < 0.05)
    NUM = {1:"uma",2:"duas",3:"três",4:"quatro",5:"cinco",6:"seis",7:"sete",8:"oito",
           9:"nove",10:"dez",11:"onze",12:"doze",13:"treze",14:"catorze",15:"quinze"}
    L += ["", f"McNemar com Holm — {NUM.get(passam, passam)} das {NUM.get(m, m)} "
              f"comparações passam:", "",
          "| comparação | dif | Holm | signif. |", "|---|---:|---:|:--:|"]
    for x, y, dif, pv, b, c, holm in sorted(pares, key=lambda r: -abs(r[2])):
        L.append(f"| {x} > {y} | {('+' if dif >= 0 else '')}{pct(dif)} | "
                 f"{pval(holm)} | {'sim' if holm < 0.05 else 'não'} |")

    # ---- por classe
    porc = collections.defaultdict(list)
    for i in universo:
        porc[porid_classe.get(i) or "?"].append(i)
    L += ["", "Por classe:", "",
          "| classe | n | " + " | ".join(ordem) + " |",
          "|---|---:|" + "---:|" * len(ordem)]
    for c, itens in sorted(porc.items(), key=lambda kv: -len(kv[1])):
        cel = " | ".join(pct(sum(sistemas[s][i] for i in itens)/len(itens)) for s in ordem)
        L.append(f"| {c} | {len(itens)} | {cel} |")

    texto = "\n".join(L)
    ANALISES.mkdir(parents=True, exist_ok=True)
    a.saida.write_text(texto + "\n")
    print(texto)
    print(f"\n-> {a.saida}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
