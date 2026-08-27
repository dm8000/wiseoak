#!/usr/bin/env python3
"""
Leitura dos resultados. Uma tabela por modelo, ganho PAREADO com significancia.

Herda as convencoes de Cloud_assistant/eval/relatorio.py, que existem porque cada uma
custou um erro naquele projeto:

  - truncagem em COLUNA SEPARADA, nunca somada ao acerto;
  - comparacao PAREADA (McNemar) entre celulas, nao diferenca de duas taxas soltas: as
    mesmas questoes rodaram nas duas, e ignorar isso joga fora a informacao que mais
    discrimina;
  - o erro padrao da diferenca pareada usa SO os discordantes:
        SE = sqrt(b + c - (c-b)^2/n) / n
    Nao e o desvio das duas taxas. Confundir os dois infla o intervalo e esconde efeito.
  - correcao de Holm-Bonferroni sobre TODAS as comparacoes da rodada. Sem ela, 20
    comparacoes produzem uma "significativa" por acaso.

    ./relatorio.py                       tudo
    ./relatorio.py --bench vf-dev --aa   so o A/A
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    meio = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centro - meio), min(1.0, centro + meio))


def mcnemar(b: int, c: int) -> float:
    """Binomial exato bicaudal sobre os discordantes. b e c sao as discordancias."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cauda = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * cauda)


def holm(pvalores: dict[str, float]) -> dict[str, tuple[float, float, bool]]:
    itens = sorted(pvalores.items(), key=lambda kv: kv[1])
    m = len(itens)
    saida: dict[str, tuple[float, float, bool]] = {}
    maior = 0.0
    for i, (chave, p) in enumerate(itens):
        ajustado = max(maior, min(1.0, (m - i) * p))  # monotonicidade
        maior = ajustado
        saida[chave] = (p, ajustado, ajustado < 0.05)
    return saida


def pareado(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, int]:
    """Devolve (n_comum, b_so_a_acertou, c_so_b_acertou)."""
    comuns = set(a) & set(b)
    so_a = sum(1 for k in comuns if a[k] and not b[k])
    so_b = sum(1 for k in comuns if b[k] and not a[k])
    return len(comuns), so_a, so_b


def ganho_pareado(a: dict[str, bool], b: dict[str, bool]) -> tuple[float, float, float]:
    """(ganho de b sobre a, erro padrao, p de McNemar)."""
    n, so_a, so_b = pareado(a, b)
    if n == 0:
        return (0.0, 0.0, 1.0)
    ganho = (so_b - so_a) / n
    var = so_a + so_b - (so_b - so_a) ** 2 / n
    se = math.sqrt(max(var, 0.0)) / n
    return (ganho, se, mcnemar(so_a, so_b))


def carregar(banco: Path, bench: str | None) -> list[dict]:
    db = sqlite3.connect(banco)
    db.row_factory = sqlite3.Row
    sql = "SELECT * FROM resultado"
    args: tuple = ()
    if bench:
        sql += " WHERE bench=?"
        args = (bench,)
    linhas = []
    for r in db.execute(sql + " ORDER BY modelo, setup, quando", args):
        d = dict(r)
        d["metricas"] = json.loads(d["metricas"])
        d["itens"] = json.loads(d["itens"] or "{}")
        linhas.append(d)
    return linhas


def tabela(linhas: list[dict]) -> str:
    saida = []
    por_modelo = defaultdict(list)
    for l in linhas:
        por_modelo[l["modelo"]].append(l)

    for modelo, cels in sorted(por_modelo.items()):
        saida.append(f"\n## {modelo}\n")
        saida.append("| setup | n | acerto | IC 95% | truncou | fidelidade | pág. ok | p50 s |")
        saida.append("|---|---:|---:|---|---:|---:|---:|---:|")
        for c in cels:
            m = c["metricas"]
            k = round(m["acerto"] * m["n"])
            lo, hi = wilson(k, m["n"])
            saida.append(
                f"| {c['setup']} | {m['n']} | {m['acerto']:.1%} | "
                f"{lo:.1%}–{hi:.1%} | {m['truncou']:.1%} | {m['fidelidade']:.1%} | "
                f"{m.get('pagina_ok', 0):.1%} | {m['seg_p50']:.1f} |")

        # comparacoes pareadas contra a PRIMEIRA celula do modelo (normalmente v0)
        base = cels[0]
        pvals, ganhos = {}, {}
        for c in cels[1:]:
            g, se, p = ganho_pareado(base["itens"], c["itens"])
            pvals[c["setup"]] = p
            ganhos[c["setup"]] = (g, se)
        if pvals:
            ajust = holm(pvals)
            saida.append(f"\nGanho pareado sobre `{base['setup']}` (McNemar + Holm):\n")
            saida.append("| setup | ganho | ±dp | p bruto | p corrigido | signif. |")
            saida.append("|---|---:|---:|---:|---:|:--:|")
            for setup, (g, se) in ganhos.items():
                p, pc, sig = ajust[setup]
                saida.append(f"| {setup} | {g:+.1%} | {se:.1%} | {p:.4f} | {pc:.4f} | "
                             f"{'sim' if sig else 'não'} |")
    return "\n".join(saida)


def relatorio_aa(linhas: list[dict]) -> str:
    """
    A/A: mesma celula, duas execucoes. Diferenca aqui e RUIDO, e nenhum A/B menor que
    ela significa coisa alguma. Este projeto declarou este experimento urgente e nunca
    o fez — sem ele, parte de todo "efeito" e temperatura.
    """
    grupos = defaultdict(list)
    for l in linhas:
        chave = (l["modelo"], l["setup"].split("|rep=")[0], l["bench"])
        grupos[chave].append(l)
    pares = {k: v for k, v in grupos.items() if len(v) >= 2}
    if not pares:
        return "\nNenhum A/A no banco. Rode com --repeticao 2 antes de comparar nada.\n"

    saida = ["\n## A/A — o piso de ruído\n",
             "| modelo | setup | acerto 1 | acerto 2 | diferença | discordantes | p |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for (modelo, setup, _), v in sorted(pares.items()):
        a, b = v[0], v[1]
        n, so_a, so_b = pareado(a["itens"], b["itens"])
        p = mcnemar(so_a, so_b)
        saida.append(
            f"| {modelo} | {setup} | {a['metricas']['acerto']:.1%} | "
            f"{b['metricas']['acerto']:.1%} | "
            f"{b['metricas']['acerto'] - a['metricas']['acerto']:+.1%} | "
            f"{so_a + so_b}/{n} | {p:.3f} |")
    saida.append("\n**Leia assim:** qualquer ganho de A/B menor que a diferença acima "
                 "está dentro do ruído, por mais bonito que seja o número.\n")
    return "\n".join(saida)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--banco", type=Path, default=RAIZ / "eval" / "resultados.sqlite")
    ap.add_argument("--bench", default=None)
    ap.add_argument("--aa", action="store_true", help="só o relatório de A/A")
    ap.add_argument("--md", type=Path, default=None)
    a = ap.parse_args()

    if not a.banco.exists():
        print(f"banco nao existe: {a.banco}", file=sys.stderr)
        return 1
    linhas = carregar(a.banco, a.bench)
    if not linhas:
        print("nenhum resultado no banco", file=sys.stderr)
        return 1

    texto = relatorio_aa(linhas) if a.aa else (relatorio_aa(linhas) + tabela(linhas))
    cab = (f"# WiseOak — resultados\n\n{len(linhas)} células"
           + (f" · bench `{a.bench}`" if a.bench else "") + "\n")
    if a.md:
        a.md.write_text(cab + texto + "\n")
        print(f"escrito em {a.md}", file=sys.stderr)
    else:
        print(cab + texto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
