#!/usr/bin/env python3
"""
O roteador acertou o destino? Cruza a rota GRAVADA com a classe real do item.

Existe porque um braco roteado que nao melhora tem duas causas opostas, e o placar
sozinho nao as separa:

  - o corpus normativo nao tem a resposta      -> trocar/ampliar o corpus
  - o roteador mandou a pergunta ao indice errado -> mexer no roteador

Sem este cruzamento, as duas parecem "o RAG de normas nao ajudou", e a correcao seria
chutada. As metricas `rotas` do runner guardam a rota por item exatamente para isto.

    ./roteamento.py --bloco ROTEADO
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
NORMATIVAS = {"juridico-normativo", "gestao"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bloco", default="ROTEADO")
    ap.add_argument("--base", default="COMPLETO-m10",
                    help="bloco com o braco de comparacao (v1)")
    a = ap.parse_args()

    dev = {i["id"]: i for i in (json.loads(l) for l in
           open(RAIZ / "dados" / "questoes_sba.dev.jsonl"))}
    db = sqlite3.connect(RAIZ / "eval" / "resultados.sqlite")
    db.row_factory = sqlite3.Row

    rotas: dict[str, str] = {}
    acertos: dict[str, bool] = {}
    for r in db.execute("SELECT setup, metricas, itens FROM resultado WHERE bloco=?",
                        (a.bloco,)):
        m = json.loads(r["metricas"])
        rotas.update(m.get("rotas") or {})
        acertos.update(json.loads(r["itens"]))
    if not rotas:
        print(f"nenhuma rota gravada no bloco {a.bloco!r} — a corrida terminou?",
              file=sys.stderr)
        return 1

    base: dict[str, bool] = {}
    for r in db.execute("SELECT setup, itens FROM resultado WHERE bloco=? ", (a.base,)):
        if r["setup"].split("|")[0] == "v1":
            base.update(json.loads(r["itens"]))

    # ---- o roteador acertou o destino?
    vp = fp = fn = vn = 0
    for iid, rota in rotas.items():
        real = (dev.get(iid) or {}).get("classe")
        foi = rota == "normas"
        devia = real in NORMATIVAS
        vp += foi and devia
        fp += foi and not devia
        fn += (not foi) and devia
        vn += (not foi) and not devia
    prec = vp / (vp + fp) if vp + fp else 0.0
    rec = vp / (vp + fn) if vp + fn else 0.0
    print(f"ROTEAMENTO · {len(rotas)} itens\n")
    print(f"  mandadas ao corpus normativo: {vp + fp}")
    print(f"    corretamente (classe normativa/gestão): {vp}")
    print(f"    indevidamente (questão clínica):        {fp}")
    print(f"  normativas que NÃO foram para lá:         {fn}")
    print(f"\n  precisão {prec:.1%} · cobertura {rec:.1%}")
    if fn:
        print(f"  -> {fn} questões normativas foram buscar no Miller, que não as cobre")
    if fp:
        print(f"  -> {fp} questões clínicas foram buscar em resolução do CFM")

    # ---- acerto por (rota x classe real), contra o v1
    print("\nACERTO POR DESTINO (v2 roteado vs v1 no Miller)\n")
    print(f"  {'destino':10s} {'classe real':22s} {'n':>4s} {'v2':>7s} {'v1':>7s} {'dif':>7s}")
    g = collections.defaultdict(list)
    for iid, rota in rotas.items():
        real = (dev.get(iid) or {}).get("classe") or "?"
        g[(rota, "normativa" if real in NORMATIVAS else "clínica")].append(iid)
    for (rota, grupo), itens in sorted(g.items()):
        com = [i for i in itens if i in acertos]
        par = [i for i in com if i in base]
        if not com:
            continue
        a2 = sum(acertos[i] for i in com) / len(com)
        a1 = sum(base[i] for i in par) / len(par) if par else float("nan")
        print(f"  {rota:10s} {grupo:22s} {len(com):4d} {a2:6.1%} {a1:6.1%} "
              f"{a2 - a1:+6.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
