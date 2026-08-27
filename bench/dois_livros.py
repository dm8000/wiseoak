#!/usr/bin/env python3
"""
Com e sem RAG, nas questoes que TODOS os cinco fluxos erraram — agora com dois livros.

O corpus passou de um livro introdutorio em portugues (Miller, Bases da Anestesia, scan
com OCR) para dois: mais o Barash, Clinical Anesthesia, nativo digital, 3.229 paginas,
em INGLES. E o teste mais direto da hipotese "o modelo erra porque o livro nao tem".

Exclui `juridico-normativo`: nenhum livro-texto americano cobre CFM, e o proprio usuario
apontou que ali o RAG so pode atrapalhar. Medir junto so diluiria.

BUSCA DENSA, nao hibrida. O BM25 nao casa portugues com ingles — vocabulario disjunto —
entao no Barash ele so devolveria ruido, e o RRF daria a esse ruido metade do peso.

    ./dois_livros.py --indice dados/indice/dois
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
ANALISES = RAIZ / "eval" / "analises"
ANALISES.mkdir(parents=True, exist_ok=True)

from wiseoak.grafos.comum import verificar_citacoes  # noqa: E402
from wiseoak.grafos.variantes import construir, estado_inicial  # noqa: E402
from wiseoak.store import Indice  # noqa: E402


def alvos() -> list[dict]:
    dev = {i["id"]: i for i in (json.loads(l) for l in
           open(RAIZ / "dados" / "questoes_sba.dev.jsonl"))}
    db = sqlite3.connect(RAIZ / "eval" / "resultados.sqlite")
    db.row_factory = sqlite3.Row
    cels = {r["setup"].split("|")[0]: json.loads(r["itens"]) for r in
            db.execute("SELECT setup,itens FROM resultado WHERE bloco='B-arquitetura'")}
    amostra = list(next(iter(cels.values())))
    todos = [k for k in amostra
             if all(not cels[g].get(k, True) for g in cels) and k in dev]
    return [dev[k] for k in todos if dev[k].get("classe") != "juridico-normativo"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indice", default="dados/indice/dois")
    ap.add_argument("--modelo", default="gemma4-plan")
    ap.add_argument("--k-busca", type=int, default=8)
    ap.add_argument("--k-contexto", type=int, default=4)
    ap.add_argument("--ancoragem", default="confiante")
    ap.add_argument("--grafo", default="v1",
                    help="o braco COM rag (o braco sem rag e sempre v0)")
    ap.add_argument("--saida", type=Path, default=ANALISES / "dois-livros.txt")
    a = ap.parse_args()

    itens = alvos()
    ix = Indice(a.indice)
    print(f"{len(itens)} itens que TODOS os fluxos erraram (sem os jurídicos)",
          file=sys.stderr)
    print(f"índice: {a.indice} · {ix.quantos('filho')} trechos", file=sys.stderr)

    g0, g1 = construir("v0"), construir(a.grafo)
    linhas, res = [], []
    t0 = time.time()
    for n, it in enumerate(itens, 1):
        q = f"{it['enunciado']}\n\nAssertiva: {it['assertiva']}"
        base = dict(modo="vf", modelo=a.modelo, ancoragem=a.ancoragem)
        r0 = g0.invoke(estado_inicial(q, **base))
        r1 = g1.invoke(estado_inicial(q, indice=ix, k_busca=a.k_busca,
                                      k_contexto=a.k_contexto, **base))
        ok0 = (r0.get("resposta") or "").strip().upper()[:1] == it["resposta"]
        ok1 = (r1.get("resposta") or "").strip().upper()[:1] == it["resposta"]
        livros = collections.Counter(
            (ix.obter(c)["livro"] or "?") for c in (r1.get("candidatos") or [])[:a.k_contexto])
        res.append({"id": it["id"], "classe": it["classe"], "sem": ok0, "com": ok1,
                    "livros": dict(livros)})
        v = verificar_citacoes(r1)
        linhas += [
            "-" * 78,
            f"[{n}] {it['fonte'][:34]} · {it['classe']}",
            f"  ASSERTIVA: {' '.join(it['assertiva'].split())[:230]}",
            f"  GABARITO {it['resposta']} | sem RAG: {r0.get('resposta')} "
            f"{'OK' if ok0 else 'erro'} | com RAG: {r1.get('resposta')} "
            f"{'OK' if ok1 else 'erro'}",
            f"  livros no contexto: {dict(livros)}"
            + (f"\n  erro apontado: {' '.join((r1.get('erro_apontado') or '—').split())[:170]}"
               if a.ancoragem == "falsificacao" else ""),
        ]
        for f in (r1.get("fontes") or [])[:3]:
            linhas.append(f"     {f}")
        linhas.append(f"  citações: {v['fieis']}/{v['citacoes']} literais")
        sys.stderr.write(f"\r  {n}/{len(itens)}  sem={sum(x['sem'] for x in res)} "
                         f"com={sum(x['com'] for x in res)}")
    sys.stderr.write("\n")

    n = len(res)
    sem = sum(x["sem"] for x in res)
    com = sum(x["com"] for x in res)
    tab = collections.Counter((x["sem"], x["com"]) for x in res)
    cab = [
        f"RESULTADO · grafo {a.grafo} · ancoragem {a.ancoragem} · "
        f"questões que todos os fluxos erravam",
        f"corpus: dois livros · {ix.quantos('filho')} trechos · busca densa · "
        f"k={a.k_busca}/{a.k_contexto}",
        "",
        f"  n = {n}",
        f"  SEM RAG : {sem}/{n} = {sem/n:.1%}",
        f"  COM RAG : {com}/{n} = {com/n:.1%}   ({com-sem:+d} itens, {(com-sem)/n:+.1%})",
        "",
        f"  só sem RAG acertou : {tab[(True, False)]}",
        f"  só com RAG acertou : {tab[(False, True)]}",
        f"  os dois acertaram  : {tab[(True, True)]}",
        f"  os dois erraram    : {tab[(False, False)]}",
        "",
        "por classe:",
    ]
    porc = collections.defaultdict(list)
    for x in res:
        porc[x["classe"]].append(x)
    for c, xs in sorted(porc.items(), key=lambda kv: -len(kv[1])):
        cab.append(f"  {c:22s} n={len(xs):3d}  sem {sum(x['sem'] for x in xs)/len(xs):5.1%}"
                   f"  com {sum(x['com'] for x in xs)/len(xs):5.1%}")
    livros = collections.Counter()
    for x in res:
        livros.update(x["livros"])
    cab += ["", f"de onde veio o contexto: {dict(livros)}",
            f"tempo: {(time.time()-t0)/60:.0f} min", ""]
    a.saida.write_text("\n".join(cab + linhas))
    print("\n".join(cab))
    print(f"detalhe em {a.saida}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
