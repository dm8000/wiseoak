#!/usr/bin/env python3
"""
Dossie de erro: cada questao que os bracos erraram, o CONTEXTO que o RAG entregou e a
resposta que o modelo deu.

Reexecuta os itens errados porque `eval/resultados.sqlite` guarda acerto/erro, nao texto.
Consequencia que o proprio dossie declara item a item: a geracao roda com `temp=0.3` e
3,9% dos itens trocam de resposta entre execucoes, entao a reexecucao pode DISCORDAR do
que foi gravado. Quando discorda, a linha vem marcada — nunca silenciada, porque um item
gravado como erro que agora acerta e informacao sobre o ruido, nao um erro de registro.

    ./dossie.py --arm ROTEADO:v8:v2 --arm COTA:v10:v3
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

from wiseoak.grafos.comum import verificar_citacoes  # noqa: E402
from wiseoak.grafos.variantes import construir, estado_inicial  # noqa: E402
from wiseoak.store import Indice  # noqa: E402


def item_gabarito(it: dict) -> str:
    return it["resposta"]


def enunciar(it: dict) -> str:
    p = [" ".join(it["enunciado"].split())]
    if it["tipo"] == "vf":
        p.append("ASSERTIVA: " + " ".join((it.get("assertiva") or "").split()))
    else:
        for letra, txt in sorted((it.get("alternativas") or {}).items()):
            p.append(f"  {letra}) " + " ".join(txt.split()))
    return "\n".join(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help="bloco:grafo:rotulo (repetivel)")
    ap.add_argument("--indice", default="dados/indice/m10")
    ap.add_argument("--modelo", default="gemma4-plan")
    ap.add_argument("--ancoragem", default="confiante")
    ap.add_argument("--k-busca", type=int, default=8)
    ap.add_argument("--k-contexto", type=int, default=4)
    ap.add_argument("--chars-trecho", type=int, default=700)
    ap.add_argument("--limite", type=int, default=0,
                    help="so os N primeiros; para teste de fumaca")
    ap.add_argument("--saida", type=Path, default=ANALISES / "dossie-erros.txt")
    a = ap.parse_args()

    dev = {i["id"]: i for i in (json.loads(l) for l in
           open(RAIZ / "dados" / "questoes_sba.dev.jsonl"))}
    db = sqlite3.connect(RAIZ / "eval" / "resultados.sqlite")
    db.row_factory = sqlite3.Row

    bracos: dict[str, dict] = {}
    for spec in a.arm:
        bloco, grafo, rot = spec.split(":")
        itens: dict[str, bool] = {}
        for r in db.execute("SELECT setup, itens FROM resultado WHERE bloco=?", (bloco,)):
            if r["setup"].split("|")[0] == grafo:
                itens.update(json.loads(r["itens"]))
        bracos[rot] = {"grafo": grafo, "itens": itens,
                       "erros": {i for i, ok in itens.items() if not ok}}
        print(f"  {rot} ({grafo}): {len(bracos[rot]['erros'])} erros", file=sys.stderr)

    # de comparacao: quem acertava antes
    antes: dict[str, dict] = {}
    for r in db.execute("SELECT setup, itens FROM resultado WHERE bloco='COMPLETO-m10'"):
        antes.setdefault(r["setup"].split("|")[0], {}).update(json.loads(r["itens"]))

    uniao = sorted(set().union(*(b["erros"] for b in bracos.values())),
                   key=lambda i: (dev[i]["classe"], i))
    if a.limite:
        uniao = uniao[:a.limite]
    print(f"  união: {len(uniao)} questões\n", file=sys.stderr)

    ix = Indice(a.indice)
    grafos = {rot: construir(b["grafo"]) for rot, b in bracos.items()}
    linhas: list[str] = []
    divergiu = 0
    t0 = time.time()

    for n, iid in enumerate(uniao, 1):
        it = dev[iid]
        gab = item_gabarito(it)
        linhas += ["=" * 90,
                   f"[{n}/{len(uniao)}]  {iid}  ·  {it['classe']}  ·  {it['tipo'].upper()}"
                   f"  ·  fonte: {it.get('fonte','?')[:40]}",
                   "",
                   enunciar(it), "",
                   f"GABARITO: {gab}"]
        for rot in ("v0", "v1"):
            if iid in antes.get(rot, {}):
                linhas.append(f"  {rot} (já medido): "
                              f"{'acertou' if antes[rot][iid] else 'errou'}")

        for rot, b in bracos.items():
            if iid not in b["erros"]:
                linhas.append(f"\n--- {rot}: ACERTOU (não é erro deste braço)")
                continue
            q = it["enunciado"] + ("\n\nAssertiva: " + it["assertiva"]
                                   if it["tipo"] == "vf" else "")
            if it["tipo"] == "mcq":
                q += "\n" + "\n".join(f"{k}) {v}" for k, v in
                                      sorted((it.get("alternativas") or {}).items()))
            r = grafos[rot].invoke(estado_inicial(
                q, modo=it["tipo"], modelo=a.modelo, ancoragem=a.ancoragem,
                indice=ix, k_busca=a.k_busca, k_contexto=a.k_contexto))
            dada = (r.get("resposta") or "").strip().upper()[:1]
            ok_agora = dada == gab
            if ok_agora:
                divergiu += 1
            rota = next((p.get("indice") for p in (r.get("trace") or [])
                         if p["no"] == "rotear"), None)
            linhas += ["", f"--- {rot}  ·  respondeu: {dada or '(vazio)'}"
                           + ("   [!] NA REEXECUÇÃO ACERTOU — gravado como erro; "
                              "é o ruído de temp=0.3" if ok_agora else "")
                           + (f"   ·  roteado para: {rota}" if rota else "")]
            if (r.get("ressalva") or "").strip():
                linhas.append(f"    ressalva: {' '.join(r['ressalva'].split())[:200]}")
            v = verificar_citacoes(r)
            linhas.append(f"    citações: {v['fieis']}/{v['citacoes']} conferem literalmente")
            linhas.append("    CONTEXTO QUE O RAG ENTREGOU:")
            for j, c in enumerate(r.get("contexto") or [], 1):
                nat = c.get("natureza")
                cab = (f"{c.get('livro','?')} · {c.get('capitulo')}" if nat == "NORMA"
                       else f"{c.get('livro','?')} · cap. {c.get('capitulo_num')}"
                            f" · p. {c.get('pagina_inicial')}")
                marca = f"[{nat}] " if nat else ""
                txt = " ".join(c["texto"].split())
                corte = txt[:a.chars_trecho]
                linhas.append(f"      ({j}) {marca}{cab}")
                linhas.append(f"          {corte}"
                              + (f" […mais {len(txt)-len(corte)} caracteres]"
                                 if len(txt) > len(corte) else ""))
            if not (r.get("contexto") or []):
                linhas.append("      (nenhum)")
        linhas.append("")
        sys.stderr.write(f"\r  {n}/{len(uniao)}  ({(time.time()-t0)/60:.0f} min)")
    sys.stderr.write("\n")

    porc = collections.Counter(dev[i]["classe"] for i in uniao)
    cab = [
        "DOSSIÊ DE ERRO · questões que os braços erraram, com o contexto e a resposta",
        "",
        "braços: " + ", ".join(f"{r} ({b['grafo']})" for r, b in bracos.items()),
        f"união: {len(uniao)} questões · " +
        " · ".join(f"{r}: {len(b['erros'])}" for r, b in bracos.items()),
        f"por classe: {dict(porc.most_common())}",
        "",
        f"AVISO: a reexecução usa temp=0.3, e 3,9% dos itens trocam de resposta entre",
        f"execuções. {divergiu} destas reexecuções ACERTARAM o que o banco gravou como erro.",
        "Essas linhas vêm marcadas com [!]. Não é erro de registro: é o piso de ruído.",
        "",
    ]
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    a.saida.write_text("\n".join(cab + linhas))
    print(f"\n{len(uniao)} questões · {divergiu} divergiram na reexecução "
          f"· {(time.time()-t0)/60:.0f} min -> {a.saida}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
