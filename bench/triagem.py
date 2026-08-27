#!/usr/bin/env python3
"""
Triagem de variantes sobre o conjunto de questoes que ja foram erradas.

Serve para RANQUEAR candidatos barato antes de gastar a corrida no banco inteiro. Nao
estabelece efeito, e o relatorio diz isso em voz alta, por duas razoes:

  1. o conjunto e cego a REGRESSAO — um braco que recupera 20 aqui e quebra 30 entre as
     767 questoes que ja estavam certas parece otimo nesta tabela;
  2. o conjunto de erros e enriquecido nos itens que o modelo responde na duvida, que sao
     justamente os instaveis: medido, 14,9% deles trocam de resposta entre execucoes.

Por isso cada braco roda com REPETICAO e o criterio e conservador: um item so conta como
recuperado se o braco acerta em TODAS as repeticoes E o controle erra em TODAS. Item que
oscila em qualquer um dos dois vai para a coluna INSTAVEL e nao entra em nenhum saldo.
Isso troca sensibilidade por ausencia de falso positivo — e a bancada ja produziu tres
falsos positivos com p<0,05 por nao fazer isso.

    ./triagem.py --controle v10 --braco v10:k8 --braco v12:pai --reps 2
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
ANALISES = RAIZ / "eval" / "analises"

from wiseoak.grafos.variantes import construir, estado_inicial  # noqa: E402
from wiseoak.store import Indice  # noqa: E402

# nome curto -> (grafo, ancoragem, k_contexto). None = herda do padrao da linha de comando.
PERFIS: dict[str, tuple[str, str | None, int | None]] = {
    "controle": ("v10", None, None),
    "k8":       ("v10", None, 8),
    "pai":      ("v12", None, None),
    "falsif":   ("v10", "falsificacao", None),
    "quant":    ("v10", "confiante_quantificador", None),
    "rerank":   ("v11", None, None),
    # combinacoes: a triagem mediu 14 itens que SO o k8 recupera e 27 que SO o falsif
    # recupera, sem antagonismo nos dois sentidos. Os eixos sao ortogonais — um mexe na
    # recuperacao, o outro em como o modelo raciocina. A uniao (54) e TETO, nao previsao:
    # so medindo o braco combinado se sabe se a interacao soma ou atrapalha.
    "k8falsif":  ("v10", "falsificacao", 8),
    "paifalsif": ("v12", "falsificacao", None),
}


def conjunto_de_erros(blocos: list[str]) -> list[str]:
    import sqlite3
    db = sqlite3.connect(RAIZ / "eval" / "resultados.sqlite")
    db.row_factory = sqlite3.Row
    errados: set[str] = set()
    for r in db.execute("SELECT bloco, itens FROM resultado"):
        if r["bloco"] in blocos:
            errados |= {i for i, ok in json.loads(r["itens"]).items() if not ok}
    return sorted(errados)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bloco", action="append", default=None,
                    help="blocos de onde sai o conjunto de erros")
    ap.add_argument("--braco", action="append", required=True,
                    help="nome do perfil em PERFIS (repetivel)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--modelo", default="gemma4-plan")
    ap.add_argument("--ancoragem", default="confiante")
    ap.add_argument("--indice", default="dados/indice/m10")
    ap.add_argument("--k-busca", type=int, default=8)
    ap.add_argument("--k-contexto", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--saida", type=Path, default=ANALISES / "triagem.txt")
    a = ap.parse_args()

    blocos = a.bloco or ["ROTEADO", "COTA"]
    dev = {i["id"]: i for i in (json.loads(l) for l in
           open(RAIZ / "dados" / "questoes_sba.dev.jsonl"))}
    itens = [i for i in conjunto_de_erros(blocos) if i in dev]
    if a.limite:
        itens = itens[:a.limite]
    nomes = ["controle"] + [b for b in a.braco if b != "controle"]
    for n in nomes:
        if n not in PERFIS:
            print(f"perfil desconhecido: {n} · conhecidos: {sorted(PERFIS)}", file=sys.stderr)
            return 1

    ix = Indice(a.indice)
    print(f"  conjunto: {len(itens)} questões erradas em {blocos}", file=sys.stderr)
    print(f"  braços: {nomes} · {a.reps} repetições · temp={a.temp}", file=sys.stderr)
    print(f"  chamadas: {len(itens) * len(nomes) * a.reps}\n", file=sys.stderr)

    # resultado[nome][item] = lista de acertos, uma por repeticao
    resultado: dict[str, dict[str, list[bool]]] = {n: {} for n in nomes}
    t0 = time.time()
    total = len(itens) * len(nomes) * a.reps
    feito = 0
    for nome in nomes:
        grafo_id, anc, kctx = PERFIS[nome]
        g = construir(grafo_id)
        anc = anc or a.ancoragem
        kctx = kctx or a.k_contexto
        for it_id in itens:
            it = dev[it_id]
            q = it["enunciado"]
            if it["tipo"] == "vf":
                q += "\n\nAssertiva: " + (it.get("assertiva") or "")
            else:
                q += "\n" + "\n".join(f"{k}) {v}" for k, v in
                                      sorted((it.get("alternativas") or {}).items()))
            oks = []
            for _ in range(a.reps):
                try:
                    r = g.invoke(estado_inicial(
                        q, modo=it["tipo"], modelo=a.modelo, ancoragem=anc, indice=ix,
                        k_busca=a.k_busca, k_contexto=kctx, temp=a.temp))
                    dada = (r.get("resposta") or "").strip().upper()[:1]
                except Exception as e:  # uma falha nao derruba o braco
                    print(f"\n  ERRO {it_id} em {nome}: {type(e).__name__}", file=sys.stderr)
                    dada = ""
                oks.append(dada == it["resposta"])
                feito += 1
            resultado[nome][it_id] = oks
            sys.stderr.write(f"\r  {nome}: {feito}/{total} ({(time.time()-t0)/60:.0f} min)")
    sys.stderr.write("\n")

    def estavel(oks: list[bool]) -> bool:
        return all(oks) or not any(oks)

    L = [f"TRIAGEM · {len(itens)} questões que já foram erradas · {a.reps} repetições "
         f"· temp={a.temp}", "",
         "CRITÉRIO CONSERVADOR: um item só conta se for ESTÁVEL nos dois braços — o braço",
         "acerta em todas as repetições e o controle erra em todas, ou o inverso. Item que",
         "oscila em qualquer um dos dois vai para INSTÁVEL e não entra em nenhum saldo.",
         "",
         "ESTA TABELA RANQUEIA, NÃO ESTABELECE. O conjunto é cego a regressão: um braço que",
         "recupera aqui pode quebrar questões que já estavam certas, e isso só aparece na",
         "corrida do banco inteiro.", "",
         f"  {'braço':10s} {'recuperou':>10s} {'quebrou':>9s} {'saldo':>7s} "
         f"{'instável':>9s} {'acerto*':>8s}"]

    ctrl = resultado["controle"]
    for nome in nomes:
        if nome == "controle":
            inst = sum(1 for v in ctrl.values() if not estavel(v))
            acc = sum(1 for v in ctrl.values() if all(v)) / len(itens)
            L.append(f"  {'controle':10s} {'—':>10s} {'—':>9s} {'—':>7s} "
                     f"{inst:9d} {acc:7.1%}")
            continue
        b = resultado[nome]
        rec = que = inst = 0
        for i in itens:
            if not estavel(b[i]) or not estavel(ctrl[i]):
                inst += 1
                continue
            if all(b[i]) and not any(ctrl[i]):
                rec += 1
            elif not any(b[i]) and all(ctrl[i]):
                que += 1
        acc = sum(1 for i in itens if all(b[i])) / len(itens)
        L.append(f"  {nome:10s} {rec:10d} {que:9d} {rec-que:+7d} {inst:9d} {acc:7.1%}")
    L += ["", "* acerto = acertou em TODAS as repetições, sobre o conjunto inteiro.",
          "  Não é comparável ao placar: este conjunto é só de questões já erradas.", ""]

    # por classe, so para os bracos
    L.append("POR CLASSE (recuperou / quebrou, só itens estáveis)")
    classes = sorted({dev[i]["classe"] for i in itens},
                     key=lambda c: -sum(1 for i in itens if dev[i]["classe"] == c))
    L.append(f"  {'classe':22s} {'n':>4s} " + " ".join(f"{n[:9]:>11s}" for n in nomes[1:]))
    for c in classes:
        ids = [i for i in itens if dev[i]["classe"] == c]
        linha = f"  {c:22s} {len(ids):4d} "
        for nome in nomes[1:]:
            b = resultado[nome]
            rec = sum(1 for i in ids if estavel(b[i]) and estavel(ctrl[i])
                      and all(b[i]) and not any(ctrl[i]))
            que = sum(1 for i in ids if estavel(b[i]) and estavel(ctrl[i])
                      and not any(b[i]) and all(ctrl[i]))
            linha += f"{f'+{rec}/-{que}':>11s} "
        L.append(linha)

    L += ["", f"tempo: {(time.time()-t0)/60:.0f} min"]
    bruto = {n: {i: v for i, v in d.items()} for n, d in resultado.items()}
    (a.saida.with_suffix(".json")).write_text(json.dumps(bruto))
    a.saida.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nbruto em {a.saida.with_suffix('.json')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
