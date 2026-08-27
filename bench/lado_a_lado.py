#!/usr/bin/env python3
"""
A MESMA assertiva respondida com e sem RAG, lado a lado.

Responde a pergunta "o modelo precisa mesmo do contexto?": se o v0 acerta sozinho e a
citacao do v2 nao sustenta nada que ele ja nao soubesse, o RAG esta pagando latencia
para confirmar o que o modelo ja tinha.

Amostra: N itens que o v2 ACERTOU e N que ERROU, da celula ja medida. Roda os dois
grafos em cada um para capturar o texto — o banco so guarda certo/errado.

    ./lado_a_lado.py gemma4-plan confiante 10
"""
import json
import sqlite3
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
# Analise vai para o PROJETO, nao para /tmp: e o registro de por que cada
# numero e o que e, e some junto com a sessao se ficar no scratchpad.
ANALISES = RAIZ / "eval" / "analises"
ANALISES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RAIZ))
from wiseoak.grafos.comum import verificar_citacoes  # noqa: E402
from wiseoak.grafos.variantes import construir, estado_inicial  # noqa: E402

MODELO = sys.argv[1] if len(sys.argv) > 1 else "gemma4-plan"
ANC = sys.argv[2] if len(sys.argv) > 2 else "confiante"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 10

db = sqlite3.connect(RAIZ / "eval" / "resultados.sqlite")
db.row_factory = sqlite3.Row
cel = {}
for r in db.execute("SELECT modelo,setup,itens,bloco FROM resultado "
                    "WHERE bench='vf-dev' AND n=80"):
    cel[(r["modelo"], r["setup"].split("|")[0],
         [x for x in r["setup"].split("|") if x.startswith("anc=")][0][4:],
         r["bloco"])] = json.loads(r["itens"])


def pega(g):
    for k, v in cel.items():
        if k[0] == MODELO and k[1] == g and k[2] == ANC:
            return v
    return {}


v0, v2 = pega("v0"), pega("v2")
porid = {i["id"]: i for i in (json.loads(l) for l in
         open(RAIZ / "dados" / "questoes_sba.dev.jsonl")) if i["tipo"] == "vf"}

certos = [k for k in v2 if v2[k] and k in porid][:N]
errados = [k for k in v2 if not v2[k] and k in porid][:N]

g0, g2 = construir("v0"), construir("v2")
for titulo, chaves in (("V2 ACERTOU", certos), ("V2 ERROU", errados)):
    print("\n" + "#" * 78)
    print(f"#  {titulo}  ({len(chaves)} itens) · {MODELO} · ancoragem {ANC}")
    print("#" * 78)
    for n, k in enumerate(chaves, 1):
        it = porid[k]
        q = f"{it['enunciado']}\n\nAssertiva: {it['assertiva']}"
        e = dict(modo="vf", modelo=MODELO, ancoragem=ANC)
        r0 = g0.invoke(estado_inicial(q, **e))
        r2 = g2.invoke(estado_inicial(q, indice="dados/indice/h512",
                                      k_busca=10, k_contexto=5, **e))
        v = verificar_citacoes(r2)
        ok = lambda r: "OK   " if r["resposta"] == it["resposta"] else "ERRO "
        print("\n" + "-" * 78)
        print(f"[{n}] {it['fonte'][:36]} q{it['numero']}{it['letra']}")
        print(f"ENUNC : {' '.join(it['enunciado'].split())[:165]}")
        print(f"ASSERT: {' '.join(it['assertiva'].split())[:215]}")
        print(f"GABAR : {it['resposta']}   |   sem RAG: {r0['resposta']} {ok(r0)}"
              f"|   com RAG: {r2['resposta']} {ok(r2)}")
        if r0.get("ressalva"):
            print(f"  sem RAG diz: {' '.join(r0['ressalva'].split())[:190]}")
        print(f"  RAG trouxe: {', '.join((r2.get('fontes') or [])[:3])[:150]}")
        for c in (r2.get("citacoes") or [])[:2]:
            print(f'    "{" ".join(str(c.get("trecho")).split())[:185]}"')
        print(f"  citações: {v['fieis']}/{v['citacoes']} literais")
        if r2.get("ressalva"):
            print(f"  com RAG diz: {' '.join(r2['ressalva'].split())[:190]}")
