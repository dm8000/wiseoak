#!/usr/bin/env python3
"""
Analise de erro: o que o v2 errou, e o que o RAG tinha entregado a ele.

Reexecuta so os itens que o v2 ERROU na celula ja medida, para capturar o contexto e a
ressalva — o banco guarda acerto/erro, nao o texto. Marca em cada caso se o v0 (sem RAG)
tinha acertado: quando o v0 acerta e o v2 erra, o contexto ATRAPALHOU, e e ai que esta a
explicacao do resultado.

    ./erros.py gemma4-plan 10
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
from wiseoak.grafos.variantes import construir, estado_inicial  # noqa: E402

modelo = sys.argv[1] if len(sys.argv) > 1 else "gemma4-plan"
quantos = int(sys.argv[2]) if len(sys.argv) > 2 else 10
anc = sys.argv[3] if len(sys.argv) > 3 else "confiante"

db = sqlite3.connect(RAIZ / "eval" / "resultados.sqlite")
db.row_factory = sqlite3.Row
cel = {}
for r in db.execute("SELECT modelo,setup,itens FROM resultado WHERE bench='vf-dev' AND n=40"):
    cel[(r["modelo"], r["setup"])] = json.loads(r["itens"])


def pega(g):
    for (m, s), v in cel.items():
        if m == modelo and s.startswith(g + "|") and f"anc={anc}" in s and "rep=2" not in s:
            return v
    return {}


v0, v2 = pega("v0"), pega("v2")
porid = {i["id"]: i for i in (json.loads(l) for l in
         open(RAIZ / "dados" / "questoes_sba.dev.jsonl")) if i["tipo"] == "vf"}

# prioriza os casos em que o v0 acertou e o v2 errou: ali o contexto foi o culpado
erros = [k for k in v2 if not v2[k] and k in porid]
erros.sort(key=lambda k: (not v0.get(k, False)))  # v0 acertou primeiro
erros = erros[:quantos]

print(f"# {modelo} · ancoragem {anc} · {len(erros)} erros do v2\n")
g = construir("v2")
for n, k in enumerate(erros, 1):
    it = porid[k]
    q = f"{it['enunciado']}\n\nAssertiva: {it['assertiva']}"
    r = g.invoke(estado_inicial(q, modo="vf", modelo=modelo, ancoragem=anc,
                                indice="dados/indice/h512", k_busca=10, k_contexto=5))
    culpa = "CONTEXTO ATRAPALHOU" if v0.get(k) else "os dois erraram"
    print("=" * 78)
    print(f"[{n}] {it['fonte'][:38]} q{it['numero']}{it['letra']}   ({culpa})")
    print(f"PERGUNTA : {' '.join(it['enunciado'].split())[:150]}")
    print(f"ASSERTIVA: {' '.join(it['assertiva'].split())[:230]}")
    print(f"GABARITO : {it['resposta']}    v0 respondeu {'certo' if v0.get(k) else 'errado'}"
          f"    v2 respondeu: {r['resposta']}")
    print("RAG TROUXE:")
    for f in (r.get("fontes") or [])[:5]:
        print(f"   - {f}")
    if r.get("ressalva"):
        print(f"RESSALVA : {' '.join(r['ressalva'].split())[:260]}")
    cit = (r.get("citacoes") or [])
    if cit:
        print(f"CITOU    : cap{cit[0].get('capitulo')} p{cit[0].get('pagina')}: "
              f"\"{' '.join(str(cit[0].get('trecho')).split())[:200]}\"")
