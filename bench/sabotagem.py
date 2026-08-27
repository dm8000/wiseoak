#!/usr/bin/env python3
"""
O modelo esta USANDO o contexto, ou so passando por ele?

Tres condicoes sobre as MESMAS perguntas e o MESMO no de resposta:

    certo      o contexto recuperado para aquela pergunta
    trocado    o contexto recuperado para OUTRA pergunta (rotacao de 1)
    vazio      sem contexto nenhum (equivale ao v0)

Se 'trocado' empatar com 'certo', o contexto e decorativo: o modelo responde de cabeca.
Se 'trocado' cair abaixo de 'vazio', o contexto errado ATIVAMENTE atrapalha, o que prova
dependencia — e e o resultado mais informativo dos tres.

Isola o no de resposta de proposito: a recuperacao ja foi inspecionada a mao e esta boa.
"""
import json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wiseoak.grafos import comum
from wiseoak.store import Indice

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
itens = [json.loads(l) for l in open("dados/questoes_sba.dev.jsonl")]
vf = [i for i in itens if i["tipo"] == "vf"]
am = random.Random(1).sample(vf, N)
ix = Indice("dados/indice/h512")

# recupera uma vez, reaproveita nas tres condicoes
ctxs = []
for k, it in enumerate(am):
    q = f"{it['enunciado']}\n\nAssertiva: {it['assertiva']}"
    est = {"pergunta": q, "modo": "vf", "modelo": "medgemma-clinical",
           "raciocinio": "nenhum", "indice": ix, "k_busca": 10, "k_contexto": 5,
           "hibrido": True, "expandir_pai": True, "trace": []}
    est.update(comum.no_recuperar(est))
    est.update(comum.no_rerankear(est))
    est.update(comum.no_montar_contexto(est))
    ctxs.append(est["contexto"])
    sys.stderr.write(f"\r  recuperando {k+1}/{N}")
sys.stderr.write("\n")

def rodar(nome, pegar_ctx):
    acertos = 0
    for i, it in enumerate(am):
        q = f"{it['enunciado']}\n\nAssertiva: {it['assertiva']}"
        est = {"pergunta": q, "modo": "vf", "modelo": "medgemma-clinical",
               "raciocinio": "nenhum", "contexto": pegar_ctx(i), "trace": []}
        r = comum.no_responder(est)
        if (r.get("resposta") or "").strip().upper()[:1] == it["resposta"]:
            acertos += 1
        sys.stderr.write(f"\r  {nome} {i+1}/{N} acerto={acertos}")
    sys.stderr.write("\n")
    return acertos / N

print(f"certo   : {rodar('certo',   lambda i: ctxs[i]):.1%}")
print(f"trocado : {rodar('trocado', lambda i: ctxs[(i + 1) % N]):.1%}")
print(f"vazio   : {rodar('vazio',   lambda i: []):.1%}")
