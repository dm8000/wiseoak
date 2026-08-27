#!/usr/bin/env python3
"""
Mostra a MESMA pergunta respondida por varias ancoragens, com a bibliografia de cada uma.

Serve para ler a diferenca de postura com os olhos, ao lado do numero — o benchmark diz
qual ganha, isto diz por que.

    ./comparar.py gemma4-plan
"""
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
ANCORAGENS = sys.argv[2].split(",") if len(sys.argv) > 2 else \
    ["estrita", "confiante", "analista"]

PERGUNTAS = [
    "Qual anestésico inalatório tem o menor coeficiente de partição sangue-gás, e o que "
    "isso significa na prática?",
    "Como o ácido tranexâmico reduz o sangramento perioperatório?",
    "Quando a cafeína é usada em neonatologia perioperatória?",
    "Que critérios se usam para dar alta da sala de recuperação pós-anestésica?",
    "O que é o efeito segundo gás e quando ele é clinicamente relevante?",
]

g = construir("v2")
for n, q in enumerate(PERGUNTAS, 1):
    print("\n" + "#" * 78)
    print(f"# PERGUNTA {n}: {q}")
    for anc in ANCORAGENS:
        r = g.invoke(estado_inicial(q, modo="livre", modelo=MODELO, ancoragem=anc,
                                    indice="dados/indice/h512",
                                    k_busca=10, k_contexto=5))
        v = verificar_citacoes(r)
        print(f"\n--- {anc.upper()} " + "-" * (60 - len(anc)))
        print(" ".join((r.get("resposta") or "").split())[:900])
        if r.get("ressalva"):
            print(f"\n  RESSALVA: {' '.join(r['ressalva'].split())[:400]}")
        cits = r.get("citacoes") or []
        if cits:
            print(f"\n  REFERÊNCIAS ({v['fieis']}/{v['citacoes']} conferem literalmente):")
            for c in cits:
                livro = c.get("livro") or "Miller, Bases da Anestesia, 6ed"
                print(f"   · {livro}, cap. {c.get('capitulo')}, p. {c.get('pagina')}")
                print(f'     "{" ".join(str(c.get("trecho")).split())[:230]}"')
        else:
            print("\n  (sem citações)")
        print(f"\n  latência {sum(t['segundos'] for t in r['trace']):.1f}s")
