#!/usr/bin/env python3
"""
Um passo do FlowBot com a flagship como LLM-meta.

O paper usa Claude-3.5 Sonnet como meta e um modelo menor como executor. Aqui o executor
e local (gemma4-plan / medgemma-clinical) e o meta e a flagship — o que e MAIS fiel ao
artigo que usar o 31B para criticar os proprios prompts. O preco e que o laco deixa de
ser automatico: cada iteracao para e entrega o material.

Tres subcomandos, um por fase do laco:

    lote       roda um lote, grava o dossie do que aconteceu (as passagens diretas,
               os erros, o contexto que cada um recebeu). E a "passagem direta" do
               paper: produz {x_i}, y e L(x_K, y) para o meta escrever o gradiente.

    avaliar    mede um prompt candidato na VALIDACAO. E o criterio de selecao do
               paper — nunca no teste.

    aplicar    promove o candidato a prompt vigente, com registro de quem substituiu
               quem e por quanto.

Todo candidato fica gravado, inclusive o recusado: um otimizador so e auditavel se as
tentativas fracassadas sobreviverem.

    ./flowbot_lote.py lote --n 5
    ./flowbot_lote.py avaliar --prompt cand-01.txt --n 25
    ./flowbot_lote.py aplicar --prompt cand-01.txt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
ANALISES = RAIZ / "eval" / "analises"
FB = ANALISES / "flowbot"
FB.mkdir(parents=True, exist_ok=True)

from wiseoak.grafos import comum  # noqa: E402
from wiseoak.grafos.variantes import de_sketch  # noqa: E402
from wiseoak.store import Indice  # noqa: E402

SKETCH_PADRAO = ["recuperar_hibrido", "rerankear", "expandir_pai", "responder"]
DIARIO = FB / "diario.jsonl"


def amostra(n_treino: int, n_val: int, semente: int):
    vf = [i for i in (json.loads(l) for l in
                      open(RAIZ / "dados" / "questoes_sba.dev.jsonl")) if i["tipo"] == "vf"]
    a = random.Random(semente).sample(vf, n_treino + n_val)
    return a[:n_treino], a[n_treino:]


def rodar(itens, sketch, modelo, prompt, ix, k_busca, k_contexto):
    """Passagem direta sobre o StateGraph compilado. Devolve um registro por item."""
    grafo = de_sketch(sketch)
    antigo = comum.SISTEMA["confiante"]
    if prompt:
        comum.SISTEMA["confiante"] = prompt
    saida = []
    try:
        for k, it in enumerate(itens, 1):
            q = f"{it['enunciado']}\n\nAssertiva: {it['assertiva']}"
            r = grafo.invoke({"pergunta": q, "modo": "vf", "modelo": modelo,
                              "raciocinio": "nenhum", "ancoragem": "confiante",
                              "indice": ix, "k_busca": k_busca,
                              "k_contexto": k_contexto, "trace": []})
            dada = (r.get("resposta") or "").strip().upper()[:1]
            saida.append({
                "id": it["id"], "fonte": it["fonte"],
                "enunciado": " ".join(it["enunciado"].split()),
                "assertiva": " ".join(it["assertiva"].split()),
                "gabarito": it["resposta"], "resposta": dada,
                "certo": dada == it["resposta"],
                "ressalva": " ".join((r.get("ressalva") or "").split()),
                "fontes": r.get("fontes") or [],
                "citacoes": [" ".join(str(c.get("trecho", "")).split())[:220]
                             for c in (r.get("citacoes") or [])[:2]],
            })
            sys.stderr.write(f"\r  {k}/{len(itens)} "
                             f"acerto={sum(x['certo'] for x in saida)}")
    finally:
        comum.SISTEMA["confiante"] = antigo
    sys.stderr.write("\n")
    return saida


def registrar(evento: dict):
    with DIARIO.open("a") as f:
        f.write(json.dumps({"quando": datetime.now().isoformat(timespec="seconds"),
                            **evento}, ensure_ascii=False) + "\n")


def cmd_lote(a):
    treino, _ = amostra(a.n, 0, a.semente)
    ix = Indice(str(RAIZ / "dados" / "indice" / "h512"))
    prompt = Path(a.prompt).read_text() if a.prompt else comum.SISTEMA["confiante"]
    print(f"lote de {len(treino)} · executor {a.modelo} · sketch {SKETCH_PADRAO}",
          file=sys.stderr)
    t0 = time.time()
    res = rodar(treino, SKETCH_PADRAO, a.modelo, prompt, ix, a.k_busca, a.k_contexto)
    acerto = sum(r["certo"] for r in res) / len(res)

    dossie = FB / f"lote-{datetime.now():%H%M%S}.md"
    L = [f"# Dossiê do lote · {a.modelo} · acerto {acerto:.0%} ({len(res)} itens)",
         "", "## Prompt vigente do nó `responder`", "", "```", prompt.strip(), "```", ""]
    for n, r in enumerate(res, 1):
        L += [f"## [{n}] {'ACERTOU' if r['certo'] else 'ERROU'} · {r['fonte'][:34]}",
              f"- **enunciado:** {r['enunciado'][:400]}",
              f"- **assertiva:** {r['assertiva'][:400]}",
              f"- **gabarito:** {r['gabarito']}  ·  **respondeu:** {r['resposta']}",
              f"- **o RAG trouxe:** {', '.join(r['fontes'][:4])}"]
        for c in r["citacoes"]:
            L.append(f"  - citou: \"{c}\"")
        if r["ressalva"]:
            L.append(f"- **ressalva:** {r['ressalva'][:400]}")
        L.append("")
    dossie.write_text("\n".join(L))
    registrar({"evento": "lote", "modelo": a.modelo, "n": len(res),
               "acerto": acerto, "dossie": dossie.name,
               "segundos": round(time.time() - t0)})
    print(f"\nacerto no lote: {acerto:.1%}", file=sys.stderr)
    print(f"dossiê: {dossie}", file=sys.stderr)
    return 0


def cmd_avaliar(a):
    _, val = amostra(a.treino, a.n, a.semente)
    ix = Indice(str(RAIZ / "dados" / "indice" / "h512"))
    prompt = Path(a.prompt).read_text() if a.prompt else comum.SISTEMA["confiante"]
    nome = Path(a.prompt).name if a.prompt else "vigente"
    print(f"avaliando '{nome}' na VALIDACAO ({len(val)} itens) · {a.modelo}",
          file=sys.stderr)
    t0 = time.time()
    res = rodar(val, SKETCH_PADRAO, a.modelo, prompt, ix, a.k_busca, a.k_contexto)
    acerto = sum(r["certo"] for r in res) / len(res)
    registrar({"evento": "avaliar", "candidato": nome, "modelo": a.modelo,
               "n": len(res), "validacao": acerto,
               "segundos": round(time.time() - t0),
               "itens": {r["id"]: r["certo"] for r in res}})
    print(f"\nVALIDAÇÃO de '{nome}': {acerto:.1%}  ({len(res)} itens, "
          f"{time.time()-t0:.0f}s)", file=sys.stderr)
    return 0


def cmd_aplicar(a):
    novo = Path(a.prompt).read_text()
    vigente = FB / "prompt-vigente.txt"
    antigo = vigente.read_text() if vigente.exists() else comum.SISTEMA["confiante"]
    vigente.write_text(novo)
    registrar({"evento": "aplicar", "candidato": Path(a.prompt).name,
               "chars_antes": len(antigo), "chars_depois": len(novo)})
    print(f"promovido: {Path(a.prompt).name} ({len(antigo)} -> {len(novo)} chars)",
          file=sys.stderr)
    print(f"vigente em {vigente}", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("comando", choices=("lote", "avaliar", "aplicar"))
    ap.add_argument("--prompt", default=None, help="arquivo com o prompt candidato")
    ap.add_argument("--modelo", default="gemma4-plan")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--treino", type=int, default=20,
                    help="quantos itens ficam no treino (a validação vem depois deles)")
    ap.add_argument("--semente", type=int, default=7)
    ap.add_argument("--k-busca", type=int, default=10)
    ap.add_argument("--k-contexto", type=int, default=5)
    a = ap.parse_args()
    return {"lote": cmd_lote, "avaliar": cmd_avaliar, "aplicar": cmd_aplicar}[a.comando](a)


if __name__ == "__main__":
    sys.exit(main())
