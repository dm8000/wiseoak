#!/usr/bin/env python3
"""
Gera um SUMARIO por capitulo, para o modelo poder navegar o livro como uma pessoa.

Por que existe: o catalogo do Miller so tem nivel de capitulo (87 titulos), e capitulo tem
mediana de 128 paginas — escolher o capitulo nao chega perto de localizar o assunto. O
campo de subtitulos veio sujo da ingestao ("Instructions for online access", "FANZCA"),
entao a estrutura fina precisa ser reconstruida.

Custo: 87 chamadas, uma por capitulo — contra as 38.736 que Contextual Retrieval no nivel
de trecho exigiria. E a mesma tecnica aplicada onde ela e barata.

O texto de cada capitulo e AMOSTRADO uniformemente, nao truncado: um capitulo de 170
paginas nao cabe no contexto, e pegar so o comeco produziria sumario que descreve a
introducao e ignora o resto.

    python3 -m wiseoak.ingest.sumario --indice dados/indice/m10 --saida dados/sumario_m10.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wiseoak import clientes  # noqa: E402

SISTEMA = (
    "Voce recebe trechos AMOSTRADOS de um capitulo de livro-texto de anestesiologia em "
    "ingles. Liste os assuntos que o capitulo cobre, em PORTUGUES, para servir de sumario "
    "de navegacao. De 6 a 12 topicos, cada um uma frase curta e concreta (o assunto, nao "
    "'discussao sobre'). Nomeie sindromes, farmacos, tecnicas e valores quando aparecerem. "
    "Nao invente topico que nao esteja nos trechos."
)
SCHEMA = {"type": "object",
          "properties": {"topicos": {"type": "array", "items": {"type": "string"}}},
          "required": ["topicos"]}


def amostrar(db: sqlite3.Connection, cap: int, alvo_chars: int) -> str:
    linhas = [r[0] for r in db.execute(
        "SELECT texto FROM chunk WHERE nivel='filho' AND capitulo_num=? "
        "ORDER BY pagina_inicial, ordem", (cap,))]
    if not linhas:
        return ""
    # amostragem UNIFORME: pegar so o inicio descreveria a introducao e nada mais
    passo = max(1, len(linhas) // 12)
    escolhidos = linhas[::passo][:12]
    saida, total = [], 0
    for t in escolhidos:
        t = " ".join(t.split())[:700]
        if total + len(t) > alvo_chars:
            break
        saida.append(t)
        total += len(t)
    return "\n---\n".join(saida)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indice", default="dados/indice/m10")
    ap.add_argument("--saida", type=Path, default=Path("dados/sumario_m10.json"))
    ap.add_argument("--modelo", default="gemma4-plan")
    ap.add_argument("--chars", type=int, default=7000)
    a = ap.parse_args()

    db = sqlite3.connect(a.indice + ".sqlite")
    caps = list(db.execute(
        "SELECT capitulo_num, capitulo, MIN(pagina_inicial), MAX(pagina_final), COUNT(*) "
        "FROM chunk WHERE nivel='filho' AND capitulo_num IS NOT NULL "
        "GROUP BY capitulo_num ORDER BY capitulo_num"))
    print(f"  {len(caps)} capítulos", file=sys.stderr)

    saida, t0 = {}, time.time()
    for n, (num, titulo, p0, p1, qtd) in enumerate(caps, 1):
        amostra = amostrar(db, num, a.chars)
        topicos = []
        if amostra:
            try:
                r = clientes.chat(
                    [{"role": "system", "content": SISTEMA},
                     {"role": "user", "content": f"Capítulo {num}: {titulo}\n\n{amostra}"}],
                    modelo=a.modelo, think=False, max_tokens=600, temp=0.0, schema=SCHEMA)
                topicos = [t.strip() for t in json.loads(r["content"]).get("topicos") or []
                           if t.strip()][:12]
            except Exception as e:
                print(f"\n  cap {num}: {type(e).__name__}", file=sys.stderr)
        saida[str(num)] = {"titulo": titulo, "pagina_inicial": p0, "pagina_final": p1,
                           "trechos": qtd, "topicos": topicos}
        sys.stderr.write(f"\r  {n}/{len(caps)} ({(time.time()-t0)/60:.0f} min)")
    sys.stderr.write("\n")
    a.saida.write_text(json.dumps(saida, ensure_ascii=False, indent=1))
    vazios = sum(1 for v in saida.values() if not v["topicos"])
    print(f"  {len(saida)} capítulos · {vazios} sem tópicos -> {a.saida}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
