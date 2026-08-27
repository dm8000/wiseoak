#!/usr/bin/env python3
"""
Indexa um JSONL de chunks. Embedding roda no llama-swap, em CPU: indexar o livro inteiro
nao toca a VRAM e nao derruba o modelo grande da GPU.

    python3 -m wiseoak.ingest.indexar dados/chunks_h512.jsonl --indice dados/indice/h512
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wiseoak.store import Indice


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("entrada", type=Path)
    ap.add_argument("--indice", type=Path, required=True)
    a = ap.parse_args()
    if not a.entrada.exists():
        print(f"nao existe: {a.entrada}", file=sys.stderr)
        return 1
    chunks = [json.loads(l) for l in a.entrada.read_text().splitlines() if l.strip()]
    a.indice.parent.mkdir(parents=True, exist_ok=True)
    ix = Indice(a.indice)
    t0 = time.time()
    n = ix.indexar(chunks, progresso=lambda i, t: sys.stderr.write(f"\r  {i}/{t}"))
    sys.stderr.write("\n")
    print(f"{n} filhos vetorizados em {time.time()-t0:.0f}s | "
          f"pais {ix.quantos('pai')} | capitulos {len(ix.capitulos())}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
