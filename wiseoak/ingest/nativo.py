#!/usr/bin/env python3
"""
Ingestao de livro-texto NATIVO DIGITAL com outline. Serve Barash e Miller 10e.

Tres diferencas do Miller que justificam um ingestor proprio em vez de parametrizar o
outro:

  estrutura   o Barash tem OUTLINE de verdade (70 marcadores: "Section I", "1 - The
              History of Anesthesia"). O Miller so tinha os nomes dos arquivos do
              scanner, e por isso la a estrutura teve de sair do cabecalho corrente.
  colunas     coluna unica, A4. O Miller e duas colunas com corte em x=352, e toda a
              reconstrucao por bandas existe por causa disso. Aqui nao ha o que
              reconstruir.
  OCR         nao ha. `pdftotext` le a camada nativa; nenhuma das regras de ligadura
              (tlu/ftu/l-por-I) se aplica, e aplica-las so poderia estragar.

IDIOMA: o livro esta em INGLES e as perguntas em PORTUGUES. Consequencia direta para a
busca: o BM25 nao casa nada entre idiomas (vocabulario disjunto), entao neste corpus so
a busca DENSA atravessa — o EmbeddingGemma e multilingue. Quem indexar isto e usar
`hibrido=True` estara pagando o custo do BM25 para receber ruido.

    python3 -m wiseoak.ingest.barash --saida dados/barash_paginas.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PADRAO = Path("/home/phobos/Downloads/Barash_Clinical_Anesthesia.pdf")

# O rotulo do livro vem por parametro: o mesmo ingestor atende os dois, e a citacao
# precisa dizer de qual deles veio o trecho.

# "Section II - Basic Principles" / "9 - Acid-Base, Fluids, and Electrolytes"
# Dois formatos reais de outline:
#   Barash   "Section II - Basic Principles"   /  "9 - Acid-Base, Fluids"
#   Miller   "Section I. INTRODUCTION"         /  "1. The Scope of Modern"
_SECAO = re.compile(r"^\s*Section\s+([IVXLC]+)\s*[-–—.]\s*(.+)$", re.I)
_CAPITULO = re.compile(r"^\s*(\d+)\s*[-–—.]\s*(.+)$")

_ROMANO = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _romano(s: str) -> int | None:
    total = anterior = 0
    for c in reversed(s.upper()):
        v = _ROMANO.get(c, 0)
        if not v:
            return None
        total = total - v if v < anterior else total + v
        anterior = v
    return total or None


def mapa_de_paginas(pdf: Path) -> dict[int, dict]:
    """
    Pagina -> (secao, capitulo), a partir do OUTLINE.

    Cada marcador vale da sua pagina ate a do proximo. E por isso que o outline vale
    mais que qualquer heuristica: a fronteira e declarada pelo editor, nao inferida.
    """
    from pypdf import PdfReader

    r = PdfReader(str(pdf))
    marcos: list[tuple[int, str]] = []

    def anda(itens):
        for x in itens:
            if isinstance(x, list):
                anda(x)
                continue
            try:
                p = r.get_destination_page_number(x) + 1
            except Exception:
                continue
            marcos.append((p, str(x.title).strip()))

    anda(r.outline)
    marcos.sort()

    total = len(r.pages)
    mapa: dict[int, dict] = {}
    atual = {"secao_num": None, "secao": None, "capitulo_num": None, "capitulo": None}
    for i, (pagina, titulo) in enumerate(marcos):
        m = _SECAO.match(titulo)
        if m:
            atual = {"secao_num": _romano(m.group(1)),
                     "secao": re.sub(r"\s+", " ", m.group(2).replace("\r", " ")).strip(),
                     "capitulo_num": None, "capitulo": None}
        else:
            m = _CAPITULO.match(titulo)
            if m:
                # o outline quebra palavra ("Practice and OperatingRoom Management")
                # o outline traz \r no meio e palavra colada na quebra
                # ("Practice and OperatingRoom Management")
                nome = re.sub(r"\s+", " ", m.group(2).replace("\r", " ")).strip()
                nome = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", nome)
                atual = {**atual, "capitulo_num": int(m.group(1)), "capitulo": nome}
        fim = marcos[i + 1][0] if i + 1 < len(marcos) else total + 1
        for p in range(pagina, fim):
            mapa[p] = dict(atual)
    return mapa


def texto_por_pagina(pdf: Path, ini: int | None, fim: int | None) -> dict[int, str]:
    """
    Uma chamada so ao pdftotext, com o separador de pagina (\\f) para fatiar depois.
    Chamar por pagina seriam 3.229 processos.
    """
    cmd = ["pdftotext", "-layout"]
    if ini:
        cmd += ["-f", str(ini)]
    if fim:
        cmd += ["-l", str(fim)]
    cmd += [str(pdf), "-"]
    bruto = subprocess.run(cmd, capture_output=True, text=True).stdout
    base = ini or 1
    return {base + i: t for i, t in enumerate(bruto.split("\f"))}


_LIMPAR = [
    re.compile(r"^\s*\d+\s*$", re.M),                       # numero de pagina sozinho
    re.compile(r"^\s*P\.?\s*\d+\s*$", re.M),
]


def limpar(texto: str) -> str:
    for p in _LIMPAR:
        texto = p.sub("", texto)
    texto = re.sub(r"[ \t]{2,}", " ", texto)
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", type=Path, default=PADRAO)
    ap.add_argument("--livro", default="barash-clinical-9ed",
                    help="rotulo que vai no metadado de cada trecho")
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--pagina-inicial", type=int, default=None)
    ap.add_argument("--pagina-final", type=int, default=None)
    a = ap.parse_args()

    if not a.pdf.exists():
        print(f"nao existe: {a.pdf}", file=sys.stderr)
        return 1

    print("lendo o outline...", file=sys.stderr)
    mapa = mapa_de_paginas(a.pdf)
    print(f"  {len(mapa)} paginas mapeadas para "
          f"{len({m['capitulo_num'] for m in mapa.values() if m['capitulo_num']})} capitulos",
          file=sys.stderr)

    print("extraindo o texto...", file=sys.stderr)
    paginas = texto_por_pagina(a.pdf, a.pagina_inicial, a.pagina_final)

    a.saida.parent.mkdir(parents=True, exist_ok=True)
    escritas = vazias = 0
    with a.saida.open("w") as f:
        for n in sorted(paginas):
            texto = limpar(paginas[n])
            if len(texto) < 120:
                vazias += 1
                continue
            meta = mapa.get(n) or {"secao_num": None, "secao": None,
                                   "capitulo_num": None, "capitulo": None}
            f.write(json.dumps({"pagina": n, **meta, "titulos": [], "texto": texto,
                                "livro": a.livro}, ensure_ascii=False) + "\n")
            escritas += 1
    print(f"{escritas} paginas escritas, {vazias} descartadas por vazias -> {a.saida}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
