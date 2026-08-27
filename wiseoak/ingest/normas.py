#!/usr/bin/env python3
"""
Ingestao de NORMA: resolucao do CFM, estatuto e regimento da SBA, diretriz.

Tres diferencas do ingestor de livro (`nativo.py`), e cada uma muda o desenho:

  sem outline   Norma nao tem sumario. A hierarquia esta no TEXTO, na numeracao legal:
                "Art. 1º", "§ 2º", "I -", "Anexo". E o parser tem de ler isso.
  documento     Uma resolucao tem 3 a 8 paginas. O `--min-pai 200` que o Miller usa
  curto         fragmentaria uma norma inteira em pedacos sem sentido; aqui o PAI
                natural e o ARTIGO.
  citacao       Nao e "capitulo 8, pagina 97" e sim "Resolucao CFM 2.174/2017, Art. 3º".
                O campo `caminho` do chunk absorve isso sem mudar o schema.

O artigo e a unidade certa porque e como a norma e citada na vida real e como a questao
de prova pergunta ("segundo a Resolucao CFM 2.174/2017, e obrigatoria a existencia de...").

    python3 -m wiseoak.ingest.normas --entrada dados/normas --saida dados/normas_paginas.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# "Art. 1º" · "Art 12" · "ARTIGO 3º" — o marcador de artigo, no inicio da linha
_ARTIGO = re.compile(r"^\s*(?:Art\.?|ARTIGO|Artigo)\s*(\d+)\s*[º°.\-–]?", re.M)
# "Anexo I" · "ANEXO II" — anexo e irmao do artigo, nao filho
_ANEXO = re.compile(r"^\s*(ANEXO\s+[IVXLC0-9]+)\b", re.I | re.M)
# "CAPITULO II" · "SECAO I" — agrupam artigos em norma longa (estatuto)
_CAPITULO = re.compile(r"^\s*(CAP[IÍ]TULO\s+[IVXLC0-9]+.*|SE[ÇC][ÃA]O\s+[IVXLC0-9]+.*)$",
                       re.I | re.M)

# lixo de rodape que aparece em toda pagina do CFM
_LIXO = [
    re.compile(r"^\s*\d+\s*$", re.M),
    re.compile(r"(?im)^.*di[áa]rio oficial da uni[ãa]o.*$"),
    re.compile(r"(?im)^\s*p[áa]gina\s+\d+\s*(de\s+\d+)?\s*$"),
]


_ART_NUM = re.compile(r"(?m)^\s*Art\.?\s*(\d{1,3})\s*[°ºo]?\s*[-–—.]")


def _monotonia(texto: str) -> float:
    """
    Fracao dos artigos que aparecem em ordem crescente.

    E o verificador que decide qual extracao usar, e ele e PROGRAMATICO de proposito.
    Um PDF de duas colunas lido em faixas horizontais funde as colunas e a numeracao sai
    embaralhada — foi assim que "Art. 4° - Os membros associados" apareceu grudado em
    "Art. 15 - Sao membros Remidos". Documento normativo numera artigo em ordem, entao a
    ordem e um sinal barato e confiavel de que a leitura saiu certa.
    """
    nums = [int(m.group(1)) for m in _ART_NUM.finditer(texto)]
    if len(nums) < 3:
        return 0.0
    return sum(1 for a, b in zip(nums, nums[1:]) if b >= a) / (len(nums) - 1)


def texto_do_pdf(pdf: Path) -> list[str]:
    """
    Uma pagina por elemento, na melhor das duas leituras disponiveis.

    `-layout` preserva a indentacao dos incisos e e melhor em documento de coluna unica.
    `-raw` segue a ordem do fluxo de conteudo, que e o que salva o documento de DUAS
    colunas: em `-layout` as colunas se intercalam no meio da frase. Seis documentos
    deste corpus sao de duas colunas, o Estatuto da SBA entre eles — e o Estatuto e o
    primeiro item da bibliografia declarada pela banca.

    Em vez de adivinhar o layout, extrai as duas e deixa `_monotonia` escolher.
    """
    saidas = {}
    for modo in ("-layout", "-raw"):
        r = subprocess.run(["pdftotext", modo, str(pdf), "-"],
                           capture_output=True, text=True)
        saidas[modo] = r.stdout
    ml, mr = _monotonia(saidas["-layout"]), _monotonia(saidas["-raw"])
    # empate fica com -layout: mantem a indentacao, e e o que ja estava medido
    melhor = "-raw" if mr > ml else "-layout"
    return saidas[melhor].split("\f")


def limpar(t: str) -> str:
    for p in _LIXO:
        t = p.sub("", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def rotulo(pdf: Path, primeira_pagina: str) -> tuple[str, str]:
    """
    (identificador curto, titulo legivel). O identificador vai no campo `livro` e e o
    que aparece na citacao — precisa ser reconhecivel por quem le a resposta.
    """
    nome = pdf.stem
    m = re.match(r"CFM-(\d+)-(\d{4})", nome)
    if m:
        return f"CFM {m.group(1)}/{m.group(2)}", f"Resolução CFM nº {m.group(1)}/{m.group(2)}"
    if nome.startswith("SBA-"):
        limpo = re.sub(r"^SBA-\d*_?", "", nome).replace("_", " ").strip()
        return f"SBA · {limpo}", f"SBA — {limpo}"
    if nome.startswith("AMB-"):
        return "AMB Diretrizes", "Projeto Diretrizes AMB"
    if nome.startswith("SBC-"):
        return "SBC Diretriz", "Diretriz da Sociedade Brasileira de Cardiologia"
    # primeira linha nao vazia da pagina 1, como ultimo recurso
    linha = next((l.strip() for l in primeira_pagina.splitlines() if l.strip()), nome)
    return nome, linha[:90]


def fatiar_por_artigo(texto: str) -> list[tuple[str, str]]:
    """
    Devolve [(rotulo_do_trecho, texto)]. O corte e no ARTIGO; capitulo e secao viram
    contexto do artigo, nao trechos proprios — um capitulo sozinho nao responde nada.
    """
    marcas: list[tuple[int, str, str]] = []
    for m in _ARTIGO.finditer(texto):
        marcas.append((m.start(), "artigo", f"Art. {m.group(1)}"))
    for m in _ANEXO.finditer(texto):
        marcas.append((m.start(), "anexo", m.group(1).upper()))
    for m in _CAPITULO.finditer(texto):
        marcas.append((m.start(), "capitulo", " ".join(m.group(1).split())[:70]))
    marcas.sort()

    if not any(t == "artigo" for _, t, _ in marcas):
        # preambulo, considerandos, diretriz em prosa: o documento inteiro e um trecho
        return [("", texto)] if texto.strip() else []

    saida: list[tuple[str, str]] = []
    cap = ""
    cortes = [m for m in marcas if m[1] != "capitulo"]
    caps = [m for m in marcas if m[1] == "capitulo"]
    # tudo antes do primeiro artigo e a ementa: vale como trecho proprio
    if cortes and cortes[0][0] > 200:
        saida.append(("Ementa", texto[:cortes[0][0]].strip()))
    for i, (pos, _, rot) in enumerate(cortes):
        fim = cortes[i + 1][0] if i + 1 < len(cortes) else len(texto)
        for cpos, _, crot in caps:
            if cpos <= pos:
                cap = crot
        corpo = texto[pos:fim].strip()
        if len(corpo) > 25:
            saida.append((f"{cap} · {rot}" if cap else rot, corpo))
    return saida


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", type=Path, required=True)
    ap.add_argument("--saida", type=Path, required=True)
    a = ap.parse_args()

    pdfs = sorted(a.entrada.glob("*.pdf"))
    if not pdfs:
        print(f"nenhum PDF em {a.entrada}", file=sys.stderr)
        return 1

    escritas = vazios = 0
    with a.saida.open("w") as f:
        for n, pdf in enumerate(pdfs, 1):
            paginas = texto_do_pdf(pdf)
            inteiro = limpar("\n".join(paginas))
            if len(inteiro) < 200:
                vazios += 1
                continue
            ident, titulo = rotulo(pdf, paginas[0] if paginas else "")
            # mapa posicao -> pagina, para a citacao apontar a pagina certa
            limites, acc = [], 0
            for p in paginas:
                acc += len(limpar(p)) + 1
                limites.append(acc)
            for rot, corpo in fatiar_por_artigo(inteiro):
                pos = inteiro.find(corpo[:80])
                pag = next((i + 1 for i, lim in enumerate(limites) if lim > max(pos, 0)), 1)
                f.write(json.dumps({
                    "pagina": pag,
                    "secao_num": None, "secao": titulo,
                    "capitulo_num": None, "capitulo": rot or titulo,
                    "titulos": [], "texto": corpo,
                    "livro": ident,
                }, ensure_ascii=False) + "\n")
                escritas += 1
            sys.stderr.write(f"\r  {n}/{len(pdfs)} documentos · {escritas} trechos")
    sys.stderr.write("\n")
    print(f"{escritas} trechos de {len(pdfs)-vazios} documentos "
          f"({vazios} sem texto) -> {a.saida}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
