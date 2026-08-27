#!/usr/bin/env python3
"""
Gera o PDF do banco de questoes para aplicar a prova em outro modelo.

SEM GABARITO, de proposito: quem responde nao pode ver a resposta, e o gabarito fica so
do nosso lado para pontuar depois. Cada item leva um ID CURTO e estavel, que e como a
resposta devolvida volta a casar com a questao — sem isso, uma resposta fora de ordem ou
uma questao pulada desalinha tudo silenciosamente.

O formato de saida pedido e uma linha por item, `ID: RESPOSTA`, porque:
  - e trivial de parsear com regex, sem depender de JSON bem formado
  - sobrevive a texto solto antes e depois
  - permite conferir cobertura (quais IDs faltaram) em vez de assumir

    ./gerar_pdf.py                          # tudo, num PDF so
    ./gerar_pdf.py --partes 4               # divide em 4, para modelo de contexto menor
    ./gerar_pdf.py --split teste            # o conjunto de teste, ainda intocado
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer)

RAIZ = Path(__file__).resolve().parents[1]

INSTRUCOES = """
Este documento traz <b>{n} questões de anestesiologia</b> extraídas de provas da
Sociedade Brasileira de Anestesiologia. Elas vêm em dois formatos:

<b>V/F</b> — um enunciado seguido de uma assertiva. Responda <b>V</b> se a assertiva
for verdadeira e <b>F</b> se for falsa.<br/>
<b>ME</b> — um enunciado seguido de quatro alternativas. Responda com a
<b>letra</b> da alternativa correta: A, B, C ou D.

<b>Como devolver as respostas</b><br/>
Uma linha por questão, no formato <b>ID: RESPOSTA</b>, exatamente assim:

<font face="Courier">q0001: V<br/>q0002: F<br/>q0003: C</font>

Responda <b>todas</b> as questões, na ordem, sem pular. Se estiver em dúvida, escolha
a resposta mais provável em vez de deixar em branco — item sem resposta conta como
erro. Não escreva justificativa: só o ID e a resposta.
"""


def carregar(split: str) -> list[dict]:
    p = RAIZ / "dados" / f"questoes_sba.{split}.jsonl"
    itens = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    # ordem estavel: V/F primeiro, depois multipla escolha, cada bloco por id
    itens.sort(key=lambda i: (0 if i["tipo"] == "vf" else 1, i["id"]))
    return itens


def escapar(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def estilos():
    base = getSampleStyleSheet()
    return {
        "titulo": ParagraphStyle("t", parent=base["Title"], fontSize=17, spaceAfter=4),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=9.5,
                              textColor=colors.HexColor("#555555"), spaceAfter=14),
        "corpo": ParagraphStyle("c", parent=base["Normal"], fontSize=10.5, leading=14.5,
                                alignment=TA_JUSTIFY, spaceAfter=8),
        "id": ParagraphStyle("i", parent=base["Normal"], fontName="Courier-Bold",
                             fontSize=9.5, textColor=colors.HexColor("#0E7C7B"),
                             spaceAfter=3),
        "enun": ParagraphStyle("e", parent=base["Normal"], fontSize=10, leading=13.5,
                               alignment=TA_JUSTIFY, spaceAfter=3),
        "asser": ParagraphStyle("a", parent=base["Normal"], fontSize=10, leading=13.5,
                                alignment=TA_JUSTIFY, leftIndent=10, spaceAfter=2),
        "alt": ParagraphStyle("l", parent=base["Normal"], fontSize=10, leading=13,
                              leftIndent=14, spaceAfter=1),
    }


def montar(itens: list[dict], saida: Path, titulo: str,
           offset: int = 0) -> dict:
    st = estilos()
    doc = SimpleDocTemplate(str(saida), pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title=titulo, author="WiseOak")
    fluxo = [Paragraph(titulo, st["titulo"]),
             Paragraph("Prova para avaliação de modelo · sem gabarito", st["sub"]),
             Paragraph(INSTRUCOES.format(n=len(itens)), st["corpo"]),
             PageBreak()]

    chave: dict[str, str] = {}
    for n, it in enumerate(itens, offset + 1):
        rot = f"q{n:04d}"
        chave[rot] = it["id"]
        tipo = "V/F" if it["tipo"] == "vf" else "ME"
        bloco = [Paragraph(f"{rot}  [{tipo}]", st["id"]),
                 Paragraph(escapar(" ".join(it["enunciado"].split())), st["enun"])]
        if it["tipo"] == "vf":
            bloco.append(Paragraph(
                "<i>Assertiva:</i> " + escapar(" ".join(it["assertiva"].split())),
                st["asser"]))
        else:
            for letra, txt in sorted((it.get("alternativas") or {}).items()):
                bloco.append(Paragraph(
                    f"<b>{letra})</b> " + escapar(" ".join(txt.split())), st["alt"]))
        bloco.append(Spacer(1, 7))
        # KeepTogether: uma questao nao se parte entre paginas — enunciado numa pagina e
        # alternativas na seguinte e fonte classica de resposta trocada
        fluxo.append(KeepTogether(bloco))

    doc.build(fluxo)
    return chave


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("dev", "teste"))
    ap.add_argument("--partes", type=int, default=1,
                    help="divide em N arquivos, para modelo de contexto menor")
    ap.add_argument("--saida", type=Path, default=RAIZ / "dados" / "prova")
    a = ap.parse_args()

    itens = carregar(a.split)
    a.saida.mkdir(parents=True, exist_ok=True)
    chave_total: dict[str, str] = {}

    tam = (len(itens) + a.partes - 1) // a.partes
    for p in range(a.partes):
        fatia = itens[p * tam:(p + 1) * tam]
        if not fatia:
            continue
        sufixo = "" if a.partes == 1 else f"-parte{p+1}de{a.partes}"
        arq = a.saida / f"prova-{a.split}{sufixo}.pdf"
        titulo = ("Anestesiologia — banco SBA"
                  + (f" (parte {p+1} de {a.partes})" if a.partes > 1 else ""))
        # o deslocamento mantem o ID unico entre as partes: a parte 2 comeca onde a 1
        # terminou, e uma resposta nunca casa com a questao errada
        chave_total.update(montar(fatia, arq, titulo, offset=p * tam))
        print(f"  {arq.name}: {len(fatia)} questões", file=sys.stderr)

    gab = a.saida / f"gabarito-{a.split}.json"
    porid = {i["id"]: i for i in itens}
    gab.write_text(json.dumps(
        {"chave": chave_total,
         "gabarito": {rot: porid[iid]["resposta"] for rot, iid in chave_total.items()},
         "tipo": {rot: porid[iid]["tipo"] for rot, iid in chave_total.items()},
         "classe": {rot: porid[iid].get("classe") for rot, iid in chave_total.items()}},
        ensure_ascii=False, indent=1))
    print(f"\ngabarito (NAO enviar): {gab}", file=sys.stderr)
    print(f"{len(chave_total)} questões no total", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
