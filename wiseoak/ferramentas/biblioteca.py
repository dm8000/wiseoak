#!/usr/bin/env python3
"""
A biblioteca: o modelo ve o catalogo e pede o que quiser, como uma pessoa usaria os livros.

Alternativa a busca puramente vetorial, motivada por um caso MEDIDO. Numa questao sobre
cirrotico com sindrome hepatorrenal, cujas alternativas eram manitol/dobutamina/
terlipressina/fenoldopam, a busca densa devolveu quatro paragrafos sobre MANITOL — o
distrator — porque a palavra estava escrita na consulta. Uma pessoa olhando um sumario
iria ao capitulo de doenca hepatica e nunca cairia nisso: **coincidencia de palavra nao
seduz quem navega por estrutura**.

TRES ferramentas, e nao mais: a bancada deste projeto mediu que a precisao de tool calling
cai conforme o numero cresce.

  biblioteca()            o catalogo — obras, capitulos, assuntos de cada um
  ler(obra, referencia)   o conteudo daquele capitulo/artigo
  buscar(consulta)        a busca vetorial de sempre, se o modelo preferir

`ler` no corpus NORMATIVO devolve o artigo inteiro — sao curtos, cabem. No Miller devolve a
busca RESTRITA ao capitulo pedido, porque capitulo tem mediana de 128 paginas e nao cabe em
contexto nenhum. Essa restricao e o que torna a captura por distrator impossivel por
construcao: escolhido o capitulo de figado, os paragrafos de manitol dos capitulos renais
ficam fora do alcance.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
_SUMARIO: dict | None = None


def _sumario() -> dict:
    global _SUMARIO
    if _SUMARIO is None:
        p = RAIZ / "dados" / "sumario_m10.json"
        _SUMARIO = json.loads(p.read_text()) if p.exists() else {}
    return _SUMARIO


def catalogo(ix_livro, ix_normas, max_topicos: int = 6) -> str:
    """O que existe na biblioteca. Compacto de proposito: vai inteiro no contexto."""
    s = _sumario()
    linhas = ["OBRA: miller — Miller's Anesthesia, 10a edicao (livro-texto, ingles)"]
    for num in sorted(s, key=lambda x: int(x)):
        c = s[num]
        t = "; ".join(c["topicos"][:max_topicos])
        linhas.append(f"  cap {num}: {c['titulo']}" + (f" — {t}" if t else ""))
    linhas.append("")
    linhas.append("OBRA: normas — resolucoes do CFM, estatuto e regimentos da SBA, "
                  "diretrizes AMB/SBC (portugues)")
    vistos: dict[str, list[str]] = {}
    for livro, cap in ix_normas.db.execute(
            "SELECT DISTINCT livro, capitulo FROM chunk WHERE nivel='filho' "
            "ORDER BY livro, capitulo"):
        vistos.setdefault(livro, []).append(str(cap))
    for livro, caps in vistos.items():
        linhas.append(f"  {livro}: {', '.join(caps[:14])}"
                      + (" …" if len(caps) > 14 else ""))
    return "\n".join(linhas)


# Acima disto, `ler` para de devolver a referencia inteira e busca DENTRO dela. Um artigo
# de resolucao cabe; um capitulo do Miller tem mediana de 128 paginas e nao cabe em
# contexto nenhum.
ORCAMENTO_CHARS = 9000


def ler(ix_livro, ix_normas, obra: str, referencia: str, consulta: str, k: int) -> list[dict]:
    obra = (obra or "").strip().lower()
    ref = (referencia or "").strip()
    if obra.startswith("norm"):
        linhas = ix_normas.db.execute(
            "SELECT id, LENGTH(texto) FROM chunk WHERE nivel='filho' "
            "AND (livro LIKE ? OR capitulo LIKE ?) ORDER BY ordem",
            (f"%{ref}%", f"%{ref}%")).fetchall()
        total = sum(x[1] or 0 for x in linhas)
        if total <= ORCAMENTO_CHARS:
            # cabe inteiro: devolve tudo, que e o ponto de navegar por estrutura
            return [{**ix_normas.obter(r[0]), "natureza": "NORMA"} for r in linhas]
        permitidos = {r[0] for r in linhas}
        saida = []
        for cid, _ in ix_normas.buscar(consulta, 200, hibrido=False):
            if cid in permitidos:
                saida.append({**ix_normas.obter(cid), "natureza": "NORMA"})
                if len(saida) >= k:
                    break
        return saida
    # Miller: capitulo nao cabe em contexto, entao busca DENTRO dele
    num = "".join(ch for ch in ref if ch.isdigit())
    if not num:
        return []
    permitidos = {r[0] for r in ix_livro.db.execute(
        "SELECT id FROM chunk WHERE nivel='filho' AND capitulo_num=?", (int(num),))}
    if not permitidos:
        return []
    saida = []
    for cid, _ in ix_livro.buscar(consulta, 200, hibrido=False):
        if cid in permitidos:
            saida.append({**ix_livro.obter(cid), "natureza": "LIVRO"})
            if len(saida) >= k:
                break
    return saida


# DUAS ferramentas, nao tres. A `biblioteca()` foi removida: o catalogo agora vai no
# `system`, identico entre perguntas, o que ativa o cache de prefixo do llama-swap (13,5 s
# na primeira chamada, 1,3 s nas seguintes). Como tool ela custaria uma rodada inteira
# para entregar o que ja esta no contexto.
ESPECIFICACOES = [
    {"type": "function", "function": {
        "name": "ler",
        "description": ("Le uma parte especifica da biblioteca. Use depois de consultar o "
                        "catalogo, quando souber onde o assunto mora."),
        "parameters": {"type": "object", "properties": {
            "obra": {"type": "string", "description": "'miller' ou 'normas'"},
            "referencia": {"type": "string",
                           "description": "numero do capitulo (miller) ou "
                                          "documento/artigo (normas)"},
            "assunto": {"type": "string",
                        "description": "o que procurar dentro dessa parte"}},
            "required": ["obra", "referencia", "assunto"]}}},
    {"type": "function", "function": {
        "name": "buscar",
        "description": ("Busca por similaridade em toda a biblioteca, sem escolher "
                        "capitulo. Util quando voce nao sabe onde o assunto mora. "
                        "CUIDADO: se a sua consulta contiver nomes de candidatos a "
                        "resposta, ela tende a trazer material sobre o candidato errado — "
                        "descreva o PROBLEMA, nao as opcoes."),
        "parameters": {"type": "object", "properties": {
            "consulta": {"type": "string"}}, "required": ["consulta"]}},
     },
]
