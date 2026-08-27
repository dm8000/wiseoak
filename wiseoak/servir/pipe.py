"""
title: WiseOak — anestesiologia com RAG
author: phobos
version: 0.1.0
required_open_webui_version: 0.5.0
description: Consulta ao livro-texto de anestesiologia. Cada grafo do experimento vira um modelo na lista.
"""

# Manifold: o metodo `pipes()` faz cada variante do grafo aparecer como um modelo
# separado no seletor do Open WebUI. E o que permite comparar v2 e v3 conversando, sem
# mexer em configuracao.
#
# O contexto vai como `__event_emitter__` (status por no) e a resposta sai em streaming
# de texto. A citacao e renderizada depois da resposta, com capitulo, pagina e trecho —
# e o proposito do sistema; sem ela a resposta nao e verificavel pelo usuario.

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, AsyncGenerator

from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


class Pipe:
    class Valves(BaseModel):
        # O Open WebUI procura Pipe.Valves para renderizar o editor de configuracao.
        indice: str = Field(
            default=str(RAIZ / "dados" / "indice" / "m10"),
            description="Caminho base do indice (sem extensao)")
        modelo: str = Field(
            default="gemma4-plan",
            description="Modelo no llama-swap. gemma4-plan e o medido como melhor aqui.")
        raciocinio: str = Field(
            default="nenhum",
            description="nenhum | prompt | nativo (nativo so funciona no gemma4-plan)")
        k_busca: int = Field(default=20, description="candidatos recuperados")
        k_contexto: int = Field(default=5, description="trechos enviados ao modelo")

    def __init__(self):
        self.valves = self.Valves()
        self._indice = None

    def pipes(self) -> list[dict]:
        # v1 primeiro: e o que ganhou medindo. Rerank e expansao para o pai custavam
        # 17 s e nao melhoravam o acerto; o v2 fica por comparacao, nao por merito.
        return [
            {"id": "v1", "name": "WiseOak · busca + citação"},
            {"id": "v2", "name": "WiseOak · com rerank (mais lento)"},
            {"id": "v7", "name": "WiseOak · consulta traduzida para inglês"},
            {"id": "v0", "name": "WiseOak · sem RAG (controle)"},
        ]

    def _obter_indice(self):
        from wiseoak.store import Indice
        if self._indice is None:
            self._indice = Indice(self.valves.indice)
        return self._indice

    async def pipe(self, body: dict, __event_emitter__=None) -> AsyncGenerator[str, None]:
        from wiseoak.grafos.comum import verificar_citacoes
        from wiseoak.grafos.variantes import construir, estado_inicial

        nome = (body.get("model") or "v2").split(".")[-1]
        mensagens = body.get("messages") or []
        pergunta = next((m["content"] for m in reversed(mensagens)
                         if m.get("role") == "user"), "")
        if not pergunta:
            yield "Faça uma pergunta sobre anestesiologia."
            return

        async def status(texto: str, done: bool = False):
            if __event_emitter__:
                await __event_emitter__({"type": "status",
                                         "data": {"description": texto, "done": done}})

        await status("montando o grafo…")
        try:
            grafo = construir(nome)
        except KeyError:
            yield f"Grafo desconhecido: {nome}"
            return

        indice = None if nome == "v0" else self._obter_indice()
        estado = estado_inicial(
            pergunta, modo="livre", modelo=self.valves.modelo,
            raciocinio=self.valves.raciocinio, indice=indice,
            k_busca=self.valves.k_busca, k_contexto=self.valves.k_contexto)

        await status("consultando o livro…")
        # O grafo e sincrono e faz I/O de rede; roda numa thread para nao travar o loop.
        resultado = await asyncio.to_thread(grafo.invoke, estado)

        for passo in resultado.get("trace") or []:
            await status(f"{passo['no']} · {passo['segundos']}s")
        await status("pronto", done=True)

        yield resultado.get("resposta") or "(sem resposta)"

        contexto = resultado.get("contexto") or []
        citacoes = resultado.get("citacoes") or []
        if not citacoes:
            if nome != "v0":
                yield "\n\n---\n*Sem citação: a resposta não foi ancorada no livro.*"
            return

        v = verificar_citacoes(resultado)
        yield "\n\n---\n### Fontes\n"
        for c in citacoes:
            yield (f"\n- **{c.get('livro', 'Miller\'s Anesthesia 10e')}**, "
                   f"capítulo {c.get('capitulo')}, página {c.get('pagina')}\n"
                   f"  > {str(c.get('trecho', '')).strip()}\n")
        # Fidelidade verificada por casamento de string contra o trecho recuperado —
        # nunca por juizo do proprio modelo.
        yield (f"\n*{v['fieis']} de {v['citacoes']} citações conferem literalmente com "
               f"o texto recuperado; {v['pagina_ok']} com a página correta.*\n")
        yield f"\n<details><summary>Trechos consultados ({len(contexto)})</summary>\n\n"
        for i, c in enumerate(contexto, 1):
            caminho = " > ".join(c.get("caminho") or [])
            yield (f"{i}. cap. {c.get('capitulo_num')} · p. {c.get('pagina_inicial')}"
                   f"{' · ' + caminho if caminho else ''}\n")
        yield "\n</details>\n"
