"""
Clientes HTTP para o llama-swap em 127.0.0.1:9292.

Tres funcoes, uma por peca do sistema:

    chat()    gemma4-plan, na GPU. thinking por requisicao.
    embed()   embed-small (EmbeddingGemma 300M), na CPU.
    rerank()  rerank-small (Qwen3-Reranker 0.6B), na CPU.

O contrato de `chat` e o mesmo de Cloud_assistant/eval/bench/__init__.py:chat() de
proposito: o que for medido na bancada tem que valer aqui sem traducao.

Por que nao sentence-transformers: embed e rerank ja estao servidos pelo llama-swap no
"andar de RAM" (persistent, ttl 0, CUDA_VISIBLE_DEVICES vazio). Indexar o livro inteiro
nao toca a VRAM e nao derruba o MedGemma da GPU.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx

BASE = os.environ.get("WISEOAK_URL", "http://127.0.0.1:9292/v1")

# gemma4-plan e o default MEDIDO, nao uma preferencia. Contra o MedGemma, mesma
# amostra e mesmo prompt: mais acerto sem RAG (72,5% x 67,5%) e fidelidade de
# citacao muito melhor (65-70% x 34-48%) — o MedGemma parafraseia, e a
# verificacao programatica exige copia literal. Alem disso o MedGemma nao tem
# canal de thinking (e Gemma 3 por baixo) nem draft para speculative decoding.
MODELO_CHAT = os.environ.get("WISEOAK_MODELO", "gemma4-plan")
MODELO_EMBED = os.environ.get("WISEOAK_EMBED", "embed-small")
MODELO_RERANK = os.environ.get("WISEOAK_RERANK", "rerank-small")

# embed-small roda com --ctx-size 2048. Texto acima disso e recusado pelo servidor, e o
# erro chega como 500 no meio de um lote de milhares de chunks. Cortar antes, e alto.
CTX_EMBED = 2048
CHARS_MAX_EMBED = int(CTX_EMBED * 3.2)  # ~3,2 chars/token em pt

# O default do Open WebUI (RAG_EMBEDDING_BATCH_SIZE=1) faz uma requisicao HTTP por chunk e
# torna a indexacao lenta o bastante para parecer travada. 16 e o valor ja adotado no
# config/openwebui.env deste projeto.
LOTE_EMBED = 16

# O rerank-small roda com --ctx-size 4096 mas SEM --ubatch-size, entao o batch fisico
# fica no default de 512 tokens e o servidor recusa com HTTP 500 qualquer par
# (pergunta + documento) acima disso. Medido: um chunk de 608 tokens derruba a chamada.
#
# Truncar aqui e a correcao que NAO mexe em config/llama-swap.yaml, que e infraestrutura
# compartilhada. A alternativa correta e acrescentar `--ubatch-size 2048` ao perfil
# rerank-small — decisao do dono da maquina, nao minha.
#
# Efeito na medicao: o reranker julga o INICIO do chunk. Com mediana de 1.222 chars por
# filho, a maioria cabe inteira; so a cauda e cortada. Documentar, nao esconder.
CTX_RERANK = 512
CHARS_MAX_RERANK = int(CTX_RERANK * 2.2)


def _chave() -> str:
    """A chave do llama-swap. Mesma fonte da bancada, mesmo fallback."""
    try:
        return (Path(__file__).resolve().parents[2] / "config" / "llama-swap.key").read_text().strip()
    except Exception:
        return "llamaswap"


_cliente: httpx.Client | None = None


def cliente() -> httpx.Client:
    """Cliente unico, reaproveitado. Indexar abre milhares de requisicoes."""
    global _cliente
    if _cliente is None:
        _cliente = httpx.Client(
            base_url=BASE,
            headers={"Authorization": f"Bearer {_chave()}"},
            timeout=httpx.Timeout(900.0, connect=10.0),
        )
    return _cliente


class ErroLlamaSwap(RuntimeError):
    pass


def _post(rota: str, corpo: dict) -> dict:
    r = cliente().post(rota, json=corpo)
    if r.status_code != 200:
        raise ErroLlamaSwap(f"{rota} devolveu {r.status_code}: {r.text[:400]}")
    return r.json()


def chat(
    mensagens: list[dict],
    *,
    modelo: str = MODELO_CHAT,
    think: bool | None = None,
    max_tokens: int = 8192,
    temp: float | None = None,
    schema: dict | None = None,
    tools: list[dict] | None = None,
) -> dict:
    """
    Uma chamada de chat. Devolve content, reasoning, tokens e segundos.

    `think` vai em chat_template_kwargs, por REQUISICAO. Trocar de perfil para ligar
    thinking custaria um swap de 18 GB por no do grafo.

    max_tokens 8192 e o padrao da bancada: fica acima do teto de thinking observado, e
    resposta vazia por truncagem conta como falha na medicao. Nao baixar sem medir.
    """
    corpo: dict = {"model": modelo, "messages": mensagens,
                   "max_tokens": max_tokens, "stream": False}
    if think is not None:
        corpo["chat_template_kwargs"] = {"enable_thinking": bool(think)}
    if temp is not None:
        corpo["temperature"] = temp
    if tools:
        # schema e tools sao mutuamente exclusivos: um forca a saida a casar a gramatica
        # do JSON, o outro precisa da saida livre para emitir a chamada. Pedir os dois
        # faz o modelo devolver JSON e nunca chamar a ferramenta.
        if schema is not None:
            raise ValueError("schema e tools nao podem ir juntos na mesma chamada")
        corpo["tools"] = tools
        corpo["tool_choice"] = "auto"
    if schema is not None:
        corpo["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "r", "schema": schema, "strict": True},
        }

    t0 = time.time()
    d = _post("/chat/completions", corpo)
    escolha = d["choices"][0]
    msg = escolha["message"]
    uso = d.get("usage") or {}
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "finish": escolha.get("finish_reason"),
        "truncou": escolha.get("finish_reason") == "length",
        "out_tokens": uso.get("completion_tokens", 0),
        "in_tokens": uso.get("prompt_tokens", 0),
        "segundos": time.time() - t0,
    }


def embed(textos: list[str], *, modelo: str = MODELO_EMBED,
          lote: int = LOTE_EMBED) -> list[list[float]]:
    """
    Vetores para uma lista de textos, na ordem de entrada.

    Texto acima de CHARS_MAX_EMBED e cortado: o servidor recusaria o lote inteiro por
    causa de um chunk, no meio de uma indexacao de milhares.
    """
    if not textos:
        return []
    saida: list[list[float]] = []
    for i in range(0, len(textos), lote):
        bruto = textos[i:i + lote]
        # TRUNCAGEM ADAPTATIVA, como no rerank. Estimar chars/token nao funciona: 3,2 e a
        # razao da prosa corrida, mas texto denso (norma, tabela, termo tecnico) chega a
        # 2,6 — um corte calculado em 6.553 chars produziu 2.512 tokens e derrubou uma
        # indexacao inteira no meio. Encolhe-se ate o servidor aceitar.
        teto = CHARS_MAX_EMBED
        for _ in range(5):
            pedaco = [t[:teto] for t in bruto]
            try:
                d = _post("/embeddings", {"model": modelo, "input": pedaco})
                break
            except ErroLlamaSwap as e:
                if "too large to process" not in str(e) or teto <= 600:
                    raise
                teto = max(600, int(teto * 0.6))
        dados = sorted(d["data"], key=lambda x: x["index"])
        saida.extend(x["embedding"] for x in dados)
    if len(saida) != len(textos):
        raise ErroLlamaSwap(f"pedi {len(textos)} vetores, vieram {len(saida)}")
    return saida


def rerank(pergunta: str, documentos: list[str], *, top_n: int | None = None,
           modelo: str = MODELO_RERANK) -> list[tuple[int, float]]:
    """
    Reordena `documentos` por relevancia. Devolve [(indice_original, score)], do melhor
    para o pior. O indice e o da lista que entrou — quem chama resolve os metadados.
    """
    if not documentos:
        return []
    # Truncagem ADAPTATIVA. Estimar chars/token nao funciona aqui: 3,2 e a razao da
    # prosa, mas termo medico tokeniza muito pior, e um corte calculado em 1.568 chars
    # ainda produziu 601 tokens. Em vez de calibrar uma constante contra um tokenizador
    # que nao vejo, encolhe-se ate o servidor aceitar.
    teto = max(200, CHARS_MAX_RERANK - len(pergunta))
    ultimo: Exception | None = None
    for _ in range(5):
        corpo: dict = {"model": modelo, "query": pergunta,
                       "documents": [d[:teto] for d in documentos]}
        if top_n is not None:
            corpo["top_n"] = top_n
        try:
            d = _post("/rerank", corpo)
            break
        except ErroLlamaSwap as e:
            if "too large to process" not in str(e) or teto <= 200:
                raise
            ultimo = e
            teto = max(200, int(teto * 0.6))
    else:
        raise ultimo  # type: ignore[misc]
    pares = [(x["index"], x["relevance_score"]) for x in d["results"]]
    pares.sort(key=lambda p: p[1], reverse=True)
    return pares
