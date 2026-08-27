# Corpus: o que está aqui e o que você precisa trazer

Duas coisas ficaram **de fora deste repositório de propósito**: o modelo e os livros. Ambos
são de terceiros e não são nossos para redistribuir. Este arquivo diz exatamente o que são e
como reconstruir o que falta — em um comando cada.

Tudo o mais está versionado: código, bancada de medição com todos os resultados
(`eval/resultados.sqlite`), análises, o corpus normativo público e as provas.

---

## 1. O modelo

Nada aqui depende de peso de modelo baixado. O sistema fala com um servidor
**llama-swap** local, compatível com a API da OpenAI, em `127.0.0.1:9292`.

| papel | modelo | onde roda | observação |
|---|---|---|---|
| geração | **Gemma 4 31B** (`gemma4-plan`) | GPU, ~18 GB | tem canal de *thinking* e emite `tool_calls` |
| embedding | **EmbeddingGemma 300M** (`embed-small`) | CPU | contexto 2048, é o que indexa |
| rerank | **Qwen3-Reranker 0.6B** (`rerank-small`) | CPU | medido e NÃO usado no fluxo vencedor |
| comparação | **MedGemma 27B** (`medgemma-clinical`) | GPU | sem canal de thinking, sem `tool_calls` |

Os nomes na primeira coluna são os `model` que o código envia. Se o seu servidor usar
outros, ajuste `MODELO_CHAT` e `MODELO_EMBED` em `wiseoak/clientes.py`, ou passe
`--modelos` no runner.

**Medição relevante:** só um modelo de ~18 GB cabe na GPU por vez, e trocar custa ~5,5 s.
Grafos que misturam dois modelos grandes pagam esse custo por nó.

---

## 2. Os livros

O corpus clínico é o **Miller's Anesthesia, 10ª edição** (Gruber, Miller et al.).
É livro comprado; o texto integral não está aqui.

O que ESTÁ aqui é o pipeline inteiro de ingestão. Com o seu exemplar em PDF:

```bash
# 1. PDF -> páginas com capítulo e seção (usa o outline do PDF)
python3 -m wiseoak.ingest.nativo --pdf SEU_MILLER.pdf --saida dados/miller10_paginas.jsonl

# 2. páginas -> chunks hierárquicos pai/filho
python3 wiseoak/ingest/chunk.py dados/miller10_paginas.jsonl \
        --saida dados/chunks_miller10.jsonl --min-pai 200 --tamanho-filho 512

# 3. chunks -> índice (embedding em CPU, não toca a VRAM)
python3 -m wiseoak.ingest.indexar dados/chunks_miller10.jsonl --indice dados/indice/m10
```

Deve sair **14.628 trechos filho**. Se o seu número divergir muito, o PDF provavelmente
tem outro layout — confira o `--tamanho-filho`.

sha256 do PDF que usamos: `0cc8d5cd82173797437560a1c5df0456cbf0a749541bc28de8b427f1440aff5c`

**Isto importa:** a primeira versão do projeto usou o *Miller — Bases da Anestesia* (o
resumo introdutório, em português) e o RAG não ajudava em nada. Trocar pelo Miller completo
levou o acerto de 74,2% para 81,1%. O livro certo foi a variável mais importante do projeto
inteiro. Não substitua pelo resumo.

O **Barash, Clinical Anesthesia** aparece no histórico (`wiseoak/ingest/barash.py`,
índice `dois`) e foi medido e descartado — está documentado no `ESTADO.md`.

---

## 3. O que já vem pronto

| caminho | o que é |
|---|---|
| `dados/normas/` | 54 documentos normativos: resoluções do CFM, estatuto/regimentos/regulamentos da SBA, Código de Ética Médica 2019, diretrizes AMB e SBC. **Todos públicos.** |
| `dados/indice/normas.*` | índice pronto desses documentos (607 trechos). Funciona sem reconstruir nada. |
| `dados/provas/` | provas da SBA em PDF, a origem do banco de questões |
| `dados/questoes_sba.*.jsonl` | banco parseado: `dev` (1.003 itens) e `teste` (**intocado**, reservado para validação final) |
| `eval/resultados.sqlite` | a bancada inteira: toda célula medida, com resultado por item |
| `eval/analises/` | o registro de por que cada número é o que é, incluindo `bancada.html` |

Para rebaixar o corpus normativo do zero: `python3 bench/coletar_normas.py`
(tem `--dry-run`, que lista as URLs sem baixar).

---

## 4. Rodar

```bash
python3 -m venv .venv && .venv/bin/pip install -e .

# uma pergunta, pelo grafo vencedor
.venv/bin/python bench/runner.py --modelos gemma4-plan --grafos v10 --bench vf \
    --split dev --n 20 --ancoragem confiante --indice dados/indice/m10 --bloco FUMACA

# a tabela comparativa completa (lê o que já está medido, não roda nada)
.venv/bin/python bench/placar.py
```

O `placar.py` funciona sem o Miller: ele só lê `eval/resultados.sqlite`.
