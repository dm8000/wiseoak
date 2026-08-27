# WiseOak

RAG de anestesiologia rodando **inteiramente local**: um Gemma 4 31B responde questões de
provas da Sociedade Brasileira de Anestesiologia consultando um livro-texto e um corpus de
normas, citando capítulo, página e artigo. Nenhum dado sai da máquina.

Aqui, o objetivo é **medir qual desenho de RAG + fluxo de trabalho ganha** comparado com
modelos flagships grandes, com teste estatístico, e registrar o que foi refutado com a mesma 
seriedade do que funcionou.

Queremos um fluxo que possa ser consultado para fins médicos ou de estudo. O modelo deve
retornar não apenas com uma resposta, mas também a fonte exata de onde a resposta foi tirada.
As provas da Sociedade são usadas *para testar a capacidade do fluxo de trabalho*. Este projeto
não é pra ser um "passador de provas". 

📊 **[Relatório com os resultados](eval/analises/bancada.html)** · 📓 **[ESTADO.md](ESTADO.md)** —
memória de trabalho, decisões e medições · 📦 **[CORPUS.md](CORPUS.md)** — o que você precisa
trazer (modelo e livro) e como reconstruir

---

## Onde estamos

998 questões do conjunto de desenvolvimento, mesmo universo para todos os sistemas.
Wilson para o intervalo, McNemar exato pareado com correção de Holm.

| sistema | total | V/F | ME | IC 95% |
|---|---:|---:|---:|---:|
| Opus 5 · effort low, sem busca | 85,4% | 85,7% | 83,8% | 83,0–87,4% |
| **WiseOak v3** · cota normas+livro | **80,6%** | 82,1% | 73,1% | 78,0–82,9% |
| WiseOak v1 · RAG Miller 10e | 80,2% | 81,5% | 73,7% | 77,6–82,5% |
| WiseOak v2 · RAG normas, roteado | 79,9% | 81,2% | 73,1% | 77,3–82,2% |
| DeepSeek-V3 · thinking + search | 77,1% | 78,5% | 70,1% | 74,3–79,6% |
| DeepSeek-V3 · thinking | 76,4% | 76,8% | 74,3% | 73,6–78,9% |
| WiseOak v0 · sem RAG | 73,0% | 74,5% | 65,9% | 70,2–75,7% |

Reproduza com `python bench/placar.py` — lê `eval/resultados.sqlite`, não roda nada.

**O que está estabelecido:** RAG bate o controle sem RAG com folga (Holm < 0,001), e o
Opus 5 sem busca ainda lidera por 4,8 pp. **O que NÃO está:** nenhuma das três variantes de
RAG é distinguível das outras — vivem dentro de 0,7 pp, e o piso de ruído é ~1 pp.

---

## Os três achados que valem mais que o placar

**1. O livro certo importou mais que qualquer ajuste de RAG.** Dois dias medindo com o
*Miller — Bases da Anestesia* (o resumo introdutório) não mostravam ganho nenhum de RAG.
Trocar pelo Miller's Anesthesia completo levou de 74,2% para 81,1%, e a fidelidade de
citação de 62% para 92%.

**2. A bancada tem piso de ruído de 3,9%.** Questões roteadas ao livro executam pipeline
idêntico entre variantes — a diferença deveria ser zero, e 29 de 747 itens trocam de resposta
mesmo assim, por `temperature=0.3`. Consequência dura: análise de subgrupo produziu **três
falsos positivos com p < 0,05** em classes de pipeline idêntico. `bench/porclasse.py` marca
isso sozinho.

**3. 95% das respostas estão no corpus — a falha é encontrar, não ter.** Dos erros clínicos
testáveis: 41% têm a resposta no top-50 mas fora do contexto (ranking), 23% no livro mas fora
do top-50 (busca), **31% chegam ao modelo e ele erra assim mesmo** (raciocínio), e só 5% são
ausência real.

---

## Como funciona

```
pergunta → rotear → recuperar → contexto → responder → resposta + citação verificada
```

O **roteador** classifica a pergunta em duas camadas: regex de custo zero (dispara só em
âncora literal — "CFM", "Resolução nº", "Código de Ética") e, se não casar, uma chamada curta
ao LLM com schema fechado. Medido: precisão 100%, cobertura 97,6%.

A **cota** é o desenho vencedor. Pergunta normativa recebe metade das vagas de contexto do
corpus de normas e metade do livro, em vez de tudo de um só. Não é o mesmo que fundir os
índices: em índice único as vagas saem de competição global, e 607 trechos de norma perdem por
volume para 14.628 do Miller. Com cota, norma compete só com norma.

A **verificação de citação é programática**, nunca por juízo do próprio modelo: casamento de
string contra o trecho recuperado, mínimo de 6 palavras.

### Grafos

| grafo | o que é | veredito |
|---|---|---|
| `v0` | sem RAG | controle |
| `v1` | busca densa + citação | linha de base forte |
| `v2` | híbrido BM25 + rerank | rerank custa 17 s/questão e não melhora |
| `v7` | consulta traduzida para inglês | **piorou muito** (51,2% → 32,6%) |
| `v8` | roteamento exclusivo de corpus | conserta o jurídico, **quebra a gestão** |
| `v9` | v8 + ferramenta de formulário | pronto, alcance ~0,5% — não mensurável |
| `v10` | **cota por corpus** | melhor configuração medida |

---

## Estrutura

```
wiseoak/
  clientes.py          fala com o llama-swap (chat, embed, rerank, tool_calls)
  store.py             índice: SQLite + numpy, cosseno exato (não ANN)
  grafos/comum.py      estado, nós, ancoragens, verificação de citação
  grafos/variantes.py  v0–v10 e compilação de grafo a partir de sketch
  ingest/              Miller (nativo/scan), Barash, normas, chunking, indexação
  ferramentas/         formulário de anestesiologia (uma tool, sem calculadora)
  servir/pipe.py       manifold do Open WebUI — cada grafo vira um modelo na lista
bench/
  runner.py            harness de medição, estratificação, guarda térmica
  placar.py            a tabela comparativa de todos os sistemas
  porclasse.py         comparação pareada por classe, marca ruído sozinho
  roteamento.py        o roteador acertou o destino?
  coletar_normas.py    baixa o corpus normativo público
eval/
  resultados.sqlite    a bancada: toda célula medida, resultado por item
  analises/            o registro de por que cada número é o que é
```

## Rodar

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python bench/placar.py                 # a tabela, sem rodar nada
langgraph dev                                    # LangGraph Studio em :2024
```

Veja **[CORPUS.md](CORPUS.md)** para o modelo e o livro, que não estão versionados.

### LangGraph Studio

`langgraph.json` expõe cada variante como um grafo próprio. Todo nó emite no estado uma
linha `faz` dizendo o que ele fez e por quê — clicar num nó no Studio mostra a decisão, não
só o tempo. Basta `{"pergunta": "..."}` para rodar; o resto tem padrão.

### Open WebUI

`wiseoak/servir/pipe.py` é um manifold: cada grafo aparece como um modelo separado no
seletor, então dá para comparar v1 e v10 conversando, sem mexer em configuração. A resposta
vem com as fontes e com a contagem de citações que conferem literalmente com o texto
recuperado.

---

## Disciplina de medição

Regras que este projeto segue, e que explicam por que várias coisas foram refutadas:

- verificação **programática**, nunca LLM como juiz
- truncagem conta como **falha**, em coluna separada
- `n` exclui item invalidado — "não respondeu" nunca vira "errou"
- **A/A antes de A/B**
- o conjunto de **teste segue intocado** até a configuração estar escolhida no dev
- o que foi refutado fica registrado no `ESTADO.md` com o número que o refutou
