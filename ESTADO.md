# ESTADO — retomada do WiseOak

Documento de retomada. Se a sessão reiniciar, leia ISTO e o `README.md`; não é preciso
reler o histórico da conversa.

Última atualização: 2026-08-26 — corpus trocado para Miller's Anesthesia 10e; gemma4-plan é o default.

## Pronto e verificado

- **Fase 0** — `.venv` (py3.12, 266 MB), `wiseoak/clientes.py` com `embed`/`rerank`/`chat`.
  Verificar com: `.venv/bin/python bench/smoke_clientes.py` → 8/8.
- **Fase 1** — `dados/questoes_sba.jsonl` (2.331 itens de 87 provas) e
  `dados/questoes_healthqa.jsonl` (5.632, controle). Split `.dev.jsonl`/`.teste.jsonl`
  por hash. Verificar com: `cd bench && ../.venv/bin/python -m unittest -q test_parse_sba.py` → 7/7.

- **Fase 2 (parcial)** — `wiseoak/ingest/normalizar.py` (correção de OCR) e
  `wiseoak/ingest/estrutura.py` (reconstrução de colunas + capítulo/seção/títulos).
  `dados/miller_paginas.jsonl`: 791 páginas, 3,5 M chars, 47 capítulos, 6 seções,
  575 títulos. Verificar: `.venv/bin/python -m unittest -q bench.test_estrutura` → 10/10.
- **Fase 2** — `chunk.py` (hierárquico pai/filho, 10/10 em `bench.test_chunk`),
  `store.py` (SQLite + numpy + BM25 + RRF) e `ingest/indexar.py`.
  `dados/chunks_h512.jsonl`: 2.062 pais / 3.403 filhos, mediana 1.241 chars.
  `dados/chunks_plano512.jsonl` (controle `--sem-hierarquia`): 789 pais / 2.861 filhos.
- **Fase 3** — `grafos/comum.py` (estado, nós, verificação programática de citação) e
  `grafos/variantes.py` (v0–v4 em LangGraph). `bench/runner.py` grava no schema da
  bancada. Fumaça: v0 responde em 0,8 s p50.

## Decisões do usuário (não re-perguntar)

| tema | decisão |
|---|---|
| **prioridade** | **acerto primeiro. Latência é só registro, não critério de decisão** |
| ancoragem | dimensão medida: estrita / com_ressalva / confiante / analista |
| formato de citação | bibliografia no fim: livro, capítulo, página e trecho exato entre aspas |

| tema | decisão |
|---|---|
| corpus | **Miller** `~/Downloads/Bases Da Anestesia - 6ª Edição - Miller (1).pdf`. O Barash falhou o download; usuário tentará outro depois |
| modelos | medir **`medgemma-clinical` E `gemma4-plan`** como dimensão do experimento |
| telemetria | harness local, sem LangSmith em nuvem |
| Studio | não é prioridade. Começar por `draw_mermaid()` (custo zero); Phoenix só se fizer falta |
| benchmark | provas da SBA + HealthQA-BR como controle |

## Fatos medidos que não devem ser re-descobertos

- **MedGemma não tem canal de thinking** (controle positivo em qwen-code). Gemma 4 tem.
  Vira contraste entre os braços, não uma escolha.
- **OCR do Miller corrompe a ligadura `fl`→`tl`** em nomes de fármaco: 24% de isoflurano,
  34% de desflurano, 38% de sevoflurano (medido em 120 páginas). Atinge o BM25, não tanto
  o denso. "Com e sem normalização de OCR" é dimensão a medir.
- **Outline do Miller é lixo** (nomes de arquivo do scanner). A estrutura vem do
  **cabeçalho corrente** de cada página (`Capítulo N Título` / `Seção N TÍTULO`).
- **O OCR tem QUATRO padrões, não um.** Além de `fl`→`tl`: `fl`→`ft`/`ff`
  (`desfturano`, `metoxiffurano`), `fi`→`ft` (`insuftciência`) e **`I`→`l` no início da
  palavra** (`lntubação` 32x, `lsquemia` 19x, `lsoflurano` 11x — 578 ocorrências).
  Todas as regras se apoiam em sequências impossíveis em português (`tlu`, `ftu`, `ffu`,
  `ftc`, e os onsets `ln`/`ls`/`lm`), por isso são seguras. Todas zeradas.
- **Título de seção NÃO sai da altura de linha** — aquilo pega legenda de figura. Sai da
  FORMA da linha: curta, só letras, seguida de prosa. Dois níveis: MAIÚSCULO e Title-case.
- **Tabela vira falso título** (uma tabela de agentes gera `Sevoflurano`, `Desflurano`,
  `Data de introdução` como seções). Daí `--min-pai=200`, que funde pai curto no anterior:
  mediana do filho vai de 853 para 1.241 chars.
- **`pdftohtml -xml` não lê a camada de OCR deste PDF** (200 elementos em 791 páginas).
  O que funciona é **`pdftotext -bbox-layout`** (541.802 palavras com coordenadas).
- **Duas colunas, corte em x=352** de 704,25 pt. Sem separar as colunas, rótulos de eixo
  de figura (`0,01`) entram no meio das frases — foi o defeito que motivou o módulo.
- **Numeral romano da seção vem OCRizado como dígito**: `11`=II, `111`=III.
- Comparação de cabeçalho tem que ser **insensível a acento** (`Seção` vs `Secao`).
- **FlowBot não tem código público.** Bilevel: laço externo muta a estrutura
  (acrescentar/remover/fundir chamadas), interno otimiza prompt por gradiente textual.
  Precisa de gabarito + conjunto de validação, que a Fase 1 já produziu.
- Acaso a bater: **V/F 51,3%**, **MCQ 25%**.

- **Ordem de leitura em duas colunas é POR BANDAS**, não por faixa de y. Ordenar tudo
  por `y` intercala as colunas: na p96 a lista de rótulos de uma figura entrava no meio
  das frases do corpo. Bloco de largura total separa bandas; dentro da banda, esquerda
  inteira e depois direita.
- **Meu primeiro teste de coluna era fraco.** Validava só a p100, onde os rótulos eram
  numéricos e caíam no filtro de ruído — passou verde com o defeito ativo. O caso da p96
  (rótulos que são texto) agora está travado em
  `test_coluna_de_figura_nao_entra_no_meio_da_frase`.
- **`pgrep -f`/`pkill -f` casam com o próprio comando** que contém a string. Já matei o
  meu shell com isso. Para processo em segundo plano, use sentinela em disco
  (`dados/indice/.pronto`), não PID. Ver [[pid-de-background-nao-e-do-processo]].

- **O reranker custa ~1,2 s POR DOCUMENTO na CPU** (23,6 s para 20 docs; 11,7 s para
  10; linear). Com `k_busca=20` o rerank sozinho gasta 24 s — inviável para "consulta
  extremamente rápida". Vira dimensão medida, não default.
- **`rerank-small` tem batch físico de 512 tokens** (`--ctx-size 4096` sem
  `--ubatch-size`): documento acima disso devolve HTTP 500. `clientes.rerank` trunca de
  forma ADAPTATIVA, encolhendo até o servidor aceitar — estimar chars/token não funciona
  (3,2 é a razão da prosa; termo médico deu 2,6). A correção mais limpa seria
  `--ubatch-size 2048` no perfil, mas isso é `config/`, decisão do dono da máquina.
- **Células de tabela isoladas se perdem** no filtro de ruído: 58 linhas só-numéricas em
  200 páginas, 90% descartadas (~240 no livro). Valores INLINE sobrevivem bem (1.013
  decimais, 79 doses em mg/kg). Como a ordem por bandas já separa a coluna de figura, o
  filtro talvez possa ser afrouxado — candidato a melhoria, custa reindexar (28 min).
- **Sinal de que o desenho funciona:** na pergunta "menor coeficiente de partição
  sangue-gás", o v0 respondeu "óxido nitroso" com confiança (errado) e o v2 recuperou a
  seção certa (cap8 SOLUBILIDADE) e **recusou-se a responder** porque o valor está numa
  tabela perdida. Abstenção honesta contra alucinação confiante.

## BUGS ENCONTRADOS NO BANCO DE QUESTÕES (2026-08-25) — invalidam medições anteriores

1. **O parser INVENTAVA gabarito.** `current_resp = 'F'` quando não conseguia ler a
   coluna Resposta. Fabricou ~135 assertivas, todas F.
2. **Gabarito vazava para o texto** em 46 itens: célula de tabela centralizada
   verticalmente põe o marcador `B)` sozinho na linha, com a primeira linha do texto
   ACIMA dele.
3. **94 questões de correlação sem o par** (`'hipercalcemia'` solto).
4. **12% da amostra medida estava comprometida** — contra diferenças de 5 a 12 pp.

**O acaso NÃO era 51,3%.** Os F fabricados falseavam o equilíbrio. No banco limpo:
**54,2%** no dev. Todo número anterior foi comparado contra a linha errada.

Meu teste checava `(verdadeiro|falso)\s*$` — só o FIM da assertiva. Vazamento no começo
passava verde. Segunda vez na sessão que meu verificador era fraco na direção do defeito.

## LATÊNCIA (registro, não critério)

93–95% do tempo de resposta é o RAG; o reranker em CPU é ~17 s de 20. Gerar custa ~1 s.
`gemma4-plan` tem speculative decoding (MTP, 1,84–2,14x medido na bancada) e
`medgemma-clinical` NÃO — não existe draft do MedGemma em disco. Não afeta acerto
(spec decoding é exato), afeta toda comparação de tempo entre os dois.

## PRIMEIRA MEDIÇÃO (INVALIDADA pelos bugs acima) (n=40, V/F, dev, medgemma-clinical)

| variante | acerto | IC 95% | p50 | fidelidade da citação |
|---|---:|---|---:|---:|
| v0 sem RAG | **60,0%** | 44,6–73,7% | 0,8 s | — |
| v1 denso puro | 57,5% | 42,2–71,5% | 4,7 s | 61,5% |
| v2 híbrido+rerank+pai | 52,5% | 37,5–67,1% | 20,5 s | 48–55% |

**O RAG não ajudou; a tendência é negativa.** Nada é significativo (Holm corrigido = 1,000
em tudo) e os ICs se sobrepõem quase inteiros. **n=40 só detecta diferença acima de ~20 pp** —
a própria `methodology.md` da pasta diz que 30×3 serve para >20 pp e que ~5 pp pede ~500.
Então o correto é: *não há evidência de que o RAG ajude*, e não *está provado que atrapalha*.

**A/A: ruído ZERO** (0 discordantes em 39). Responde a pergunta nº 2 em aberto da pasta.
Qualquer efeito real acima do ruído é detectável; falta é AMOSTRA, não estabilidade.

Fatos que sobrevivem à falta de significância:
- **v2 custa 25× a latência do v0** (20,5 s contra 0,8 s) e não entrega acerto.
- **Metade das citações não confere literalmente** (48–62%): o modelo parafraseia em vez
  de copiar, mesmo com a instrução explícita e `json_schema`.
- A recuperação está BOA (tópico/capítulo/seção certos, inspecionado à mão). O gargalo
  não é achar o trecho.

Hipóteses a testar, em ordem de custo:
1. **Amostra.** Repetir com n=200 (há 986 V/F no dev). ~70 min para v0+v2.
2. **O prompt restritivo pode estar enviesando para F.** "Responda SOMENTE com base no
   contexto" num binário forçado empurra para negar quando o contexto não confirma.
   Este projeto já mediu wrapper genérico derrubando RAG de 93,3% para 80%.
3. **O livro é o introdutório** (*Bases da Anestesia*), e as assertivas são de prova de
   título. Pode faltar granularidade — testável medindo answerability à mão numa amostra.

## ROADMAP

**Ferramentas em vez de contexto colado** (pedido 2026-08-26, para bloco pequeno).
Hoje a busca acontece ANTES do modelo, uma vez, e ele recebe o que vier. Com
`tool_calls` ele decidiria quando e o que buscar, poderia refinar depois de ler, e a
CITAÇÃO viraria exata por construção — hoje ela é copiada à mão e a fidelidade fica em
62–68%.

Pré-requisito verificado em 2026-08-26: `gemma4-plan` **emite tool_calls** (e traduziu
a consulta para inglês por conta própria); `medgemma-clinical` emite **zero**.

Duas ferramentas, no máximo — a bancada deste projeto mediu que a precisão de tool
calling cai conforme o número de ferramentas cresce:
  `buscar(consulta, k)` e `citar(id_do_trecho)`.

**Outros pendentes**
- medir o prompt `falsificacao` no banco INTEIRO (foi de 2,3% para 14,0% nos 43 difíceis,
  mas lá 66% era Verdadeiro; falta ver se virou "sim para tudo")
- v3 e v4 nunca medidos com o corpus novo
- SAESP: única fonte brasileira da bibliografia da banca, ainda ausente

## Próximo passo

A/A em curso (sentinela `eval/.aa-pronto`): v2, 40 assertivas V/F do dev, 2 repetições.
Ler com `bench/relatorio.py --aa`. Só depois a grade v0–v4 × {medgemma-clinical,
gemma4-plan} × raciocínio — e nenhum ganho menor que o ruído do A/A conta como efeito.

**O sub-agente tem teto de complexidade.** Spec de 3,3 KB devolveu resposta vazia duas
vezes (thinking estoura o orçamento). Tarefa acima de ~2 KB de spec: escreva você mesma
ou parta em duas. Ver [[subagente-razao-spec-artefato]].

## Contabilidade do sub-agente (sempre o LÍQUIDO)

| marco | acumulado |
|---|---|
| antes do WiseOak | −1.978 |
| fim da Fase 1 | −10.365 |
| Fase 2 (estrutura) | −14.184 |

## Fase normativa + formulário (2026-08-26)

### Corpus de normas
61 documentos, 653 trechos, 166 pais em `dados/indice/normas`. Divisão por `Art. N`/`§`/
`ANEXO`, não por capítulo; a citação nomeia resolução e artigo (`CFM 2174/2017`, `Art. 3`).

**Seis arquivos de prova foram removidos antes de indexar** — `SBA-gabarito{1,2,3}.pdf`,
`SBA-ME1/me2/me3.pdf`. A varredura da SBA pegava todo PDF do site (496 arquivos, 586 MB).
Gabarito indexado faria o RAG recuperar a resposta da questão sendo avaliada, e o
benchmark deixaria de medir qualquer coisa. `bench/coletar_normas.py` agora filtra na
origem, além do filtro na pasta.

### Três defeitos encontrados nesta fase

1. **`clientes.embed` truncava por 3,2 chars/token**; texto normativo denso é ~2,6, e o
   lote estourava o batch físico de 2048. Agora encolhe até ser aceito, como o `rerank`
   já fazia. Mesma miscalibração, segunda vez.

2. **`store.indexar` vetorizava ids duplicados.** O id é hash do conteúdo e o
   `INSERT OR REPLACE` colapsa no SQLite, mas a lista de vetores não colapsava junto: 11
   dos 664 trechos ficavam com dois vetores para o mesmo id, e o mesmo trecho podia ocupar
   duas das k vagas de contexto. Causa real do corpus, não hipotética: resoluções repetem
   texto de praxe literalmente. `m10` e `dois` estavam limpos (0 duplicados) — o defeito
   só aparece em corpus com repetição literal.

3. **`bench/runner.py` tinha `h512` como índice padrão** — o *Bases da Anestesia*, o livro
   REFUTADO. Uma corrida sem `--indice` media o corpus errado em silêncio; só o rótulo
   `ix=` no setup denunciava, e depois do fato. Padrão agora é `m10`, o script passa
   `--indice` explícito, e o runner **anuncia o corpus** (nome, nº de trechos, livros) na
   partida. Nenhum resultado anterior foi invalidado: todos gravam `ix=m10` no rótulo.

### Formulário — alcance declarado antes de medir
`wiseoak/ferramentas/formulario.py`, 21 verbetes, UMA ferramenta `formula(consulta)`, sem
calculadora. A busca casa nome + sinônimos + **situação clínica**, porque os 6 erros de
fórmula medidos falharam em saber *qual* fórmula — e um índice por nome só serve a quem já
sabe o nome. Verificado: 11/11 situações clínicas descritas sem o nome acham a fórmula
certa; e no v9 ponta a ponta o `gemma4-plan` chamou a ferramenta com a situação
("pressão de pico alta com platô normal"), não com um nome.

**Não é mensurável nesta amostra.** Dos 161 itens de valor-numérico do dev, 16 trazem
marca de cálculo e 9 são dose por peso (multiplicação que nunca falhou). Sobram ~5 itens
em 1.003 = 0,5%, contra IC de ±2,5 pp. Medir o v9 custaria ~100 min para devolver ruído.
Existe pelo produto, não pela medição.

`v9` = rotear → recuperar → contexto → **formula** → responder. O nó é separado do
responder porque `schema=` (gramática JSON) e `tools` se excluem: pedir os dois faz o
modelo devolver JSON e nunca chamar a ferramenta.

### Sub-agente
Não usado nesta fase. O formulário é conteúdo de domínio: a spec teria de conter cada
fórmula, ou seja, spec ≈ saída — a condição em que o ledger já mediu prejuízo. Ledger
segue em −14.184 líquido.

### O que a pergunta "esses arquivos não têm questionário?" descobriu (2026-08-26)

A resposta é não — varredura do TEXTO indexado (não do nome do arquivo) por alternativas
A–D, "assinale", "gabarito", numeração de questão e V/F: zero questões. Os dois positivos
são falsos: o REGULAMENTO DO TEA rege *como* a prova é aplicada ("cada resposta correta
marcará um ponto"), e "item 5.1" no Regulamento SAVA é numeração de artigo.

Mas a checagem expôs três defeitos maiores que a contaminação que eu procurava:

**1. `chunk.py` fundia documentos diferentes num mesmo chunk.** `montar_pais` só fechava o
pai na troca de `capitulo_num`, e a ingestão de normas põe `capitulo_num=None` em todos os
registros de todos os documentos — então a condição nunca disparava e o pai atravessava a
fronteira. Resultado medido: texto do **Estatuto da SBA dentro de um chunk rotulado
`CFM 2174/2017, ANEXO IX`**, e 17 dos 61 documentos sumindo absorvidos pelo vizinho. Num
corpus cuja razão de existir é citar resolução e artigo, citar o documento errado é falha
total. Corrigido: o pai fecha na troca de `livro`. `m10` nunca foi afetado (livro único,
`capitulo_num` preenchido).

**2. Guarda nova em `chunk.py`:** documento que entra tem de sair. Se algum não sair, o
script imprime ALERTA e devolve exit 2. Testado com entrada que perde um documento de
propósito — dispara.

**3. Seis documentos eram de DUAS COLUNAS e `-layout` intercalava as colunas no meio da
frase** ("Art. 4° - Os membros associados da SBA, que não Art. 15 - São membros Remidos").
O Estatuto — primeiro item da bibliografia declarada pela banca — era um deles. Agora
`normas.py` extrai com `-layout` E `-raw` e escolhe pela **monotonia da numeração de
artigo**: embaralhar colunas destrói a ordem crescente, então a ordem é um verificador
programático barato. Estatuto: layout 91% → raw 100%. Efeito: 829 → 1.120 trechos por
artigo; o Estatuto passou de 35 trechos embaralhados para 73 artigos limpos.

**Versões superadas removidas** (7 documentos): Regimentos de Assembleia Geral 2023, CNT
2024, Conselho de Defesa Profissional 2024, Conselho Fiscal 2024, Conselho Superior 2024,
Assembleia de Representantes 2020, e a cópia da CFM 2174 republicada pela SBA. Num RAG
normativo a redação revogada seria citada como vigente. As resoluções do CFM foram
CONFERIDAS e não são versões umas das outras — só uma revogação é declarada no corpus
(CFM 1931/2009 pelo cem2019), e a revogada não está no índice.

Corpus final: **54 documentos, 607 trechos, 0 vetores duplicados**. Sanidade: "periodicidade
das reuniões da Diretoria" agora devolve `SBA · ESTATUTO 2026, CAPÍTULO II, Art. 3` — antes
devolvia o chunk contaminado do CFM.

Dois falsos positivos meus, registrados para não repetir: "membros Remidos" NÃO é exclusivo
do Estatuto (é categoria estatutária citada em vários regulamentos), e `portalmedico.org.br`
dentro de um documento "SBA" não é vazamento — é a resolução do CFM republicada pela SBA.

### Corrida do v2 e ferramentas de leitura (2026-08-26)

Decisão do usuário, e ele estava certo: **não reexecutar v0 e v1**. Já estão medidos nos
mesmos itens, mesma config; a corrida é só do braço novo, e a comparação sai contra o que
está gravado. Cortou de 3,3 h para ~2,1 h.

Nomenclatura: o usuário chama de **WiseOak v2** o braço roteado; internamente o grafo é
`v8` (oitava variante). O `placar.py` faz o mapeamento.

**`bench/placar.py`** — placar único e reprodutível. Junta os braços do WiseOak
(`resultados.sqlite`) e os modelos externos (`dados/prova/*.txt`) no MESMO universo, que é
a interseção estrita de quem respondeu o quê: 998 itens, não os 1.002 da tabela anterior —
aquela deixava passar quatro itens que algum sistema não respondeu. Wilson, McNemar exato
sobre discordantes, Holm monótono. A tabela anterior era script avulso, perdido.

**`bench/roteamento.py`** — cruza a rota GRAVADA com a classe real. Existe porque um braço
roteado que não melhora tem duas causas opostas — corpus sem a resposta, ou roteador
mandando ao índice errado — e o placar sozinho não as separa. Testado com bloco sintético
descartável antes de precisar dele.

Para isso o **runner passou a gravar a rota por item** (`metricas.rotas`). Custou reiniciar
a corrida perdendo 117 questões (~14 min): a célula V/F já rodava com o módulo antigo
carregado, e é onde estão a maioria das 65 questões jurídicas.

**Desenho do roteamento, declarado:** dois índices EXCLUSIVOS, não fundidos. `m10` (14.628
trechos, Miller) ou `normas` (607 trechos, 54 documentos) — um `if/else`, um por questão.
Separados porque a bancada já mediu que corpus minoritário some no índice combinado (o
Barash era 52% dos trechos e chegava a 7% do contexto). Custo: questão que dependa das duas
fontes recebe só uma. Não medido.

**O alvo, visível na tabela por classe:** em `juridico-normativo` (n=65) o v1 com RAG faz
66,2%, PIOR que o v0 sem RAG (70,8%) — buscar no Miller o que só existe em resolução do CFM
atrapalha ativamente. Opus 5 faz 81,5%. É essa inversão que o v2 tem de desfazer.

**Aviso estatístico registrado antes de ver o resultado:** com o v2 o Holm passa de 10 para
15 comparações. Se o v2 subir ~2 pp sobre o v1, é bem possível que `v2 > v1` NÃO passe —
o efeito está concentrado em 96 de 998 itens. A leitura tem de ser a linha por classe e o
`roteamento.py`, não só o total.

### Lacuna conhecida no corpus normativo (2026-08-26, NÃO corrigida ainda)

Dois documentos normativos ficaram de fora por erro do meu filtro:
`SBA-1_CODIGO_DE_PROCESSO_ADMINISTRATIVO_DA_SBA_2026.pdf` e
`SBA-2_CODIGO_PROFISSIONAL_DA_SBA_2026.pdf`, ambos em `dados/normas/descartados/`.
A SBA numera a própria série normativa (0_ESTATUTO, 1_CODIGO..., 2_CODIGO...) e o filtro
tinha `codigo de etica` mas não `codigo` — guardou o 0 e jogou fora o 1 e o 2. O Código
Profissional é material direto de `juridico-normativo`.

O filtro em `coletar_normas.py` foi corrigido (`c[óo]digo` inteiro, mais `diretriz`), mas
o ÍNDICE ATUAL (`dados/indice/normas`, 54 documentos, 607 trechos) foi construído SEM eles
e a corrida do v2 está usando esse índice. Não reindexei no meio para não invalidar a
medição.

Sequência deliberada: só faz sentido acrescentá-los DEPOIS de saber se o roteamento
funciona. Se o v2 mostrar que o roteador manda as questões ao índice certo e ainda assim
`juridico-normativo` não sobe, aí o corpus é o gargalo e os dois Códigos entram. Se o
roteador estiver errando o destino, mexer no corpus antes seria corrigir a coisa errada.

Os arquivos `SBA-C####_##.pdf` foram CONFERIDOS e estão corretamente fora: são "Nota de
Repúdio" e "Nota de Posicionamento" — manifestações institucionais, não normas.

## RESULTADO do WiseOak v2 (grafo v8, roteamento exclusivo) — 2026-08-27

Placar completo em `eval/analises/placar.txt`, reprodutível com `bench/placar.py`.

| sistema | total | V/F | ME |
|---|---:|---:|---:|
| Opus 5 (effort low, sem busca) | 85,4% | 85,7% | 83,8% |
| WiseOak v1 (RAG Miller 10e) | 80,2% | 81,5% | 73,7% |
| **WiseOak v2 (RAG normas, roteado)** | **79,9%** | 81,2% | 73,1% |
| DeepSeek-V3 (thinking+search) | 77,1% | 78,5% | 70,1% |
| DeepSeek-V3 (thinking) | 76,4% | 76,8% | 74,3% |
| WiseOak v0 (sem RAG) | 73,0% | 74,5% | 65,9% |

`v1 > v2`: +0,3 pp, Holm = 1,0. **Empate no total.** Por classe, duas coisas grandes em
direções opostas:

| classe | destino | n | v1 | v2 | dif | só v2 / só v1 | p |
|---|---|---:|---:|---:|---:|:--:|---:|
| juridico-normativo | normas | 65 | 66,2% | **80,0%** | **+13,8 pp** | 14 / 5 | 0,064 |
| gestao | normas | 31 | 90,3% | **77,4%** | **−12,9 pp** | 0 / 4 | 0,125 |
| tecnica | livro | 149 | 80,5% | 75,2% | −5,4 pp | 1 / 9 | 0,022 |

**O que ficou provado**
- O roteamento CONSERTOU o defeito que motivou a fase: em `juridico-normativo` o v1
  (66,2%) era pior que o v0 SEM RAG (70,8%); o v2 faz 80,0%, encosta no Opus 5 (81,5%) e
  passa os dois DeepSeek.
- O mecanismo do roteador é bom: precisão 100%, cobertura 97,6%, zero questões clínicas
  mandadas às normas.

**PISO DE RUÍDO DA BANCADA, medido pela primeira vez.** Nas 747 questões roteadas ao
livro, o v2 executa pipeline IDÊNTICO ao v1 — mesmo grafo, índice, k e ancoragem. A
diferença deveria ser zero e foi −0,9 pp: **29 de 747 itens (3,9%) trocam de resposta
entre execuções**, por causa de `temp=0.3` em `no_responder`. Consequências:
- diferença abaixo de ~1 pp no total é indistinguível de ruído;
- a linha de `tecnica` com p=0,022 é FALSO POSITIVO — é pipeline idêntico. Serve de régua
  para ler qualquer análise de subgrupo neste projeto.

**Por que o total não se move, e não é fracasso:** `juridico-normativo` é 6,5% do banco.
+13,8 pp lá valem +0,9 pp no agregado — o próprio piso de ruído. Projeção com gestão
corrigida: 80,3% vs 80,2%, p=1,0. **Esta fase conserta uma classe; ela não pode mover o
agregado.** Registrado antes de rodar, não depois.

### Diagnóstico dos 13 erros restantes em juridico-normativo
- 3 são risco ocupacional (abuso de substâncias ×2, radiação) — tema do Miller cap. 84,
  rotulados como jurídicos. Mesma doença da gestão: o RÓTULO não corresponde a onde a
  resposta está.
- 2 são lacuna de corpus: ética em pesquisa (CNS/CONEP) e Lei 10.205 (hemoterapia).
- os demais recuperam documento normativo e ainda erram.

**Concentração da recuperação, medida nas 65 jurídicas (260 vagas):** CFM 2174/2017 ocupa
37,7% das vagas com 4,1% dos trechos (9× demais); CFM 1802/2006, 12,3% com 0,8% (15×);
`cem2019` — o Código de Ética, fonte natural das questões de ética — só 5,4% com 16% dos
trechos (3× de menos). **25 dos 54 documentos nunca aparecem.** Duas resoluções sobre "a
prática do ato anestésico" viraram atratores semânticos. BM25 híbrido foi TESTADO e ajuda
pouco (2174 cai para 26,9%, cem2019 sobe para 8,5%) — não resolve.

## WiseOak v3 (grafo v10, COTA por corpus) — rodando

Diagnóstico de fundo: **o roteamento exclusivo é destrutivo.** Mandar a pergunta às normas
TIRA o livro; quando o rótulo não corresponde à fonte da resposta, a pergunta perde a única
fonte que a continha. É a causa comum da queda em gestão e dos 3 erros de risco ocupacional.

`v10` = rotear → `no_recuperar_cota` → `no_contexto_cota` → responder. Pergunta roteada às
normas recebe metade das vagas de cada corpus (2+2 em k=4); pergunta clínica segue com o
livro inteiro — dar-lhe resolução do CFM seria trocar contexto útil por ruído.

NÃO é o mesmo que fundir os índices: em índice único as vagas saem de competição global e
607 trechos de norma perdem para 14.628 do Miller por volume (medido com o Barash: 52% dos
trechos, 7% do contexto). Com cota as normas competem só com normas.

Contexto INTERCALADO entre fontes, não concatenado: se as normas viessem todas antes, o
corte em `k_contexto` decapitaria uma fonte e a cota deixaria de existir na prática.

**Rotulagem de natureza da fonte** (ideia do usuário): cada trecho vai como `[NORMA]` ou
`[LIVRO]`, e duas frases entram no prompt SÓ quando o contexto é misto de fato — que norma
define obrigatoriedade e livro descreve prática. Norma passa a ser citada por artigo, não
por "capítulo e página" (antes saía `capitulo None`). Mantido curto: a bancada mediu
envelope genérico de prompt derrubando o RAG de 93,3% para 80%.

**`gestao` VOLTOU ao roteamento**, de propósito. Saíra como remendo do esquema exclusivo;
com cota o remendo atrapalha, porque deixaria a classe recebendo 4 vagas do livro e a cota
nunca seria testada onde o v2 quebrou.

**CONFUNDIMENTO DECLARADO:** o v3 empacota duas mudanças (cota + rotulagem). Se melhorar,
não dá para atribuir a uma delas sem uma terceira corrida. Juntadas porque entregar
contexto misto sem dizer ao modelo que é misto é meio-desenho, não variante.

**Critério de leitura, fixado ANTES do resultado:** o total não decide — 96 de 998 itens
não movem o agregado acima do ruído. Decide a linha por classe: `gestao` tem de subir dos
77,4% e `juridico-normativo` tem de segurar os 80%.

## RESULTADO do WiseOak v3 (grafo v10, cota) — 2026-08-27

| sistema | total | V/F | ME |
|---|---:|---:|---:|
| Opus 5 (effort low, sem busca) | 85,4% | 85,7% | 83,8% |
| **WiseOak v3 (cota: normas + livro)** | **80,6%** | 82,1% | 73,1% |
| WiseOak v1 (RAG Miller) | 80,2% | 81,5% | 73,7% |
| WiseOak v2 (RAG normas, roteado) | 79,9% | 81,2% | 73,1% |
| DeepSeek-V3 (thinking+search) | 77,1% | 78,5% | 70,1% |
| DeepSeek-V3 (thinking) | 76,4% | 76,8% | 74,3% |
| WiseOak v0 (sem RAG) | 73,0% | 74,5% | 65,9% |

**A cota fez o que foi desenhada para fazer.** Contra o v2:

| classe | n | v2 | v3 | dif | só v3 / só v2 |
|---|---:|---:|---:|---:|:--:|
| gestao | 31 | 77,4% | **90,3%** | **+12,9 pp** | 4 / 0 |
| juridico-normativo | 65 | 80,0% | 75,4% | −4,6 pp | 4 / 7 |

`gestao` voltou EXATAMENTE ao nível do v1 (90,3%), com 4 discordantes a favor e zero
contra. `juridico-normativo` mantém +9,2 pp sobre o v1 (66,2% → 75,4%) mas cede 4,6 pp
para o v2 — mecanicamente esperado: a classe passa a receber 2 vagas de norma em vez de 4,
e ali a norma é mesmo a fonte certa. A cota **troca parte do ganho jurídico por robustez a
rótulo errado**, e essa troca é o desenho, não um defeito.

### O que NÃO ficou provado, e é o principal
`v3 > v1` = +0,4 pp, Holm = **1,000**. `v3 > v2` = +0,7 pp, Holm = 1,000. As três variantes
de RAG são **estatisticamente indistinguíveis entre si** (80,2 / 79,9 / 80,6). O que os
dados sustentam é só: as três batem o v0 (Holm < 0,001) e o Opus 5 bate as três
(+4,8 pp sobre o v3, Holm = 0,010).

### O piso de ruído voltou a produzir falso positivo, DUAS vezes
`fisiopatologia` (n=384, roteada ao livro, pipeline IDÊNTICO nos dois braços) deu p=0,0127
contra o v2 e p=0,0391 contra o v1. Efeito impossível; é `temp=0.3`. Somados ao p=0,022 de
`tecnica` no v2, são três falsos positivos em três corridas. **Neste banco, análise de
subgrupo a temp=0,3 produz "significância" espúria como regra, não exceção.**
`bench/porclasse.py` marca isso sozinho.

### Conclusão da fase e próximo passo
O gargalo de MEDIÇÃO passou a ser o ruído, não o corpus nem o roteamento. Com 3,9% dos
itens oscilando, nenhuma diferença de ~1 pp entre variantes pode ser estabelecida, e é
nessa faixa que as três variantes vivem. O passo com maior retorno agora é **baixar
`temp` para 0 em `no_responder`** e remedir — não como melhoria de acerto, mas para tornar
as comparações decidíveis. O conjunto de teste segue INTOCADO.

## Onde estão os 19,4% de erro do v3, e por que os documentos "não têm a resposta"

| classe | erros | % dos erros | acerto |
|---|---:|---:|---:|
| fisiopatologia | 50 | 25,8% | 87,0% |
| valor-numerico | 48 | 24,7% | 70,2% |
| tecnica | 34 | 17,5% | 77,2% |
| farmacologia | 31 | 16,0% | 82,0% |
| juridico-normativo | 16 | 8,2% | 75,4% |
| imagem | 12 | 6,2% | 67,6% |
| gestao | 3 | 1,5% | 90,3% |

**84% dos erros são clínicos.** O corpus normativo nunca é consultado neles — vão ao Miller
por desenho, e corretamente: resolução do CFM não tem fisiopatologia.

Três causas distintas, e só uma é dos documentos:

**1. Falha minha de coleta.** O plano prometia o *Projeto Diretrizes* da AMB e as *Diretrizes*
da SBC — programas com dezenas de documentos. O coletor trouxe UM PDF de cada: 62 trechos,
10,2% do corpus normativo. A metade que deveria cobrir conteúdo clínico é inexistente.

**2. O rótulo da classe não corresponde a onde a resposta mora.** Terceira vez que isto
aparece (gestao, risco ocupacional, agora aqui): "unidade ambulatorial do tipo I ... dose
inferior a 5,5 mg" é rotulada `valor-numerico` porque tem número, vai ao Miller, e a resposta
está em resolução do CFM. A taxonomia classifica pelo que a pergunta PARECE, não pela fonte.

**3. Nos clínicos, o livro TEM a resposta — a busca traz o parágrafo vizinho.** Verificado em
6 erros: a recuperação acerta capítulo e tópico e erra a frase. ARISCAT devolveu "vários
índices de risco foram desenvolvidos" em vez da composição do escore; hipotermia devolveu
"metabolismo cerebral na hipotermia" em vez do valor de 20 °C. **Granularidade, não cobertura.**

### CAVEAT GRANDE: os experimentos de arquitetura foram medidos no livro ERRADO
Auditoria de `eval/resultados.sqlite` por índice: **todo bloco anterior a `COMPLETO-m10` usou
`ix=h512`** — o *Bases da Anestesia*, 3.406 trechos, o livro refutado. Inclui `B-arquitetura`,
que é a origem das conclusões "rerank não ajuda", "expansão para o pai não ajuda", "híbrido
não ajuda", "portão é ruído":

    v0 70,3% · v1 72,3% · v2 70,0% · v5 71,7% · v6 72,0%   (n=300, ix=h512)

Amplitude de 2,3 pp em n=300, contra um piso de ruído de 3,9%. Não era conclusivo nem naquele
corpus, e nunca foi refeito no Miller (14.628 trechos, 4× maior, livro certo). **A refutação
do BM25 JÁ falhou em transferir** — no corpus normativo ele ajudou (2174 de 37,7% para 26,9%).
Não há razão para confiar que as outras transferem.

Os mecanismos descartados — expansão para o pai, rerank, k maior — são exatamente os que
atacam "tópico certo, frase errada". **Retestá-los no m10 é o experimento de maior valor
esperado agora**, acima de ampliar corpus.

## As respostas ESTÃO nos documentos: onde exatamente a falha ocorre

Teste programático, cruzando idioma por ÂNCORA NÚMERO+UNIDADE (número atravessa tradução;
o Miller está em inglês e as questões em português). 39 dos 163 erros clínicos são
testáveis assim.

| onde está a resposta | n | % | natureza |
|---|---:|---:|---|
| já no contexto entregue (top-4) | 12 | 30,8% | **raciocínio** |
| no top-50, fora do contexto | 16 | 41,0% | **ranking** |
| no livro, fora do top-50 | 9 | 23,1% | **busca** |
| não está no livro | 2 | 5,1% | cobertura |

**95% das respostas estão no corpus.** A falha reparte-se em ~64% recuperação, ~31%
raciocínio, ~5% cobertura.

Exemplos verificados, ambos com a frase NO CONTEXTO e resposta errada mesmo assim:
- "a 20 °C induz supressão do EEG" (V) — contexto trazia *"complete suppression of the EEG
  (at approximately 18 °C to 20 °C)"*.
- "CPDA-1, hematócrito entre 65% e 75%" (V) — contexto trazia *"When CPDA is the
  anticoagulant used, the Hct is greater than 65%"*.

### ARMADILHA DE VERIFICADOR, terceira nesta bancada
A primeira versão do teste casava NÚMERO SOLTO e dava 72,5% de "já no contexto". Era falso:
o Miller traz as citações bibliográficas como dígitos colados ao texto, então a âncora "20"
casava com a referência "207" e "14" com "140". **Exigir a unidade junto derrubou de 72,5%
para 30,8%.** Regra que fica: âncora numérica em corpus científico SEM unidade não vale
nada — as referências poluem tudo.

### Consequência para o roadmap
Não é falta de documento. Ampliar corpus tem retorno baixo (5% dos erros). O retorno está em:
1. **Recuperação (64%)** — os mecanismos descartados no `h512` (expansão para o pai, rerank,
   k maior) atacam exatamente "está no top-50 e não chega ao contexto". Nunca testados no m10.
2. **Raciocínio (31%)** — a resposta chega e o modelo erra. Isso é ancoragem/prompt, ou
   limite do Gemma 4 31B. O Opus 5 acerta várias dessas sem RAG nenhum.

### Relatório publicado
`eval/analises/bancada.html` — mesma fonte que a página em
https://claude.ai/code/artifact/051e6b92-d269-4e9a-8b9e-d1e10dca6cbf
Editar o arquivo NO PROJETO e republicar mantém a URL. Publicar de outro caminho sem passar
a URL criaria um artefato separado.

## Dossiê de erro e um achado sobre a instabilidade do conjunto de erros

`bench/dossie.py` gera `eval/analises/dossie-erros.txt`: as 231 questões que o v2 e/ou o v3
erraram, com o CONTEXTO que o RAG entregou trecho a trecho, a resposta dada, a ressalva, a
rota e a fidelidade das citações. O runner só grava acerto/erro, então os itens são
reexecutados para capturar o texto.

**Achado não previsto: 59 das 395 reexecuções (14,9%) ACERTARAM o que o banco gravou como
erro** — 3,8× o piso de ruído de 3,9% medido sobre todos os itens.

Não é contradição, é seleção. O piso de 3,9% foi medido no conjunto INTEIRO; o conjunto de
ERROS é enriquecido em itens que o modelo responde na dúvida, e é justamente esses que
oscilam. Consequência dura para a leitura de qualquer análise de erro neste projeto:

**cerca de um sexto do "conjunto de erros" não é estável.** Uma análise qualitativa de erros
está olhando, em parte, ruído de amostragem — e conclusões tiradas de "o modelo errou X"
precisam disso em conta. As linhas afetadas vêm marcadas com `[!]` no dossiê, nunca omitidas.

Isso reforça a recomendação já registrada: baixar `temp` para 0 antes de qualquer análise
fina de erro.

## A causa do ruído da bancada: texto antes do veredito (2026-08-27)

O portão de verificação do plano ("A/A a temp=0 tem de dar zero discordantes") FALHOU, e
foi para isso que ele existia. A `temp=0` a discordância foi de 12,5% em 40 itens V/F —
não menor que a 0,3, e sim comparável.

A causa não é a temperatura deixar de chegar ao modelo (verificado: chega, e a API isolada
a `temp=0` devolve 1 saída distinta em 3). É a ORDEM DOS CAMPOS DO SCHEMA:

| ancoragem | campos | discordantes a temp=0, n=40 |
|---|---|---:|
| `estrita` | `resposta` → `citacoes` | **0 (0,0%)** |
| `confiante` | **`ressalva`** → `resposta` → `citacoes` | **5 (12,5%)** |

**A pilha é determinística.** O `gemma4-plan` usa decodificação especulativa (MTP), que
introduz variação numérica de GPU; em texto livre longo isso desempata tokens quase iguais.
Como `ressalva` é gerada ANTES de `resposta`, a divergência do texto entra no contexto do
veredito e transborda para ele. Em três pares inspecionados, resposta e contexto eram
idênticos e só a ressalva divergia — mas em 12,5% dos casos chega ao veredito.

**A consequência é desconfortável e precisa ficar registrada:** a ordem `ressalva` antes de
`resposta` foi adotada por medição (campo depois da resposta vira racionalização pós-fato;
mediu-se o modelo escrevendo a justificativa correta e ainda respondendo errado), e a
ancoragem `confiante` vale +10 pp. **O ganho de acerto e o ruído de medição são o mesmo
mecanismo.** Não existe configuração que tenha os dois.

Isso reinterpreta os 3,9% e os 14,9% já medidos: não eram "ruído de temperatura", eram o
custo de estabilidade do raciocínio-antes-da-resposta. Baixar `temp` NÃO resolve, e a
recomendação anterior de "rodar tudo a temp=0 para decidir" estava errada.

### Consequência para o desenho de qualquer comparação daqui em diante
Execução única não decide nada entre variantes próximas. Toda triagem passa a usar
**repetição por item**, com critério conservador: um item só conta como recuperado se o
braço acerta em TODAS as repetições e o controle erra em TODAS. Isso troca sensibilidade
por ausência de falso positivo, que é o que a bancada precisa.

## Triagem de variantes nas 231 questões já erradas (2026-08-27)

`bench/triagem.py`, 2 repetições por item, `temp=0`, critério conservador (item só conta
se estável nos dois braços). Resultado em `eval/analises/triagem.txt`, bruto no `.json`.

| braço | recuperou | quebrou | saldo | instável | razão rec/queb |
|---|---:|---:|---:|---:|---:|
| controle (v10) | — | — | — | 26 | — |
| **falsif** (ancoragem `falsificacao`) | **40** | 15 | **+25** | 30 | 2,7 |
| **k8** (k_contexto 8) | 27 | **7** | +20 | 37 | **3,9** |
| pai (v12, seção-pai) | 29 | 11 | +18 | 42 | 2,6 |
| quant (`confiante_quantificador`) | 11 | 13 | **−2** | 37 | 0,8 |

**Instabilidade do controle: 26/231 = 11,3%.** Duas repetições bastaram; terceira
desnecessária.

### REFUTADO: instruir o modelo a conferir quantificador/negação/relação
O braço `quant` saiu de −2. A ancoragem foi construída a partir do diagnóstico linguístico
de um modelo externo sobre o dossiê — que identificou padrões REAIS (o modelo trata
"frequente" como "pode", lê negação por cima, inverte "constante grande" e "tempo longo").
O padrão existe; **mandar conferir não corrige**. Mantida a versão mínima justamente para
não confundir com o custo de envelope longo, e ainda assim não funcionou.

### O achado mais limpo: valor-numérico
| classe | n | k8 | pai | falsif |
|---|---:|---|---|---|
| **valor-numerico** | 53 | +7/**−0** | +11/**−0** | +10/**−0** |
| juridico-normativo | 20 | +4/−0 | +6/−2 | +3/−1 |
| imagem | 13 | +1/−0 | +0/−0 | +5/−0 |

Os três braços ganham em valor-numérico **sem quebrar uma única questão**: 28 recuperadas,
zero perdidas. Confirma diretamente a medição anterior de que 41% dos erros numéricos tinham
a resposta no top-50 e fora das 4 vagas. Era gargalo de CONTEXTO, e alargar contexto resolve.

### Complementaridade entre falsif e k8
- ambos recuperam: 13 · só `k8`: **14** · só `falsif`: 27
- **união 54**, contra 40 do melhor sozinho
- **zero antagonismo**: nenhum conserta o que o outro quebra, nas duas direções
- eixos ortogonais: `k8` mexe na recuperação, `falsif` em como o modelo raciocina

União é TETO, não previsão — por isso os braços combinados (`k8falsif`, `paifalsif`) estão
sendo medidos antes de comprometer as ~6 h da corrida no banco inteiro.

### O que a triagem NÃO decide
É cega a regressão. Um braço que recupera 40 aqui pode quebrar mais que isso entre as 767
questões que já estavam certas, e isso só aparece no banco inteiro. Nenhum braço é adotado
sem essa corrida.

## ROADMAP — fora de escopo agora

### Testar `reasoning-medical-27B` (Qwen 3.6-27B com fine-tune médico)
Pedido do usuário em 2026-08-27, explicitamente fora de escopo no momento.

**Por que ele é interessante à luz do achado de hoje.** Mediu-se que o ganho de +10 pp da
ancoragem "raciocine antes de responder" e o ruído de 12,5% são o MESMO mecanismo: a
`ressalva` é gerada antes da `resposta` DENTRO do JSON, e a variação numérica da
decodificação especulativa transborda do texto para o veredito.

Um modelo com **canal de thinking nativo** poderia desacoplar as duas coisas: o raciocínio
acontece fora do JSON, o campo `resposta` continua sendo o primeiro do schema, e a
`estrita` (que mediu 0/40 de discordância) volta a ser viável sem perder o raciocínio.
Se isso se confirmar, resolve um problema que hoje não tem solução na configuração atual.

**A verificar ANTES de planejar medição** — a mesma checagem que reprovou o MedGemma 27B:
1. emite `tool_calls`? (o `medgemma-clinical` emite ZERO; o `gemma4-plan` emite)
2. tem canal de thinking de verdade, separado do conteúdo? (o MedGemma não tem)
3. cabe no orçamento de VRAM? Só um modelo de ~18 GB por vez na GPU, e a troca custa 5,5 s
4. responde em português com qualidade? (o `v7`, que traduzia a consulta para inglês,
   despencou de 51,2% para 32,6% — idioma não é detalhe nesta bancada)

Falhar em (1) ou (2) não desqualifica: o modelo ainda serviria como braço de comparação,
como o MedGemma serviu. Mas muda o desenho do experimento.

**Custo de disco:** verificar antes de baixar. O usuário já sinalizou disco apertado, e o
projeto tem a regra de não duplicar modelos.

## Combinação k8+falsif: efeitos aditivos, quebra não aumenta (2026-08-27)

Critério FIXADO ANTES de ver o resultado: combinação vence se saldo ≥ +30 e quebra ≤ 20;
qualquer coisa abaixo disso, `falsif` sozinho. Empate técnico contava como derrota da
combinação, por ela custar contexto dobrado.

| braço | recuperou | quebrou | saldo | acerto no conjunto |
|---|---:|---:|---:|---:|
| controle (v10) | — | — | — | 13,9% |
| k8 sozinho | 27 | 7 | +20 | 25,5% |
| falsif sozinho | 40 | 15 | +25 | 29,4% |
| **k8+falsif** | **53** | **15** | **+38** | **34,6%** |

**Os efeitos são aditivos.** A união dos dois medidos separadamente dava 54 recuperadas; a
combinação entrega 53. Não há interferência — o receio de que o contexto dobrado diluísse
a instrução do `falsificacao` (que pede para procurar o erro na afirmação) não se
confirmou.

**E a quebra NÃO aumentou:** 15 no `falsif` sozinho, 15 na combinação. O componente `k8`
acrescentou 13 recuperações a custo zero de quebra. Por classe: juridico-normativo +6/−0,
imagem +6/−0, valor-numerico +12/−1.

### WiseOak v4 = v10 + `falsificacao` + k_contexto 8
NÃO é grafo novo: são dois parâmetros sobre o v10. O rótulo no banco sai
`v10|anc=falsificacao|rac=nenhum|ix=m10|k=8/8`, auto-documentado.

Rodando no banco inteiro com a temperatura PADRÃO, não com `temp=0`. Contraintuitivo depois
do achado do schema, mas é o certo: todas as outras linhas do placar foram medidas assim, e
trocar a condição só na linha nova invalidaria a comparação. A instabilidade fica como
ressalva à parte, nunca misturada dentro da tabela.

**Expectativa registrada ANTES do resultado:** não esperar +38 no total. Aquele número é
sobre 231 itens selecionados por terem falhado, e a triagem é cega ao que o v4 quebra entre
as 767 que já estavam certas. Ganho de 2 a 4 pp sobre os 80,6% do v3 já seria bom; muito
acima disso merece desconfiança antes de comemoração.

## REFUTADO: o WiseOak v4. E por que a triagem mentiu (2026-08-27)

O v4 (`v10` + ancoragem `falsificacao` + k_contexto 8) foi medido no banco inteiro e é
**PIOR que o v3**: 78,1% contra 82,1% no V/F pareado, **−4,0 pp**. A triagem prometia +38.

### O mecanismo: viés de negação reintroduzido
Das 87 questões que o v4 quebrou, **82 (94%) tinham gabarito V** — o modelo disse F em
assertivas verdadeiras. Das 54 que ganhou, 49 (91%) tinham gabarito F. Taxa base do
conjunto: 57% V.

**O v4 deslocou o modelo para negar.** Ganha nas falsas, perde mais nas verdadeiras, e
como V é maioria, o saldo é negativo. É exatamente o viés que a ancoragem `confiante`
existia para corrigir (valeu +10 pp) e que a `falsificacao`, ao instruir "procure o erro",
reintroduz.

É o MESMO mecanismo pelo qual o "Passo 0" da análise externa foi rejeitado — empurrar para
F. A `falsificacao` faz isso de forma mais sutil, e o efeito é idêntico.

### Por que a triagem no conjunto de erros não podia ver
| conjunto | n | V | F |
|---|---:|---:|---:|
| banco V/F inteiro | 832 | 56,6% | 43,4% |
| as 231 erradas (V/F) | 177 | 49,2% | **50,8%** |

**O conjunto de erros é enriquecido em gabarito F.** Um braço que empurra para "F" parece
ótimo ali e péssimo no banco todo. A triagem não errou por azar: ela tem um viés
ESTRUTURAL que favorece qualquer mudança que desloque a distribuição de respostas.

**Lição generalizável, e a mais importante desta fase:** triagem em conjunto de erros só
serve para ordenar candidatos que NÃO mexem na distribuição de saída. Para qualquer
mudança de ancoragem ou de prompt, ela é pior que inútil — é ativamente enganosa. Mudanças
de recuperação (`k8`, `pai`) são mais seguras de triar assim, porque não tocam no viés de
resposta.

### O que se salva do v4
Por classe, o v4 GANHA onde a recuperação era o gargalo e PERDE nas classes grandes:
`juridico-normativo` +6,8 pp, `imagem` +8,0 pp; `fisiopatologia` −6,5 pp (n=355),
`farmacologia` −5,1, `tecnica` −5,4.

O componente `k8` sozinho tinha dado +20 na triagem sem tocar em ancoragem — ele continua
candidato legítimo, e agora é o ÚNICO que sobra: `v10` + `--k-contexto 8`, mantendo
`confiante`. Precisa da corrida no banco inteiro para valer.

### Correção de ferramenta
`bench/porclasse.py` marcava "RUÍDO (pipeline idêntico)" para qualquer classe roteada ao
livro, assumindo que os dois braços tinham a mesma configuração. Na comparação v3→v4 isso
marcou como ruído um efeito real de −6,5 pp. Corrigido: o aviso só dispara quando
ancoragem, k e índice coincidem nos dois lados; caso contrário imprime o aviso oposto.

## O problema da recuperação, nomeado e medido à mão (2026-08-30)

### O nome técnico
**Semantic dilution** (diluição semântica) em recuperação densa de **vetor único** com
**mean pooling**. O chunk de 1.536 caracteres vira UM vetor que representa o tópico
dominante do parágrafo; um fato específico dentro dele quase não contribui. A consulta tem
mediana de 230 caracteres — **assimetria de 6,7×**. Em RI: alta relevância TÓPICA, baixa
relevância de RESPOSTA.

Dois agravantes deste projeto: recuperação **translíngue** (pergunta pt, Miller en), que
torna o alinhamento fino ainda mais fraco; e chunks dimensionados quando o corpus era outro
livro, nunca revisitados.

### A demonstração mais limpa
Três assertivas sobre gravidez que diferem SÓ no substantivo recebem quase o mesmo contexto:

    "ventilação minuto +10%"        -> 1. minute ventilation elevated 45% to 50%   ACERTOU
    "capacidade vital +40%"         -> 1. tidal volume increases 20%...            mesmo cluster
    "capacidade pulmonar total +30%"-> 1. tidal volume increases 20%...            mesmo cluster

A busca resolve o tópico ("alterações respiratórias na gravidez") e é cega ao substantivo,
que é onde a resposta mora.

### recall@4 = 25%, VERIFICADO À MÃO (n=12)
Só 3 das 12 questões recebem, nas 4 vagas de contexto, um trecho que permita decidir a
assertiva. As estimativas automáticas (40,5% e 37,8%) estavam INFLADAS.

### DUAS tentativas de automatizar a verdade de referência FALHARAM
Registrado para ninguém repetir:

| critério | precisão contra rotulagem manual |
|---|---:|
| âncora numérica sozinha | ~21% |
| âncora + cognato pt/en | 21% |
| âncora + cognato + proximidade (janelas de 25 a 120 chars) | **teto de 33%** |
| juiz LLM (`gemma4-plan`, schema fechado, temp 0) | **64%** (7/11) |

O juiz erra na MESMA direção do critério lexical: diz "sim" para trecho que apenas trata do
assunto. Nenhum dos dois serve de base para comparar braços — um critério que premia trecho
tópico favorece sistematicamente o braço que recupera mais trecho tópico, que é justamente
o que queremos deixar de fazer.

**Consequência de método:** avaliação de recuperação neste projeto exige *qrels* construídos
à mão (questão -> id do trecho certo), feitos UMA vez e reusados por todos os braços. É o
padrão de RI e é o único caminho defensável aqui.

### O que a literatura oferece (pesquisado)
- **Contextual Retrieval** (Anthropic): prefixo de 50–100 tokens situando cada chunk, gerado
  por LLM, antes de embutir. −35% de falha no top-20; −49% com BM25 contextual; −67% com
  rerank. Custo aqui: 14.628 chamadas.
- **Small-to-big / parent retrieval**: indexar filho pequeno, entregar pai grande. **A
  hierarquia já existe neste projeto** — só falta encolher o filho (`--tamanho-filho 180`).
- **Late chunking** (Jina): inviável, o EmbeddingGemma tem ctx 2048.
- **Multi-vetor / ColBERT**: resolve na raiz, mas é outro modelo e outro índice.

`bench/recuperacao.py` existe e mede recuperação isolada (0,06 s/consulta contra 1,7 h de
uma corrida com o modelo), mas **os números dele só valem depois dos qrels manuais**.

## Fluxo alternativo a modelar: recuperação navegacional (agentic retrieval)

Ideia do usuário, 2026-08-30. Em vez de busca por embedding, o modelo recebe o CATÁLOGO do
corpus (sumário, títulos, índices, anotações) e usa `tool_calls` para pedir os trechos que
ele julga relevantes. Nome na área: **agentic retrieval** / busca navegacional — é como o
Claude Code funciona (grep e leitura, sem índice vetorial).

**Por que é atraente aqui:** contorna o modo de falha que nomeamos. A diluição semântica
acontece na compressão do trecho num vetor e na comparação translíngue; navegar sumário não
usa vetor nenhum, usa o conhecimento do modelo sobre onde as coisas ficam num livro. E as
questões da banca nomeiam o assunto explicitamente, que é o insumo de que uma consulta a
sumário precisa.

### MEDIDO: são dois modos de falha, e cada abordagem conserta um

| questão | capítulos entregues no top-4 | modo |
|---|---|---|
| capacidade vital na gravidez | 58, 58, 58, 58 | **intra-capítulo** |
| creatinina / Child-Pugh | 38, 15, 55, 14 | **inter-capítulo** |
| hipotermia / EEG | 35, 71, 10, 75 | misto (cap. certo veio, parágrafo não) |

**Navegação conserta o inter-capítulo e NÃO conserta o intra.** Na capacidade vital, um
modelo consultando o catálogo iria ao capítulo 58 — exatamente onde a busca já foi. Trecho
menor conserta o intra e não conserta o inter. **São complementares, não alternativos.**

### Desenho combinado, que é o que vale testar
1. modelo escolhe 1–3 capítulos no catálogo de 87 (uma chamada, ~2 s)
2. busca densa **restrita a esses capítulos**, sobre trechos pequenos

Mede-se com a mesma verdade de referência do `bench/onde.py`, sem rodar geração.

### O que existe hoje e o que falta
- **existe**: 87 capítulos com títulos reais no `m10.sqlite` (cabem num prompt);
  `gemma4-plan` já verificado emitindo `tool_calls`
- **falta**: hierarquia abaixo de capítulo. O campo `caminho` veio sujo da ingestão
  ("Instructions for online access", "FANZCA") — navegação fina exigiria reextrair estrutura
- **anotar catálogo/PDF com metadados** é a mesma família do Contextual Retrieval da
  Anthropic (−35% de falha), aplicada ao documento em vez do trecho: bem mais barato que
  anotar 38 mil trechos

### Ressalvas declaradas
- vira o conhecimento do modelo em roteador; um 31B sabe menos que o Opus. Sinal a favor: o
  Opus faz 85,4% SEM RAG, o que sugere que "onde o assunto mora" é conhecimento que esses
  modelos têm
- cada rodada de ferramenta é uma chamada: 2–3 rodadas levam a corrida de 1,7 h para ~4 h
- a bancada mediu que a precisão de tool calling cai com o número de ferramentas — manter
  poucas: `sumario()`, `ler(capitulo)`, e a busca densa atual como recurso

## O gargalo era a CONSULTA, não o documento (2026-08-30)

Investigando a diluição semântica, a hipótese natural era chunk grande demais. Reindexei o
Miller com filho de 180 tokens (38.736 trechos, mediana 551 chars contra 1.536) e o caso
diagnóstico **não mudou nada**. As três assertivas sobre gravidez continuaram recebendo o
mesmo pacote.

A causa estava do outro lado:

    ENUNCIADO (235 chars, IDÊNTICO nas três): "Desde o momento da concepção, inúmeras
                                               alterações fisiológicas ocorrem na gestante..."
    ASSERTIVA (40 chars, o que distingue):    "A ventilação minuto está aumenta em 10%."

A consulta enviada ao embedding era **83% texto comum entre questões irmãs**. O vetor era
dominado pelo cenário, que por construção carrega ZERO informação discriminativa.

**790 das 836 questões V/F do dev (94%) compartilham enunciado com outra**, em grupos de
até 12. O defeito atingia quase todo o banco V/F desde o início do projeto.

### As duas mudanças só funcionam JUNTAS
recall@4 conferido à mão nas mesmas 12 questões, critério estrito:

| consulta | índice | recall@4 |
|---|---|---:|
| enunciado + assertiva | grande (1.536) | 3/12 |
| enunciado + assertiva | pequeno (551) | 3/12 |
| só assertiva | grande | 3/12 |
| **só assertiva** | **pequeno** | **~6/12** |

Mecanismo: o trecho pequeno dá granularidade para existir um vetor que represente o FATO em
vez do tópico do parágrafo; a consulta focada dá um vetor de busca que discrimina entre
irmãs. Granularidade fina com consulta genérica continua caindo no cluster genérico;
consulta discriminativa contra parágrafo de 1.536 chars continua batendo no tópico.

Acertos que não existiam antes e não são discutíveis:
- "meia-vida de eliminação 2 a 4 horas" -> *"short elimination half-life of about 2 to 4 hours"*
- "6 mL/kg de peso predito, platô < 30 cmH2O" -> *"lung-protective ventilation with 6 mL/kg of PBW"*
- "20 °C induz supressão do EEG" -> *"18°C to 20°C is probably adequate"* (o número entrou)

**Declarado: minha régua de rotulagem afrouxou no meio.** Pelo critério frouxo dá 8/12.
Reporto os dois números; o estrito é o que vale.

### WiseOak v5 = grafo v13
`rotear -> focar -> recuperar(cota) -> contexto -> responder`, índice `m10p`, k=20/8.

`no_focar` altera só `consulta` (o que a busca usa); `pergunta` segue intacta para o
responder, que precisa do cenário para interpretar a assertiva. Não se aplica a múltipla
escolha — lá o enunciado carrega o caso.

k_contexto=8 é deliberado: com trecho pequeno e k=4 o modelo receberia 1.839 chars contra
~5.000 da linha de base, e uma queda seria indistinguível de "recebeu menos texto".

## RESULTADO do v5 e a captura por distrator na múltipla escolha (2026-08-31)

| braço | total | V/F | ME |
|---|---:|---:|---:|
| **WiseOak v5** (consulta focada + trecho pequeno) | **81,0%** | **83,8%** | **67,1%** |
| WiseOak v3 (cota) | 80,6% | 82,1% | 73,1% |

**V/F: o melhor de todos (83,8%)**, com `juridico-normativo` indo de 76,3% para **89,8%**,
8 ganhas e ZERO perdidas, p=0,0078 — o único p significativo por classe da fase. Fidelidade
de citação subiu junto: 91,5% -> 92,3%.

**ME: caiu 6 pp**, e a causa foi identificada lendo o caso.

### O mecanismo: a consulta contém os distratores
Questão do cirrótico com síndrome hepatorrenal, alternativas manitol/dobutamina/
terlipressina/fenoldopam. O índice PEQUENO devolveu **quatro parágrafos sobre manitol** —
a alternativa A, errada — porque a palavra estava escrita na consulta. No índice grande as
menções a manitol estavam diluídas em parágrafos maiores e não dominavam.

**O trecho pequeno não é melhor nem pior: é mais LITERAL.** Acha com precisão o que se
pediu. Se o que se pediu contém a resposta errada dentro, acha a resposta errada.

Removendo as alternativas da consulta, a captura some e aparece o parágrafo certo:
*"…increasing the risk for HEPATORENAL SYNDROME. A range of therapies…"*

**A regra geral que sai disso:** buscar pela parte que DESCREVE O PROBLEMA, nunca pela que
LISTA CANDIDATOS. Não é ajuste a formato de prova — vale para qualquer entrada com opções
("dou manitol, dobutamina ou terlipressina?" envenena a busca do mesmo jeito).

### REFUTADOS nesta rodada
- **expansão para o pai como conserto da ME**: entregaria mais texto em volta do parágrafo
  errado. Não ataca a captura.
- **rotear por formato de prova (V/F vs ME)**: ajuste ao benchmark, não ao produto. Quem
  consulta o sistema não chega rotulado.
- **encadear `focar` -> `reformular`**: o foco tira o cenário e o reformulador INVENTA um
  errado. Em "a capacidade pulmonar total está aumentada em 30%" (contexto: gravidez), a
  partir da assertiva sozinha ele escreveu "hiperinsuflação pulmonar" e buscou hiperoxemia.
  A assertiva sozinha basta para casamento vetorial direto e NÃO basta para quem interpreta.

## Recuperação navegacional: construída e testada (v6, grafo v14)

`wiseoak/ferramentas/biblioteca.py` — três ferramentas (`biblioteca`, `ler`, `buscar`),
teto deliberado porque a bancada mediu precisão de tool calling caindo com o número.

**`wiseoak/ingest/sumario.py`**: gera sumário de cada um dos 87 capítulos do Miller, uma
chamada por capítulo (~5 min), amostrando o texto uniformemente. Sem isso não há navegação
fina: capítulo tem mediana de 128 páginas e o campo de subtítulos veio sujo da ingestão.
Resultado em `dados/sumario_m10.json`, 87/87 preenchidos. O capítulo 14 lista
*"Insuficiência renal aguda (AKI) e Síndrome Hepatorrenal (HRS-AKI) na cirrose"* — exatamente
o que a busca vetorial não achava.

**Fumaça no caso do cirrótico: acertou (C).** Duas rodadas — `biblioteca()` e depois
`buscar()` com uma consulta que ELE escreveu descrevendo o problema, sem os quatro fármacos.
Chegou sozinho à regra deduzida. Custo: 17,6 s/questão contra 5,5 s do v5.

**Mas ele não usou `ler`**, então a restrição por capítulo não foi exercitada — o ganho veio
só da reformulação da consulta. Daí o v7.

## WiseOak v7 (grafo v15): o modelo escreve a própria consulta
`reformular(problema) -> recuperar(cota) -> contexto -> responder`, prompt novo
`_REFORMULAR["problema"]` que PROÍBE citar as alternativas.

Mesmo índice e k do v5: a única diferença é a estratégia de consulta. Custo 7,9 s/questão
(+1,7 s da reformulação) contra 17,6 s da navegação — sete vezes menos pelo mesmo acerto no
caso testado.

**Reformular NÃO é o mesmo que o modelo escrever a própria busca no sentido do v6.**
Reformular é cego e de um tiro; navegar é informado e iterativo (vê o que voltou e pode
tentar de novo). No caso do cirrótico convergiram porque a primeira consulta já era boa —
isso nada diz sobre as questões em que a primeira tentativa falha.

## RESULTADO do v7 e a assimetria entre tipos de pergunta (2026-08-31)

| braço | V/F | ME |
|---|---:|---:|
| v3 (cota) | 81,7% | 73,1% |
| **v5** (focar + trecho pequeno) | **83,4%** | 67,1% |
| **v7** (reformular + trecho pequeno) | 80,7% | **74,3%** |

Cada estratégia de consulta ganha na população para a qual foi desenhada e perde na outra.
`focar` resolve o enunciado compartilhado (94% das V/F) e leva a V/F ao melhor número do
projeto; a reformulação resolve a captura por distrator e leva a ME ao melhor entre nossas
versões. Aplicar uma nas duas estraga metade.

**Aplicando cada uma onde ganha: 82,2%** — o melhor WiseOak até agora, +1,6 pp sobre o v3.
Não foi rodado como versão única; é a combinação aritmética dos dois blocos medidos.

A regra não é sobre formato de prova, é sobre estrutura da entrada: **se a pergunta lista
candidatos, reformule descrevendo o problema e exclua os candidatos; se traz cenário
genérico mais afirmação específica, busque pela afirmação.**

## MEDIDO: cache de prefixo torna o catálogo quase grátis

O custo de um fluxo navegacional parecia proibitivo — catálogo de 45.628 chars (~12.900
tokens) em toda rodada. Medições:

| condição | entrada | saída | tempo |
|---|---:|---:|---:|
| contexto mínimo | 35 | 204 | 3,1 s |
| catálogo, saída curta | 12.959 | 2 | 12,1 s |
| catálogo, saída longa | 12.965 | 261 | 18,1 s |

O custo é **prefill**, não geração: ler o catálogo custa ~12 s, gerar a resposta ~6 s.

Mas com o catálogo como **prefixo FIXO** (`--cache-reuse 256` já estava no llama-swap):

    1ª chamada (fria)  12.975 tokens  13,5 s
    2ª                 12.978 tokens   1,3 s
    3ª                 12.973 tokens   1,3 s

**Requisito de projeto:** o `system` tem de ser IDÊNTICO entre perguntas. Nada específico
da pergunta nele — nem a pergunta, nem a classe, nem o modo. Se variar, o prefixo diverge
no primeiro token e nada é reaproveitado. Foi o que aconteceu na primeira medição, quando o
`system` mudava e o catálogo ia no `user`.

Consequência: **dispensa LoRA, catálogo em dois níveis e leitura item a item por SED** —
as três alternativas levantadas para contornar um custo que não existe.

## WiseOak v8 (grafo v16): navegacional com verificação de suficiência

`navegar (laço com teto) -> responder`. Duas ferramentas apenas: `ler` (parte específica;
acima de 9.000 chars busca dentro dela) e `buscar` (a densa de sempre). A `biblioteca()`
foi REMOVIDA — com o catálogo no `system`, ela custaria uma rodada para entregar o que já
está no contexto.

**Efeito colateral que vale registrar:** no v6, com o catálogo atrás de uma tool call, o
modelo NUNCA usava `ler` — ia direto ao `buscar`. Com o catálogo no `system` ele passou a
usar `ler` e a acertar o capítulo (cirrótico → cap 14 fígado; capacidade vital → cap 58
obstétrica). **Ver o mapa antes de agir mudou a decisão**; precisar pedir o mapa não bastava.

Fumaça: os três casos conhecidos acertaram, com ~8,5 s/questão após o aquecimento.

**Alerta registrado antes de medir:** o juiz de suficiência aprovou as 6 leituras de fumaça,
sem exceção. A volta atrás nunca disparou. Bate com os 64% de concordância medidos (ele erra
aprovando trecho que só trata do assunto). Se isso se mantiver em 300 questões, o mecanismo
de backtracking está construído e INERTE, e qualquer ganho vem de navegar melhor — não de
tentar de novo. Os vereditos vão para o trace justamente para isso ser medido.

## RESULTADO do v8 navegacional: NÃO passa no critério (2026-08-31)

Amostra estratificada de 300 (250 V/F + 50 ME). Critério fixado ANTES de ver: mais
recuperadas que quebradas no pareado, E a direção se sustentando nos dois formatos.

| contra | v8 | outro | dif | ganhou/perdeu | p |
|---|---:|---:|---:|:--:|---:|
| v5 | 83,3% | 82,7% | +0,7 pp | 21 / 19 | 0,875 |
| v3 | 83,3% | 82,3% | +1,0 pp | 22 / 19 | 0,755 |
| v1 | 83,3% | 81,7% | +1,7 pp | 24 / 19 | 0,542 |
| v7 | 83,3% | 79,3% | +4,0 pp | 25 / 13 | 0,073 |

Por formato:
- **V/F: empate** com o v5 (15 ganhas, 17 perdidas). A navegação não ajuda ali — o problema
  da V/F era o enunciado compartilhado, que o `focar` resolve mecanicamente por um centésimo
  do custo.
- **ME: 82,0% contra 74,0% do v5** nos mesmos 50 (6 ganhas, 2 perdidas). Melhor número de ME
  do projeto, mas n=50 e p=0,29.

**O que o v8 conseguiu:** é a primeira versão que não troca ganho de um formato por perda no
outro. Mas "não piorar" não era o critério, e não justifica 8,5 s/questão contra 5,5 s.

### LACUNA DE INSTRUMENTAÇÃO, minha
O plano dizia para medir o juiz de suficiência. Construí o registro no `no_navegar`
(`vereditos` no trace) mas NÃO estendi a persistência no `runner.py`, que só grava
`seg_por_no`. Os vereditos desta corrida se perderam. O que sobra são as 6 leituras de
fumaça, todas aprovadas — consistente com os 64% de concordância medidos e com a hipótese
de que o backtracking está inerte.

Para a próxima: persistir `vereditos` do mesmo jeito que `rotas` foi persistido.

## O quadro depois de oito versões

    v0 sem RAG   73,0%
    v1 RAG       80,2%   <- o unico ganho grande, e foi TROCAR O LIVRO
    v2 79,9% · v3 80,6% · v4 77,3% · v5 81,0% · v7 80,0% · v8 ~83% (n=300)
    Opus 5 SEM RAG NENHUM  85,4%

**Sete iterações de arquitetura de recuperação, todas dentro de ~3 pp umas das outras.** O
único salto real do projeto foi trocar o corpus (Bases da Anestesia -> Miller completo).
Roteamento, cota, ancoragem, tamanho de trecho, consulta focada, reformulação e navegação
agentica: nenhuma delas moveu o número de forma demonstrável.

Isso reforça a teoria já registrada em `bancada.html`: **o gargalo não é informação.** Um
modelo mais capaz, sem recuperação nenhuma e sem busca, faz 85,4%. As quatro medições que
apontam para isso continuam de pé, e a alavanca não testada continua sendo o MODELO.

## O thinking esteve DESLIGADO nas 60 células do projeto (2026-08-31)

Auditoria de `eval/resultados.sqlite`: **todas as 60 células medidas usam `rac=nenhum`.**
v0 a v8, Gemma e MedGemma, todos os blocos. A dimensão existe no código
(`nenhum | prompt | nativo`) e nunca foi exercitada com o corpus atual — foi medida só nas
fases iniciais, no `h512`, o livro refutado.

Isso importa porque a outra bancada do usuário mediu exatamente esse fator:

| setup | gemma4-plan | qwen-code |
|---|---:|---:|
| `prompt` (sem thinking) — **o que usamos** | 88,2% | 84,4% |
| `think_prompt` (com thinking) | **90,2%** | **89,3%** |

**~2 pontos no Gemma, ~5 no Qwen.** Maior que qualquer coisa que sete iterações de
arquitetura de recuperação renderam aqui, e estava disponível o tempo todo por uma flag.

Erro de priorização meu: mencionei "a dimensão realmente inexplorada é thinking ligado"
horas antes e segui por recuperação, atrás do que eu tinha acabado de diagnosticar em vez
do que tinha maior retorno esperado.

### A interação com o achado do schema
Os 12,5% de instabilidade vêm de gerar texto livre DENTRO do JSON antes da resposta
(`ressalva` antes de `resposta`). **O thinking nativo sai FORA do JSON.** Em princípio, ele
dá o ganho de raciocínio sem o custo de estabilidade — o que abriria caminho para voltar à
ancoragem `estrita` (0/40 de discordância no teste de determinismo) e ter as duas coisas.

Não testado ainda. Se o acerto subir com thinking, é o próximo experimento óbvio.

### Fumaça do thinking (gemma4-plan, contexto pequeno)
    think=False   reasoning=   0 chars · out= 106 tok · truncou=False
    think=True    reasoning= 883 chars · out= 325 tok · truncou=False

~3x mais tokens de saída. **Risco concreto: truncagem** com contexto grande mais raciocínio
longo — o `max_tokens` do responder é 4.096, e truncagem conta como falha em coluna
separada nesta bancada.

### Qwen: pré-requisitos passaram, corrida abortada por prioridade
`qwen-code` e `qwen-fast` emitem `tool_calls` com argumentos bem formados, respeitam
`json_schema`, escrevem português correto. Contraste com o `medgemma-clinical`, reprovado
por emitir ZERO tool_calls. A corrida foi interrompida em 4/250 para dar lugar ao teste de
thinking; nenhuma célula gravada.

## REFUTADO: thinking nativo (2026-09-01)

Gemma no fluxo v8, amostra estratificada de 300, única variável `--raciocinio nativo`.

| | sem thinking (v8) | com thinking |
|---|---:|---:|
| V/F | 83,6% | 82,0% |
| ME | 82,0% | 80,0% |
| **combinado (n=300)** | **83,3%** | **81,7%** |
| pareado | — | **13 ganhas / 18 perdidas**, p=0,47 |
| tempo do responder | 6,1 s | **15,8 s** (2,6×) |
| truncagem V/F | 0% | 0,8% |
| **truncagem ME** | 0% | **12,0%** |

Perde contra todas as configurações anteriores (13/18 vs v8, 17/20 vs v5, 18/20 vs v3),
sempre dentro do ruído, sempre para baixo.

**Os 12% de truncagem na ME são mecânicos:** o raciocínio consome orçamento, a questão de
múltipla escolha já traz contexto e quatro alternativas, e `max_tokens=4096` não comporta os
dois. Corrigível subindo o teto — mas não muda o quadro, porque no V/F a truncagem foi de
0,8% e o resultado também foi negativo.

**A medição externa não transferiu.** Naquela bancada o `think_prompt` valia ~2 pontos, mas
o preprompt dela é de AGENTE DE SOFTWARE (rege verbosidade, escopo, uso de ferramenta,
`~/workspace`). A nossa ancoragem rege INFERÊNCIA — uma regra sobre pesar ausência de
evidência. Julgar assertiva contra parágrafo não parece se beneficiar de deliberação longa;
possivelmente o oposto, já que a ancoragem diz como decidir e o raciocínio livre pode se
afastar dela.

Nota de infraestrutura: o `llama-swap.yaml` sempre teve `--reasoning on` no perfil
`gemma4-plan`. O thinking estava disponível desde sempre; quem o desligava era o nosso
código, mandando `enable_thinking: false` por requisição.

## O padrão de nove configurações

    v0 73,0% · v1 80,2% · v2 79,9% · v3 80,6% · v4 77,3% · v5 81,0%
    v7 80,0% · v8 ~83,3% (n=300) · v10 thinking ~81,7% (n=300)
    Opus 5 SEM RAG: 85,4%

Roteamento por classe, cota entre corpora, ancoragem por falsificação, tamanho de trecho,
consulta focada, reformulação pelo modelo, navegação agêntica, verificação de suficiência e
raciocínio nativo. **Nenhuma moveu o número de forma demonstrável.** O único salto do
projeto segue sendo a troca do corpus (Bases da Anestesia → Miller completo, +7,2 pp).

## Qwen3.6-27B no fluxo v8: EMPATE (2026-09-01)

Mesma amostra estratificada de 300, mesmo grafo, mesmos parâmetros. Única variável: o modelo.

| comparação | Qwen | Gemma | ganhou/perdeu | p |
|---|---:|---:|:--:|---:|
| vs Gemma v8 (mesmo fluxo) | 82,0% | 83,3% | 23 / 27 | 0,672 |
| vs Gemma v5 | 82,0% | 82,7% | 22 / 24 | 0,883 |
| vs Gemma v3 | 82,0% | 82,3% | 21 / 22 | **1,000** |
| vs Gemma com thinking | 82,0% | 81,7% | 23 / 22 | **1,000** |

**O formato do empate é o mais informativo:** ~22 questões cada lado resolve que o outro não
resolve, saldo próximo de zero em todas as comparações. Não são modelos equivalentes —
acertam conjuntos DIFERENTES, e nenhum é sistematicamente melhor.

Detalhes: navegação mais rápida (9,0 s contra 15,4 s do Gemma no mesmo nó); zero truncagem
no V/F; **1 chamada de ferramenta malformada em 250** (0,4%), quebrando numa string com
acentos em português — modo de falha que o Gemma não teve, e que contraria os "zero
malformadas" medidos na outra bancada (lá os argumentos eram caminhos de arquivo, não texto
livre em português); **fidelidade de citação menor**, 89,8% contra ~92%, o que pesa dado que
o objetivo declarado é resposta com fonte verificável.

## BALANÇO: onde cada alavanca chegou

    corpus       trocar o livro-texto      +7,2 pp   <- o UNICO ganho do projeto
    recuperacao  7 arquiteturas            ~0
    prompt       ancoragem, reformulacao   ~0   (e `falsificacao`: -4 pp)
    raciocinio   thinking nativo           -1,6 pp
    modelo       Gemma -> Qwen             ~0

Nove configurações e dois modelos, todos entre 80% e 83,5%. Opus 5 **sem recuperação
nenhuma**: 85,4%.

**A conclusão que a evidência sustenta:** o teto de ~83% pertence à CLASSE DE MODELO (27–31B
nesta tarefa), não ao arranjo dos componentes. Nove configurações e dois modelos chegando ao
mesmo lugar não é coincidência — é platô.

## O que resta, e que é diferente em espécie do que já foi testado

1. ~~Mandar as questões COM o nosso contexto a um modelo forte~~ — **FORA DO ROADMAP**,
   decisão do usuário em 2026-09-01. Custo real: ~5.000 chars de contexto x 1.003 questões =
   **~1,2 milhão de tokens de entrada**, pagos da assinatura dele. Eu estimei o custo de
   RODAR e não o de PAGAR ao propor.

   O teste segue sendo o único que separa "o contexto presta e o modelo local não o
   aproveita" de "o contexto não acrescenta nada" — nenhuma iteração de RAG nossa consegue
   responder isso, porque em todas o gerador é o mesmo. Fica registrado como pergunta em
   aberto, não como tarefa.
2. **Conjunto de teste** — 972 questões, nunca tocado. É a validação, não um experimento.
3. **Modelo maior**, não lateral. Restrição real: um modelo de ~18 GB por vez na GPU.
4. **Questões de imagem** (37 no dev): `qwen-vision` existe com `mmproj`, mas as figuras
   nunca foram extraídas dos PDFs das provas — é trabalho de ingestão, não troca de modelo.
5. **Dois Códigos da SBA** ainda fora do corpus normativo (lacuna registrada em 2026-08-27).
