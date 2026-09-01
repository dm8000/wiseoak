"""
Estado e nos compartilhados pelos grafos v0-v4.

Duas decisoes que valem para todos os grafos:

**Um so modelo grande por grafo.** Cabe um modelo de ~18 GB na 3090 e trocar custa 5,5 s.
Um grafo que reformula com gemma4-plan e responde com medgemma-clinical pagaria swap A
CADA PERGUNTA, duas vezes. O modelo e parametro do grafo inteiro, nao de cada no.

**MedGemma nao tem canal de thinking** (medido, com controle positivo em qwen-code:
Gemma 3 por baixo). O remedio conhecido deste projeto para "sob contexto longo o campo
numerico recebe o numero saliente" nao esta disponivel nele. Por isso `raciocinio` e uma
CONFIGURACAO, com tres valores possiveis, e nao uma escolha fixa:
    "nenhum"    responde direto
    "prompt"    pede raciocinio explicito no texto antes da resposta
    "nativo"    usa chat_template_kwargs (so funciona no gemma4-plan)

A citacao sai por `response_format: json_schema` porque este projeto ja mediu conformidade
de contrato indo de 0/15 para 15/15 com schema. Verificar citacao com juiz LLM esta
proibido pela regra da pasta; a verificacao e casamento de string contra o chunk que foi
de fato recuperado.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
import operator
from typing import Annotated, Any, Literal, TypedDict

from typing_extensions import NotRequired

from .. import clientes
from ..store import Indice

Modo = Literal["livre", "mcq", "vf"]
Raciocinio = Literal["nenhum", "prompt", "nativo"]

# Quanto o modelo pode se afastar do contexto. E uma DIMENSAO, nao uma escolha fixa.
#
# "estrita" e a instrucao original ("responda SOMENTE com base no contexto"). Medido em
# 5 casos a mao: ela faz o modelo tratar "o contexto nao confirma" como prova de que a
# assertiva e FALSA. Acerta quando a assertiva e mesmo falsa e erra quando ela e
# verdadeira mas o livro nao desce naquele detalhe — o livro diz "estimulantes
# respiratorios" e a assertiva diz "cafeina"; o livro diz que o tranexamico e
# antifibrinolitico e a assertiva descreve os sitios de ligacao de lisina.
#
# "com_ressalva" separa as duas coisas: ancora no contexto, mas quando o contexto trata
# do tema sem decidir o ponto exato, REGISTRA a lacuna no campo `ressalva` e responde
# assim mesmo. O campo existe para ser CONTADO — quantas vezes dispara, e qual o acerto
# com e sem ela — nao para ser lido por um juiz.
Ancoragem = Literal["estrita", "com_ressalva", "confiante", "confiante_quantificador", "analista",
                    "analista_leve", "bib_so", "falsificacao"]


# Defaults de TODA configuracao. Existem porque pelo LangGraph Studio o estado inicial
# e um JSON escrito a mao: exigir sete campos para fazer uma pergunta transforma a UI em
# formulario. Com isto, {"pergunta": "..."} basta.
PADRAO = {
    "idioma_busca": "pt",
    "modo": "livre",
    "modelo": "gemma4-plan",
    "raciocinio": "nenhum",
    "ancoragem": "estrita",
    # m10 = Miller's Anesthesia 10e, 14.628 trechos. Substituiu o h512 (Bases da
    # Anestesia): nas 43 questoes que todos os fluxos erravam, o acerto foi de
    # 14,0% para 51,2% so trocando o corpus.
    "indice": "dados/indice/m10",
    # indice do corpus normativo, escolhido pelo roteador. Separado do m10 de proposito:
    # no indice combinado com o Barash, o corpus minoritario foi de 52% dos trechos para
    # 7% do contexto — documento curto some ao lado de 14 mil trechos de livro.
    "indice_normas": "dados/indice/normas",
    "k_busca": 10,
    "k_contexto": 5,
    "hibrido": True,
    "expandir_pai": True,
    # 0.3 e o historico da bancada. A triagem roda com 0: nas questoes que o modelo
    # responde na duvida — que sao exatamente as do conjunto de erros — 14,9% trocam de
    # resposta entre execucoes, e comparar bracos sob esse ruido nao decide nada.
    "temp": 0.3,
    "max_rodadas": 4,
}


def cfg(estado: "Estado", chave: str):
    """Le configuracao com default. `None` explicito tambem cai no default."""
    v = estado.get(chave)
    return PADRAO[chave] if v is None else v


class Estado(TypedDict):
    pergunta: str
    # --- configuracao do braco experimental; tudo opcional, ver PADRAO
    modo: NotRequired[Modo]
    modelo: NotRequired[str]
    raciocinio: NotRequired[Raciocinio]
    ancoragem: NotRequired[Ancoragem]
    idioma_busca: NotRequired[str]
    indice_normas: NotRequired[str]
    indice_livro: NotRequired[Any]
    rota: NotRequired[str]
    indice: NotRequired[Any]
    k_busca: NotRequired[int]
    k_contexto: NotRequired[int]
    hibrido: NotRequired[bool]
    temp: NotRequired[float]
    max_rodadas: NotRequired[int]
    expandir_pai: NotRequired[bool]
    # --- preenchido pelos nos
    consulta: NotRequired[str]
    formulario: NotRequired[str]
    candidatos: NotRequired[list[str]]
    contexto: NotRequired[list[dict]]
    resposta: NotRequired[str]
    letra: NotRequired[str]
    citacoes: NotRequired[list[dict]]
    ressalva: NotRequired[str]
    erro_apontado: NotRequired[str]
    fontes: NotRequired[list[str]]
    preprompt: NotRequired[str]
    truncou: NotRequired[bool]
    trace: Annotated[list[dict], operator.add]


# --------------------------------------------------------------------- schemas

_CITACAO = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "capitulo": {"type": "integer"},
            "pagina": {"type": "integer"},
            "trecho": {"type": "string"},
        },
        "required": ["capitulo", "pagina", "trecho"],
        "additionalProperties": False,
    },
}

SCHEMAS: dict[Modo, dict] = {
    "livre": {"type": "object", "additionalProperties": False,
              "required": ["resposta", "citacoes"],
              "properties": {"resposta": {"type": "string"}, "citacoes": _CITACAO}},
    "mcq": {"type": "object", "additionalProperties": False,
            "required": ["letra", "citacoes"],
            "properties": {"letra": {"type": "string", "enum": ["A", "B", "C", "D"]},
                           "citacoes": _CITACAO}},
    "vf": {"type": "object", "additionalProperties": False,
           "required": ["resposta", "citacoes"],
           "properties": {"resposta": {"type": "string", "enum": ["V", "F"]},
                          "citacoes": _CITACAO}},
}

_RESSALVA = {"ressalva": {"type": "string"}}

# Campo da ancoragem "falsificacao". Vem ANTES de `resposta` no schema: em geracao
# restrita por gramatica o modelo emite na ordem declarada, entao preencher isto e
# raciocinio, e nao racionalizacao pos-fato.
_ERRO = {"erro_apontado": {"type": "string"}}

# A ancoragem "analista" pede bibliografia: o livro entra na citacao em vez de ser
# subentendido pelo renderizador.
_CITACAO_BIB = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "livro": {"type": "string"},
            "capitulo": {"type": "integer"},
            "pagina": {"type": "integer"},
            "trecho": {"type": "string"},
        },
        "required": ["livro", "capitulo", "pagina", "trecho"],
        "additionalProperties": False,
    },
}


def schema_para(modo: "Modo", ancoragem: "Ancoragem") -> dict:
    """
    No modo estrito o schema fica IDENTICO ao original — nenhuma medicao ja feita muda
    de significado por causa desta funcao.
    """
    base = SCHEMAS[modo]
    if ancoragem == "estrita":
        return base  # nao mexer: celulas ja medidas dependem deste schema
    # ORDEM DOS CAMPOS IMPORTA. Em geracao restrita por gramatica o modelo emite os
    # campos NA ORDEM do schema; um campo depois da resposta e racionalizacao pos-fato,
    # nao raciocinio. Medido: com `ressalva` DEPOIS, o modelo escreveu "o acido
    # tranexamico e um analogo sintetico da lisina" na ressalva e ainda assim respondeu
    # F. `ressalva` vem PRIMEIRO para poder influenciar o veredito.
    if ancoragem == "falsificacao":
        return {**base,
                "required": ["erro_apontado"] + base["required"],
                "properties": {**_ERRO, **base["properties"]}}
    props = {**_RESSALVA, **base["properties"]}
    if ancoragem in ("analista", "analista_leve", "bib_so"):
        props["citacoes"] = _CITACAO_BIB
    return {**base,
            "required": ["ressalva"] + base["required"],
            "properties": props}


SISTEMA: dict[str, str] = {
    "estrita": ("Voce e um assistente de anestesiologia. Responda SOMENTE com base no "
                "contexto fornecido. Se o contexto nao contiver a resposta, diga isso "
                "em vez de completar com conhecimento proprio."),
    "com_ressalva": (
        "Voce e um assistente de anestesiologia. Ancore a resposta no contexto "
        "fornecido. Quando o contexto tratar do tema mas NAO decidir o ponto exato da "
        "pergunta — por exemplo, confirma a classe do farmaco mas nao o mecanismo, ou "
        "fala da categoria mas nao do agente especifico — escreva essa lacuna em "
        "'ressalva' e AINDA ASSIM de a melhor resposta possivel, completando com seu "
        "conhecimento o que faltar. Nao deixe de responder. Se o contexto decidir a "
        "questao, deixe 'ressalva' vazia."),
    # Terceira variante: ataca de frente o modo de falha medido. O modelo tratava
    # "o contexto nao confirma" como prova de falsidade, e chegou a escrever o mecanismo
    # correto do acido tranexamico na ressalva antes de responder F. Aqui a regra de
    # inferencia e dita explicitamente, em vez de ficar implicita na permissao.
    "confiante": (
        "Voce e um assistente de anestesiologia. Use o contexto do livro como APOIO, "
        "nao como limite.\n\n"
        "REGRA CENTRAL: a ausencia de uma informacao no contexto NAO e evidencia de que "
        "a afirmacao seja falsa. O livro nao cobre todo detalhe. So julgue algo falso "
        "quando o contexto CONTRADIZ a afirmacao, ou quando seu proprio conhecimento "
        "indica que ela e falsa.\n\n"
        "Se o contexto confirma a categoria mas nao o detalhe — cita a classe do farmaco "
        "mas nao o mecanismo, fala do grupo mas nao do agente — isso e COMPATIVEL com a "
        "afirmacao, nao contra ela. Anote a lacuna em 'ressalva' e decida somando o "
        "contexto ao seu conhecimento.\n\n"
        "Decida sempre. Nao se abstenha."),

    # Ataca o vies MEDIDO: nos itens que todos os fluxos erram, 66% tem gabarito
    # VERDADEIRO — ou seja, o erro conjunto e quase sempre negar algo verdadeiro. A
    # ancoragem "confiante" ja dizia que ausencia nao e prova de falsidade e nao bastou,
    # porque dizia o que NAO fazer sem dar a regra do que fazer.
    #
    # A regra aqui e logica, nao retorica: uma afirmacao so e falsa se algo NELA estiver
    # errado. O schema obriga a apontar o erro ANTES de responder — sem erro apontado,
    # a resposta e Verdadeiro. Isso troca o default: hoje o modelo nega por duvida;
    # aqui ele so nega por evidencia.
    "falsificacao": (
        "Voce julga afirmacoes de anestesiologia. Use o contexto do livro como apoio, "
        "nao como limite.\n\n"
        "REGRA DE DECISAO, nesta ordem:\n"
        "1. Procure o ERRO. Uma afirmacao so e FALSA se algo NELA estiver incorreto: um "
        "numero trocado, um mecanismo errado, uma indicacao que na verdade e "
        "contraindicacao, uma relacao invertida, uma parte que contradiz a outra.\n"
        "2. Escreva em 'erro_apontado' QUAL e esse erro, de forma concreta e "
        "verificavel. Nao vale escrever 'o contexto nao confirma' — isso NAO e um erro "
        "da afirmacao, e uma limitacao da sua fonte.\n"
        "3. Se voce nao conseguir apontar um erro concreto, a resposta e VERDADEIRO, e "
        "'erro_apontado' fica vazio.\n\n"
        "Nao confunda 'nao consegui confirmar' com 'e falso'. Sao coisas diferentes, e "
        "confundi-las e o erro mais comum nesta tarefa. Se a afirmacao e plausivel, "
        "coerente com o que voce sabe, e nada nela esta errado, ela e verdadeira mesmo "
        "que o contexto nao a mencione."),

    # Mistura do preprompt de producao do gemma4-plan (prompts/gemma4-plan.md) com a
    # regra de inferencia da ancoragem "confiante". O RAG e apresentado como FERRAMENTA
    # do analista, nao como limite: e a diferenca de postura que se quer medir.
    "analista": (
        "Analista de anestesiologia. Entrega decisao, nao menu de opcoes.\n\n"
        "FERRAMENTA: voce consultou o livro-texto e recebeu os trechos abaixo. Eles sao "
        "sua fonte primaria — use-os. Nao sao seu limite.\n\n"
        "REGRA DE INFERENCIA: a ausencia de uma informacao nos trechos NAO e evidencia "
        "de que ela seja falsa. So negue quando o trecho CONTRADIZ, ou quando seu "
        "conhecimento indica que e falso. Trecho que confirma a categoria mas nao o "
        "detalhe e COMPATIVEL com a afirmacao, nao contra ela.\n\n"
        "RIGOR:\n"
        "- Marque o que e FATO (esta nos trechos, e voce cita), o que e ESTIMATIVA "
        "(sua inferencia a partir deles) e o que e CONHECIMENTO PROPRIO (nao esta nos "
        "trechos).\n"
        "- Numero sem fonte e chute: ou cita, ou marca como estimativa, ou nao usa.\n"
        "- 'Nao sei' e diferente de 'nao e possivel saber'. Use a certa.\n"
        "- Premissa falsa na pergunta: corrija antes de responder.\n"
        "- Densidade acima de volume. Uma frase que decide vale mais que um paragrafo "
        "que contextualiza. Pergunta simples, resposta simples.\n\n"
        "ORDEM DA RESPOSTA: primeiro a resposta e o racional; as RESSALVAS vem por "
        "ultimo, ao final do texto, nunca no meio. Decida sempre."),

    # ABLACAO do "analista", que perdeu 21 pp. Duas variantes que isolam de onde veio a
    # perda — persona/rigor, ou a exigencia de bibliografia.
    #
    # analista_leve: mantem a persona e a bibliografia, TIRA o bloco de rigor
    # (fato/estimativa/opiniao, numero sem fonte, premissa falsa, densidade).
    "analista_leve": (
        "Analista de anestesiologia. Entrega decisao, nao menu de opcoes.\n\n"
        "FERRAMENTA: voce consultou o livro-texto e recebeu os trechos abaixo. Eles sao "
        "sua fonte primaria — use-os. Nao sao seu limite.\n\n"
        "REGRA DE INFERENCIA: a ausencia de uma informacao nos trechos NAO e evidencia "
        "de que ela seja falsa. So negue quando o trecho CONTRADIZ, ou quando seu "
        "conhecimento indica que e falso. Trecho que confirma a categoria mas nao o "
        "detalhe e COMPATIVEL com a afirmacao, nao contra ela.\n\n"
        "ORDEM DA RESPOSTA: primeiro a resposta; as RESSALVAS por ultimo. "
        "Decida sempre. Nao se abstenha."),

    # bib_so: o texto do "confiante" LETRA POR LETRA, mais uma unica frase pedindo a
    # bibliografia. Se a queda continuar aqui, a culpa e da bibliografia; se sumir, era
    # a persona.
    "bib_so": (
        "Voce e um assistente de anestesiologia. Use o contexto do livro como APOIO, "
        "nao como limite.\n\n"
        "REGRA CENTRAL: a ausencia de uma informacao no contexto NAO e evidencia de que "
        "a afirmacao seja falsa. O livro nao cobre todo detalhe. So julgue algo falso "
        "quando o contexto CONTRADIZ a afirmacao, ou quando seu proprio conhecimento "
        "indica que ela e falsa.\n\n"
        "Se o contexto confirma a categoria mas nao o detalhe — cita a classe do farmaco "
        "mas nao o mecanismo, fala do grupo mas nao do agente — isso e COMPATIVEL com a "
        "afirmacao, nao contra ela. Anote a lacuna em 'ressalva' e decida somando o "
        "contexto ao seu conhecimento.\n\n"
        "Cada citacao vai no formato de bibliografia: livro, capitulo, pagina e o trecho "
        "EXATO entre aspas, copiado literalmente.\n\n"
        "Decida sempre. Nao se abstenha."),
}

# `confiante` LETRA POR LETRA mais tres verificacoes. Os padroes vem do dossie de erro:
# o modelo ignora quantificador (trata "frequente" como "pode"), le negacao por cima
# ("non-polluting") e inverte relacao ("ke0 grande" significa meia-vida CURTA).
#
# Deliberadamente CURTO e sem roteiro de passos: esta bancada mediu envelope generico de
# prompt derrubando o RAG de 93,3% para 80%. Diz o que CONFERIR, nao como pensar.
SISTEMA["confiante_quantificador"] = SISTEMA["confiante"] + (
    "\n\nAntes de decidir, confira tres pontos NA AFIRMACAO:\n"
    "1. QUANTIFICADOR - 'sempre', 'todos', 'nunca', 'frequente' afirmam mais que 'pode' "
    "ou 'ocorre'. Se a fonte so sustenta a forma fraca, a forma forte e FALSA.\n"
    "2. NEGACAO - leia 'nao', 'sem', 'exceto', 'contraindicado' literalmente; uma "
    "negacao trocada inverte a afirmacao inteira.\n"
    "3. RELACAO - 'maior', 'mais rapido', 'aumenta' tem de apontar para o mesmo lado que "
    "a fonte. Constante grande costuma significar tempo CURTO.")


INSTRUCAO: dict[Modo, str] = {
    "livre": "Responda a pergunta. Em 'citacoes', liste os trechos do livro em que voce "
             "se baseou, COPIADOS LITERALMENTE do contexto, com o capitulo e a pagina.",
    "mcq": "Escolha a alternativa correta e devolva apenas a letra em 'letra'. Em "
           "'citacoes', copie LITERALMENTE do contexto o trecho que sustenta a escolha.",
    "vf": "Julgue a assertiva: 'V' se verdadeira, 'F' se falsa. Em 'citacoes', copie "
          "LITERALMENTE do contexto o trecho que sustenta o julgamento.",
}


_INDICES: dict[str, Indice] = {}


def indice_de(estado: Estado) -> Indice | None:
    """
    Aceita o indice como objeto OU como caminho em texto.

    Pelo LangGraph Studio o estado inicial e JSON: nao da para passar um objeto Python.
    Sem isto, v1-v4 sao impossiveis de rodar pela UI. O cache evita recarregar 10 MB de
    vetores a cada invocacao.
    """
    ix = estado.get("indice")
    if ix is None:
        ix = PADRAO["indice"]
    if isinstance(ix, Indice):
        return ix
    caminho = str(ix)
    if caminho not in _INDICES:
        _INDICES[caminho] = Indice(caminho)
    return _INDICES[caminho]


INSTRUCAO_ANALISTA: dict[str, str] = {
    "livre": (
        "Responda a pergunta. Toda afirmacao que vier dos trechos precisa aparecer em "
        "'citacoes', no formato de bibliografia: nome do livro, capitulo, pagina e o "
        "TRECHO EXATO, copiado literalmente entre aspas — nao parafraseado. O que voce "
        "afirmar de conhecimento proprio, marque como tal no texto e NAO cite. "
        "Em 'ressalva', o que os trechos nao decidiram."),
    "mcq": ("Escolha a alternativa correta e devolva a letra em 'letra'. Em 'citacoes', "
            "copie LITERALMENTE o trecho que sustenta a escolha, com livro, capitulo e "
            "pagina. Em 'ressalva', o que os trechos nao decidiram."),
    "vf": ("Julgue a assertiva: 'V' se verdadeira, 'F' se falsa. Em 'citacoes', copie "
           "LITERALMENTE o trecho que sustenta o julgamento, com livro, capitulo e "
           "pagina. Em 'ressalva', o que os trechos nao decidiram."),
}


# O QUE CADA NO FAZ, em uma linha. Vai junto no trace para o painel de estado do
# LangGraph Studio explicar o no sem exigir que quem audita leia o codigo.
FAZ = {
    "reformular": "reescreve a pergunta como consulta de busca, com termos tecnicos",
    "focar": "usa so a assertiva como consulta de busca; o enunciado e comum entre irmas",
    "recuperar": "busca no indice do livro (hibrida BM25+vetor, ou so vetor)",
    "rerankear": "reordena os trechos por relevancia, no Qwen3-Reranker em CPU",
    "contexto": "monta o contexto final; opcionalmente troca o trecho pela secao-pai",
    "criticar": "julga se os trechos BASTAM (nao se a resposta esta certa)",
    "formula": "da ao modelo UMA ferramenta: consultar o formulario (sem calculadora)",
    "navegar": "o modelo ve o catalogo e escolhe o que ler, com ferramentas",
    "responder": "produz resposta, citacoes e ressalva, com schema JSON",
}


def _marcar(no: str, t0: float, **extra) -> dict:
    """
    Devolve o passo como ATUALIZACAO de estado, para o redutor concatenar.

    `faz` acompanha cada passo de proposito: no Studio, clicar num no mostra o estado,
    nao o codigo. Sem esta linha, quem audita ve `rerankear` e um numero, e tem de ir
    ler o fonte para saber o que aconteceu ali.
    """
    return {"trace": [{"no": no, "faz": FAZ.get(no, ""),
                       "segundos": round(time.time() - t0, 2), **extra}]}


# ------------------------------------------------------------------------ nos

_REFORMULAR = {
    # Descreve o PROBLEMA e proibe os candidatos. Motivado por caso medido: numa questao
    # cujas alternativas eram manitol/dobutamina/terlipressina/fenoldopam, a busca densa
    # devolveu quatro paragrafos sobre MANITOL — o distrator — porque a palavra estava na
    # consulta. O prompt "pt" pede explicitamente nome de farmaco, que na multipla escolha
    # sao as alternativas: pedir sinonimo e farmaco e pedir para envenenar a busca.
    "problema": ("Voce reescreve uma questao de anestesiologia como consulta de busca em "
                 "livro-texto. Descreva o PROBLEMA CLINICO: quadro, achados, mecanismo e "
                 "o que se pergunta. NUNCA inclua as alternativas de resposta nem os "
                 "farmacos listados como opcao — buscar por eles traz material sobre a "
                 "opcao errada. Devolva SO a consulta, sem explicacao."),
    "pt": ("Voce reescreve perguntas de anestesiologia como consulta para busca "
           "em um livro-texto. Devolva SO a consulta reescrita: termos tecnicos, "
           "sinonimos e nome de farmaco. Sem explicacao, sem preambulo."),
    # A busca densa prefere trechos no MESMO IDIOMA da consulta. Medido: com o Barash
    # em ingles valendo 52% do indice, ele ficou com 7% do contexto quando a consulta
    # ia em portugues, e posicao mediana 11 no top-20 — quase sempre abaixo do corte.
    # Traduzir a CONSULTA (a resposta segue em portugues) e o conserto barato.
    "en": ("You rewrite Portuguese anesthesiology questions as an ENGLISH search query "
           "for a medical textbook. Return ONLY the query: technical terms, synonyms, "
           "drug names in English. Use the standard English terminology of the field "
           "(e.g. 'bloqueador neuromuscular adespolarizante' -> 'nondepolarizing "
           "neuromuscular blocking agent'). No explanation, no preamble."),
}


def no_reformular(estado: Estado) -> dict:
    """
    Reescreve a pergunta como consulta de busca. Unico no que gasta uma chamada de LLM
    ANTES de recuperar qualquer coisa — por isso a latencia dele e medida a parte.

    `idioma_busca="en"` traduz a consulta. Serve para corpus em ingles: o embedding e
    multilingue, mas nao neutro — ele casa melhor dentro do mesmo idioma, e por isso um
    livro em ingles fica invisivel para uma pergunta em portugues.
    """
    t0 = time.time()
    idioma = cfg(estado, "idioma_busca")
    r = clientes.chat(
        [{"role": "system", "content": _REFORMULAR.get(idioma, _REFORMULAR["pt"])},
         # se `no_focar` ja isolou a parte que descreve o problema, reescreve A PARTIR
         # dela: reformular a pergunta inteira reintroduz o enunciado compartilhado que o
         # foco tinha removido, e a recuperacao volta a cair no cluster generico
         {"role": "user", "content": estado.get("consulta") or estado["pergunta"]}],
        modelo=cfg(estado, "modelo"),
        think=True if cfg(estado, "raciocinio") == "nativo" else False,
        max_tokens=256, temp=0.3)
    consulta = " ".join(r["content"].split())[:400] or estado["pergunta"]
    return {"consulta": consulta,
            **_marcar("reformular", t0, tokens=r["out_tokens"], idioma=idioma,
                      pergunta_original=estado["pergunta"][:160],
                      consulta_gerada=consulta[:200])}


# Classes cuja fonte NAO e livro-texto. Medido: o Opus 5 abriu 15,4% de vantagem em
# `juridico-normativo` e 9,7% em `gestao` — sao normas do CFM, estatuto da SBA e
# literatura de gestao hospitalar, e o Miller nao cobre nenhuma das tres.
# `gestao` esta de volta DE PROPOSITO. Ela saiu quando o roteamento era exclusivo, porque
# la mandar gestao as normas custava o livro inteiro e a classe caiu de 90,3% para 77,4%.
# Com COTA o remendo vira estorvo: deixando gestao fora, ela recebe 4 vagas do livro e a
# cota nunca e testada exatamente onde o desenho anterior quebrou. Se a cota presta,
# gestao tem de voltar a subir recebendo 2 normas + 2 livro.
_ROTA_NORMAS = {"juridico-normativo", "gestao"}


def no_rotear(estado: Estado) -> dict:
    """
    Escolhe o indice pela CLASSE da pergunta. Duas camadas, e a primeira e de graca.

    1. REGRA (regex). So dispara em ancora literal — "CFM", "Resolucao nº", "Lei nº",
       "Codigo de Etica", "Sociedade Brasileira de Anestesiologia". Precisao alta,
       cobertura baixa, de proposito: e a mesma regra de bench/classificar.py, e ela
       foi escrita para nao chutar.
    2. LLM com schema fechado, so quando a regra nao casa. Uma chamada de 48 tokens.

    O risco e simetrico e vale declarar: rotear errado manda a pergunta ao indice errado,
    e ai o RAG piora em vez de melhorar. Por isso a regra vem primeiro.
    """
    t0 = time.time()
    import sys as _sys
    from pathlib import Path as _Path
    _b = _Path(__file__).resolve().parents[2] / "bench"
    if str(_b) not in _sys.path:
        _sys.path.insert(0, str(_b))
    from classificar import classe_llm, classe_regra  # noqa: E402

    item = {"enunciado": estado["pergunta"], "assertiva": "", "alternativas": {}}
    classe = classe_regra(item)
    origem = "regra"
    if classe is None:
        classe = classe_llm(item, cfg(estado, "modelo"))
        origem = "llm"

    normas = classe in _ROTA_NORMAS
    destino = cfg(estado, "indice_normas") if normas else cfg(estado, "indice")
    return {"indice": destino, "rota": classe,
            # o indice do livro ANTES de ser sobrescrito: sem guardar, o esquema de cota
            # nao teria como voltar a ele — o roteador e destrutivo por natureza
            "indice_livro": cfg(estado, "indice"),
            **_marcar("rotear", t0, classe=classe, origem=origem,
                      indice="normas" if normas else "livro",
                      decisao=("corpus normativo (CFM/SBA)" if normas
                               else "livro-texto (Miller 10e)"))}


def no_formula(estado: Estado) -> dict:
    """
    Da ao modelo UMA ferramenta: consultar o formulario.

    POR QUE UM NO SEPARADO, e nao tool calling dentro do responder: o responder gera sob
    `schema=` (gramatica JSON forcada), e schema e tools se excluem — pedir os dois faz o
    modelo devolver JSON e nunca chamar a ferramenta. Separando, a resposta final continua
    estruturada e mensuravel, que e do que a bancada inteira depende.

    ALCANCE DECLARADO: ~5 itens em 1.003 (0,5%), contra IC de +/-2,5 pp. Serve ao produto;
    nao e mensuravel nesta amostra. Ver o cabecalho de ferramentas/formulario.py.
    """
    t0 = time.time()
    from wiseoak.ferramentas import formulario

    r = clientes.chat(
        [{"role": "system",
          "content": "Voce prepara material para responder uma questao de "
                     "anestesiologia. Se a questao depender de uma relacao "
                     "quantitativa (formula, valor de referencia calculado), chame a "
                     "ferramenta `formula` descrevendo a situacao clinica. Se nao "
                     "depender, responda apenas NAO."},
         {"role": "user", "content": estado["pergunta"][:1500]}],
        modelo=cfg(estado, "modelo"), think=False, max_tokens=256, temp=0.0,
        tools=[formulario.ESPECIFICACAO])

    chamadas = r.get("tool_calls") or []
    if not chamadas:
        return {"formulario": "", **_marcar("formula", t0, chamou=False,
                                            decisao="modelo nao pediu formula")}
    textos, consultas = [], []
    for c in chamadas[:2]:
        try:
            args = json.loads((c.get("function") or {}).get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        consultas.append(str(args.get("consulta") or "")[:80])
        textos.append(formulario.executar(args))
    return {"formulario": "\n\n".join(t for t in textos if t),
            **_marcar("formula", t0, chamou=True, consultas=consultas,
                      decisao=f"consultou o formulario: {consultas}")}


def no_focar(estado: Estado) -> dict:
    """
    A consulta de BUSCA passa a ser so a assertiva; o gerador continua vendo tudo.

    MEDIDO: 790 das 836 questoes V/F do dev (94%) compartilham o enunciado com outra
    questao, em grupos de ate 12. O enunciado e o cenario ("Desde o momento da concepcao,
    inumeras alteracoes fisiologicas ocorrem na gestante...") e por construcao e IDENTICO
    entre questoes irmas — carrega zero informacao discriminativa e ocupava a maior parte
    do vetor de consulta: 235 caracteres de texto comum contra 40 da assertiva.

    O efeito era tres assertivas diferentes ("ventilacao minuto", "capacidade vital",
    "capacidade pulmonar total") recuperarem o MESMO contexto. Focando na assertiva, as
    tres divergem e cada uma acerta o alvo.

    So altera `consulta`, que e o que `no_recuperar` usa; `pergunta` segue intacta para o
    responder, que precisa do cenario para interpretar a assertiva.

    Nao se aplica a multipla escolha: la o enunciado carrega o caso e as alternativas sao
    o que varia, entao a consulta completa continua sendo a certa.
    """
    t0 = time.time()
    p = estado["pergunta"]
    marca = "\n\nAssertiva:"
    if marca not in p:
        return {**_marcar("focar", t0, focou=False,
                          decisao="sem assertiva destacada; consulta inalterada")}
    asr = p.split(marca, 1)[1].strip()
    return {"consulta": asr,
            **_marcar("focar", t0, focou=True, chars_antes=len(p), chars_depois=len(asr),
                      decisao="busca so pela assertiva; enunciado segue no prompt")}


def no_recuperar(estado: Estado) -> dict:
    t0 = time.time()
    ix = indice_de(estado)
    consulta = estado.get("consulta") or estado["pergunta"]
    k = cfg(estado, "k_busca")
    hibrido = cfg(estado, "hibrido")
    res = ix.buscar(consulta, k, hibrido=hibrido)
    return {"candidatos": [cid for cid, _ in res],
            **_marcar("recuperar", t0, n=len(res), k=k,
                      busca="hibrida (BM25 + denso, RRF)" if hibrido else "densa pura",
                      consulta_usada=consulta[:200])}


def _fonte_de(estado: Estado, qual: str):
    """Resolve "normas" ou "livro" para o objeto de indice correspondente."""
    chave = "indice_normas" if qual == "normas" else "indice_livro"
    alvo = estado.get(chave) or PADRAO.get(chave) or PADRAO["indice"]
    return indice_de({**estado, "indice": alvo})


def no_recuperar_cota(estado: Estado) -> dict:
    """
    COTA FIXA por corpus, em vez de escolher um so.

    O roteamento exclusivo do v8 e destrutivo: ao mandar a pergunta para as normas ele
    TIRA o livro. Quando o rotulo da classe nao corresponde a onde a resposta esta, a
    pergunta perde a unica fonte que a continha — medido: 3 dos 13 erros restantes em
    `juridico-normativo` sao risco ocupacional (abuso de substancias, radiacao), tema do
    Miller cap. 84, e o v1 acertava varios deles.

    A cota reserva vagas em vez de disputa-las. NAO e o mesmo que fundir os indices: num
    indice unico as vagas saem de competicao global, e 607 trechos de norma perdem para
    14.628 do Miller por volume — a bancada mediu isso com o Barash (52% dos trechos, 7%
    do contexto). Com cota as normas competem so com normas.

    So se aplica a pergunta ROTEADA as normas. Questao clinica continua com o livro
    inteiro: dar-lhe 2 vagas de resolucao do CFM seria trocar contexto util por ruido.

    O id do candidato leva o prefixo da fonte ("normas|abc123"), porque a partir daqui
    dois indices estao em jogo e um id solto nao diz mais de onde veio.
    """
    t0 = time.time()
    consulta = estado.get("consulta") or estado["pergunta"]
    k = cfg(estado, "k_busca")
    normativa = estado.get("rota") in _ROTA_NORMAS

    if not normativa:
        ix = _fonte_de(estado, "livro")
        res = [("livro", cid) for cid, _ in ix.buscar(consulta, k, hibrido=False)]
        cotas = {"livro": k}
    else:
        metade = max(1, k // 2)
        res = ([("normas", cid) for cid, _ in
                _fonte_de(estado, "normas").buscar(consulta, metade, hibrido=False)]
               + [("livro", cid) for cid, _ in
                  _fonte_de(estado, "livro").buscar(consulta, metade, hibrido=False)])
        cotas = {"normas": metade, "livro": metade}

    return {"candidatos": [f"{f}|{c}" for f, c in res],
            **_marcar("recuperar", t0, n=len(res), k=k, cotas=cotas,
                      busca="densa pura, com cota por corpus",
                      decisao=("normas + livro, vagas garantidas" if normativa
                               else "livro inteiro (pergunta clínica)"),
                      consulta_usada=consulta[:200])}


def no_contexto_cota(estado: Estado) -> dict:
    """
    Monta o contexto ALTERNANDO entre as fontes, nao concatenando.

    Se as vagas das normas viessem todas antes das do livro, o corte em k_contexto
    poderia decapitar uma fonte inteira, e a cota deixaria de existir na pratica.
    """
    t0 = time.time()
    k = cfg(estado, "k_contexto")
    por_fonte: dict[str, list[str]] = {}
    for marcado in estado["candidatos"]:
        fonte, _, cid = marcado.partition("|")
        por_fonte.setdefault(fonte, []).append(cid)

    intercalado: list[tuple[str, str]] = []
    for i in range(max((len(v) for v in por_fonte.values()), default=0)):
        for fonte, ids in por_fonte.items():
            if i < len(ids):
                intercalado.append((fonte, ids[i]))

    vistos: set[str] = set()
    contexto: list[dict] = []
    for fonte, cid in intercalado[:k]:
        c = _fonte_de(estado, fonte).obter(cid)
        if c["id"] in vistos:
            continue
        vistos.add(c["id"])
        alvo = c
        if cfg(estado, "expandir_pai"):
            # o filho ganha a busca, o pai da o entorno. Medido: em 41% dos erros
            # clinicos a resposta esta no top-50 e nao entra nas vagas — o fato
            # especifico costuma estar a um paragrafo do trecho recuperado.
            pai = _fonte_de(estado, fonte).pai_de(cid)
            if pai:
                alvo = {**pai, "id_filho": cid}
        # natureza da fonte, nao so o nome: o modelo precisa saber que um artigo de
        # resolucao DEFINE obrigatoriedade, enquanto o livro DESCREVE pratica. Sem isso
        # ele pode responder "e obrigatorio" a partir do que o Miller recomenda.
        contexto.append({**alvo, "natureza": "NORMA" if fonte == "normas" else "LIVRO"})
    return {"contexto": contexto,
            **_marcar("contexto", t0, n=len(contexto),
                      chars=sum(len(c["texto"]) for c in contexto),
                      expandiu_para_pai=cfg(estado, "expandir_pai"),
                      fontes={f: sum(1 for x, _ in intercalado[:k] if x == f)
                              for f in por_fonte},
                      trechos=[f"{c.get('livro','?')[:24]} p{c.get('pagina_inicial')}"
                               for c in contexto])}


def no_rerankear(estado: Estado) -> dict:
    t0 = time.time()
    ix = indice_de(estado)
    consulta = estado.get("consulta") or estado["pergunta"]
    k = cfg(estado, "k_contexto")
    antes = len(estado["candidatos"])
    pares = ix.rerankear(consulta, estado["candidatos"], k=k)
    return {"candidatos": [cid for cid, _ in pares],
            **_marcar("rerankear", t0,
                      modelo="rerank-small (Qwen3-Reranker 0.6B, CPU)",
                      de=antes, para=len(pares),
                      scores=[round(x, 3) for _, x in pares[:5]])}


def no_montar_contexto(estado: Estado) -> dict:
    """
    O filho ganha a busca; o pai da o entorno. Sem expandir, a resposta fica presa a uma
    janela de ~390 tokens e perde a frase anterior que definia o termo.
    """
    t0 = time.time()
    ix = indice_de(estado)
    k = cfg(estado, "k_contexto")
    vistos: set[str] = set()
    contexto: list[dict] = []
    for cid in estado["candidatos"][:k]:
        c = ix.obter(cid)
        alvo = c
        if cfg(estado, "expandir_pai"):
            pai = ix.pai_de(cid)
            if pai:
                alvo = {**pai, "id_filho": cid}
        if alvo["id"] in vistos:
            continue
        vistos.add(alvo["id"])
        contexto.append(alvo)
    passo = _marcar("contexto", t0, n=len(contexto),
            chars=sum(len(c["texto"]) for c in contexto),
            expandiu_para_pai=cfg(estado, "expandir_pai"),
            trechos=[f"cap{c.get('capitulo_num')} p{c.get('pagina_inicial')}"
                     f"{' · ' + ' > '.join(c.get('caminho') or []) if c.get('caminho') else ''}"
                     for c in contexto])
    return {"contexto": contexto, **passo, "fontes": [
        f"cap{c.get('capitulo_num')} p{c.get('pagina_inicial')}-{c.get('pagina_final')}"
        f"{' · ' + ' > '.join(c.get('caminho') or []) if c.get('caminho') else ''}"
        for c in contexto]}


def formatar_contexto(contexto: list[dict]) -> str:
    partes = []
    for i, c in enumerate(contexto, 1):
        caminho = " > ".join(c.get("caminho") or [])
        nat = c.get("natureza")
        if nat == "NORMA":
            # norma se cita por artigo, nao por capitulo e pagina
            partes.append(f"[{i}] NORMA · {c.get('livro')} · {c.get('capitulo')}\n"
                          + c["texto"])
            continue
        cab = ((f"[{i}] LIVRO · " if nat else f"[{i}] ")
               + f"{c.get('livro')} | capitulo {c.get('capitulo_num')}: "
               f"{c.get('capitulo')} | pagina {c.get('pagina_inicial')}"
               + (f"-{c.get('pagina_final')}" if c.get("pagina_final") != c.get("pagina_inicial") else "")
               + (f" | {caminho}" if caminho else ""))
        partes.append(cab + "\n" + c["texto"])
    return "\n\n".join(partes)


_CATALOGO_CACHE: dict[str, str] = {}


def _sistema_navegar(ix_livro, ix_normas) -> str:
    """
    O `system` da navegacao. IDENTICO entre perguntas, de proposito.

    MEDIDO: o catalogo tem ~12.900 tokens e custa 13,5 s de prefill na primeira chamada.
    Com `--cache-reuse` no llama-swap e o catalogo como PREFIXO FIXO, a segunda chamada
    custa 1,3 s — dez vezes menos. Se qualquer parte do `system` variar por pergunta, o
    prefixo diverge no primeiro token e nada e reaproveitado; foi o que aconteceu na
    primeira medicao, quando o system mudava e o catalogo ia no `user`.

    Por isso: nada especifico da pergunta aqui. Nem a pergunta, nem a classe, nem o modo.
    """
    from wiseoak.ferramentas import biblioteca as B
    chave = f"{ix_livro.caminho}|{ix_normas.caminho}"
    if chave not in _CATALOGO_CACHE:
        _CATALOGO_CACHE[chave] = B.catalogo(ix_livro, ix_normas)
    return (
        "Voce responde questoes de anestesiologia consultando uma biblioteca.\n\n"
        "Use `ler` quando o catalogo indicar onde o assunto mora — e o caminho preferido, "
        "porque restringe a busca aquela parte e evita trazer material sobre assunto "
        "vizinho. Use `buscar` so quando o catalogo nao indicar.\n\n"
        "AO FORMULAR A BUSCA, descreva o PROBLEMA. Nunca inclua as alternativas de "
        "resposta nem os farmacos listados como opcao: buscar por eles traz material "
        "sobre a opcao errada.\n\n"
        "Depois de cada leitura voce recebe um veredito automatico de suficiencia. Se for "
        "INSUFICIENTE, tente OUTRO item do catalogo. Quando tiver material que decida a "
        "questao, responda PRONTO sem chamar mais nada.\n\n"
        "CATALOGO DA BIBLIOTECA:\n" + _CATALOGO_CACHE[chave]
    )


def no_navegar(estado: Estado) -> dict:
    """
    O modelo navega a biblioteca e junta o contexto que quiser, com verificacao de
    suficiencia e volta atras.

    Motivado por caso MEDIDO: na questao do cirrotico com sindrome hepatorrenal, cujas
    alternativas eram manitol/dobutamina/terlipressina/fenoldopam, a busca densa devolveu
    quatro paragrafos sobre MANITOL — porque a palavra estava na consulta. Quem navega por
    estrutura nao tem esse modo de falha.

    O laco roda SEM `schema`, porque schema e tools se excluem; o veredito estruturado sai
    no `no_responder`, sobre o contexto que este no juntou.

    TETO DE RODADAS obrigatorio: 5% das perguntas nao tem resposta no corpus (medido), e
    sem teto sao exatamente elas que consomem a corrida procurando o que nao existe. O
    `v4_autocritica` ja registrava a licao.

    O juiz de suficiencia e o elo fraco e vai instrumentado: concordou com rotulagem
    manual em 7 de 11 casos (64%), errando na direcao de aprovar trecho que so trata do
    assunto. Cada veredito vai para o trace, para poder ser medido depois.
    """
    t0 = time.time()
    from wiseoak.ferramentas import biblioteca as B

    ix_livro = indice_de({**estado, "indice": cfg(estado, "indice")})
    ix_normas = indice_de({**estado, "indice": cfg(estado, "indice_normas")})
    k = cfg(estado, "k_contexto")
    max_rodadas = int(cfg(estado, "max_rodadas"))

    msgs = [{"role": "system", "content": _sistema_navegar(ix_livro, ix_normas)},
            {"role": "user", "content": estado["pergunta"][:2500]}]
    contexto: list[dict] = []
    passos: list[str] = []
    vereditos: list[str] = []

    for _ in range(max_rodadas):
        r = clientes.chat(msgs, modelo=cfg(estado, "modelo"), think=False,
                          max_tokens=500, temp=cfg(estado, "temp"),
                          tools=B.ESPECIFICACOES)
        chamadas = r.get("tool_calls") or []
        if not chamadas:
            break
        msgs.append({"role": "assistant", "content": r.get("content") or "",
                     "tool_calls": chamadas})
        novos: list[dict] = []
        for c in chamadas[:2]:
            f = c.get("function") or {}
            nome = f.get("name")
            try:
                args = json.loads(f.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if nome == "ler":
                novos = B.ler(ix_livro, ix_normas, args.get("obra", ""),
                              args.get("referencia", ""),
                              args.get("assunto") or estado["pergunta"], k)
                passos.append(f"ler({args.get('obra')},{args.get('referencia')})"
                              f"->{len(novos)}")
            elif nome == "buscar":
                q = args.get("consulta") or estado["pergunta"]
                novos = [{**ix_livro.obter(cid), "natureza": "LIVRO"}
                         for cid, _ in ix_livro.buscar(q, k, hibrido=False)]
                passos.append(f"buscar({q[:44]})->{len(novos)}")
            else:
                passos.append(f"?{nome}")
                msgs.append({"role": "tool", "tool_call_id": c.get("id", ""),
                             "content": f"ferramenta desconhecida: {nome}"})
                continue
            contexto.extend(novos)
            texto = "\n\n".join(f"[{i}] {' '.join(x['texto'].split())[:900]}"
                                 for i, x in enumerate(novos, 1)) or "nada encontrado"
            # veredito de suficiencia sobre TUDO que ja se tem, nao so o ultimo lote
            crit = no_criticar({**estado, "contexto": contexto})
            basta = crit["contexto_basta"]
            vereditos.append("suficiente" if basta else "insuficiente")
            msgs.append({"role": "tool", "tool_call_id": c.get("id", ""),
                         "content": texto[:6000] + "\n\n[veredito automatico: "
                         + ("SUFICIENTE — pode responder]" if basta
                            else "INSUFICIENTE — tente outro item do catalogo]")})
        if vereditos and vereditos[-1] == "suficiente":
            break

    vistos, final = set(), []
    for c in contexto:
        if c["id"] in vistos:
            continue
        vistos.add(c["id"])
        final.append(c)
    return {"contexto": final[:max(k, 8)],
            **_marcar("navegar", t0, rodadas=len(passos), passos=passos,
                      vereditos=vereditos, n=len(final),
                      parou_por=("suficiente" if vereditos[-1:] == ["suficiente"]
                                 else "teto" if len(passos) >= max_rodadas
                                 else "modelo parou"),
                      decisao="o modelo escolheu o que ler")}


def no_responder(estado: Estado) -> dict:
    t0 = time.time()
    modo: Modo = cfg(estado, "modo")
    contexto = estado.get("contexto") or []

    ancoragem: Ancoragem = cfg(estado, "ancoragem")
    sistema = SISTEMA[ancoragem]
    if not contexto:  # v0: sem RAG
        sistema = ("Voce e um assistente de anestesiologia. Responda com base no seu "
                   "conhecimento. Deixe 'citacoes' como lista vazia.")

    instrucao = (INSTRUCAO_ANALISTA[modo]
                 if ancoragem in ("analista", "analista_leve", "bib_so")
                 else INSTRUCAO[modo])
    partes = [instrucao]
    if cfg(estado, "raciocinio") == "prompt":
        # MedGemma nao tem canal de thinking; este e o substituto, e e uma DIMENSAO
        # medida, nao um default.
        partes.append("Antes de decidir, raciocine passo a passo internamente sobre o "
                      "contexto. Devolva apenas o JSON final.")
    if contexto:
        naturezas = {c.get("natureza") for c in contexto if c.get("natureza")}
        if len(naturezas) > 1:
            # curta de proposito: esta bancada mediu que envelope generico de prompt
            # derrubou o RAG de 93,3% para 80%. Duas frases factuais, sem instrucao
            # de postura.
            partes.append(
                "O contexto traz duas naturezas de fonte. [NORMA] é texto legal ou "
                "regulamentar brasileiro e define o que é obrigatório, permitido ou "
                "vedado. [LIVRO] é livro-texto e descreve prática clínica e evidência. "
                "Para o que a lei ou a resolução exige, vale a NORMA; para "
                "fisiopatologia, técnica e dados, vale o LIVRO.")
        partes.append("CONTEXTO:\n" + formatar_contexto(contexto))
    if estado.get("formulario"):
        # depois do contexto e antes da pergunta: e material de apoio, nao fonte a citar
        partes.append("FORMULÁRIO (referência, não é fonte para citação):\n"
                      + estado["formulario"])
    partes.append("PERGUNTA:\n" + estado["pergunta"])

    r = clientes.chat(
        [{"role": "system", "content": sistema},
         {"role": "user", "content": "\n\n".join(partes)}],
        modelo=cfg(estado, "modelo"),
        think=True if cfg(estado, "raciocinio") == "nativo" else False,
        max_tokens=4096, temp=cfg(estado, "temp"),
        schema=schema_para(modo, ancoragem))

    saida: dict = {"truncou": r["truncou"]}
    try:
        d = json.loads(r["content"])
    except json.JSONDecodeError:
        d = {}
    saida["citacoes"] = d.get("citacoes") or []
    saida["ressalva"] = (d.get("ressalva") or d.get("erro_apontado") or "").strip()
    saida["erro_apontado"] = (d.get("erro_apontado") or "").strip()
    if modo == "mcq":
        saida["letra"] = (d.get("letra") or "").strip().upper()[:1]
        saida["resposta"] = saida["letra"]
    else:
        saida["resposta"] = (d.get("resposta") or "").strip()
    saida.update(_marcar("responder", t0, tokens=r["out_tokens"], truncou=r["truncou"],
            modelo=cfg(estado, "modelo"), ancoragem=ancoragem,
            raciocinio=cfg(estado, "raciocinio"),
            thinking_nativo=(cfg(estado, "raciocinio") == "nativo"),
            reasoning_chars=len(r["reasoning"]),
            contexto_chars=sum(len(c["texto"]) for c in contexto),
            preprompt=sistema, instrucao=instrucao,
            schema=sorted(schema_para(modo, ancoragem)["properties"])))
    saida["preprompt"] = sistema
    return saida


def no_portao(estado: Estado) -> dict:
    """
    Portao de relevancia: se o contexto NAO trata do assunto, DESCARTA o contexto.

    Motivado por auditoria item a item. Numa assertiva sobre a Lei 9.434/97 (doacao de
    orgaos), a busca trouxe "cap36 REFERENCIAS" e "cap13 TESTES" — o livro e um texto
    americano traduzido e nao cobre legislacao brasileira. Sem RAG o modelo acertou,
    citando a lei de cabeca; COM aquele contexto irrelevante ele concluiu "o contexto
    nao detalha as regras juridicas" e respondeu Falso.

    O `criticar` do v4 rebusca quando o contexto nao basta. Isso nao resolve este caso:
    rebuscar num livro que nao tem o assunto traz mais irrelevancia. O portao faz o
    contrario — solta o contexto e deixa o modelo responder do que sabe, que e
    exatamente o comportamento do v0, que acertou.
    """
    t0 = time.time()
    contexto = estado.get("contexto") or []
    if not contexto:
        return _marcar("portao", t0, decisao="sem contexto para julgar")
    r = clientes.chat(
        [{"role": "system",
          "content": "Responda so TRATA ou NAO_TRATA. Julgue se os trechos abaixo tratam "
                     "do ASSUNTO da pergunta. Nao julgue se respondem — so se sao sobre "
                     "o mesmo tema."},
         {"role": "user", "content":
          "PERGUNTA: " + estado["pergunta"][:900] + "\n\nTRECHOS:\n"
          + formatar_contexto(contexto)[:6000]}],
        modelo=cfg(estado, "modelo"), think=False, max_tokens=12, temp=0.0)
    trata = "NAO" not in r["content"].upper()
    return {**({} if trata else {"contexto": [], "fontes": []}),
            **_marcar("portao", t0, trata=trata,
                      decisao="mantem o contexto" if trata else "DESCARTA o contexto")}


def no_criticar(estado: Estado) -> dict:
    """
    v4: o modelo julga se o contexto BASTA. Nao julga se a resposta esta certa — isso
    seria LLM-as-judge, que este projeto ja mediu como superestimado.
    """
    t0 = time.time()
    r = clientes.chat(
        [{"role": "system", "content": "Responda so SUFICIENTE ou INSUFICIENTE."},
         {"role": "user", "content":
          "O contexto abaixo permite responder a pergunta?\n\nPERGUNTA: "
          + estado["pergunta"] + "\n\nCONTEXTO:\n"
          + formatar_contexto(estado.get("contexto") or [])[:12000]}],
        modelo=cfg(estado, "modelo"), think=False, max_tokens=16, temp=0.0)
    basta = "INSUF" not in r["content"].upper()
    return {"contexto_basta": basta,
            **_marcar("criticar", t0, basta=basta, veredito=r["content"].strip()[:40])}


# ------------------------------------------------- verificacao de citacao

def _normalizar(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", s)


def verificar_citacoes(estado: Estado, *, min_palavras: int = 6) -> dict:
    """
    Programatica, nunca juiz. Uma citacao e FIEL quando:
      1. o trecho citado aparece literalmente (normalizado) em algum chunk que foi de
         fato recuperado — nao no livro inteiro, no CONTEXTO que o modelo viu; e
      2. a pagina citada cai na faixa de paginas desse chunk.

    Citacao curta demais casaria por acaso, por isso `min_palavras`.
    """
    contexto = estado.get("contexto") or []
    corpus = [(c, _normalizar(c["texto"])) for c in contexto]
    total = fieis = pagina_ok = 0
    for cit in estado.get("citacoes") or []:
        trecho = _normalizar(str(cit.get("trecho", "")))
        if len(trecho.split()) < min_palavras:
            total += 1
            continue
        total += 1
        alvo = " ".join(trecho.split())
        for c, texto in corpus:
            if alvo in " ".join(texto.split()):
                fieis += 1
                pi, pf = c.get("pagina_inicial"), c.get("pagina_final")
                if pi is not None and pi <= int(cit.get("pagina", -1)) <= (pf or pi):
                    pagina_ok += 1
                break
    return {"citacoes": total, "fieis": fieis, "pagina_ok": pagina_ok,
            "fidelidade": fieis / total if total else 0.0}
