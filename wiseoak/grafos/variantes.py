"""
Os cinco grafos do experimento.

    v0  MedGemma sozinho, sem RAG            baseline OBRIGATORIO
    v1  denso puro, top-k, sem rerank        RAG ingenuo
    v2  hibrido + rerank + expansao para pai uma passada
    v3  reformula -> v2                      a proposta do usuario
    v4  v3 + critica de suficiencia          rebusca quando o contexto nao basta

Sem v0 o experimento nao diz nada: uma prova de multipla escolha mede modelo E RAG
juntos, e o MedGemma acerta muita coisa de cabeca. O que interessa e a DIFERENCA.

v1 e v2 diferem em tres coisas de uma vez (hibrido, rerank, pai). Isso e proposital
nesta primeira rodada: primeiro se estabelece que existe efeito, depois se desmonta o
efeito em fatores. Medir 8 combinacoes antes de saber se ha sinal gasta GPU a toa.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import comum
from .comum import Estado


def _compilar(construir) -> "object":
    g = StateGraph(Estado)
    construir(g)
    return g.compile()


def v0_sem_rag():
    def montar(g: StateGraph):
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "responder")
        g.add_edge("responder", END)
    return _compilar(montar)


def v1_denso():
    def montar(g: StateGraph):
        def recuperar(estado: Estado) -> dict:
            return comum.no_recuperar({**estado, "hibrido": False})

        def contexto(estado: Estado) -> dict:
            return comum.no_montar_contexto({**estado, "expandir_pai": False})

        g.add_node("recuperar", recuperar)
        g.add_node("contexto", contexto)
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "recuperar")
        g.add_edge("recuperar", "contexto")
        g.add_edge("contexto", "responder")
        g.add_edge("responder", END)
    return _compilar(montar)


def v2_hibrido():
    def montar(g: StateGraph):
        g.add_node("recuperar", comum.no_recuperar)
        g.add_node("rerankear", comum.no_rerankear)
        g.add_node("contexto", comum.no_montar_contexto)
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "recuperar")
        g.add_edge("recuperar", "rerankear")
        g.add_edge("rerankear", "contexto")
        g.add_edge("contexto", "responder")
        g.add_edge("responder", END)
    return _compilar(montar)


def v3_reformula():
    def montar(g: StateGraph):
        g.add_node("reformular", comum.no_reformular)
        g.add_node("recuperar", comum.no_recuperar)
        g.add_node("rerankear", comum.no_rerankear)
        g.add_node("contexto", comum.no_montar_contexto)
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "reformular")
        g.add_edge("reformular", "recuperar")
        g.add_edge("recuperar", "rerankear")
        g.add_edge("rerankear", "contexto")
        g.add_edge("contexto", "responder")
        g.add_edge("responder", END)
    return _compilar(montar)


def v4_autocritica():
    """
    Uma unica rebusca, com a pergunta ORIGINAL e k maior. Sem o teto de uma passada o
    grafo entra em laco quando o livro simplesmente nao tem a resposta — e um livro de
    ~800 paginas nao tem resposta para tudo.
    """
    def montar(g: StateGraph):
        def rebuscar(estado: Estado) -> dict:
            novo = {**estado, "consulta": estado["pergunta"],
                    "k_busca": estado.get("k_busca", 20) * 2}
            saida = comum.no_recuperar(novo)
            return {**saida, "rebuscou": True}

        def decidir(estado: Estado) -> str:
            if estado.get("contexto_basta", True) or estado.get("rebuscou"):
                return "responder"
            return "rebuscar"

        g.add_node("reformular", comum.no_reformular)
        g.add_node("recuperar", comum.no_recuperar)
        g.add_node("rerankear", comum.no_rerankear)
        g.add_node("contexto", comum.no_montar_contexto)
        g.add_node("criticar", comum.no_criticar)
        g.add_node("rebuscar", rebuscar)
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "reformular")
        g.add_edge("reformular", "recuperar")
        g.add_edge("recuperar", "rerankear")
        g.add_edge("rerankear", "contexto")
        g.add_edge("contexto", "criticar")
        g.add_conditional_edges("criticar", decidir,
                                {"rebuscar": "rebuscar", "responder": "responder"})
        g.add_edge("rebuscar", "rerankear")
        g.add_edge("responder", END)
    return _compilar(montar)


# ---------------------------------------------------- grafo a partir de sketch

# Uma operacao da sketch -> o no que a executa. E o que permite ao FlowBot otimizar
# grafos LangGraph DE VERDADE, em vez de uma simulacao em laco Python: o vencedor da
# busca e um StateGraph compilado, igual aos v0-v4, e entra no benchmark e no Pipe sem
# traducao.
DESCRICAO = {
    "rotear":            "Escolhe o corpus pela classe: norma vai ao CFM/SBA, clínica vai ao Miller.",
    "reformular":        "Reescreve a pergunta como consulta de busca (1 chamada de LLM).",
    "traduzir":          "Reescreve a pergunta como consulta em INGLÊS, para corpus em inglês.",
    "recuperar_hibrido": "Busca no livro: BM25 + vetores, fundidos por posição (RRF).",
    "recuperar_denso":   "Busca no livro só por similaridade de vetor.",
    "rerankear":         "Reordena os trechos no Qwen3-Reranker, em CPU (~1,2 s por trecho).",
    "expandir_pai":      "Troca o trecho curto pela seção inteira que o contém.",
    "contexto_raso":     "Usa o trecho recuperado como está, sem expandir.",
    "criticar":          "Julga se os trechos bastam; não julga se a resposta está certa.",
    "portao":            "Se os trechos não tratam do assunto, DESCARTA o contexto e responde de conhecimento próprio.",
    "formula":           "Oferece UMA ferramenta ao modelo: consultar o formulário. Sem calculadora — os erros medidos não eram de aritmética.",
    "responder":         "Produz resposta, citações e ressalva, com schema JSON.",
}

OPERACOES = {
    "rotear":            lambda e: comum.no_rotear(e),
    "reformular":        lambda e: comum.no_reformular(e),
    "traduzir":          lambda e: comum.no_reformular({**e, "idioma_busca": "en"}),
    "recuperar_hibrido": lambda e: comum.no_recuperar({**e, "hibrido": True}),
    "recuperar_denso":   lambda e: comum.no_recuperar({**e, "hibrido": False}),
    "rerankear":         lambda e: comum.no_rerankear(e),
    "expandir_pai":      lambda e: comum.no_montar_contexto({**e, "expandir_pai": True}),
    "contexto_raso":     lambda e: comum.no_montar_contexto({**e, "expandir_pai": False}),
    "criticar":          lambda e: comum.no_criticar(e),
    "portao":            lambda e: comum.no_portao(e),
    "formula":           lambda e: comum.no_formula(e),
    "responder":         lambda e: comum.no_responder(e),
}


def de_sketch(sketch: list[str]):
    """
    Compila uma lista de operacoes num StateGraph linear.

    Garante o invariante que o `responder` depende: se houve busca e ninguem montou o
    contexto, monta antes de responder. Sem isso a sketch 'busca -> responde' entrega
    candidatos sem texto e o modelo responde no vazio.
    """
    passos = [p for p in sketch if p in OPERACOES]
    if "responder" not in passos:
        passos.append("responder")
    tem_busca = any(p.startswith("recuperar") for p in passos)
    tem_contexto = any(p in ("expandir_pai", "contexto_raso") for p in passos)
    if tem_busca and not tem_contexto:
        passos.insert(passos.index("responder"), "expandir_pai")

    g = StateGraph(Estado)
    anterior = START
    vistos: dict[str, int] = {}
    for p in passos:
        vistos[p] = vistos.get(p, 0) + 1
        nome = p if vistos[p] == 1 else f"{p}_{vistos[p]}"
        # o no leva a descricao como docstring: e o que o Studio mostra no painel
        fn = OPERACOES[p]
        fn.__doc__ = DESCRICAO.get(p, "")
        g.add_node(nome, fn)
        g.add_edge(anterior, nome)
        anterior = nome
    g.add_edge(anterior, END)
    return g.compile()


def v5_portao():
    """
    v2 mais um PORTAO DE RELEVANCIA antes de responder.

    Nasceu da auditoria: os dois modos de erro do v2 sao (a) contexto irrelevante que o
    modelo le como evidencia contra a assertiva, e (b) contexto relevante do qual ele
    infere demais. O portao ataca (a) — o unico dos dois em que existe uma decisao
    objetiva a tomar.
    """
    def montar(g: StateGraph):
        g.add_node("recuperar", comum.no_recuperar)
        g.add_node("rerankear", comum.no_rerankear)
        g.add_node("contexto", comum.no_montar_contexto)
        g.add_node("portao", comum.no_portao)
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "recuperar")
        g.add_edge("recuperar", "rerankear")
        g.add_edge("rerankear", "contexto")
        g.add_edge("contexto", "portao")
        g.add_edge("portao", "responder")
        g.add_edge("responder", END)
    return _compilar(montar)


def v6_denso_portao():
    """v1 (denso puro, sem rerank e sem pai) mais o portao. O mais barato com portao."""
    def montar(g: StateGraph):
        def recuperar(e):
            return comum.no_recuperar({**e, "hibrido": False})

        def contexto(e):
            return comum.no_montar_contexto({**e, "expandir_pai": False})

        g.add_node("recuperar", recuperar)
        g.add_node("contexto", contexto)
        g.add_node("portao", comum.no_portao)
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "recuperar")
        g.add_edge("recuperar", "contexto")
        g.add_edge("contexto", "portao")
        g.add_edge("portao", "responder")
        g.add_edge("responder", END)
    return _compilar(montar)


def v7_traduz():
    """
    v6 (denso + portao) com a consulta TRADUZIDA para ingles antes de buscar.

    Existe porque o corpus util esta em ingles (Barash, Miller 10e) e as perguntas em
    portugues. Medido no indice combinado: o livro em ingles era 52% dos trechos e
    chegava a 7% do contexto. Se traduzir a consulta resolver, o gargalo era idioma; se
    nao, o conteudo nao esta la.
    """
    def montar(g: StateGraph):
        def traduzir(e):
            return comum.no_reformular({**e, "idioma_busca": "en"})

        def recuperar(e):
            return comum.no_recuperar({**e, "hibrido": False})

        def contexto(e):
            return comum.no_montar_contexto({**e, "expandir_pai": False})

        g.add_node("traduzir", traduzir)
        g.add_node("recuperar", recuperar)
        g.add_node("contexto", contexto)
        g.add_node("portao", comum.no_portao)
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "traduzir")
        g.add_edge("traduzir", "recuperar")
        g.add_edge("recuperar", "contexto")
        g.add_edge("contexto", "portao")
        g.add_edge("portao", "responder")
        g.add_edge("responder", END)
    return _compilar(montar)


def v8_roteado():
    """
    v1 com ROTEAMENTO de corpus: norma vai ao indice de normas, clinica vai ao Miller.

    Nasceu da analise de onde o Opus 5 ganhou de nos: +15,4% da classe
    `juridico-normativo` e +9,7% de `gestao`. Nessas duas a fonte nao e livro-texto, e
    nenhuma melhoria de recuperacao sobre o Miller resolveria — o conteudo nao esta la.
    """
    def montar(g: StateGraph):
        def recuperar(e):
            return comum.no_recuperar({**e, "hibrido": False})

        def contexto(e):
            return comum.no_montar_contexto({**e, "expandir_pai": False})

        g.add_node("rotear", comum.no_rotear)
        g.add_node("recuperar", recuperar)
        g.add_node("contexto", contexto)
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "rotear")
        g.add_edge("rotear", "recuperar")
        g.add_edge("recuperar", "contexto")
        g.add_edge("contexto", "responder")
        g.add_edge("responder", END)
    return _compilar(montar)


def v9_formulario():
    """
    v8 mais o formulario: rotear -> recuperar -> contexto -> formula -> responder.

    O no `formula` vem DEPOIS do contexto e ANTES do responder porque e material de
    apoio, nao fonte a citar — a verificacao de citacao continua batendo so contra o
    trecho recuperado do livro.

    ALCANCE DECLARADO, para nao vender o que nao entrega: dos 161 itens de valor-numerico
    do dev, 16 trazem marca de calculo e 9 sao dose por peso (multiplicacao que nunca
    falhou). Sobram ~5 itens em 1.003 — 0,5%, contra IC de +/-2,5 pp. Este grafo existe
    pelo produto; o efeito dele NAO e detectavel nesta amostra, e medi-lo custaria 100 min
    para devolver ruido.
    """
    def montar(g: StateGraph):
        def recuperar(e):
            return comum.no_recuperar({**e, "hibrido": False})

        def contexto(e):
            return comum.no_montar_contexto({**e, "expandir_pai": False})

        g.add_node("rotear", comum.no_rotear)
        g.add_node("recuperar", recuperar)
        g.add_node("contexto", contexto)
        g.add_node("formula", comum.no_formula)
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "rotear")
        g.add_edge("rotear", "recuperar")
        g.add_edge("recuperar", "contexto")
        g.add_edge("contexto", "formula")
        g.add_edge("formula", "responder")
        g.add_edge("responder", END)
    return _compilar(montar)


def v10_cota():
    """
    WiseOak v3 — roteamento com COTA em vez de escolha exclusiva.

    rotear -> recuperar(cota) -> contexto(intercalado) -> responder

    Diferenca unica para o v8: a pergunta roteada as normas recebe metade das vagas do
    corpus normativo e metade do Miller, em vez de as quatro das normas. Pergunta clinica
    segue com o livro inteiro.

    Nasceu de uma medicao do v8: o roteamento exclusivo consertou `juridico-normativo`
    (66,2% -> 80,0%) e QUEBROU `gestao` (90,3% -> 77,4%), porque tirava o livro de
    perguntas cuja resposta estava nele. Com cota, errar o rotulo custa metade do
    contexto em vez de todo ele.
    """
    def montar(g: StateGraph):
        def contexto(e):
            # FIXO em False: e o comportamento medido como WiseOak v3. `estado_inicial`
            # traz expandir_pai=True por padrao, e o no de cota passou a honrar a flag —
            # sem fixar aqui, o v10 deixaria de reproduzir o proprio resultado.
            return comum.no_contexto_cota({**e, "expandir_pai": False})

        g.add_node("rotear", comum.no_rotear)
        g.add_node("recuperar", comum.no_recuperar_cota)
        g.add_node("contexto", contexto)
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "rotear")
        g.add_edge("rotear", "recuperar")
        g.add_edge("recuperar", "contexto")
        g.add_edge("contexto", "responder")
        g.add_edge("responder", END)
    return _compilar(montar)


def v11_cota_rerank():
    """
    v3 (cota) mais REORDENACAO: recupera um pool maior e o rerankeador escolhe as vagas.

    O v10 preenche as vagas por similaridade de cosseno, que ordena por PARECENCA de
    topico. Medido no dossie: em 41% dos erros clinicos a resposta esta no top-50 e nao
    entra nas 4 vagas — o trecho certo e recuperado e perde a vaga para um mais parecido.
    O cross-encoder pontua pergunta-contra-trecho, que e o que separa "fala do assunto"
    de "responde a pergunta".

    Foi descartado antes medindo no `h512` (o livro refutado, n=300, amplitude menor que
    o ruido). Nunca foi testado no Miller completo, e a refutacao do BM25 ja falhou em
    transferir de um corpus para outro.

    O rerank roda em CPU e custa ~1,2 s por trecho: o pool e o gargalo, nao o k final.
    """
    def montar(g: StateGraph):
        def recuperar(e):
            # pool maior SO aqui: e o rerankeador que corta para k_contexto
            return comum.no_recuperar_cota({**e, "k_busca": max(20, comum.cfg(e, "k_busca"))})

        def rerankear(e):
            # os candidatos vem marcados "fonte|id"; o rerank precisa do id puro
            marcados = e["candidatos"]
            de = {m.split("|", 1)[1]: m.split("|", 1)[0] for m in marcados}
            puros = [m.split("|", 1)[1] for m in marcados]
            r = comum.no_rerankear({**e, "candidatos": puros})
            r["candidatos"] = [f"{de[c]}|{c}" for c in r["candidatos"]]
            return r

        g.add_node("rotear", comum.no_rotear)
        g.add_node("recuperar", recuperar)
        g.add_node("rerankear", rerankear)
        g.add_node("contexto",
                   lambda e: comum.no_contexto_cota({**e, "expandir_pai": False}))
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "rotear")
        g.add_edge("rotear", "recuperar")
        g.add_edge("recuperar", "rerankear")
        g.add_edge("rerankear", "contexto")
        g.add_edge("contexto", "responder")
        g.add_edge("responder", END)
    return _compilar(montar)


def v12_cota_pai():
    """
    v3 (cota) trocando o trecho curto pela SECAO-PAI que o contem.

    Mesma medicao que motiva o rerank, atacada por outro lado: em 41% dos erros clinicos
    a resposta esta no top-50 e nao entra nas vagas, e nos casos inspecionados a busca
    acerta o topico e erra a frase — o fato especifico esta a um paragrafo do trecho
    recuperado. Trazer a secao inteira traz o vizinho junto.

    Barato: nenhuma chamada extra de modelo, so um SELECT do pai. O custo e prompt maior.
    """
    def montar(g: StateGraph):
        g.add_node("rotear", comum.no_rotear)
        g.add_node("recuperar", comum.no_recuperar_cota)
        g.add_node("contexto",
                   lambda e: comum.no_contexto_cota({**e, "expandir_pai": True}))
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "rotear")
        g.add_edge("rotear", "recuperar")
        g.add_edge("recuperar", "contexto")
        g.add_edge("contexto", "responder")
        g.add_edge("responder", END)
    return _compilar(montar)


def v13_focado():
    """
    WiseOak v5 — v3 (cota) mais consulta FOCADA e indice de trecho PEQUENO.

    rotear -> focar -> recuperar(cota) -> contexto -> responder

    As duas mudancas foram medidas a mao nas mesmas 12 questoes e SO FUNCIONAM JUNTAS:

        consulta          indice     recall@4 conferido a mao
        enunciado+asser   grande     3/12   (linha de base)
        enunciado+asser   pequeno    3/12
        so assertiva      grande     3/12
        so assertiva      PEQUENO    ~6/12

    Faz sentido mecanicamente: o trecho pequeno da granularidade para existir um vetor que
    represente o FATO em vez do topico do paragrafo; a consulta focada da um vetor de busca
    que discrimina entre questoes irmas. Granularidade fina com consulta generica continua
    caindo no cluster generico; consulta discriminativa contra paragrafo de 1.536 chars
    continua batendo no topico.

    O indice pequeno entra por `--indice dados/indice/m10p` (38.736 trechos, mediana 551
    chars, contra 14.628 e 1.536 do m10).
    """
    def montar(g: StateGraph):
        g.add_node("rotear", comum.no_rotear)
        g.add_node("focar", comum.no_focar)
        g.add_node("recuperar", comum.no_recuperar_cota)
        g.add_node("contexto",
                   lambda e: comum.no_contexto_cota({**e, "expandir_pai": False}))
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "rotear")
        g.add_edge("rotear", "focar")
        g.add_edge("focar", "recuperar")
        g.add_edge("recuperar", "contexto")
        g.add_edge("contexto", "responder")
        g.add_edge("responder", END)
    return _compilar(montar)


def v14_biblioteca():
    """
    WiseOak v6 — recuperacao NAVEGACIONAL: o modelo ve o catalogo e escolhe o que ler.

    navegar -> responder

    Nao ha no de busca: quem decide o que entra no contexto e o modelo, via ferramentas.
    A busca vetorial continua disponivel como UMA das tres, se ele preferir.

    Motivacao medida: a busca densa foi capturada pelo distrator numa questao de multipla
    escolha (quatro paragrafos sobre manitol porque "manitol" era uma das alternativas).
    Navegar por estrutura nao tem esse modo de falha.
    """
    def montar(g: StateGraph):
        g.add_node("navegar", comum.no_navegar)
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "navegar")
        g.add_edge("navegar", "responder")
        g.add_edge("responder", END)
    return _compilar(montar)


def v15_consulta_propria():
    """
    WiseOak v7 — o modelo escreve a propria consulta de busca, sem navegar.

    reformular(problema) -> recuperar(cota) -> contexto -> responder

    REFUTADO encadear com `no_focar`: focar tira o cenario e o reformulador INVENTA um
    errado. Em "a capacidade pulmonar total esta aumentada em 30%" (contexto: gravidez),
    a partir da assertiva sozinha ele escreveu "hiperinsuflacao pulmonar" e foi buscar
    hiperoxemia. A assertiva sozinha basta para casamento vetorial direto e NAO basta para
    quem precisa interpreta-la.

    Testa a hipotese barata que saiu do v6: no caso do cirrotico, o ganho da navegacao veio
    de o modelo ter REESCRITO a consulta descrevendo o problema (sem os distratores), nao
    de ele ter escolhido capitulo — ele nem chegou a usar `ler`. Se for isso mesmo, uma
    chamada curta entrega o mesmo que quatro rodadas de ferramenta, a um terco do custo.

    `idioma_busca="problema"` seleciona o prompt que PROIBE incluir as alternativas.
    """
    def montar(g: StateGraph):
        g.add_node("reformular",
                   lambda e: comum.no_reformular({**e, "idioma_busca": "problema"}))
        g.add_node("recuperar", comum.no_recuperar_cota)
        g.add_node("contexto",
                   lambda e: comum.no_contexto_cota({**e, "expandir_pai": False}))
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "reformular")
        g.add_edge("reformular", "recuperar")
        g.add_edge("recuperar", "contexto")
        g.add_edge("contexto", "responder")
        g.add_edge("responder", END)
    return _compilar(montar)


def v16_navegacional():
    """
    WiseOak v8 — o modelo navega a biblioteca, com verificacao de suficiencia.

    navegar (laco com teto) -> responder

    Nao ha no de busca fixo: quem decide o que entra no contexto e o modelo. Duas
    ferramentas — `ler` (parte especifica; se exceder o orcamento, busca dentro dela) e
    `buscar` (a densa de sempre) — e um veredito automatico de suficiencia depois de cada
    leitura, que o faz tentar outro item quando o material nao decide a questao.

    O catalogo vai no `system`, IDENTICO entre perguntas: medido, isso leva o custo de
    13,5 s para 1,3 s por chamada a partir da segunda, via cache de prefixo.
    """
    def montar(g: StateGraph):
        g.add_node("navegar", comum.no_navegar)
        g.add_node("responder", comum.no_responder)
        g.add_edge(START, "navegar")
        g.add_edge("navegar", "responder")
        g.add_edge("responder", END)
    return _compilar(montar)


GRAFOS = {
    "v0": v0_sem_rag,
    "v1": v1_denso,
    "v2": v2_hibrido,
    "v3": v3_reformula,
    "v4": v4_autocritica,
    "v5": v5_portao,
    "v6": v6_denso_portao,
    "v7": v7_traduz,
    "v8": v8_roteado,
    "v9": v9_formulario,
    "v10": v10_cota,
    "v11": v11_cota_rerank,
    "v12": v12_cota_pai,
    "v13": v13_focado,
    "v14": v14_biblioteca,
    "v15": v15_consulta_propria,
    "v16": v16_navegacional,
}


def grafo_flowbot():
    """
    O melhor workflow que o FlowBot encontrou, para auditoria no LangGraph Studio.

    Le a sketch de eval/analises/flowbot-execucao.json. Enquanto o FlowBot nao rodou,
    devolve o v2, que e o ponto de partida da busca — assim o Studio nunca quebra por
    falta de arquivo.
    """
    import json as _json
    from pathlib import Path as _Path
    alvo = _Path(__file__).resolve().parents[2] / "eval" / "analises" / "flowbot-execucao.json"
    try:
        sketch = _json.loads(alvo.read_text())["melhor"]["sketch"]
    except Exception:
        sketch = ["recuperar_hibrido", "rerankear", "expandir_pai", "responder"]
    return de_sketch(sketch)


def construir(nome: str):
    if nome not in GRAFOS:
        raise KeyError(f"grafo desconhecido: {nome}. Ha {sorted(GRAFOS)}")
    return GRAFOS[nome]()


def estado_inicial(pergunta: str, *, modo: comum.Modo = "livre",
                   modelo: str = "gemma4-plan",
                   raciocinio: comum.Raciocinio = "nenhum",
                   ancoragem: comum.Ancoragem = "estrita",
                   indice=None, k_busca: int = 20, k_contexto: int = 5,
                   temp: float | None = None) -> Estado:
    return {
        "pergunta": pergunta, "modo": modo, "modelo": modelo,
        "raciocinio": raciocinio, "ancoragem": ancoragem, "indice": indice,
        "k_busca": k_busca, "k_contexto": k_contexto,
        "hibrido": True, "expandir_pai": True, "trace": [],
        **({"temp": temp} if temp is not None else {}),
    }
