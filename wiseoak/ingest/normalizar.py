"""
Correcao dirigida dos erros de OCR do Miller PT-BR.

Medido em 120 paginas (2026-08-25): o OCR le a ligadura `fl` como `tl`, e isso cai em
cheio nos halogenados — 24% das mencoes a isoflurano, 34% a desflurano e 38% a
sevoflurano estao corrompidas. O embedding denso tolera (subword); o BM25 nao, e o BM25
e justamente a metade da busca hibrida que carrega termo tecnico raro.

CUIDADO com a regra geral `tl -> fl`: ela esta ERRADA em portugues. "atlas", "atleta",
"atlantico", "atletico" sao palavras reais. O que nao existe em portugues e a sequencia
`tlu` — por isso a regra automatica se limita a ela, e o resto e lista curada.

Nada aqui e cosmetico: cada regra corrige um termo que a busca lexical precisa achar.
"""

from __future__ import annotations

import re
import unicodedata

# `tlu` nao ocorre em nenhuma palavra do portugues. Cobre fluxo, fluido, flutuar,
# influencia, isoflurano, desflurano, sevoflurano, flumazenil...
_TLU = re.compile(r"tlu", re.IGNORECASE)

# A mesma ligadura tambem e lida como `ft` e `ff`, e a ligadura `fi` como `ft`.
# Nenhuma das tres sequencias abaixo existe em portugues, entao a troca e segura:
#   ftu/ffu -> flu   (desfturano, sevofturano, metoxiffurano, ftuido)
#   ftc     -> fic   (insuftciencia)
_FTU = re.compile(r"f[tf]u", re.IGNORECASE)
_FTC = re.compile(r"ftc", re.IGNORECASE)

# `I` maiusculo lido como `l` minusculo no inicio da palavra. Portugues NAO admite
# palavra comecando com ln, ls ou lm — sao onsets impossiveis, logo todo caso e erro de
# OCR. Atinge termos centrais: lntubacao (32x), lsquemia (19x), lsoflurano (11x).
_L_POR_I = re.compile(r"\bl(?=[nsm][a-zà-ÿ])")

# Termos em que o OCR erra fora do padrao `tlu`. Chave em minusculas.
_CURADOS = {
    "protoxido": "protóxido",
    "antlogistico": "antiflogístico",
    "retlexo": "reflexo",
    "retlexos": "reflexos",
    "intlamacao": "inflamação",
    "intlamatorio": "inflamatório",
    "retluxo": "refluxo",
    "contlito": "conflito",
    "tlebite": "flebite",
    "tlexao": "flexão",
    "tlexibilidade": "flexibilidade",
    "anafilatico": "anafilático",
    # cauda longa medida no corpo do livro: `tl` por `fl` fora de `tlu` (nao se pode
    # automatizar, porque `atleta`/`atlantico` sao reais), `i` acentuado lido como `r`
    # ou `rn`, e `o` acentuado lido como `6`.
    "retlete": "reflete",
    "retletem": "refletem",
    "retletir": "refletir",
    "retletida": "refletida",
    "cardraco": "cardíaco",
    "cardracos": "cardíacos",
    "sangurneo": "sanguíneo",
    "sangurnea": "sanguínea",
    "inalat6rio": "inalatório",
    "inalat6rios": "inalatórios",
    "inalatorlos": "inalatórios",
}

# Marca d'agua do scan, em toda pagina. Sem remover, vira ruido em todo chunk e o BM25
# passa a casar a marca d'agua em vez do conteudo.
_LIXO = [
    re.compile(r"^\s*APOSTILASMEDICINA@HOTMAIL\.COM\s*$", re.I | re.M),
    re.compile(r"^\s*Produtos:\s*https?://\S+\s*$", re.I | re.M),
    re.compile(r"^\s*lista\.mercadolivre\.com\.br\S*\s*$", re.I | re.M),
]

# Palavra quebrada no fim da linha com hifen: "metoxiflu-\nrano" -> "metoxiflurano".
_HIFEN_QUEBRA = re.compile(r"(\w)[-\u00ad]\s*\n\s*(\w)")

# Mesmo defeito quando a quebra ja virou espaco: "recu- peracao". Em portugues o
# hifen de composto NUNCA e seguido de espaco ("pre-operatorio"), e o travessao vem
# com espaco dos DOIS lados — por isso exigir letra colada antes do hifen e seguro.
_HIFEN_ESPACO = re.compile(r"(\w)- (\w)")

# Palavra partida por espaco no meio, sem hifen: "metoxiflu rano". So se junta quando o
# resultado existe no lexico abaixo — juntar por heuristica quebraria texto legitimo.
_PARTIDAS = [
    (re.compile(r"\bmetoxiflu\s+rano\b", re.I), "metoxiflurano"),
    (re.compile(r"\bisoflu\s+rano\b", re.I), "isoflurano"),
    (re.compile(r"\bdesflu\s+rano\b", re.I), "desflurano"),
    (re.compile(r"\bsevoflu\s+rano\b", re.I), "sevoflurano"),
    (re.compile(r"\bhalo\s+tano\b", re.I), "halotano"),
    (re.compile(r"\bpropo\s+fol\b", re.I), "propofol"),
]

_ESPACOS = re.compile(r"[ \t ]{2,}")
_LINHAS_VAZIAS = re.compile(r"\n{3,}")


def _preserva_caixa(original: str, corrigido: str) -> str:
    if original.isupper():
        return corrigido.upper()
    if original[:1].isupper():
        return corrigido[:1].upper() + corrigido[1:]
    return corrigido


def corrigir_ligadura(texto: str) -> str:
    """Trocas automaticas de sequencia impossivel em portugues, preservando a caixa."""
    texto = _TLU.sub(lambda m: _preserva_caixa(m.group(0), "flu"), texto)
    texto = _FTU.sub(lambda m: _preserva_caixa(m.group(0), "flu"), texto)
    texto = _FTC.sub(lambda m: _preserva_caixa(m.group(0), "fic"), texto)
    return _L_POR_I.sub("I", texto)


def corrigir_curados(texto: str) -> str:
    def troca(m: re.Match) -> str:
        p = m.group(0)
        base = unicodedata.normalize("NFKD", p.lower())
        base = "".join(c for c in base if not unicodedata.combining(c))
        alvo = _CURADOS.get(base)
        return _preserva_caixa(p, alvo) if alvo else p

    if not _CURADOS:
        return texto
    padrao = re.compile(r"\b(" + "|".join(re.escape(k) for k in _CURADOS) + r")\b", re.I)
    return padrao.sub(troca, texto)


def remover_lixo(texto: str) -> str:
    for p in _LIXO:
        texto = p.sub("", texto)
    return texto


def juntar_quebras(texto: str) -> str:
    texto = _HIFEN_QUEBRA.sub(r"\1\2", texto)
    texto = _HIFEN_ESPACO.sub(r"\1\2", texto)
    for padrao, alvo in _PARTIDAS:
        texto = padrao.sub(lambda m, a=alvo: _preserva_caixa(m.group(0), a), texto)
    return texto


def normalizar(texto: str, *, ocr: bool = True) -> str:
    """
    Pipeline completo. `ocr=False` desliga so as correcoes de OCR e mantem a limpeza
    estrutural — e assim que "com e sem correcao de OCR" vira uma dimensao medida em vez
    de um remendo silencioso.
    """
    texto = remover_lixo(texto)
    # ORDEM IMPORTA: as correcoes de LETRA vem antes da juncao de PALAVRA. 'lsoflu rano'
    # so casa a regra de juncao depois que o `l` inicial vira `I`; na ordem inversa a
    # palavra fica partida para sempre.
    if ocr:
        texto = corrigir_ligadura(texto)
        texto = corrigir_curados(texto)
    texto = juntar_quebras(texto)
    texto = _ESPACOS.sub(" ", texto)
    texto = _LINHAS_VAZIAS.sub("\n\n", texto)
    return texto.strip()
