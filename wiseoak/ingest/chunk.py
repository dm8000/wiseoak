#!/usr/bin/env python3
"""
Chunking hierarquico pai/filho do livro ja estruturado por pagina.

A hierarquia deste corpus, de fora para dentro:

    secao (6)  ->  capitulo (47)  ->  titulo MAIUSCULO  ->  titulo Title-case  ->  prosa

Os dois niveis de subtitulo saem das LINHAS do corpo, nao do campo `titulos` de
estrutura.py — aquele campo detecta por altura de linha e, neste scan, pega legenda e
rotulo de eixo de figura, nao secao. A deteccao daqui e por forma da linha, e foi
conferida a mao nas paginas 95-130: devolve "EFEITO SEGUNDO GAS", "SOLUBILIDADE",
"Ventilacao alveolar", "Metabolismo".

`--sem-hierarquia` e o braco de controle do experimento: derruba o caminho de titulos e
faz o pai virar a pagina. Serve para medir quanto a hierarquia vale de fato, em vez de
supor que vale.

    python3 chunk.py dados/miller_paginas.jsonl --saida dados/miller_chunks.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

# Casa linha que PODE ser subtitulo: comeca com maiuscula e so tem letra, espaco e hifen.
_SUBTITULO = re.compile(r"^[A-ZÁÂÃÀÉÊÍÓÔÕÚÜÇ][A-Za-zÀ-ÿ \-]{2,45}$")

# ~3,2 chars por token em portugues; mesma constante de clientes.py, onde ela protege o
# ctx 2048 do embed-small.
CHARS_POR_TOKEN = 3.2

_FIM_FRASE = re.compile(r"[.!?](?=\s|$)")


def _e_subtitulo(linha: str, proxima: str | None) -> bool:
    """
    Tres condicoes, todas necessarias. A terceira e a que separa subtitulo de linha
    solta de figura: subtitulo de verdade e seguido de PROSA.
    """
    if not _SUBTITULO.match(linha) or len(linha.split()) > 6:
        return False
    return proxima is not None and len(proxima.split()) > 6


def _nivel(titulo: str) -> int:
    """MAIUSCULO e nivel 1; qualquer outra caixa e nivel 2."""
    letras = [c for c in titulo if c.isalpha()]
    return 1 if letras and all(c.isupper() for c in letras) else 2


def detectar_titulos(texto: str) -> list[tuple[int, str]]:
    """
    Devolve [(indice_da_linha, titulo)]. Linhas de subtitulo CONSECUTIVAS sao um titulo
    so: no original, 'POTENCIA RELATIVA DOS ANESTESICOS' e 'INALATORIOS' formam um.
    """
    linhas = texto.split("\n")
    achados: list[tuple[int, str]] = []
    i = 0
    while i < len(linhas):
        atual = linhas[i].strip()
        proxima = next((l.strip() for l in linhas[i + 1:] if l.strip()), None)
        if atual and _e_subtitulo(atual, proxima):
            partes, j = [atual], i + 1
            while j < len(linhas):
                seguinte = linhas[j].strip()
                depois = next((l.strip() for l in linhas[j + 1:] if l.strip()), None)
                if seguinte and _e_subtitulo(seguinte, depois):
                    partes.append(seguinte)
                    j += 1
                else:
                    break
            achados.append((i, " ".join(partes)))
            i = j
        else:
            i += 1
    return achados


def _ident(texto: str, caminho: list[str]) -> str:
    return hashlib.sha1(("|".join(caminho) + "\n" + texto).encode()).hexdigest()[:12]


def fundir_pequenos(pais: list[dict], minimo: int) -> list[dict]:
    """
    Funde no anterior o pai curto demais para ser secao.

    Medido neste corpus: 32% dos pais ficavam abaixo de 200 chars, e a inspecao mostrou
    que sao LINHAS DE TABELA lidas como titulo (uma tabela de agentes inalatorios vira
    'Sevoflurano', 'Desflurano', 'Data de introducao'...). Fragmentar assim encurta o
    filho e espalha o contexto.

    O titulo do pai fundido NAO se perde: entra como texto no comeco do trecho, para a
    busca lexical continuar achando por ele.
    """
    saida: list[dict] = []
    for pai in pais:
        curto = len(pai["texto"].strip()) < minimo
        cabe = (saida and curto
                and saida[-1]["capitulo_num"] == pai["capitulo_num"])
        if cabe:
            titulo = pai["caminho"][-1] if pai["caminho"] else ""
            saida[-1]["texto"] += "\n" + (titulo + " " if titulo else "") + pai["texto"]
            saida[-1]["pagina_final"] = max(saida[-1]["pagina_final"], pai["pagina_final"])
        else:
            saida.append(pai)
    return saida


def montar_pais(paginas: list[dict], *, hierarquia: bool = True) -> list[dict]:
    """
    Um pai por trecho entre subtitulos. Fecha tambem na troca de capitulo — um pai que
    atravessa capitulo carrega contexto errado para o filho, e o filho e o que vai para
    o modelo.
    """
    pais: list[dict] = []
    atual: dict | None = None
    n1: str | None = None
    n2: str | None = None

    def fechar():
        nonlocal atual
        if atual and atual["texto"].strip():
            pais.append(atual)
        atual = None

    def abrir(pg: dict, caminho: list[str]):
        nonlocal atual
        atual = {
            # o livro vem da PAGINA, nao fixo no codigo: o mesmo chunker atende o
            # Miller e o Barash, e a citacao precisa dizer de qual dos dois veio
            "livro": pg.get("livro", "miller-bases-6ed"),
            "secao_num": pg["secao_num"], "secao": pg["secao"],
            "capitulo_num": pg["capitulo_num"], "capitulo": pg["capitulo"],
            "caminho": list(caminho),
            "pagina_inicial": pg["pagina"], "pagina_final": pg["pagina"],
            "texto": "",
        }

    cap_anterior = object()
    livro_anterior = object()
    for pg in paginas:
        # Fechar tambem na troca de LIVRO. Nao e redundante com a troca de capitulo: o
        # corpus normativo tem capitulo_num=None em todos os registros de todos os
        # documentos, entao a condicao de capitulo nunca disparava e um pai atravessava
        # a fronteira entre documentos — texto do Estatuto da SBA foi parar dentro de um
        # chunk rotulado "CFM 2174/2017, ANEXO IX", e a citacao nomearia o documento
        # errado. Num corpus cuja razao de existir e citar resolucao e artigo, isso e
        # fatal. 17 dos 61 documentos sumiam absorvidos pelo vizinho.
        if pg.get("livro") != livro_anterior:
            fechar()
            n1 = n2 = None
            cap_anterior = object()
            livro_anterior = pg.get("livro")
        if pg["capitulo_num"] != cap_anterior:
            fechar()
            n1 = n2 = None
            cap_anterior = pg["capitulo_num"]

        if not hierarquia:
            fechar()
            abrir(pg, [])
            atual["texto"] = pg["texto"]
            continue

        linhas = pg["texto"].split("\n")
        cortes = dict(detectar_titulos(pg["texto"]))
        if atual is None:
            abrir(pg, [t for t in (n1, n2) if t])

        for idx, linha in enumerate(linhas):
            if idx in cortes:
                fechar()
                titulo = cortes[idx]
                if _nivel(titulo) == 1:
                    n1, n2 = titulo, None
                else:
                    n2 = titulo
                abrir(pg, [t for t in (n1, n2) if t])
                continue
            if atual is not None:
                atual["texto"] += linha + "\n"
                atual["pagina_final"] = pg["pagina"]

    fechar()
    return pais


def janelar(texto: str, alvo_chars: int, overlap: float) -> list[str]:
    """
    Corta em janelas por palavra, terminando em fim de frase quando ha um perto do fim.
    O overlap repete a cauda da janela anterior: sem ele, o fato que cai exatamente na
    emenda fica irrecuperavel pelas duas janelas.
    """
    palavras = texto.split()
    if not palavras:
        return []
    janelas: list[str] = []
    i = 0
    while i < len(palavras):
        pedaco: list[str] = []
        tam = 0
        j = i
        while j < len(palavras) and tam < alvo_chars:
            pedaco.append(palavras[j])
            tam += len(palavras[j]) + 1
            j += 1
        trecho = " ".join(pedaco)
        # prefere cortar em fim de frase, se houver um no ultimo quarto da janela
        if j < len(palavras):
            limite = int(len(trecho) * 0.75)
            fins = [m.end() for m in _FIM_FRASE.finditer(trecho) if m.end() >= limite]
            if fins:
                trecho = trecho[:fins[-1]]
                j = i + len(trecho.split())
        janelas.append(trecho.strip())
        if j >= len(palavras):
            break
        recuo = int(len(trecho.split()) * overlap)
        i = max(j - recuo, i + 1)
    return [x for x in janelas if x]


def chunkar(paginas: list[dict], *, tamanho_filho: int, overlap: float,
            hierarquia: bool, min_pai: int = 200) -> list[dict]:
    alvo = int(tamanho_filho * CHARS_POR_TOKEN)
    saida: list[dict] = []
    pais = montar_pais(paginas, hierarquia=hierarquia)
    if hierarquia and min_pai > 0:
        pais = fundir_pequenos(pais, min_pai)
    for pai in pais:
        texto_pai = " ".join(pai["texto"].split())
        pai_id = _ident(texto_pai, pai["caminho"])
        saida.append({**pai, "id": pai_id, "nivel": "pai", "pai_id": None,
                      "ordem": 0, "texto": texto_pai})
        for k, trecho in enumerate(janelar(texto_pai, alvo, overlap)):
            saida.append({**pai, "id": _ident(trecho, pai["caminho"] + [str(k)]),
                          "nivel": "filho", "pai_id": pai_id, "ordem": k,
                          "texto": trecho})
    return saida


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada", type=Path)
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--pagina-inicial", type=int, default=None)
    ap.add_argument("--pagina-final", type=int, default=None)
    ap.add_argument("--tamanho-filho", type=int, default=512, help="em tokens")
    ap.add_argument("--overlap", type=float, default=0.15)
    ap.add_argument("--min-pai", type=int, default=200,
                    help="funde no anterior o pai com menos chars que isto (0 desliga)")
    ap.add_argument("--sem-hierarquia", action="store_true",
                    help="braco de controle: sem caminho de titulos, pai = pagina")
    a = ap.parse_args()

    if not a.entrada.exists():
        print(f"nao existe: {a.entrada}", file=sys.stderr)
        return 1

    paginas = [json.loads(l) for l in a.entrada.read_text().splitlines() if l.strip()]
    if a.pagina_inicial is not None:
        paginas = [p for p in paginas if p["pagina"] >= a.pagina_inicial]
    if a.pagina_final is not None:
        paginas = [p for p in paginas if p["pagina"] <= a.pagina_final]
    if not paginas:
        print("nenhuma pagina na faixa pedida", file=sys.stderr)
        return 1

    chunks = chunkar(paginas, tamanho_filho=a.tamanho_filho, overlap=a.overlap,
                     hierarquia=not a.sem_hierarquia, min_pai=a.min_pai)
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    with a.saida.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    filhos = [c for c in chunks if c["nivel"] == "filho"]
    med = statistics.median([len(c["texto"]) for c in filhos]) if filhos else 0
    print(f"paginas {len(paginas)} | pais {len(chunks) - len(filhos)} | "
          f"filhos {len(filhos)} | mediana {med:.0f} chars/filho", file=sys.stderr)

    # Documento que entra tem de sair. Um livro que some foi absorvido pelo vizinho, e
    # ai o texto dele passa a ser citado com o nome do outro — falha silenciosa que ja
    # aconteceu: o Estatuto da SBA acabou dentro de "CFM 2174/2017, ANEXO IX".
    ent = {pg.get("livro") for pg in paginas}
    sai = {c.get("livro") for c in chunks}
    if ent - sai:
        print(f"  ALERTA: {len(ent - sai)} documentos entraram e nao saem: "
              f"{sorted(x for x in (ent - sai) if x)[:5]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
