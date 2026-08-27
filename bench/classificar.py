#!/usr/bin/env python3
"""
Anota as questoes em TOPICO, para estratificar a analise.

Motivo: a media sobre o banco inteiro esconde classes que se comportam de forma oposta.
Auditoria item a item mostrou uma assertiva sobre a Lei 9.434/97 em que a busca trouxe
"cap36 REFERENCIAS" e o modelo, vendo lixo, concluiu Falso — enquanto SEM RAG acertou
citando a lei. Um livro-texto americano traduzido nao cobre legislacao brasileira, e
nenhum ajuste de prompt conserta isso. Medir junto com farmacologia mistura populacoes.

DUAS CAMADAS, e a ordem importa:

  regra   so dispara no que e LITERAL e inequivoco — citacao de lei/resolucao, mencao a
          figura com legenda, numero colado a unidade de medida. Alta precisao, baixa
          cobertura, de proposito.
  llm     o resto, com `json_schema` e rotulo fechado. Classificar NAO e julgar: a regra
          da pasta proibe juiz de LLM para VERIFICAR, e isto e anotacao. O schema existe
          porque este projeto mediu conformidade de contrato indo de 0/15 para 15/15.

A primeira versao usava so regra e errou visivelmente: 'ocorre pouca alteracao da
capacidade pulmonar total' virou "gestao" (casou "alteracao"/"capacidade" com o padrao
de *utilizacao*) e 'hiponatremia' virou "farmacologia". Regra gulosa e pior que nenhuma,
porque estratificar por rotulo errado engana com aparencia de rigor.

Cada item guarda `classe` e `classe_origem`, para dar para auditar quem rotulou o que.

    ./classificar.py --so-regra          # de graca, sem GPU
    ./classificar.py                     # regra + llm no resto
    ./classificar.py --conferir 20       # amostra estratificada para conferir a mao
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
ANALISES = RAIZ / "eval" / "analises"
ANALISES.mkdir(parents=True, exist_ok=True)

from wiseoak import clientes  # noqa: E402

TOPICOS = {
    "juridico-normativo": "lei, resolucao do CFM, codigo de etica, normas da SBA, "
                          "responsabilidade profissional, organizacao da sociedade",
    "imagem": "depende de ver uma figura, foto, traçado ou esquema",
    "valor-numerico": "a resposta depende de um numero exato: dose, escore, faixa de "
                      "referencia, coeficiente",
    "gestao": "administracao do centro cirurgico, custo, agendamento, produtividade, "
              "indicadores de qualidade",
    "farmacologia": "farmaco: mecanismo, farmacocinetica, interacao, efeito adverso",
    "tecnica": "como se faz um procedimento: bloqueio, via aerea, puncao, monitorizacao, "
               "equipamento",
    "fisiopatologia": "mecanismo de doenca, fisiologia, repercussao clinica, conduta",
}

# ---------------------------------------------------------------- camada 1: regra
#
# So o que e literal. Cada padrao precisa de uma ANCORA textual que nao existe fora da
# classe — nome proprio de norma, mencao a legenda, numero colado a unidade.

REGRAS: list[tuple[str, re.Pattern]] = [
    ("juridico-normativo", re.compile(
        r'(?i)\bCFM\b|\bCRM\b|conselho (federal|regional) de medicina'
        r'|resolu[çc][ãa]o\s+(n?[º°.]?\s*)?\d|\blei\s+n?[º°.]?\s*\d'
        r'|c[óo]digo de [ée]tica m[ée]dica|\bANVISA\b|portaria\s+n?[º°.]?\s*\d'
        r'|sociedade brasileira de anestesiologia|congresso brasileiro de anestesio'
        # a sigla e o que as questoes usam de fato ("a Diretoria da SBA"); por extenso
        # quase nao aparece. Em texto de anestesiologia em portugues, SBA e sempre a
        # sociedade — o risco de falso positivo e baixo.
        r'|\bSBA\b|\bAMB\b|\bCRM\b'
        r'|assembleia de representantes|diretoria|estatuto|regimento')),

    ("imagem", re.compile(
        r'(?i)com legenda|\blegenda\s*["“]|\bestrutura\s+\d\b|\bfigura\s+\d'
        r'|\b(nesta|nesse|neste|na)\s+(figura|imagem|tra[çc]ado)\b'
        r'|\bindicad[oa]\s+pela\s+seta\b')),

    ("valor-numerico", re.compile(
        r'\b\d+(?:[,.]\d+)?\s*(mg/kg|mcg/kg|mL/kg|mg|mL|mcg|µg|kg|mmHg|cmH2O|mEq/L|UI|%)\b'
        r'|\bvariam?\s+de\s+\d+\s+a\s+\d+\b', re.I)),
]


# Onde cada regra pode casar. A distincao nao e detalhe:
#
#   juridico e imagem  o ENUNCIADO define o enquadramento. Sob "Segundo a Resolucao CFM
#                      2.174/2017:", a assertiva E normativa mesmo sem citar a norma.
#   valor-numerico     so a ASSERTIVA vale. Casar no enunciado fez "idade avancada e
#                      sexo feminino sao fatores de risco" — que nao tem numero nenhum —
#                      herdar a classe de outra assertiva do mesmo bloco.
ESCOPO = {"juridico-normativo": "bloco", "imagem": "bloco", "valor-numerico": "assertiva"}


def classe_regra(item: dict) -> str | None:
    assertiva = item.get("assertiva") or ""
    bloco = f"{item.get('enunciado', '')} {assertiva} " \
            f"{' '.join((item.get('alternativas') or {}).values())}"
    for nome, padrao in REGRAS:
        alvo = assertiva if ESCOPO.get(nome) == "assertiva" and assertiva else bloco
        if padrao.search(alvo):
            return nome
    return None


# ------------------------------------------------------------------ camada 2: llm

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["topico"],
          "properties": {"topico": {"type": "string", "enum": list(TOPICOS)}}}

SISTEMA = ("Voce rotula questoes de prova de anestesiologia por TOPICO. Um rotulo por "
           "questao, o que melhor descreve o assunto principal. Nao julgue se a "
           "assertiva e verdadeira — so classifique o tema.\n\n"
           + "\n".join(f"- {k}: {v}" for k, v in TOPICOS.items()))


def classe_llm(item: dict, modelo: str) -> str:
    corpo = f"{item.get('enunciado', '')}\n\n{item.get('assertiva') or ''}"
    alts = item.get("alternativas") or {}
    if alts:
        corpo += "\n" + "\n".join(f"{k}) {v}" for k, v in sorted(alts.items()))
    try:
        r = clientes.chat([{"role": "system", "content": SISTEMA},
                           {"role": "user", "content": corpo[:2500]}],
                          modelo=modelo, think=False, max_tokens=48, temp=0.0,
                          schema=SCHEMA)
        t = json.loads(r["content"]).get("topico")
        return t if t in TOPICOS else "fisiopatologia"
    except Exception:
        return "fisiopatologia"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--so-regra", action="store_true")
    ap.add_argument("--modelo", default="gemma4-plan")
    ap.add_argument("--conferir", type=int, default=0,
                    help="grava amostra estratificada para conferencia a mao")
    a = ap.parse_args()

    caminho = RAIZ / "dados" / "questoes_sba.jsonl"
    itens = [json.loads(l) for l in caminho.read_text().splitlines() if l.strip()]

    por_regra = 0
    pendentes = []
    for it in itens:
        c = classe_regra(it)
        if c:
            it["classe"], it["classe_origem"] = c, "regra"
            por_regra += 1
        else:
            pendentes.append(it)
    print(f"regra rotulou {por_regra} de {len(itens)} "
          f"({100 * por_regra / len(itens):.0f}%); restam {len(pendentes)}",
          file=sys.stderr)

    if a.so_regra:
        for it in pendentes:
            it["classe"], it["classe_origem"] = "?", "pendente"
    else:
        for n, it in enumerate(pendentes, 1):
            it["classe"], it["classe_origem"] = classe_llm(it, a.modelo), "llm"
            if n % 25 == 0 or n == len(pendentes):
                sys.stderr.write(f"\r  llm {n}/{len(pendentes)}")
        sys.stderr.write("\n")

    caminho.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in itens) + "\n")

    # propaga para os splits, casando por id (o id e estavel desde a correcao)
    classe = {i["id"]: (i["classe"], i["classe_origem"]) for i in itens}
    for sufixo in ("dev", "teste"):
        p = RAIZ / "dados" / f"questoes_sba.{sufixo}.jsonl"
        if not p.exists():
            continue
        linhas = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        for i in linhas:
            i["classe"], i["classe_origem"] = classe.get(i["id"], ("?", "ausente"))
        p.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in linhas) + "\n")

    c = collections.Counter(i["classe"] for i in itens)
    o = collections.Counter(i["classe_origem"] for i in itens)
    print(f"\n{'classe':22s} {'itens':>6s} {'%':>6s}")
    for nome in TOPICOS:
        print(f"{nome:22s} {c.get(nome, 0):6d} {100 * c.get(nome, 0) / len(itens):5.1f}%")
    print(f"\norigem: {dict(o)}", file=sys.stderr)

    if a.conferir:
        rnd = random.Random(11)
        L = ["# Conferência da anotação", "",
             "Marque ERRADO e escreva a classe certa onde discordar. A concordância "
             "medida aqui é o erro conhecido da estratificação.", ""]
        for nome in TOPICOS:
            pool = [i for i in itens if i["classe"] == nome and i["tipo"] == "vf"]
            if not pool:
                continue
            L.append(f"## {nome} ({len(pool)} no banco)")
            for i in rnd.sample(pool, min(a.conferir, len(pool))):
                L.append(f"- [{i['classe_origem']}] {' '.join(i['assertiva'].split())[:150]}")
            L.append("")
        p = ANALISES / "conferencia-classes.md"
        p.write_text("\n".join(L))
        print(f"conferência em {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
