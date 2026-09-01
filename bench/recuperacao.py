#!/usr/bin/env python3
"""
Avalia a RECUPERAÇÃO isolada. Nenhuma geração, nenhum modelo de chat.

Por que existe: comparar um braço rodando o modelo custa ~1,7 h. Recuperar custa 0,06 s por
consulta. Medir a busca sozinha e ~1.000x mais barato e responde diretamente a pergunta que
importa — "o trecho certo esta chegando ao modelo?" — sem passar pelo ruido do gerador.

VERDADE DE REFERENCIA, e as suas limitacoes declaradas. Um chunk conta como portador da
resposta se satisfaz DUAS condicoes ao mesmo tempo:

  1. ancora numerica: um numero + unidade da assertiva aparece no chunk;
  2. cognato: pelo menos uma palavra de conteudo da assertiva aparece no chunk, casada por
     prefixo de 5 apos normalizar as diferencas ortograficas pt/en (ph->f, th->t, y->i...).

A condicao (1) sozinha JA MENTIU: numa questao sobre largura de manguito, "20%" casou com
"reducao de PAM em 20%"; numa sobre Child-Pugh, "0,3 mg/dL" casou com criterio de lesao
renal aguda. A condicao (2) existe para matar esse tipo de coincidencia.

O criterio erra para MENOS, nunca para mais: termo sem cognato entre os idiomas
(`manguito`/`cuff`) nao casa, entao o recall reportado e piso, nao teto. E por isso que
`bench/ouro.py` existe — a taxa de acerto do criterio contra rotulagem manual e reportada
junto de todo numero.

    ./recuperacao.py --braco atual --braco hibrido
    ./recuperacao.py --braco atual --dump-ouro 60      # extrai casos para rotular a mao
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
ANALISES = RAIZ / "eval" / "analises"

from wiseoak.store import Indice  # noqa: E402

# nome -> (indice, hibrido, rerank)
BRACOS: dict[str, tuple[str, bool, bool]] = {
    "atual":    ("dados/indice/m10", False, False),
    "hibrido":  ("dados/indice/m10", True,  False),
    "rerank":   ("dados/indice/m10", False, True),
    "pequeno":  ("dados/indice/m10p", False, False),   # so existe apos reindexar
    "pequeno-h":("dados/indice/m10p", True,  False),
}

_UNI_PT = r"(?:%|mg|mcg|µg|mL|L|cmH2O|mmHg|°C|graus|horas?|minutos?|dias?|semanas?)"
_ANCORA = re.compile(r"(?<![\w.,])(\d{1,4}(?:[.,]\d{1,2})?)\s*(" + _UNI_PT + r")", re.I)
_UNI_EN = {
    "%": r"\s*%", "mg": r"\s*mg", "mcg": r"\s*(?:mcg|µg)", "ml": r"\s*mL", "l": r"\s*L\b",
    "cmh2o": r"\s*cm\s?H2O", "mmhg": r"\s*mm\s?Hg", "graus": r"\s*°\s?C", "°c": r"\s*°\s?C",
    "horas": r"\s*(?:hours?|h\b)", "hora": r"\s*(?:hours?|h\b)",
    "minutos": r"\s*min", "minuto": r"\s*min", "dias": r"\s*days?", "dia": r"\s*days?",
    "semanas": r"\s*weeks?", "semana": r"\s*weeks?",
}

_PARAR = {"assertiva", "sobre", "seguinte", "seguintes", "questao", "prova", "considere",
          "relacao", "respeito", "acerca", "julgue", "trimestral", "resposta", "paciente",
          "durante", "quando", "porque", "entre", "maior", "menor", "pode", "podem",
          "deve", "devem", "apresenta", "apresentam"}


def _norm(p: str) -> str:
    p = unicodedata.normalize("NFD", p.lower())
    p = "".join(c for c in p if unicodedata.category(c) != "Mn")
    for a, b in (("ph", "f"), ("th", "t"), ("y", "i"), ("ct", "t"), ("cc", "c"),
                 ("qu", "c"), ("k", "c"), ("ss", "s"), ("z", "s"), ("mm", "m"), ("nn", "n")):
        p = p.replace(a, b)
    return re.sub(r"[^a-z]", "", p)


def ancoras(texto: str) -> list[tuple[str, str]]:
    a = [(m.group(1), m.group(2)) for m in _ANCORA.finditer(texto)]
    return [(n, u) for n, u in a if not re.fullmatch(r"0?[.,]?0+", n)]


def padroes(num: str, uni: str) -> list[str]:
    a = num.replace(",", ".")
    formas = {a} | ({a[:-2]} if a.endswith(".0") else set())
    return [re.escape(f) + _UNI_EN.get(uni.lower(), r"\b") for f in formas]


def cognatos(assertiva: str) -> set[str]:
    """Radicais de 5 caracteres das palavras de conteudo, normalizados pt/en."""
    saida = set()
    for p in re.findall(r"[A-Za-zÀ-ÿ]{6,}", assertiva):
        if p.lower() in _PARAR:
            continue
        n = _norm(p)
        if len(n) >= 5:
            saida.add(n[:5])
    return saida


# Distancia maxima, em caracteres do texto original, entre o numero e o termo.
# Sem esta restricao a precisao do criterio foi de 21% contra rotulagem manual: numa
# assertiva sobre "capacidade vital aumentada em 40%", o cognato `capac` casava num ponto
# do paragrafo e o "40%" em outro, falando de DEBITO CARDIACO. O fato e uma proposicao —
# o numero tem de estar JUNTO do termo que ele qualifica, nao no mesmo paragrafo.
JANELA = 120


def portador(texto_chunk: str, pats: list[str], stems: set[str]) -> bool:
    """
    Ancora numerica E cognato, e os dois DENTRO DA MESMA JANELA de texto.

    A proximidade e o que separa "o paragrafo fala do assunto e contem um numero" de "o
    paragrafo afirma que a grandeza vale aquele numero".
    """
    for m in (mm for p in pats for mm in re.finditer(p, texto_chunk, re.I)):
        ini = max(0, m.start() - JANELA)
        janela = _norm(texto_chunk[ini:m.end() + JANELA])
        if any(s in janela for s in stems):
            return True
    return False


def avaliar(braco: str, itens: list[dict], k_max: int, ix_cache: dict) -> dict:
    caminho, hibrido, rerank = BRACOS[braco]
    if caminho not in ix_cache:
        ix_cache[caminho] = Indice(caminho)
    ix = ix_cache[caminho]
    posicoes: list[int | None] = []
    detalhe = []
    for it in itens:
        asr = it.get("assertiva") or it["enunciado"]
        pats = [p for n, u in it["_ancoras"] for p in padroes(n, u)]
        stems = it["_stems"]
        q = it["enunciado"] + ("\n\nAssertiva: " + (it.get("assertiva") or "")
                               if it["tipo"] == "vf" else "")
        ids = [c for c, _ in ix.buscar(q, k_max, hibrido=hibrido)]
        if rerank:
            ids = [c for c, _ in ix.rerankear(q, ids, k=k_max)]
        pos = next((i for i, c in enumerate(ids, 1)
                    if portador(ix.obter(c)["texto"], pats, stems)), None)
        posicoes.append(pos)
        detalhe.append({"id": it["id"], "pos": pos, "top": ids[:4]})
    n = len(posicoes)
    achou = [p for p in posicoes if p]
    return {
        "n": n,
        "recall": {k: sum(1 for p in achou if p <= k) / n for k in (4, 8, 20, 50, k_max)},
        "mrr": sum(1 / p for p in achou) / n if n else 0.0,
        "mediana": statistics.median(achou) if achou else None,
        "nunca": n - len(achou),
        "detalhe": detalhe,
    }


def carregar(split: str, tipo: str | None) -> list[dict]:
    itens = [json.loads(l) for l in open(RAIZ / "dados" / f"questoes_sba.{split}.jsonl")]
    saida = []
    for it in itens:
        if tipo and it["tipo"] != tipo:
            continue
        alvo = (it.get("assertiva") or "") + " " + it["enunciado"]
        a = ancoras(alvo)
        s = cognatos(it.get("assertiva") or it["enunciado"])
        if not a or not s:
            continue
        saida.append({**it, "_ancoras": a, "_stems": s})
    return saida


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--braco", action="append", default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tipo", default="vf")
    ap.add_argument("--k-max", type=int, default=100)
    ap.add_argument("--dump-ouro", type=int, default=0,
                    help="grava N casos para rotulagem manual e sai")
    ap.add_argument("--saida", type=Path, default=ANALISES / "recuperacao.txt")
    a = ap.parse_args()

    itens = carregar(a.split, a.tipo)
    print(f"  {len(itens)} questões com âncora numérica E cognato "
          f"(split {a.split}, tipo {a.tipo})", file=sys.stderr)

    if a.dump_ouro:
        ix = Indice(BRACOS["atual"][0])
        casos = []
        for it in itens[:a.dump_ouro]:
            pats = [p for n, u in it["_ancoras"] for p in padroes(n, u)]
            q = it["enunciado"] + "\n\nAssertiva: " + (it.get("assertiva") or "")
            ids = [c for c, _ in ix.buscar(q, a.k_max, hibrido=False)]
            pos = next((i for i, c in enumerate(ids, 1)
                        if portador(ix.obter(c)["texto"], pats, it["_stems"])), None)
            casos.append({
                "id": it["id"], "assertiva": " ".join((it.get("assertiva") or "").split()),
                "ancoras": it["_ancoras"], "stems": sorted(it["_stems"]),
                "pos_automatica": pos,
                "trecho_apontado": (" ".join(ix.obter(ids[pos-1])["texto"].split())[:600]
                                    if pos else None),
            })
        p = ANALISES / "ouro-candidatos.json"
        p.write_text(json.dumps(casos, ensure_ascii=False, indent=1))
        print(f"  {len(casos)} casos -> {p}", file=sys.stderr)
        return 0

    bracos = a.braco or ["atual"]
    ix_cache: dict = {}
    L = [f"RECUPERAÇÃO ISOLADA · {len(itens)} questões · split {a.split} · tipo {a.tipo}",
         "",
         "Verdade de referência: âncora número+unidade E cognato pt/en (prefixo 5).",
         "O critério erra para MENOS — termo sem cognato entre os idiomas não casa.",
         "O recall abaixo é PISO. A precisão do critério é medida em bench/ouro.py.",
         "",
         f"  {'braço':12s} {'n':>4s} " + " ".join(f"{'@'+str(k):>7s}" for k in (4, 8, 20, 50))
         + f" {'@'+str(a.k_max):>7s} {'MRR':>6s} {'mediana':>8s} {'nunca':>6s}"]
    bruto = {}
    for b in bracos:
        if b not in BRACOS:
            print(f"braço desconhecido: {b} · conhecidos: {sorted(BRACOS)}", file=sys.stderr)
            return 1
        if not Path(BRACOS[b][0] + ".npy").exists():
            print(f"  {b}: índice {BRACOS[b][0]} não existe — pulando", file=sys.stderr)
            continue
        r = avaliar(b, itens, a.k_max, ix_cache)
        bruto[b] = r
        L.append(f"  {b:12s} {r['n']:4d} "
                 + " ".join(f"{r['recall'][k]:6.1%} " for k in (4, 8, 20, 50, a.k_max))
                 + f"{r['mrr']:6.3f} {str(r['mediana'] or '—'):>8s} {r['nunca']:6d}")
    L += ["", "recall@4 é a métrica que corresponde ao que o modelo recebe hoje.", ""]
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    a.saida.write_text("\n".join(L) + "\n")
    (a.saida.with_suffix(".json")).write_text(json.dumps(bruto))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
