#!/usr/bin/env python3
"""
Diagnostico do erro, sem gastar GPU: le o que ja esta no banco.

Responde, com numero em vez de impressao:
  1. a recuperacao difere entre modelos?
  2. os dois modelos erram nos MESMOS itens?
  3. com e sem RAG erram nos mesmos itens?
  4. quanto do erro e item impossivel (imagem, correlacao orfa, valor de tabela)?
  5. o erro se concentra em item cujo termo-chave NAO esta no contexto recuperado?
"""
import json
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from wiseoak.store import Indice  # noqa: E402

BLOCO = sys.argv[1] if len(sys.argv) > 1 else "limpo"
ANC = sys.argv[2] if len(sys.argv) > 2 else "confiante"

db = sqlite3.connect(RAIZ / "eval" / "resultados.sqlite")
db.row_factory = sqlite3.Row
cel = {}
for r in db.execute("SELECT modelo,setup,itens,bloco FROM resultado WHERE bench='vf-dev' AND n=80"):
    anc = [x for x in r["setup"].split("|") if x.startswith("anc=")]
    cel[(r["modelo"], r["setup"].split("|")[0], anc[0][4:] if anc else "estrita",
         r["bloco"])] = json.loads(r["itens"])


def pega(mod, g, anc=ANC, bloco=None):
    for k, v in cel.items():
        if k[0] == mod and k[1] == g and k[2] == anc and (bloco is None or k[3] == bloco):
            return v
    return {}


porid = {i["id"]: i for i in (json.loads(l) for l in
         open(RAIZ / "dados" / "questoes_sba.dev.jsonl")) if i["tipo"] == "vf"}

IMG = re.compile(r'(?i)\b(nest[ea]|na|no)\s+(figura|imagem|corte|traçado|gráfico|esquema)\b'
                 r'|\bestrutura\s+\d|\bfigura\s+\d|legenda|\bseta\b')
NUM = re.compile(r'\d+[,.]\d+|\b\d+\s*(mg|ml|mcg|µg|g|kg|%|mmHg|mEq|horas?|minutos?)\b')


def classe(it):
    alvo = f"{it['enunciado']} {it['assertiva']}"
    if IMG.search(alvo):
        return "depende de imagem"
    if NUM.search(it["assertiva"]):
        return "exige valor numerico"
    return "prosa"


def _norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z0-9 ]+", " ", "".join(c for c in s if not unicodedata.combining(c)))


def cobertura(it, ix, k=5):
    """Fracao dos termos raros da assertiva que aparecem no contexto recuperado."""
    q = f"{it['enunciado']}\n\nAssertiva: {it['assertiva']}"
    ids = [c for c, _ in ix.buscar(q, k)]
    ctx = _norm(" ".join(ix.obter(c)["texto"] for c in ids))
    termos = [t for t in set(_norm(it["assertiva"]).split()) if len(t) > 7]
    if not termos:
        return None
    return sum(1 for t in termos if t in ctx) / len(termos)


print("=" * 74)
print("1. A RECUPERACAO DIFERE ENTRE MODELOS?")
print("=" * 74)
print("  NAO. v0/v1/v2 recuperam a partir da pergunta crua — mesma consulta, mesmo")
print("  indice, mesmo k. A recuperacao e IDENTICA entre gemma4 e medgemma.")
print("  So o v3 difere: la a consulta e escrita pelo proprio modelo.")

print("\n" + "=" * 74)
print("2. OS DOIS MODELOS ERRAM NOS MESMOS ITENS?")
print("=" * 74)
for g in ("v0", "v2"):
    a, b = pega("gemma4-plan", g), pega("medgemma-clinical", g)
    com = set(a) & set(b)
    if not com:
        continue
    ea = {k for k in com if not a[k]}
    eb = {k for k in com if not b[k]}
    juntos = ea & eb
    print(f"  {g}: gemma errou {len(ea)}, medgemma errou {len(eb)}, "
          f"AMBOS erraram {len(juntos)}")
    esperado = len(ea) * len(eb) / len(com)
    print(f"      sobreposicao esperada por acaso: {esperado:.1f} → "
          f"{'MUITO acima' if len(juntos) > esperado * 1.5 else 'perto'} do acaso")

print("\n" + "=" * 74)
print("3. COM E SEM RAG ERRAM NOS MESMOS ITENS?")
print("=" * 74)
for mod in ("gemma4-plan", "medgemma-clinical"):
    v0, v2 = pega(mod, "v0"), pega(mod, "v2")
    com = set(v0) & set(v2)
    if not com:
        continue
    tab = {p: 0 for p in ((True, True), (True, False), (False, True), (False, False))}
    for k in com:
        tab[(bool(v0[k]), bool(v2[k]))] += 1
    print(f"  {mod}  (n={len(com)})")
    print(f"     os dois acertaram : {tab[(True, True)]:3d}")
    print(f"     so SEM rag acertou: {tab[(True, False)]:3d}   <- o RAG estragou")
    print(f"     so COM rag acertou: {tab[(False, True)]:3d}   <- o RAG salvou")
    print(f"     os dois erraram   : {tab[(False, False)]:3d}   <- nem um nem outro sabe")

print("\n" + "=" * 74)
print("4. O ERRO E ITEM IMPOSSIVEL (imagem / valor numerico)?")
print("=" * 74)
for mod in ("gemma4-plan", "medgemma-clinical"):
    v2 = pega(mod, "v2")
    if not v2:
        continue
    tot = {}
    err = {}
    for k, ok in v2.items():
        if k not in porid:
            continue
        c = classe(porid[k])
        tot[c] = tot.get(c, 0) + 1
        if not ok:
            err[c] = err.get(c, 0) + 1
    print(f"  {mod}")
    for c in sorted(tot):
        print(f"     {c:22s} {tot[c]:3d} itens, {err.get(c, 0):3d} erros "
              f"({100 * err.get(c, 0) / tot[c]:5.1f}%)")

print("\n" + "=" * 74)
print("5. O ERRO SE CONCENTRA ONDE O CONTEXTO NAO COBRE O TERMO?")
print("=" * 74)
ix = Indice(str(RAIZ / "dados" / "indice" / "h512"))
v2 = pega("gemma4-plan", "v2")
certos, errados = [], []
for k, ok in list(v2.items()):
    if k not in porid:
        continue
    c = cobertura(porid[k], ix)
    if c is None:
        continue
    (certos if ok else errados).append(c)
if certos and errados:
    print(f"  cobertura media do termo-chave no contexto recuperado:")
    print(f"     quando ACERTOU: {sum(certos) / len(certos):.1%}  (n={len(certos)})")
    print(f"     quando ERROU  : {sum(errados) / len(errados):.1%}  (n={len(errados)})")
    d = sum(certos) / len(certos) - sum(errados) / len(errados)
    print(f"     diferenca: {d:+.1%} → "
          f"{'o contexto REALMENTE cobre menos quando erra' if d > 0.05 else 'sem diferenca clara'}")
