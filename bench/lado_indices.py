#!/usr/bin/env python3
"""
Mesma pergunta, dois indices, lado a lado. Inspecao, nao metrica.

Existe porque comparar recuperacao com numero exige qrels feitos a mao, e os criterios
automaticos ja falharam duas vezes aqui (33% e 64% de precisao contra rotulagem manual).
Enquanto os qrels nao existem, a leitura direta dos casos que JA se entende e a evidencia
mais confiavel disponivel.

O caso de referencia sao tres assertivas sobre gravidez que diferem so no substantivo
("ventilacao minuto", "capacidade vital", "capacidade pulmonar total"). No indice de filho
grande as tres recebem quase o mesmo contexto — e a demonstracao de diluicao semantica.
Se o filho menor funcionar, as tres tem de receber contextos DIFERENTES entre si.

    ./lado_indices.py --a dados/indice/m10 --b dados/indice/m10p --caso gravidez
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from wiseoak.store import Indice  # noqa: E402

CASOS = {
    # o trio que demonstra a diluicao: mesma vinheta, substantivo diferente
    "gravidez": ["ventilação minuto está aumenta",
                 "capacidade vital está aumentada",
                 "capacidade pulmonar total está aumentada"],
    # fato numerico especifico dentro de um paragrafo sobre o tema certo
    "hipotermia": ["20 graus celsius induz a supressão do eletroencefalograma"],
    "creatinina": ["creatinina acima de 3 mg"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="dados/indice/m10")
    ap.add_argument("--b", default="dados/indice/m10p")
    ap.add_argument("--caso", action="append", default=None)
    ap.add_argument("--amostra", choices=("acertou", "errou"), default=None,
                    help="amostra questoes pelo resultado ja medido, em vez dos casos fixos")
    ap.add_argument("--bloco", default="COTA", help="de qual braco vem o resultado")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--semente", type=int, default=3)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--chars", type=int, default=150)
    a = ap.parse_args()

    dev = [json.loads(l) for l in open(RAIZ / "dados" / "questoes_sba.dev.jsonl")]

    alvos = None
    if a.amostra:
        # amostrar das DUAS populacoes e o que protege contra regressao: olhar so o que
        # falhou favorece qualquer mudanca, foi assim que a triagem no conjunto de erros
        # enganou. Aqui a amostra e sorteada com semente fixa, para ser reproduzivel.
        import random, sqlite3
        db = sqlite3.connect(RAIZ / "eval" / "resultados.sqlite")
        res = {}
        for (itens,) in db.execute("SELECT itens FROM resultado WHERE bloco=?", (a.bloco,)):
            res.update(json.loads(itens))
        querido = (a.amostra == "acertou")
        ids = [i for i, ok in res.items() if ok == querido]
        random.seed(a.semente)
        escolhidos = set(random.sample(ids, min(a.n, len(ids))))
        alvos = [i for i in dev if i["id"] in escolhidos and i["tipo"] == "vf"]
        print(f"  amostra: {len(alvos)} questoes que o bloco {a.bloco} "
              f"{a.amostra}", file=sys.stderr)

    casos = a.caso or (list(CASOS) if alvos is None else [])
    ixs = [(n, Indice(p)) for n, p in (("A " + Path(a.a).name, a.a),
                                       ("B " + Path(a.b).name, a.b))
           if Path(p + ".npy").exists()]
    if len(ixs) < 2:
        print("um dos índices não existe ainda", file=sys.stderr)
        return 1

    pares = [(None, it) for it in (alvos or [])]
    for caso in casos:
        for chave in CASOS[caso]:
            it = next((i for i in dev if chave in (i.get("assertiva") or "")), None)
            if it is None:
                print(f"  não achei: {chave}", file=sys.stderr)
                continue
            pares.append((caso, it))

    for _, it in pares:
        if True:
            q = it["enunciado"] + "\n\nAssertiva: " + (it.get("assertiva") or "")
            print(f"\n{'='*92}")
            print(f"{' '.join((it.get('assertiva') or '').split())[:170]}")
            for nome, ix in ixs:
                print(f"\n  [{nome}]")
                for i, (c, _) in enumerate(ix.buscar(q, a.k, hibrido=False), 1):
                    d = ix.obter(c)
                    print(f"    {i}. {' '.join(d['texto'].split())[:a.chars]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
