#!/usr/bin/env python3
"""
Divide os bancos de questoes em dev e teste, 50/50.

O split e por HASH DO ID, nao por sorteio: e reproduzivel sem guardar semente, e um
item nunca troca de lado quando o banco cresce. O conjunto de TESTE nao e olhado ate a
configuracao final estar escolhida — inclusive, e principalmente, por um otimizador
automatico de prompt, que faz overfit ao teste com muito mais eficiencia que uma pessoa.

    ./split.py --entrada ../dados/questoes_sba.jsonl
"""
import argparse, collections, hashlib, json, re, sys
from pathlib import Path


def lado(item_id: str, sal: str = "wiseoak") -> str:
    h = hashlib.sha1(f"{sal}:{item_id}".encode()).hexdigest()
    return "dev" if int(h[:8], 16) % 2 == 0 else "teste"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entrada", type=Path, nargs="+", required=True)
    ap.add_argument("--sal", default="wiseoak")
    a = ap.parse_args()

    for entrada in a.entrada:
        if not entrada.exists():
            print(f"nao existe: {entrada}", file=sys.stderr)
            return 1
        itens = [json.loads(l) for l in entrada.read_text().splitlines() if l.strip()]
        # DEDUPLICA por id. A SBA publica a mesma prova duas vezes, original e errata
        # (ex.: ME2-1o-Tri-0523 e ME2-1o-Trimestral-errata-0523), e 15,4% dos itens
        # vinham em dobro. Conferido: nenhuma das duplicatas tem gabarito conflitante, e
        # como o lado do split sai do hash do id, a duplicata nunca caiu nos dois lados —
        # o efeito era inflar `n` e gastar GPU, nao corromper a medicao.
        # Assertiva curta demais e questao de CORRELACAO cujo par se perdeu na extracao
        # ("Correlacione a alteracao no ECG com a alteracao eletrolitica:" -> assertiva
        # so "hipercalcemia"). Sem o par nao ha resposta possivel, e o item vira ruido
        # que dilui qualquer efeito real. 94 casos.
        antes = len(itens)
        itens = [i for i in itens
                 if i.get("tipo") != "vf" or len((i.get("assertiva") or "").split()) >= 6]
        if len(itens) != antes:
            print(f"  sem par de correlacao: {antes - len(itens)} itens", file=sys.stderr)

        # DEDUPLICA POR CONTEUDO, nao por id. Sao coisas diferentes: o id vem da
        # ORIGEM (arquivo|tipo|numero|letra) para sobreviver a correcao de parser sem
        # embaralhar o split; a duplicata vem do CONTEUDO, porque a SBA publica a mesma
        # prova como original e como errata, com nomes de arquivo diferentes.
        def conteudo(it):
            texto = f"{it.get('enunciado','')}|{it.get('assertiva','')}|" \
                    f"{json.dumps(it.get('alternativas'), sort_keys=True, ensure_ascii=False)}"
            return hashlib.sha1(re.sub(r"\s+", " ", texto).strip().lower().encode()).hexdigest()

        vistos, unicos = set(), []
        for it in itens:
            c = conteudo(it)
            if c in vistos:
                continue
            vistos.add(c)
            unicos.append(it)
        if len(unicos) != len(itens):
            print(f"  deduplicados: {len(itens) - len(unicos)} itens repetidos",
                  file=sys.stderr)
        itens = unicos
        cont = collections.Counter()
        saidas = {s: entrada.with_name(f"{entrada.stem}.{s}.jsonl").open("w")
                  for s in ("dev", "teste")}
        try:
            for it in itens:
                s = lado(it["id"], a.sal)
                it["split"] = s
                saidas[s].write(json.dumps(it, ensure_ascii=False) + "\n")
                cont[(s, it["tipo"])] += 1
        finally:
            for f in saidas.values():
                f.close()
        print(f"{entrada.name}: {len(itens)} itens", file=sys.stderr)
        for (s, t), n in sorted(cont.items()):
            print(f"   {s:5s} {t:14s} {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
