#!/usr/bin/env python3
"""
Baixa o Larxel/healthqa-br como CONTROLE FORA DE DOMINIO.

Nao e benchmark de anestesiologia: sao Revalida e ENARE, medicina geral, e o campo
`group` nao tem nenhuma categoria de anestesia (verificado 2026-08-25 pela API do HF).
Serve para uma pergunta so: se o RAG do livro de anestesia melhorar TAMBEM aqui, o ganho
nao veio do livro.

Usa a datasets-server em vez do parquet para nao exigir pyarrow/pandas.
"""
import argparse, json, sys, urllib.request
from pathlib import Path

PARQUET = "https://huggingface.co/api/datasets/Larxel/healthqa-br/parquet/default/train/0.parquet"


def baixar(limite: int | None = None) -> list[dict]:
    """
    Le o parquet de uma vez. A datasets-server (/rows) devolve 429 sob paginacao: 57
    requisicoes para 5.632 linhas estouram o limite por origem, mesmo com recuo
    exponencial de ate 32 s. Um GET so resolve.
    """
    import io
    import pyarrow.parquet as pq

    with urllib.request.urlopen(PARQUET, timeout=180) as r:
        bruto = r.read()
    tabela = pq.read_table(io.BytesIO(bruto))
    itens = tabela.to_pylist()
    return itens[:limite] if limite else itens


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--saida", type=Path, default=Path("../dados/questoes_healthqa.jsonl"))
    ap.add_argument("--limite", type=int, default=None)
    a = ap.parse_args()
    itens = baixar(a.limite)
    if not itens:
        print("nada baixado", file=sys.stderr)
        return 1
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    with a.saida.open("w") as f:
        for x in itens:
            f.write(json.dumps({
                "id": x["id"], "fonte": x["source"], "arquivo": "healthqa-br",
                "tipo": "mcq_healthqa", "numero": 0, "ano": int(x.get("year") or 0),
                "grupo": x.get("group"), "enunciado": x["question"], "resposta": x["answer"],
            }, ensure_ascii=False) + "\n")
    print(f"{len(itens)} questoes -> {a.saida}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
