#!/usr/bin/env python3
"""
Coleta o corpus NORMATIVO e de DIRETRIZES. Tudo de acesso aberto.

Motivo, medido: o Opus 5 abriu 15,4% de vantagem sobre nos na classe
`juridico-normativo` e 9,7% em `gestao`. Sao as classes cuja fonte NAO e livro-texto —
e norma do CFM, estatuto de sociedade e diretriz de especialidade. O Miller nao cobre e
nenhum ajuste de prompt resolve.

Quatro fontes:

  CFM    resolucoes em sistemas.cfm.org.br. NAO ha endpoint publico de enumeracao: a
         raiz de /normas/ e tela de login, e /pesquisa da 404. A lista vem de uma
         relacao curada por sociedade de anestesiologia, e o PDF vem da fonte oficial.
         Adivinhar numero nao serve — a 2217/2019 responde 404 nesse padrao.
  SBA    estatuto e regimentos, pela mesma REST API do WordPress que colheu as provas.
  AMB    Projeto Diretrizes, PDFs em amb.org.br.
  SBC    diretrizes em acesso aberto no SciELO.

Grava um manifesto com URL, data de acesso e sha256 de cada arquivo: sem isso nao da
para saber depois de que versao da norma veio uma citacao, e norma e revogada.

    ./coletar_normas.py --dry-run     # lista sem baixar
    ./coletar_normas.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "dados" / "normas"
CABECA = {"User-Agent": "Mozilla/5.0 (pesquisa academica; anestesiologia)"}

# Resolucoes do CFM relevantes a anestesiologia. Numero e ano conferidos um a um contra
# sistemas.cfm.org.br (todos 200, application/pdf) em 2026-08-26. O ano de 1451 e 1995
# nao vinha na lista de origem e foi descoberto por varredura.
CFM = [
    (1451, 1995, "Urgencia e emergencia: estrutura minima"),
    (1595, 2000, "Relacao com a industria farmaceutica"),
    (1640, 2002, "Procedimento cirurgico"),
    (1670, 2003, "Sedacao consciente e niveis mais profundos"),
    (1711, 2003, "Lipoaspiracao: parametros de seguranca"),
    (1720, 2004, "Termo de consentimento"),
    (1802, 2006, "Pratica do ato anestesico (revogada pela 2174)"),
    (1805, 2006, "Ortotanasia"),
    (1886, 2008, "Cirurgia ambulatorial"),
    (1995, 2012, "Diretivas antecipadas de vontade"),
    (2147, 2016, "Diretor tecnico e responsabilidade"),
    (2174, 2017, "Pratica do ato anestesico (vigente)"),
]
CFM_URL = "https://sistemas.cfm.org.br/normas/arquivos/resolucoes/BR/{ano}/{num}_{ano}.pdf"

# Termos que identificam documento normativo da SBA na busca do WordPress.
SBA_TERMOS = ["estatuto", "regimento", "regulamento", "resolucao", "norma tecnica",
              "codigo de etica", "defesa profissional"]
SBA_BUSCA = "https://www.sbahq.org/wp-json/wp/v2/search?search={termo}&per_page=50&page={pag}"

AMB = [
    ("https://amb.org.br/files/_BibliotecaAntiga/anestesia-venosa-total-para-sedacao.pdf",
     "Projeto Diretrizes: anestesia venosa total para sedacao"),
]

SBC = [
    ("https://www.scielo.br/j/abc/a/cLFwccgTWxk7fyXyFpFGx7b/?format=pdf&lang=pt",
     "SBC: ressuscitacao cardiopulmonar e emergencias cardiovasculares"),
]

MINIMO = 4096  # abaixo disto e pagina de erro salva como PDF; ja aconteceu neste projeto


def baixar(url: str, timeout: int = 90) -> bytes | None:
    """GET com recuo exponencial em 429/5xx, como em coletar_healthqa.py."""
    for tentativa in range(5):
        try:
            req = urllib.request.Request(url, headers=CABECA)
            return urllib.request.urlopen(req, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and tentativa < 4:
                espera = 2 ** tentativa
                sys.stderr.write(f"\r  {e.code}, aguardando {espera}s")
                time.sleep(espera)
                continue
            return None
        except Exception:
            if tentativa < 4:
                time.sleep(2 ** tentativa)
                continue
            return None
    return None


def alvos_sba() -> list[tuple[str, str]]:
    """Varre a REST API do WordPress da SBA atras de PDF de documento normativo."""
    import re
    paginas, achados = set(), []
    for termo in SBA_TERMOS:
        for pag in (1, 2):
            bruto = baixar(SBA_BUSCA.format(termo=termo.replace(" ", "%20"), pag=pag), 30)
            if not bruto:
                break
            try:
                dados = json.loads(bruto)
            except json.JSONDecodeError:
                break
            if not isinstance(dados, list) or not dados:
                break
            for x in dados:
                if x.get("url"):
                    paginas.add((x["url"], x.get("title", "")))
            time.sleep(0.3)
    # SO documento normativo. Sem este filtro a varredura traz todo PDF do site — 496
    # arquivos, 586 MB, incluindo relatorio de gestao, certificado ISO e, o que importa,
    # GABARITO DE PROVA. Gabarito no indice faz o RAG recuperar a resposta da questao
    # que esta sendo avaliada, e o benchmark deixa de medir qualquer coisa.
    # 'c[óo]digo' inteiro, nao so 'codigo de etica': a SBA numera a propria serie
    # normativa (0_ESTATUTO, 1_CODIGO_DE_PROCESSO_ADMINISTRATIVO, 2_CODIGO_PROFISSIONAL)
    # e o filtro antigo guardava o 0 e jogava fora o 1 e o 2.
    NORMATIVO = re.compile(r'(?i)estatuto|regimento|regulamento|resolu[çc][ãa]o|norma|'
                           r'c[óo]digo|cem\d{4}|diretriz')
    PROVA = re.compile(r'(?i)gabarito|prova|\bME[123]\b|trimestral|quadri|simulado')
    for url, titulo in sorted(paginas):
        html = baixar(url, 30)
        if not html:
            continue
        for pdf in set(re.findall(rb'https://[^"\'\s<>]+\.pdf', html)):
            u = pdf.decode("utf8", "ignore")
            if PROVA.search(u) or not NORMATIVO.search(u + " " + titulo):
                continue
            achados.append((u, f"SBA: {re.sub('<[^>]+>', '', titulo)[:70]}"))
        time.sleep(0.2)
    return achados


def gravar(dados: bytes, destino: Path, url: str, descricao: str,
           fonte: str, manifesto: list) -> bool:
    if not dados or not dados[:4] == b"%PDF" or len(dados) < MINIMO:
        return False
    destino.write_bytes(dados)
    manifesto.append({
        "arquivo": destino.name, "fonte": fonte, "descricao": descricao,
        "url": url, "acessado": date.today().isoformat(),
        "bytes": len(dados), "sha256": hashlib.sha256(dados).hexdigest(),
    })
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="lista as URLs sem baixar nada")
    ap.add_argument("--destino", type=Path, default=DESTINO)
    ap.add_argument("--pular-sba", action="store_true",
                    help="a varredura da SBA e a mais lenta; pule ao reexecutar")
    a = ap.parse_args()

    planejado: list[tuple[str, str, str, str]] = []   # fonte, nome, url, descricao
    for num, ano, desc in CFM:
        planejado.append(("cfm", f"CFM-{num}-{ano}.pdf",
                          CFM_URL.format(num=num, ano=ano), f"Resolucao CFM {num}/{ano} — {desc}"))
    for url, desc in AMB:
        planejado.append(("amb", "AMB-" + url.rsplit("/", 1)[-1], url, desc))
    for url, desc in SBC:
        planejado.append(("sbc", "SBC-rcp.pdf", url, desc))

    if not a.pular_sba:
        print("varrendo a REST API da SBA...", file=sys.stderr)
        for url, desc in alvos_sba():
            nome = "SBA-" + url.rsplit("/", 1)[-1][:60]
            planejado.append(("sba", nome, url, desc))

    print(f"\n{len(planejado)} documentos planejados:\n", file=sys.stderr)
    for fonte, nome, url, desc in planejado:
        print(f"  [{fonte:3s}] {nome[:44]:44s} {desc[:56]}")
    if a.dry_run:
        print("\n--dry-run: nada baixado.", file=sys.stderr)
        return 0

    a.destino.mkdir(parents=True, exist_ok=True)
    manifesto: list[dict] = []
    ok = pulado = falhou = 0
    for fonte, nome, url, desc in planejado:
        alvo = a.destino / nome
        if alvo.exists() and alvo.stat().st_size >= MINIMO:
            pulado += 1
            dados = alvo.read_bytes()
            gravar(dados, alvo, url, desc, fonte, manifesto)
            continue
        dados = baixar(url)
        if gravar(dados, alvo, url, desc, fonte, manifesto):
            ok += 1
            sys.stderr.write(f"\r  baixados {ok}, pulados {pulado}, falhas {falhou}   ")
        else:
            falhou += 1
            print(f"\n  FALHOU: {nome} <- {url}", file=sys.stderr)
        time.sleep(0.4)
    sys.stderr.write("\n")

    (a.destino / "manifesto.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifesto) + "\n")
    print(f"\n{ok} baixados, {pulado} ja existiam, {falhou} falharam", file=sys.stderr)
    print(f"manifesto: {a.destino / 'manifesto.jsonl'}", file=sys.stderr)
    return 0 if falhou == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
