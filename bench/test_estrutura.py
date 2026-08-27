"""
Verificador de wiseoak/ingest/estrutura.py. Comportamento, nao estrutura.

Fatos lidos a mao do PDF (nao gerados pelo codigo sob teste):

  pagina 100  traz a frase "Uma diferenca importante entre inducao da anestesia e
              recuperacao da anestesia e o impacto potencial do metabolismo na taxa de
              reducao da PA no termino da anestesia."
              Na saida do `pdftotext -layout` essa frase vem PARTIDA por rotulos de eixo
              de figura da coluna esquerda ("0,01", "0, 1"). E esse o defeito que a
              reconstrucao de colunas existe para corrigir.
  pagina 105  cabecalho corrente "Capitulo 8 Anestesicos lnalatorios"
  pagina 140  cabecalho corrente "Secao 11 FARMACOLOGIA E FISIOLOCIA"  (II, com OCR ruim)
  geometria   704.25 x 900 pt, duas colunas, corte em x ~ 352
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
BBOX = RAIZ / "dados" / "bruto" / "miller.bbox.xml"
MOD = RAIZ / "wiseoak" / "ingest" / "estrutura.py"


def rodar(*args):
    return subprocess.run([sys.executable, str(MOD), *map(str, args)],
                          capture_output=True, text=True, cwd=RAIZ)


def paginas(pi, pf) -> dict[int, dict]:
    with tempfile.TemporaryDirectory() as d:
        saida = Path(d) / "p.jsonl"
        r = rodar(BBOX, "--saida", saida, "--pagina-inicial", pi, "--pagina-final", pf)
        assert r.returncode == 0, f"saiu {r.returncode}: {r.stderr[-700:]}"
        assert saida.exists(), f"nao escreveu a saida. stderr: {r.stderr[-700:]}"
        return {j["pagina"]: j for j in
                (json.loads(l) for l in saida.read_text().splitlines() if l.strip())}


class TestCLI(unittest.TestCase):
    def test_help_sai_zero(self):
        r = rodar("--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.strip())

    def test_faixa_de_paginas_muda_o_resultado(self):
        # flag decorativa e pior que flag ausente: duas faixas, dois resultados
        poucas = paginas(100, 102)
        muitas = paginas(100, 110)
        self.assertEqual(len(poucas), 3)
        self.assertEqual(len(muitas), 11)

    def test_arquivo_inexistente_sai_diferente_de_zero(self):
        self.assertNotEqual(rodar(RAIZ / "nao-existe.xml").returncode, 0)


class TestColunas(unittest.TestCase):
    def test_prosa_continua_sem_rotulo_de_figura(self):
        p = paginas(100, 100)[100]
        txt = " ".join(p["texto"].split())
        # a frase inteira, sem nada injetado no meio
        self.assertIn("impacto potencial do metabolismo na taxa de", txt,
                      "a frase continua quebrada: colunas nao foram separadas")
        # o rotulo de eixo nao pode aparecer NO MEIO da frase
        i = txt.find("impacto potencial do metabolismo")
        self.assertGreaterEqual(i, 0)
        self.assertNotIn("0,01", txt[i:i + 200])
        self.assertNotIn("0, 1", txt[i:i + 200])

    def test_coluna_de_figura_nao_entra_no_meio_da_frase(self):
        """
        p96: a coluna esquerda traz a lista de rotulos de uma figura ("Coeficiente de
        particao sangue-gas", "Debito cardiaco"...) e a direita traz prosa. Ordenar por
        faixa de y intercalava as duas e picava a frase. Este caso NAO era coberto pelo
        teste da p100, onde os rotulos eram numericos e caiam no filtro de ruido.
        """
        txt = " ".join(paginas(96, 96)[96]["texto"].split())
        i = txt.find("que acompanha a")
        self.assertGreaterEqual(i, 0, "a frase da p96 sumiu")
        depois = txt[i:i + 90]
        self.assertNotIn("Coeficiente de parti", depois,
                         f"rotulo de figura no meio da frase: {depois!r}")
        self.assertIn("fase inicial", depois, f"frase quebrada: {depois!r}")

    def test_ordem_de_leitura_esquerda_antes_de_direita(self):
        p = paginas(100, 100)[100]
        self.assertGreater(len(p["texto"]), 800, "pagina quase vazia")


class TestEstruturaDoLivro(unittest.TestCase):
    def test_capitulo_do_cabecalho_corrente(self):
        p = paginas(105, 105)[105]
        self.assertEqual(p["capitulo_num"], 8, f"veio {p.get('capitulo_num')}")
        self.assertIn("nalat", p["capitulo"], f"veio {p.get('capitulo')!r}")

    def test_secao_em_numeral_romano_corrompido(self):
        # o OCR le III como 111 e II como 11; tem que virar inteiro certo
        p = paginas(140, 140)[140]
        self.assertEqual(p["secao_num"], 2, f"veio {p.get('secao_num')}")
        p3 = paginas(260, 260)[260]
        self.assertEqual(p3["secao_num"], 3, f"veio {p3.get('secao_num')}")

    def test_cabecalho_nao_vaza_para_o_corpo(self):
        p = paginas(105, 105)[105]
        self.assertNotIn("Capítulo 8 Anest", p["texto"][:200],
                         "cabecalho corrente ficou dentro do corpo")

    def test_marca_dagua_removida(self):
        for p in paginas(100, 110).values():
            self.assertNotIn("APOSTILASMEDICINA", p["texto"])
            self.assertNotIn("mercadolivre", p["texto"])

    def test_titulos_detectados_por_altura(self):
        # alguma pagina da faixa tem subtitulo; a lista existe e traz strings curtas
        achou = False
        for p in paginas(100, 110).values():
            self.assertIsInstance(p["titulos"], list)
            for t in p["titulos"]:
                self.assertLess(len(t), 120, f"titulo longo demais: {t!r}")
                achou = True
        self.assertTrue(achou, "nenhum titulo detectado em 11 paginas")


if __name__ == "__main__":
    unittest.main(verbosity=2)
