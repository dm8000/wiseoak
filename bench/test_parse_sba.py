"""
Verificador de parse_sba.py. Comportamento, nao estrutura.

Os valores esperados foram lidos a mao dos PDFs, nao gerados pelo proprio parser —
senao o teste so provaria que o codigo concorda consigo mesmo.

    TSA 2016            99 linhas 'RESPOSTA:', Q1=A Q2=C Q3=C
    ME1-PN2022          40 questoes,           Q1=B Q2=A Q3=D
    ME1-1o-Tri-0523     5 enunciados / 25 assertivas V-F
                        Q1: A=F B=F C=F D=V E=V
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PROVAS = AQUI.parent / "dados" / "provas"

TSA = PROVAS / "2016-11_TSA-Prova-Escrita-Amarela.pdf"
PN = PROVAS / "2023-02_ME1-PN2022-GABARITO-2.pdf"
VF = PROVAS / "2023-05_ME1-1o-Tri-0523.pdf"


def rodar(*args):
    return subprocess.run([sys.executable, str(AQUI / "parse_sba.py"), *map(str, args)],
                          capture_output=True, text=True, cwd=AQUI)


def parsear(pdf) -> list[dict]:
    """Roda a CLI de verdade e le o JSONL que ela escreveu."""
    with tempfile.TemporaryDirectory() as d:
        saida = Path(d) / "q.jsonl"
        r = rodar(pdf, "--saida", saida)
        assert r.returncode == 0, f"saiu {r.returncode}: {r.stderr[-600:]}"
        assert saida.exists(), f"nao escreveu {saida}. stderr: {r.stderr[-600:]}"
        return [json.loads(l) for l in saida.read_text().splitlines() if l.strip()]


class TestCLI(unittest.TestCase):
    def test_help_sai_zero(self):
        # sem o guard __main__ o script sai 0 sem fazer nada; isto pega isso
        r = rodar("--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.strip())

    def test_saida_muda_com_a_flag(self):
        # flag decorativa e pior que flag ausente: dois valores, dois arquivos
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.jsonl", Path(d) / "b.jsonl"
            self.assertEqual(rodar(PN, "--saida", a).returncode, 0)
            self.assertEqual(rodar(PN, "--saida", b).returncode, 0)
            self.assertTrue(a.exists() and b.exists())
            self.assertEqual(a.read_text(), b.read_text())

    def test_pdf_inexistente_sai_diferente_de_zero(self):
        self.assertNotEqual(rodar(PROVAS / "nao-existe.pdf").returncode, 0)


class TestMCQ(unittest.TestCase):
    def test_prova_nacional_2022(self):
        qs = parsear(PN)
        mcq = [q for q in qs if q["tipo"] == "mcq"]
        # 40 linhas 'Resposta:' no PDF, MENOS 4 marcadas 'QUESTAO ANULADA' pela banca.
        # Contar linhas de gabarito superestima n: a anulada tambem tem gabarito.
        self.assertEqual(len(mcq), 36, f"esperava 36 questoes validas, vieram {len(mcq)}")
        # a q6 e anulada e tem o layout quebrado no PDF de origem
        self.assertNotIn(6, {q["numero"] for q in mcq}, "questao anulada entrou na amostra")
        porn = {q["numero"]: q for q in mcq}
        self.assertEqual(porn[1]["resposta"], "B")
        self.assertEqual(porn[2]["resposta"], "A")
        self.assertEqual(porn[3]["resposta"], "D")
        # o enunciado tem que ser o texto da questao, nao o cabecalho da pagina
        self.assertIn("67 anos", porn[1]["enunciado"])
        self.assertNotIn("PROVA NACIONAL", porn[1]["enunciado"])
        # 4 alternativas, e a resposta tem que ser uma delas
        for q in mcq:
            self.assertEqual(sorted(q["alternativas"]), ["A", "B", "C", "D"], f"q{q['numero']}")
            self.assertIn(q["resposta"], q["alternativas"], f"q{q['numero']}")
            for letra, txt in q["alternativas"].items():
                self.assertGreater(len(txt.strip()), 3, f"q{q['numero']} alt {letra} vazia")

    def test_tsa_2016(self):
        qs = parsear(TSA)
        mcq = [q for q in qs if q["tipo"] == "mcq"]
        # 100 no caderno, 99 com RESPOSTA: no corpo. Aceita a faixa, nao o numero magico.
        self.assertGreaterEqual(len(mcq), 95, f"vieram so {len(mcq)}")
        self.assertLessEqual(len(mcq), 100)
        porn = {q["numero"]: q for q in mcq}
        self.assertEqual(porn[1]["resposta"], "A")
        self.assertEqual(porn[2]["resposta"], "C")
        self.assertEqual(porn[3]["resposta"], "C")
        self.assertIn("27 anos", porn[1]["enunciado"])
        # as instrucoes da prova sao numeradas 1..13 e NAO podem virar questao
        self.assertNotIn("cartão de resposta", porn[1]["enunciado"].lower())


class TestVerdadeiroFalso(unittest.TestCase):
    def test_trimestral_2023(self):
        qs = parsear(VF)
        vf = [q for q in qs if q["tipo"] == "vf"]
        self.assertEqual(len(vf), 25, f"esperava 25 assertivas, vieram {len(vf)}")
        self.assertEqual(len({q["numero"] for q in vf}), 5, "esperava 5 enunciados")

        q1 = {q["letra"]: q for q in vf if q["numero"] == 1}
        self.assertEqual(sorted(q1), ["A", "B", "C", "D", "E"])
        self.assertEqual({l: q["resposta"] for l, q in q1.items()},
                         {"A": "F", "B": "F", "C": "F", "D": "V", "E": "V"})

        # a palavra do gabarito nao pode sobrar dentro do texto da assertiva
        for q in vf:
            self.assertNotRegex(q["assertiva"], r"(?i)(verdadeiro|falso)\s*$",
                                f"q{q['numero']}{q['letra']}: gabarito vazou no texto")
            self.assertGreater(len(q["assertiva"].strip()), 20, f"q{q['numero']}{q['letra']}")
            self.assertIn(q["resposta"], ("V", "F"))
        # o enunciado compartilhado tem que estar junto de cada assertiva
        self.assertIn("2.174", q1["A"]["enunciado"])


class TestComum(unittest.TestCase):
    def test_ids_unicos_e_metadados(self):
        qs = parsear(PN) + parsear(VF)
        ids = [q["id"] for q in qs]
        self.assertEqual(len(ids), len(set(ids)), "ids repetidos")
        for q in qs:
            self.assertIn(q["tipo"], ("mcq", "vf"))
            self.assertTrue(q["fonte"])
            self.assertIsInstance(q["ano"], int)
            self.assertGreaterEqual(q["ano"], 2000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
