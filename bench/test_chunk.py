"""
Verificador de wiseoak/ingest/chunk.py.

Fatos lidos a mao de dados/miller_paginas.jsonl (nao gerados pelo codigo sob teste):

  p95 e p100   capitulo_num 8, capitulo contem "nalat", secao_num 2
  p100         a linha "Metabolismo" aparece sozinha, seguida da frase
               "Uma diferenca importante entre inducao da anestesia e recuperacao da
                anestesia e o impacto potencial do metabolismo na taxa de reducao..."
  p95-p97      titulos MAIUSCULOS reais: "EFEITO SEGUNDO GAS", "SOLUBILIDADE";
               titulos Title-case reais: "Ventilacao alveolar"
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PAGINAS = RAIZ / "dados" / "miller_paginas.jsonl"
MOD = RAIZ / "wiseoak" / "ingest" / "chunk.py"


def rodar(*args):
    return subprocess.run([sys.executable, str(MOD), *map(str, args)],
                          capture_output=True, text=True, cwd=RAIZ)


def chunks(*extra, pi=95, pf=105) -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        saida = Path(d) / "c.jsonl"
        r = rodar(PAGINAS, "--saida", saida, "--pagina-inicial", pi,
                  "--pagina-final", pf, *extra)
        assert r.returncode == 0, f"saiu {r.returncode}: {r.stderr[-700:]}"
        assert saida.exists(), f"nao escreveu a saida. stderr: {r.stderr[-700:]}"
        return [json.loads(l) for l in saida.read_text().splitlines() if l.strip()]


class TestCLI(unittest.TestCase):
    def test_help_sai_zero(self):
        r = rodar("--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.strip())

    def test_entrada_inexistente_sai_diferente_de_zero(self):
        self.assertNotEqual(rodar(RAIZ / "nao-existe.jsonl").returncode, 0)

    def test_tamanho_do_filho_muda_a_contagem(self):
        # flag decorativa e pior que flag ausente: dois valores, dois resultados
        pequenos = [c for c in chunks("--tamanho-filho", 256) if c["nivel"] == "filho"]
        grandes = [c for c in chunks("--tamanho-filho", 768) if c["nivel"] == "filho"]
        self.assertGreater(len(pequenos), len(grandes),
                           f"256 deu {len(pequenos)} e 768 deu {len(grandes)}")


class TestHierarquia(unittest.TestCase):
    def test_pais_e_filhos_existem(self):
        cs = chunks()
        pais = {c["id"] for c in cs if c["nivel"] == "pai"}
        filhos = [c for c in cs if c["nivel"] == "filho"]
        self.assertGreater(len(pais), 0, "nenhum pai")
        self.assertGreater(len(filhos), len(pais), "menos filhos que pais")
        for f in filhos:
            self.assertIn(f["pai_id"], pais, f"filho {f['id']} aponta para pai inexistente")

    def test_caminho_de_titulos(self):
        cs = chunks()
        caminhos = {tuple(c["caminho"]) for c in cs}
        achatado = {t for cam in caminhos for t in cam}
        self.assertTrue(any("SEGUNDO" in t.upper() for t in achatado),
                        f"titulo maiusculo real nao virou caminho: {sorted(achatado)[:12]}")
        self.assertTrue(any("Metabolismo" == t for t in achatado),
                        "o subtitulo 'Metabolismo' da p100 nao entrou no caminho")

    def test_chunk_do_metabolismo_tem_o_contexto_certo(self):
        cs = chunks()
        alvo = [c for c in cs if "impacto potencial do metabolismo" in c["texto"]]
        self.assertTrue(alvo, "a frase do metabolismo sumiu do corpus")
        c = alvo[0]
        self.assertEqual(c["capitulo_num"], 8)
        self.assertIn("nalat", c["capitulo"])
        self.assertEqual(c["secao_num"], 2)
        self.assertIn(100, range(c["pagina_inicial"], c["pagina_final"] + 1))
        self.assertIn("Metabolismo", c["caminho"])

    def test_sem_hierarquia_e_uma_dimensao_de_verdade(self):
        # o braco "chunking plano" tem que produzir caminho vazio, e mesmo assim texto
        plano = chunks("--sem-hierarquia")
        self.assertTrue(plano)
        self.assertTrue(all(c["caminho"] == [] for c in plano),
                        "--sem-hierarquia manteve o caminho de titulos")
        self.assertGreater(sum(len(c["texto"]) for c in plano), 10000)


class TestTamanhoEOverlap(unittest.TestCase):
    def test_filho_cabe_no_contexto_do_embed(self):
        # embed-small roda com --ctx-size 2048; estourar derruba o lote inteiro
        for c in chunks("--tamanho-filho", 512):
            if c["nivel"] == "filho":
                self.assertLessEqual(len(c["texto"]), 512 * 3.2 * 1.3,
                                     f"filho {c['id']} grande demais: {len(c['texto'])} chars")

    def test_filhos_consecutivos_se_sobrepoem(self):
        cs = [c for c in chunks("--overlap", 0.2) if c["nivel"] == "filho"]
        porpai = {}
        for c in cs:
            porpai.setdefault(c["pai_id"], []).append(c)
        achou = False
        for lista in porpai.values():
            if len(lista) < 2:
                continue
            lista.sort(key=lambda c: c["ordem"])
            a, b = lista[0], lista[1]
            cauda = " ".join(a["texto"].split()[-8:])
            self.assertIn(cauda.split()[0], b["texto"],
                          "filhos consecutivos nao se sobrepoem")
            achou = True
            break
        self.assertTrue(achou, "nenhum pai com 2+ filhos para testar overlap")

    def test_todo_chunk_tem_texto_util(self):
        for c in chunks():
            self.assertGreater(len(c["texto"].strip()), 40, f"chunk vazio: {c['id']}")
            self.assertTrue(c["livro"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
