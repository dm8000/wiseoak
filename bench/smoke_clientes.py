#!/usr/bin/env python3
"""
Teste de fumaça dos tres clientes. Roda contra o llama-swap de verdade.

Isto reprova se: o servidor estiver fora, a chave estiver errada, o endpoint de rerank
mudar de formato, o embedding vier com dimensao instavel, ou o thinking nao chegar ao
modelo. Cada uma dessas ja quebrou algo neste projeto.

    .venv/bin/python bench/smoke_clientes.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wiseoak import clientes


class TestEmbed(unittest.TestCase):
    def test_dimensao_estavel_e_ordem_preservada(self):
        a = clientes.embed(["propofol", "bloqueio subaracnoideo"])
        b = clientes.embed(["propofol"])
        self.assertEqual(len(a), 2)
        self.assertEqual(len(b), 1)
        # dimensao igual entre chamadas: se o perfil trocar de modelo, o indice ja
        # gravado deixa de casar com as consultas, silenciosamente.
        self.assertEqual(len(a[0]), len(b[0]))
        self.assertGreater(len(a[0]), 64)
        # mesmo texto, mesmo vetor: prova que a ordem da resposta foi respeitada
        self.assertAlmostEqual(a[0][0], b[0][0], places=4)

    def test_lote_maior_que_o_batch(self):
        # LOTE_EMBED=16; 20 textos forcam duas requisicoes e expoem erro de concatenacao
        textos = [f"questao numero {i} sobre anestesia" for i in range(20)]
        v = clientes.embed(textos)
        self.assertEqual(len(v), 20)
        self.assertEqual(len({len(x) for x in v}), 1)

    def test_texto_gigante_nao_derruba(self):
        # acima do ctx 2048 do embed-small: tem que cortar, nao estourar
        v = clientes.embed(["dor " * 20000])
        self.assertEqual(len(v), 1)


class TestRerank(unittest.TestCase):
    def test_documento_plantado_vence_distratores(self):
        pergunta = "Qual a dose de intubacao do rocuronio?"
        docs = [
            "A cafeteria do hospital serve almoco das 11h as 14h.",
            "O rocuronio na dose de 0,6 mg/kg produz condicoes de intubacao em 60 a 90 segundos.",
            "A tabela de plantao de julho foi afixada no mural.",
            "O estacionamento tem 40 vagas para visitantes.",
        ]
        pares = clientes.rerank(pergunta, docs)
        self.assertEqual(len(pares), 4)
        self.assertEqual(pares[0][0], 1, f"esperava o indice 1 no topo, veio {pares}")
        # separacao, nao so ordem: score do bom tem que destacar do melhor distrator
        self.assertGreater(pares[0][1], pares[1][1])

    def test_lista_vazia(self):
        self.assertEqual(clientes.rerank("qualquer", []), [])


class TestChat(unittest.TestCase):
    def test_responde_sem_thinking(self):
        r = clientes.chat(
            [{"role": "user", "content": "Responda apenas com o numero: quanto e 7 mais 5?"}],
            think=False, max_tokens=64,
        )
        self.assertIn("12", r["content"])
        self.assertFalse(r["truncou"])
        self.assertGreater(r["out_tokens"], 0)

    def test_chat_template_kwargs_chega_ao_servidor(self):
        """
        Controle positivo, em qwen-code: prova que `think` sai daqui e chega ao modelo.

        Sem este controle, o teste seguinte (MedGemma sem reasoning) seria ambiguo —
        cliente quebrado e modelo sem thinking dao o mesmo resultado vazio.

        CUSTA UM SWAP DE 18 GB. E o preco de saber qual das duas hipoteses e a certa.
        """
        p = [{"role": "user", "content": "Quanto e 17 vezes 23? Responda so o numero."}]
        com = clientes.chat(p, modelo="qwen-code", think=True, max_tokens=1024)
        sem = clientes.chat(p, modelo="qwen-code", think=False, max_tokens=1024)
        self.assertGreater(len(com["reasoning"]), 0, "think=True nao produziu reasoning")
        self.assertEqual(sem["reasoning"], "", "think=False produziu reasoning")

    def test_medgemma_nao_tem_canal_de_thinking(self):
        """
        Propriedade MEDIDA, nao suposta (2026-08-25): medgemma-clinical devolve
        reasoning vazio com think=True e com think=False. MedGemma 27B e Gemma 3 por
        baixo, e Gemma 3 nao tem modo de raciocinio.

        Consequencia de projeto: o remedio conhecido deste projeto para o modo de falha
        "sob contexto longo o campo numerico recebe o numero saliente" (0/18 sem
        thinking, 16/18 com) NAO esta disponivel no MedGemma. Raciocinio explicito tem
        que ser pedido no prompt, e isso vira uma dimensao a medir, nao uma decisao.

        Se um dia este teste falhar, o modelo ganhou thinking e o desenho muda.
        """
        p = [{"role": "user", "content": "Quanto e 17 vezes 23? Responda so o numero."}]
        for think in (True, False):
            r = clientes.chat(p, think=think, max_tokens=1024)
            self.assertEqual(r["reasoning"], "",
                             f"medgemma passou a emitir reasoning com think={think}")
            self.assertIn("391", r["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
