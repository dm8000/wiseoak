#!/usr/bin/env python3
"""
FlowBot adaptado ao WiseOak: induz o workflow por otimizacao bilevel com gradientes
textuais.

    Yu, Kim & Kim, "FlowBot: Inducing LLM Workflows with Bilevel Optimization and
    Textual Gradients", arXiv:2604.26258. Nao ha codigo publico; isto e implementacao
    a partir do paper.

O metodo, como esta no artigo:

    laco interno   g_K = LLM(x_K, y, L(x_K,y); t_grad_loss)      ultima camada
                   g_k = LLM(x_k, x_k+1, g_k+1; t_grad_call)     retropropaga
                   t_i <- LLM(t_i, {g_i}^B; t_optim_call)        atualiza o prompt
    laco externo   g_W = LLM(W, {x_i}, y, L; t_grad_workflow)    add/remove/funde
                   W   <- LLM(W, {g_W}^B; t_optim_workflow)
    treino         1 epoca, lote 5, 2 laços bilevel por lote, 1 externo + 5 internos,
                   validacao a cada lote, melhor da validacao vai ao teste.

TRES ADAPTACOES, declaradas:

1. **So chamada de LLM e camada.** `recuperar`, `rerankear` e `contexto` sao ferramentas
   deterministicas — no formalismo do paper elas sao as *tools* a que a chamada tem
   acesso, nao camadas com prompt. Camadas otimizaveis: reformular, criticar, responder.

2. **A sketch e restrita a um REGISTRO.** O paper deixa o LLM escrever a estrutura em
   texto livre. Aqui ele escolhe entre operacoes executaveis conhecidas; passo fora do
   registro e descartado e registrado. Sem isso a sketch nao roda.

3. **Orcamento.** Uma passagem direta custa ~20 s nesta maquina. Validacao a cada lote
   sobre 40 itens seriam 13 min por lote. O tamanho da validacao e parametro, e o
   default e pequeno de proposito.

SALVAGUARDA CONTRA OVERFIT: a selecao e pela VALIDACAO, como no paper, e o conjunto de
TESTE nunca e tocado por este script. Um otimizador automatico faz overfit ao que ele
enxerga com muito mais eficiencia que uma pessoa.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
ANALISES = RAIZ / "eval" / "analises"
ANALISES.mkdir(parents=True, exist_ok=True)

from wiseoak import clientes  # noqa: E402
from wiseoak.grafos import comum  # noqa: E402
from wiseoak.grafos.variantes import de_sketch  # noqa: E402
from wiseoak.store import Indice  # noqa: E402

# ------------------------------------------------------------------ o registro

# Operacoes que a sketch pode conter. O laco externo escolhe entre estas; o que ele
# inventar fora daqui nao executa.
REGISTRO = {
    "reformular": "reescreve a pergunta como consulta de busca antes de consultar o livro",
    "recuperar_hibrido": "busca no livro combinando BM25 e vetores (RRF)",
    "recuperar_denso": "busca no livro so por vetores",
    "rerankear": "reordena os trechos recuperados por relevancia",
    "expandir_pai": "troca o trecho curto pela secao inteira que o contem",
    "criticar": "julga se os trechos bastam e, se nao, busca de novo",
    "responder": "produz a resposta final com citacoes",
}
OBRIGATORIO = "responder"
CAMADAS_LLM = ("reformular", "criticar", "responder")  # so estas tem prompt otimizavel


@dataclass
class Estado:
    """W (a sketch) mais os prompts t_i. E o que a otimizacao bilevel move."""
    sketch: list[str]
    prompts: dict[str, str] = field(default_factory=dict)

    def texto(self) -> str:
        return "\n".join(f"{i}. {p}: {REGISTRO.get(p, '?')}"
                         for i, p in enumerate(self.sketch, 1))

    def copia(self) -> "Estado":
        return Estado(list(self.sketch), dict(self.prompts))


# ------------------------------------------------------- executar uma sketch

def executar(est: Estado, pergunta: str, modo: str, modelo: str, ix: Indice,
             k_busca: int, k_contexto: int) -> dict:
    """
    Passagem direta sobre um StateGraph COMPILADO a partir da sketch.

    Compilar de verdade, em vez de simular num laco Python, e o que garante que o
    workflow vencedor da busca seja o MESMO objeto que o benchmark mede e que o Pipe
    serve. Um otimizador que otimiza uma simulacao encontra o otimo da simulacao.
    """
    grafo = de_sketch(est.sketch)
    e = {"pergunta": pergunta, "modo": modo, "modelo": modelo,
         "raciocinio": "nenhum", "ancoragem": "confiante", "indice": ix,
         "k_busca": k_busca, "k_contexto": k_contexto, "trace": []}
    with prompt_temporario(est, "responder"):
        r = grafo.invoke(e)
    # {x_i}: o que cada camada produziu, que o laco interno retropropaga
    xs = [("entrada", pergunta)]
    for passo in (r.get("trace") or []):
        # o proprio no 'responder' ja esta no trace; nao duplicar
        saida = (r.get("resposta", "") if passo["no"] == "responder"
                 else str(passo.get("consulta_usada") or passo.get("trechos")
                          or passo.get("veredito") or ""))
        xs.append((passo["no"], str(saida)[:300]))
    return {"estado": r, "xs": xs, "saida": r.get("resposta", ""),
            "ressalva": r.get("ressalva", ""), "fontes": r.get("fontes") or []}


class prompt_temporario:
    """Injeta o prompt otimizado no lugar do de producao, so durante a chamada."""

    def __init__(self, est: Estado, camada: str):
        self.est, self.camada, self.antigo = est, camada, None

    def __enter__(self):
        novo = self.est.prompts.get(self.camada)
        if novo and self.camada == "responder":
            self.antigo = comum.SISTEMA["confiante"]
            comum.SISTEMA["confiante"] = novo
        return self

    def __exit__(self, *a):
        if self.antigo is not None:
            comum.SISTEMA["confiante"] = self.antigo


# ----------------------------------------------------- gradientes textuais

META = "gemma4-plan"  # o LLM que escreve os gradientes e as atualizacoes


def _meta(sistema: str, usuario: str, max_tokens: int = 900) -> str:
    r = clientes.chat([{"role": "system", "content": sistema},
                       {"role": "user", "content": usuario}],
                      modelo=META, think=False, max_tokens=max_tokens, temp=0.7)
    return r["content"].strip()


T_GRAD_LOSS = ("Voce produz GRADIENTE TEXTUAL. Recebe a saida de um passo, a resposta "
               "correta e o erro. Diga, em ate 4 linhas, O QUE no PROMPT daquele passo "
               "causou o erro e que mudanca concreta o corrigiria. Nao reescreva o "
               "prompt; descreva a direcao da mudanca. Se a saida estava certa, diga o "
               "que preservar.")

T_GRAD_CALL = ("Voce retropropaga GRADIENTE TEXTUAL. Recebe a entrada de um passo, a "
               "saida dele, e o gradiente do passo seguinte. Diga em ate 3 linhas como "
               "a SAIDA deste passo deveria mudar para ajudar o passo seguinte.")

T_OPTIM_CALL = ("Voce reescreve o prompt de um passo a partir dos gradientes de um lote. "
                "Devolva SO o prompt novo, em portugues, sem comentario e sem aspas ao "
                "redor. Mantenha o que os gradientes elogiam. Nao aumente o prompt sem "
                "necessidade: neste projeto ja foi medido que prompt inchado DERRUBA o "
                "desempenho em 21 pontos.")

T_GRAD_WORKFLOW = (
    "Voce avalia a ESTRUTURA de um workflow. Recebe a sequencia de passos, o que cada um "
    "produziu, a resposta correta e o erro. Diga em ate 3 linhas se algum passo deveria "
    "ser ACRESCENTADO, REMOVIDO ou FUNDIDO, e por que. Considere que cada passo custa "
    "tempo. Se a estrutura esta boa, diga isso.")

T_OPTIM_WORKFLOW = (
    "Voce reescreve a sequencia de passos de um workflow a partir dos gradientes de um "
    "lote. Escolha APENAS entre as operacoes disponiveis, uma por linha, na ordem de "
    "execucao, sem numerar e sem comentar. O passo 'responder' e obrigatorio e vem por "
    "ultimo. Uma busca tem de vir antes de rerankear ou expandir_pai.")


def gradiente_lote(est: Estado, lote: list[dict], modelo: str, ix: Indice,
                   k_busca: int, k_contexto: int, log) -> tuple[dict, list[str]]:
    """Uma passagem direta + retropropagacao por exemplo. Devolve grads por camada."""
    grads: dict[str, list[str]] = {c: [] for c in CAMADAS_LLM}
    grads_w: list[str] = []
    for it in lote:
        q = f"{it['enunciado']}\n\nAssertiva: {it['assertiva']}"
        r = executar(est, q, "vf", modelo, ix, k_busca, k_contexto)
        certo = (r["saida"] or "").strip().upper()[:1] == it["resposta"]
        perda = "acertou" if certo else f"ERROU (respondeu {r['saida']!r}, correto {it['resposta']!r})"
        log(f"      {'ok ' if certo else 'ERR'} {it['id']}")

        # gradiente da ultima camada
        g = _meta(T_GRAD_LOSS,
                  f"PERGUNTA:\n{q[:800]}\n\nSAIDA DO PASSO 'responder':\n{r['saida'][:300]}\n"
                  f"RESSALVA:\n{r['ressalva'][:400]}\n\nRESPOSTA CORRETA: {it['resposta']}\n"
                  f"RESULTADO: {perda}", 500)
        grads["responder"].append(g)

        # retropropaga para as camadas de LLM anteriores que existirem na sketch
        anteriores = [p for p in est.sketch if p in CAMADAS_LLM and p != "responder"]
        prox = g
        for camada in reversed(anteriores):
            saida = next((v for k, v in r["xs"] if k == camada), "")
            prox = _meta(T_GRAD_CALL,
                         f"PASSO: {camada} — {REGISTRO[camada]}\n"
                         f"ENTRADA: {q[:400]}\nSAIDA DESTE PASSO: {str(saida)[:400]}\n"
                         f"GRADIENTE DO PASSO SEGUINTE:\n{prox[:500]}", 400)
            grads[camada].append(prox)

        # gradiente de estrutura
        passos = "\n".join(f"  {k}: {str(v)[:120]}" for k, v in r["xs"])
        grads_w.append(_meta(T_GRAD_WORKFLOW,
                             f"ESTRUTURA ATUAL:\n{est.texto()}\n\nO QUE CADA PASSO "
                             f"PRODUZIU:\n{passos}\n\nRESPOSTA CORRETA: {it['resposta']}\n"
                             f"RESULTADO: {perda}", 400))
    return grads, grads_w


def atualizar_prompts(est: Estado, grads: dict[str, list[str]], log) -> None:
    for camada, gs in grads.items():
        if camada not in est.sketch or not gs:
            continue
        atual = est.prompts.get(camada) or (comum.SISTEMA["confiante"]
                                            if camada == "responder" else "")
        if not atual:
            continue
        novo = _meta(T_OPTIM_CALL,
                     f"PASSO: {camada} — {REGISTRO[camada]}\n\nPROMPT ATUAL:\n{atual}\n\n"
                     f"GRADIENTES DO LOTE:\n" + "\n---\n".join(g[:400] for g in gs), 900)
        if 80 < len(novo) < 2500:
            est.prompts[camada] = novo
            log(f"      prompt de '{camada}' atualizado ({len(atual)} -> {len(novo)} chars)")


def atualizar_sketch(est: Estado, grads_w: list[str], log) -> None:
    disp = "\n".join(f"  {k}: {v}" for k, v in REGISTRO.items())
    novo = _meta(T_OPTIM_WORKFLOW,
                 f"OPERACOES DISPONIVEIS:\n{disp}\n\nESTRUTURA ATUAL:\n{est.texto()}\n\n"
                 f"GRADIENTES DO LOTE:\n" + "\n---\n".join(g[:300] for g in grads_w), 400)
    passos, descartados = [], []
    for linha in novo.splitlines():
        nome = linha.strip().lstrip("0123456789.-) ").split(":")[0].strip()
        if nome in REGISTRO:
            if nome not in passos:
                passos.append(nome)
        elif nome:
            descartados.append(nome)
    # o registro tem uma ordem valida; a sketch precisa respeita-la
    ordem = list(REGISTRO)
    passos.sort(key=lambda p: ordem.index(p))
    if OBRIGATORIO not in passos:
        passos.append(OBRIGATORIO)
    if not any(p.startswith("recuperar") for p in passos):
        passos = [p for p in passos if p in ("reformular", OBRIGATORIO)]
    if passos and passos != est.sketch:
        log(f"      sketch: {est.sketch} -> {passos}")
        est.sketch = passos
    if descartados:
        log(f"      fora do registro, descartados: {descartados}")


# ------------------------------------------------------------------ avaliar

def avaliar(est: Estado, itens: list[dict], modelo: str, ix: Indice,
            k_busca: int, k_contexto: int) -> float:
    acertos = 0
    for it in itens:
        q = f"{it['enunciado']}\n\nAssertiva: {it['assertiva']}"
        try:
            r = executar(est, q, "vf", modelo, ix, k_busca, k_contexto)
        except Exception:
            continue
        acertos += (r["saida"] or "").strip().upper()[:1] == it["resposta"]
    return acertos / len(itens) if itens else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default="gemma4-plan", help="o executor")
    ap.add_argument("--treino", type=int, default=20)
    ap.add_argument("--validacao", type=int, default=25)
    ap.add_argument("--lote", type=int, default=5)
    ap.add_argument("--bilevel", type=int, default=1,
                    help="laços bilevel por lote (o paper usa 2; 1 cabe no orçamento)")
    ap.add_argument("--k-busca", type=int, default=10)
    ap.add_argument("--k-contexto", type=int, default=5)
    ap.add_argument("--semente", type=int, default=7)
    ap.add_argument("--saida", type=Path, default=ANALISES / "flowbot-execucao.json")
    a = ap.parse_args()

    vf = [i for i in (json.loads(l) for l in
                      open(RAIZ / "dados" / "questoes_sba.dev.jsonl")) if i["tipo"] == "vf"]
    rnd = random.Random(a.semente)
    amostra = rnd.sample(vf, a.treino + a.validacao)
    treino, validacao = amostra[:a.treino], amostra[a.treino:]
    ix = Indice(str(RAIZ / "dados" / "indice" / "h512"))

    linhas: list[str] = []

    def log(m: str):
        print(m, file=sys.stderr)
        linhas.append(m)

    est = Estado(["recuperar_hibrido", "rerankear", "expandir_pai", "responder"])
    log(f"# FlowBot · executor {a.modelo} · meta {META}")
    log(f"# treino {len(treino)} · validacao {len(validacao)} · lote {a.lote}")
    log(f"# sketch inicial: {est.sketch}")

    t0 = time.time()
    base = avaliar(est, validacao, a.modelo, ix, a.k_busca, a.k_contexto)
    log(f"\nvalidacao inicial: {base:.1%}  ({time.time()-t0:.0f}s)")
    melhor = {"pontos": base, "sketch": list(est.sketch), "prompts": dict(est.prompts)}
    historico = [{"lote": 0, "validacao": base, "sketch": list(est.sketch)}]

    for n in range(0, len(treino), a.lote):
        lote = treino[n:n + a.lote]
        log(f"\n=== lote {n // a.lote + 1} de {(len(treino) + a.lote - 1) // a.lote} ===")
        for ciclo in range(a.bilevel):
            grads, grads_w = gradiente_lote(est, lote, a.modelo, ix,
                                            a.k_busca, a.k_contexto, log)
            atualizar_sketch(est, grads_w, log)      # externo
            atualizar_prompts(est, grads, log)       # interno
        p = avaliar(est, validacao, a.modelo, ix, a.k_busca, a.k_contexto)
        log(f"    validacao: {p:.1%}" + ("  <- MELHOR" if p > melhor["pontos"] else ""))
        historico.append({"lote": n // a.lote + 1, "validacao": p,
                          "sketch": list(est.sketch)})
        if p > melhor["pontos"]:
            melhor = {"pontos": p, "sketch": list(est.sketch),
                      "prompts": dict(est.prompts)}

    log(f"\n# MELHOR NA VALIDACAO: {melhor['pontos']:.1%} (inicial {base:.1%})")
    log(f"# sketch: {melhor['sketch']}")
    a.saida.write_text(json.dumps(
        {"base": base, "melhor": melhor, "historico": historico,
         "log": linhas, "minutos": round((time.time() - t0) / 60, 1)},
        ensure_ascii=False, indent=2))
    log(f"# gravado em {a.saida}")
    log("# o conjunto de TESTE nao foi tocado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
