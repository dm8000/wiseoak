#!/usr/bin/env python3
"""
Formulario de anestesiologia, exposto como UMA ferramenta ao modelo.

POR QUE BUSCA POR SITUACAO E NAO SO POR NOME
Dos 47 erros numericos medidos, 6 envolviam formula e NENHUM errou a aritmetica —
erraram em saber QUAL formula aplicar. Uma tabela indexada so pelo nome canonico ajuda
exatamente quem ja sabe o nome, ou seja, quem nao precisava dela. Por isso cada verbete
carrega `quando`, e a busca casa contra nome + sinonimos + quando + a grandeza calculada.

POR QUE NAO HA CALCULADORA
Mesma medicao: zero erros de aritmetica. Uma ferramenta de calculo resolveria um problema
que nao existe neste banco, e cada ferramenta a mais custa precisao de tool calling.

POR QUE UMA FERRAMENTA SO
A bancada deste projeto mediu que a precisao de tool calling cai conforme o numero de
ferramentas cresce. Uma so, com busca boa, bate varias com nomes exatos.

ALCANCE, DECLARADO: no dev, 161 questoes sao de valor numerico, mas so 16 trazem marca de
calculo e 9 dessas sao dose por peso (multiplicacao que nunca falhou). O formulario
alcanca ~5 itens em 1.003 — 0,5%, contra IC de +/-2,5 pp. Serve ao produto; nao e
mensuravel nesta amostra, e nao deve ser vendido como ganho de acerto.
"""

from __future__ import annotations

import re
import unicodedata

# Cada verbete: nome canonico, expressao, unidades/faixa normal, quando usar, sinonimos.
# `quando` e material de busca tanto quanto o nome — ver o cabecalho.
FORMULAS: list[dict] = [
    {
        "nome": "Ânion gap",
        "expressao": "AG = Na⁺ − (Cl⁻ + HCO₃⁻)",
        "normal": "8–12 mEq/L (sem K⁺); 12–16 mEq/L se incluir K⁺",
        "quando": "acidose metabólica: separar as de ânion gap alto (cetoacidose, "
                  "lactato, uremia, intoxicação) das hiperclorêmicas (diarreia, soro "
                  "fisiológico em excesso, acidose tubular renal)",
        "obs": "Corrigir para hipoalbuminemia: AG + 2,5 × (4,0 − albumina g/dL). Sem "
               "essa correção o AG parece normal no paciente crítico hipoalbuminêmico.",
        "sinonimos": ["anion gap", "hiato aniônico", "AG"],
    },
    {
        "nome": "Delta gap (delta ratio)",
        "expressao": "Δ/Δ = (AG − 12) / (24 − HCO₃⁻)",
        "normal": "1–2 = acidose de AG alto pura",
        "quando": "acidose metabólica de ânion gap alto, para descobrir se há um segundo "
                  "distúrbio junto",
        "obs": "< 0,4 hiperclorêmica; 0,4–0,8 mista; > 2 sugere alcalose metabólica ou "
               "acidose respiratória crônica concomitante.",
        "sinonimos": ["delta delta", "delta ratio", "distúrbio misto"],
    },
    {
        "nome": "Peso corporal predito (Devine)",
        "expressao": "Homem: 50 + 0,91 × (altura_cm − 152,4)\n"
                     "Mulher: 45,5 + 0,91 × (altura_cm − 152,4)",
        "normal": "kg",
        "quando": "programar volume corrente na ventilação — o pulmão escala com a "
                  "ALTURA, não com o peso real; usar peso real em obeso causa "
                  "volutrauma",
        "obs": "É o peso que entra na ventilação protetora, não o peso da balança.",
        "sinonimos": ["PCP", "peso ideal", "peso predito", "ventilação protetora"],
    },
    {
        "nome": "Ventilação protetora",
        "expressao": "VC = 6 mL/kg de peso corporal predito",
        "normal": "platô < 30 cmH₂O; driving pressure (Pplatô − PEEP) < 15 cmH₂O",
        "quando": "SDRA e, por extensão, ventilação intraoperatória de rotina",
        "obs": "Os três alvos andam juntos: 6 mL/kg de PCP, platô < 30, driving < 15.",
        "sinonimos": ["volume corrente", "SDRA", "SARA", "ARDS", "6 ml/kg"],
    },
    {
        "nome": "Gradiente alvéolo-arterial de O₂",
        "expressao": "PAO₂ = FiO₂ × (Patm − PH₂O) − PaCO₂/R = FiO₂ × (760 − 47) − PaCO₂/0,8\n"
                     "Gradiente A-a = PAO₂ − PaO₂",
        "normal": "≈ (idade/4) + 4 mmHg, em ar ambiente",
        "quando": "hipoxemia: separar hipoventilação e baixa FiO₂ (gradiente NORMAL) de "
                  "shunt, distúrbio V/Q e alteração de difusão (gradiente ALARGADO)",
        "obs": "R = 0,8 é o quociente respiratório. PH₂O = 47 mmHg a 37 °C.",
        "sinonimos": ["A-a", "gradiente alveolo arterial", "hipoxemia", "PAO2"],
    },
    {
        "nome": "Conteúdo arterial de oxigênio",
        "expressao": "CaO₂ = (1,34 × Hb × SaO₂) + (0,003 × PaO₂)",
        "normal": "≈ 20 mL O₂/dL",
        "quando": "avaliar oferta de O₂; mostrar que a hemoglobina domina e a PaO₂ "
                  "dissolvida contribui pouco",
        "obs": "1,34 mL O₂ por grama de Hb saturada. SaO₂ em fração.",
        "sinonimos": ["CaO2", "conteúdo de oxigênio", "oferta de oxigênio"],
    },
    {
        "nome": "Oferta e consumo de oxigênio",
        "expressao": "DO₂ = DC × CaO₂ × 10\nVO₂ = DC × (CaO₂ − CvO₂) × 10",
        "normal": "DO₂ ≈ 1.000 mL/min; VO₂ ≈ 250 mL/min",
        "quando": "choque, para separar problema de oferta de problema de extração",
        "obs": "O fator 10 converte dL em L.",
        "sinonimos": ["DO2", "VO2", "entrega de oxigênio", "choque"],
    },
    {
        "nome": "Taxa de extração de oxigênio",
        "expressao": "TEO₂ = VO₂/DO₂ ≈ (SaO₂ − SvO₂)/SaO₂",
        "normal": "22–30%",
        "quando": "SvO₂ baixa: extração aumentada indica oferta insuficiente para a "
                  "demanda; extração baixa com lactato alto sugere falha de utilização "
                  "(sepse, cianeto)",
        "obs": "Acima de 50–60% a reserva acabou e o metabolismo vira anaeróbio.",
        "sinonimos": ["TEO2", "extração de oxigênio", "SvO2", "saturação venosa mista"],
    },
    {
        "nome": "Débito cardíaco pelo princípio de Fick",
        "expressao": "DC = VO₂ / [(CaO₂ − CvO₂) × 10]\nÍndice cardíaco = DC / superfície corporal",
        "normal": "DC 4–8 L/min; IC 2,5–4,0 L/min/m²",
        "quando": "medir débito sem termodiluição, ou conferir um valor de cateter",
        "obs": "Exige sangue venoso MISTO (artéria pulmonar), não venoso central.",
        "sinonimos": ["Fick", "débito cardíaco", "índice cardíaco", "DC"],
    },
    {
        "nome": "Resistência vascular",
        "expressao": "RVS = [(PAM − PVC)/DC] × 80\nRVP = [(PAP média − POAP)/DC] × 80",
        "normal": "RVS 800–1.200; RVP 40–180 dyn·s·cm⁻⁵",
        "quando": "classificar o choque: RVS baixa aponta distributivo; RVS alta com DC "
                  "baixo aponta cardiogênico ou hipovolêmico",
        "obs": "O 80 é o fator de conversão de mmHg·min/L para dyn·s·cm⁻⁵.",
        "sinonimos": ["RVS", "RVP", "resistência periférica", "choque distributivo"],
    },
    {
        "nome": "Pressão de perfusão cerebral",
        "expressao": "PPC = PAM − PIC   (usar PVC no lugar da PIC se a PVC for maior)",
        "normal": "alvo 60–70 mmHg",
        "quando": "traumatismo cranioencefálico, neurocirurgia, hipertensão intracraniana",
        "obs": "Zerar o transdutor no meato acústico externo, não no átrio, quando a "
               "cabeceira está elevada — senão a PAM cerebral vem superestimada.",
        "sinonimos": ["PPC", "perfusão cerebral", "PIC", "TCE", "hipertensão intracraniana"],
    },
    {
        "nome": "Pressão de perfusão coronariana",
        "expressao": "PPCo = pressão diastólica aórtica − pressão diastólica final do VE",
        "normal": "—",
        "quando": "isquemia miocárdica: explica por que taquicardia e hipotensão "
                  "diastólica são deletérias, e por que o VE esquerdo perfunde na diástole",
        "obs": "Na estenose aórtica a PDF do VE sobe e a margem desaparece.",
        "sinonimos": ["perfusão coronariana", "isquemia", "diástole", "estenose aórtica"],
    },
    {
        "nome": "Complacência do sistema respiratório",
        "expressao": "Estática = VC / (Pplatô − PEEP)\nDinâmica = VC / (Ppico − PEEP)",
        "normal": "estática 60–100 mL/cmH₂O",
        "quando": "pressão de pico alta: se o PLATÔ também subiu, é complacência "
                  "(edema, SDRA, pneumotórax, abdome); se só o PICO subiu, é resistência "
                  "(broncoespasmo, tubo dobrado, secreção)",
        "obs": "É a distinção pico-versus-platô que resolve a maioria das questões.",
        "sinonimos": ["complacência", "pressão de platô", "pressão de pico", "broncoespasmo"],
    },
    {
        "nome": "Espaço morto (Bohr-Enghoff)",
        "expressao": "Vd/Vt = (PaCO₂ − PetCO₂) / PaCO₂",
        "normal": "0,20–0,35",
        "quando": "diferença grande entre CO₂ expirado e arterial: embolia pulmonar, "
                  "baixo débito, hipovolemia",
        "sinonimos": ["espaço morto", "Vd/Vt", "capnografia", "embolia pulmonar", "PetCO2"],
    },
    {
        "nome": "Fração de shunt",
        "expressao": "Qs/Qt = (CcO₂ − CaO₂) / (CcO₂ − CvO₂)",
        "normal": "< 5%",
        "quando": "hipoxemia que não corrige com O₂ a 100% — a marca do shunt verdadeiro",
        "obs": "CcO₂ = conteúdo capilar terminal, calculado assumindo saturação de 100%.",
        "sinonimos": ["shunt", "Qs/Qt", "hipoxemia refratária", "atelectasia"],
    },
    {
        "nome": "Clearance de creatinina (Cockcroft-Gault)",
        "expressao": "ClCr = [(140 − idade) × peso_kg] / (72 × creatinina_mg/dL)\n"
                     "× 0,85 se mulher",
        "normal": "> 90 mL/min",
        "quando": "ajustar dose de fármaco de eliminação renal; estimar função renal "
                  "quando a creatinina isolada engana (idoso, sarcopênico)",
        "obs": "Creatinina normal em idoso magro pode esconder clearance muito baixo.",
        "sinonimos": ["clearance", "Cockcroft", "função renal", "creatinina", "ajuste de dose"],
    },
    {
        "nome": "Déficit e correção de sódio",
        "expressao": "Déficit = ACT × (Na desejado − Na atual)\n"
                     "ACT = 0,6 × peso (homem) ou 0,5 × peso (mulher)",
        "normal": "corrigir no máximo 8–10 mEq/L em 24 h",
        "quando": "hiponatremia — inclusive a da intoxicação hídrica pós-RTU de próstata "
                  "e a da gestante",
        "obs": "Correção rápida demais causa mielinólise pontina. Em hiponatremia aguda "
               "sintomática, salina hipertônica e alvo de subir 4–6 mEq/L rápido, e parar.",
        "sinonimos": ["hiponatremia", "sódio", "mielinólise", "RTU", "salina hipertônica"],
    },
    {
        "nome": "Sódio corrigido na hiperglicemia",
        "expressao": "Na corrigido = Na medido + 1,6 × [(glicemia − 100) / 100]",
        "normal": "—",
        "quando": "cetoacidose diabética ou estado hiperosmolar, onde o sódio medido "
                  "parece baixo por diluição",
        "sinonimos": ["pseudo-hiponatremia", "cetoacidose", "hiperglicemia", "sódio corrigido"],
    },
    {
        "nome": "Osmolaridade plasmática",
        "expressao": "Osm = 2 × Na⁺ + glicose/18 + ureia/6",
        "normal": "285–295 mOsm/kg",
        "quando": "calcular o hiato osmolar (medida − calculada): hiato alto aponta "
                  "álcool, metanol, etilenoglicol, manitol",
        "obs": "Se usar BUN em vez de ureia, o divisor é 2,8.",
        "sinonimos": ["osmolaridade", "hiato osmolar", "gap osmolar", "manitol", "metanol"],
    },
    {
        "nome": "Dose máxima de anestésico local",
        "expressao": "Volume (mL) = dose (mg) / (concentração % × 10)",
        "normal": "lidocaína 4,5 mg/kg (7 com adrenalina); bupivacaína 2,5 mg/kg "
                  "(3 com adrenalina); ropivacaína 3 mg/kg",
        "quando": "bloqueio de plexo, peridural, anestesia local — teto antes da "
                  "intoxicação sistêmica",
        "obs": "Solução a 1% = 10 mg/mL. Intoxicação: emulsão lipídica 20%, bolus de "
               "1,5 mL/kg.",
        "sinonimos": ["anestésico local", "lidocaína", "bupivacaína", "ropivacaína",
                      "intoxicação", "dose tóxica", "emulsão lipídica"],
    },
    {
        "nome": "Reposição no queimado (Parkland)",
        "expressao": "Volume 24 h = 4 mL × peso_kg × %SCQ (Ringer lactato)",
        "normal": "metade nas primeiras 8 h a partir da QUEIMADURA, resto em 16 h",
        "quando": "queimadura extensa",
        "obs": "As 8 h contam do momento da queimadura, não da chegada. Débito urinário "
               "de 0,5 mL/kg/h no adulto é o alvo que corrige a fórmula.",
        "sinonimos": ["Parkland", "queimado", "queimadura", "SCQ", "regra dos nove",
                      "Ringer", "reposição volêmica"],
    },
]


def _normalizar(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]+", " ", s)


_PARAR = {"de", "da", "do", "a", "o", "e", "em", "para", "com", "qual", "como",
          "que", "no", "na", "um", "uma", "por", "se", "the", "of"}


def _tokens(s: str) -> set[str]:
    return {t for t in _normalizar(s).split() if len(t) > 2 and t not in _PARAR}


def buscar(consulta: str, n: int = 3) -> list[dict]:
    """
    Verbetes mais proximos da consulta. Casa contra nome, sinonimos E `quando` — a
    situacao clinica encontra a formula sem que se saiba o nome dela.
    """
    q = _tokens(consulta)
    if not q:
        return []
    pontuados = []
    for f in FORMULAS:
        # nome e sinonimos pesam mais que a descricao de uso, mas `quando` conta:
        # e o que atende quem descreve o problema em vez de nomear a formula
        forte = _tokens(f["nome"] + " " + " ".join(f["sinonimos"]))
        fraco = _tokens(f["quando"] + " " + f.get("obs", "") + " " + f["expressao"])
        p = 3 * len(q & forte) + len(q & fraco)
        if p:
            pontuados.append((p, f))
    pontuados.sort(key=lambda x: -x[0])
    return [f for _, f in pontuados[:n]]


def formatar(verbetes: list[dict]) -> str:
    if not verbetes:
        return "Nenhuma fórmula do formulário corresponde à consulta."
    partes = []
    for f in verbetes:
        b = [f"{f['nome']}", f"  {f['expressao']}"]
        if f.get("normal"):
            b.append(f"  valores: {f['normal']}")
        b.append(f"  quando: {f['quando']}")
        if f.get("obs"):
            b.append(f"  nota: {f['obs']}")
        partes.append("\n".join(b))
    return "\n\n".join(partes)


# Schema OpenAI. UMA ferramenta so — ver o cabecalho.
ESPECIFICACAO = {
    "type": "function",
    "function": {
        "name": "formula",
        "description": (
            "Consulta o formulário de anestesiologia. Aceita o NOME da fórmula "
            "('ânion gap') ou a SITUAÇÃO CLÍNICA ('pressão de pico alta com platô "
            "normal', 'hipoxemia que não melhora com O2 a 100%'). Use quando a questão "
            "depender de uma relação quantitativa e você não tiver certeza de qual "
            "fórmula se aplica."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "consulta": {
                    "type": "string",
                    "description": "Nome da fórmula ou descrição da situação clínica.",
                }
            },
            "required": ["consulta"],
        },
    },
}


def executar(argumentos: dict) -> str:
    return formatar(buscar(str(argumentos.get("consulta") or "")))
