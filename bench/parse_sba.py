#!/usr/bin/env python3
"""Extrai questoes de provas SBA (PDF) para JSONL."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Extrai questoes de PDFs SBA para JSONL")
    parser.add_argument("pdfs", nargs="+", help="Caminhos dos PDFs")
    parser.add_argument("--saida", required=True, help="Arquivo JSONL de saida")
    return parser.parse_args()


def extract_text(pdf_path):
    result = subprocess.run(
        ["pdftotext", "-layout", pdf_path, "-"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"pdftotext falhou para {pdf_path}: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def extract_year(text, filename):
    m = re.search(r'(\d{4})-\d{2}_', filename)
    if m:
        return int(m.group(1))
    m = re.search(r'ANO\s+(\d{4})', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


def strip_header_footer(text):
    lines = text.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r'^PROVA\s+NACIONAL', stripped, re.IGNORECASE):
            continue
        if re.match(r'^Sociedade\s+Brasileira\s+de\s+Anestesiologia', stripped, re.IGNORECASE):
            continue
        if re.match(r'^CCA$', stripped, re.IGNORECASE):
            continue
        if re.match(r'^\d+$', stripped) and len(stripped) <= 3:
            continue
        filtered.append(line)
    return '\n'.join(filtered)


def is_instruction_block(lines):
    keywords = [
        'cartao de resposta', 'assinado', 'proibido fumar', 'duracao de',
        'instrucoes', 'instruções', 'folha de respostas', 'caneta esferografica',
        'caneta esferográfica', 'letra de forma', 'sala da prova'
    ]
    block_text = ' '.join(lines).lower()
    for kw in keywords:
        if kw in block_text:
            return True
    return False


def detect_format(text):
    vf_count = len(re.findall(r'(?:Verdadeiro|Falso)', text))
    mcq_count = len(re.findall(r'^\s*Resposta:\s*[A-D]', text, re.MULTILINE | re.IGNORECASE))
    if vf_count > mcq_count:
        return 'vf'
    return 'mcq'


def parse_mcq(text, pdf_path=''):
    text = strip_header_footer(text)
    lines = text.split('\n')
    questions = []
    blocks = []
    current_block = None
    current_lines = []

    for line in lines:
        m = re.match(r'^\s*(\d+)\s*[-)\–]\s*(.*)', line)
        if m:
            if current_block is not None:
                blocks.append((current_block, current_lines))
            current_block = int(m.group(1))
            cleaned_line = re.sub(r'^\s*\d+\s*[-)\–]\s*', '', line).strip()
            current_lines = [cleaned_line] if cleaned_line else []
        elif current_block is not None:
            current_lines.append(line)

    if current_block is not None:
        blocks.append((current_block, current_lines))

    for num, blines in blocks:
        if is_instruction_block(blines):
            continue
        # Questao anulada pela banca nao entra na amostra. O PDF ainda traz um
        # "Resposta: X" para ela, entao contar linhas de gabarito superestima n.
        if re.search(r'quest[aã]o\s+anulada', '\n'.join(blines), re.I):
            continue
        has_alts = False
        alt_count = 0
        for bl in blines:
            if re.match(r'^\s*[A-Ea-e][).]', bl):
                has_alts = True
                alt_count += 1
        if alt_count < 4:
            continue
        resposta_m = re.search(r'^\s*Resposta:\s*([A-E])', '\n'.join(blines), re.MULTILINE | re.IGNORECASE)
        if not resposta_m:
            continue
        resposta = resposta_m.group(1).upper()
        enunciado_parts = []
        alts = {}
        current_alt = None
        skip_resposta = False
        for bl in blines:
            stripped = bl.strip()
            if re.match(r'^\s*Resposta:', stripped, re.IGNORECASE):
                skip_resposta = True
                continue
            if skip_resposta:
                continue
            alt_m = re.match(r'^\s*([A-Ea-e])[).]\s*(.*)', bl)
            if alt_m:
                current_alt = alt_m.group(1).upper()
                alts[current_alt] = alt_m.group(2).strip()
            elif current_alt and current_alt in alts:
                alts[current_alt] += ' ' + stripped
            else:
                enunciado_parts.append(stripped)
        enunciado = ' '.join(enunciado_parts)
        enunciado = re.sub(r'\s+', ' ', enunciado).strip()
        if len(alts) < 4:
            continue
        if any(not alts.get(k, '').strip() for k in ('A', 'B', 'C', 'D')):
            sys.stderr.write(f"  descartada q{num}: alternativa ausente ou vazia\n")
            continue
        # ID ANCORADO NA ORIGEM, nao no texto. Com sha1 do texto, qualquer
        # correcao de parser muda o id, e como o split e por hash do id, os
        # itens TROCAM DE LADO entre dev e teste. Isso ja invalidou uma
        # comparacao (v0 deu 83,8% e 73,8% na mesma configuracao, por troca de
        # amostra). Arquivo, numero e letra nao mudam quando o parser melhora.
        chave = f"{os.path.basename(pdf_path)}|mcq|{num}"
        qid = hashlib.sha1(chave.encode()).hexdigest()[:8]
        questions.append({
            'id': qid,
            'tipo': 'mcq',
            'numero': num,
            'enunciado': enunciado,
            'alternativas': {'A': alts.get('A', ''), 'B': alts.get('B', ''), 'C': alts.get('C', ''), 'D': alts.get('D', '')},
            'resposta': resposta
        })
    return questions


def parse_vf(text, pdf_path=''):
    text = strip_header_footer(text)
    lines = text.split('\n')
    questions = []
    blocks = []
    current_block = None
    current_lines = []

    for line in lines:
        m = re.match(r'^\s*(\d+)\s*[-)\–]\s*(.*)', line)
        if m:
            if current_block is not None:
                blocks.append((current_block, current_lines))
            current_block = int(m.group(1))
            cleaned_line = re.sub(r'^\s*\d+\s*[-)\–]\s*', '', line).strip()
            current_lines = [cleaned_line] if cleaned_line else []
        elif current_block is not None:
            current_lines.append(line)

    if current_block is not None:
        blocks.append((current_block, current_lines))

    for num, blines in blocks:
        if not blines:
            continue
            
        # O enunciado vai da abertura do bloco ATE o cabecalho 'Questoes/Resposta:' ou a
        # primeira alternativa — nao e so a primeira linha. Pegar so blines[0] truncava
        # 27% dos enunciados no meio da frase ("...compoe uma avaliacao pre-").
        partes_enun = []
        for bl in blines:
            st = bl.strip()
            if re.match(r'^(?:Quest[oõ]es|Resposta:)', st, re.IGNORECASE):
                break
            if re.match(r'^\s*[A-Ea-e][).]\s', bl):
                break
            if st:
                partes_enun.append(st)
        enunciado = re.sub(r'\s+', ' ', ' '.join(partes_enun)).strip()
        assertivas = {}
        current_letra = None
        
        for bl in blines[1:]:
            stripped = bl.strip()
            if not stripped:
                continue
                
            if re.match(r'^(?:Questoes|Resposta:)', stripped, re.IGNORECASE):
                continue
                
            alt_m = re.match(r'^\s*([A-Ea-e])[).]\s*(.*)', bl)
            if alt_m:
                current_letra = alt_m.group(1).upper()
                rest = alt_m.group(2)
                
                # Sem \s+ obrigatorio antes: quando a celula e centralizada verticalmente
                # o marcador fica sozinho na linha ("B)          Falso") e o \s* do
                # proprio marcador ja consumiu os espacos.
                resp_m = re.search(r'\b(Verdadeiro|Falso)\s*$', rest, re.IGNORECASE)
                if resp_m:
                    current_resp = 'V' if resp_m.group(1).lower() == 'verdadeiro' else 'F'
                    rest = rest[:resp_m.start()].strip()
                else:
                    # NUNCA inventar gabarito. O codigo antigo assumia 'F' aqui, e isso
                    # fabricou o gabarito de 46 assertivas — todas F, todas erradas por
                    # construcao. Item sem gabarito legivel sai da amostra.
                    current_resp = None

                # Celula centralizada: a primeira linha da assertiva aparece ACIMA da
                # linha do marcador, e foi anexada por engano a letra anterior. Se esta
                # letra ficou sem texto, aquela linha era dela.
                if not rest and current_letra and assertivas:
                    anterior = assertivas.get(sorted(assertivas)[-1])
                    if anterior and anterior.get('extras'):
                        rest = anterior['extras'].pop()
                        anterior['text'] = anterior['text'][:-(len(rest) + 1)].rstrip()

                assertivas[current_letra] = {'text': rest, 'resp': current_resp,
                                             'extras': []}
            elif current_letra and current_letra in assertivas:
                assertivas[current_letra]['text'] += ' ' + stripped
                assertivas[current_letra]['extras'].append(stripped)
                
        for letra in sorted(assertivas.keys()):
            data = assertivas[letra]
            assertiva_text = re.sub(r'\s+', ' ', data['text']).strip()
            # alguns autores repetem o gabarito dentro da celula ("... -F"); isso
            # vazaria a resposta para dentro da pergunta
            assertiva_text = re.sub(r'\s*[-–—]\s*[VF]\s*$', '', assertiva_text).strip()
            resp = data['resp']
            if resp is None:
                sys.stderr.write(f"  descartada q{num}{letra}: sem gabarito legivel\n")
                continue
            if re.match(r'(?i)^(verdadeiro|falso)\b', assertiva_text):
                sys.stderr.write(f"  descartada q{num}{letra}: gabarito vazou no texto\n")
                continue
            
            # ID ANCORADO NA ORIGEM, nao no texto. Com sha1 do texto, qualquer
            # correcao de parser muda o id, e como o split e por hash do id, os
            # itens TROCAM DE LADO entre dev e teste. Isso ja invalidou uma
            # comparacao (v0 deu 83,8% e 73,8% na mesma configuracao, por troca de
            # amostra). Arquivo, numero e letra nao mudam quando o parser melhora.
            chave = f"{os.path.basename(pdf_path)}|vf|{num}|{letra}"
            qid = hashlib.sha1(chave.encode()).hexdigest()[:8]
            questions.append({
                'id': qid,
                'tipo': 'vf',
                'numero': num,
                'letra': letra,
                'enunciado': enunciado,
                'assertiva': assertiva_text,
                'resposta': resp
            })
    return questions


def get_fonte(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r'^\d{4}-\d{2}_', base)
    if m:
        base = base[m.end():]
    return base


def main():
    args = parse_args()
    for pdf_path in args.pdfs:
        if not os.path.isfile(pdf_path):
            print(f"Arquivo nao encontrado: {pdf_path}", file=sys.stderr)
            sys.exit(1)

    all_items = []
    with open(args.saida, 'w', encoding='utf-8') as f:
        for pdf_path in args.pdfs:
            text = extract_text(pdf_path)
            fmt = detect_format(text)
            ano = extract_year(text, os.path.basename(pdf_path))
            fonte = get_fonte(os.path.basename(pdf_path))
            arquivo = os.path.basename(pdf_path)

            if fmt == 'mcq':
                questions = parse_mcq(text, pdf_path)
            else:
                questions = parse_vf(text, pdf_path)

            for q in questions:
                q['fonte'] = fonte
                q['arquivo'] = arquivo
                q['ano'] = ano
                f.write(json.dumps(q, ensure_ascii=False) + '\n')

            print(f"{arquivo}: {fmt} -> {len(questions)} itens", file=sys.stderr)

    print(f"Total: {len(all_items)} itens em {args.saida}", file=sys.stderr)


if __name__ == '__main__':
    main()