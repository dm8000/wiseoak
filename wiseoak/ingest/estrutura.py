#!/usr/bin/env python3
import sys
import os
import json
import argparse
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from normalizar import normalizar


def parse_secao_num(s):
    """Converte numeral romano OCRizado (ex: '11' -> 2, '111' -> 3) ou arabe."""
    if not s:
        return None
    s = s.strip()
    if all(c == '1' for c in s) and len(s) > 0:
        return len(s)
    romans = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev = 0
    for c in reversed(s.upper()):
        val = romans.get(c, 0)
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total if total > 0 else None


def _sem_acento(s: str) -> str:
    """Casa cabecalho independente de acento: o OCR alterna 'Secao'/'Seção'."""
    import unicodedata
    n = unicodedata.normalize("NFKD", s)
    return "".join(c for c in n if not unicodedata.combining(c)).lower()


def is_noise(text):
    """Filtra linhas com menos de 4 chars ou so digitos/sinais."""
    t = text.strip()
    if len(t.replace(' ', '')) < 4:
        return True
    if not re.search(r'[^\W\d_]', t):
        return True
    return False


def process_page(page_num, page_elem, prev_meta):
    lines = []
    for line in page_elem.iter():
        if line.tag.split('}')[-1] == 'line':
            xMin = float(line.get('xMin', 0))
            yMin = float(line.get('yMin', 0))
            xMax = float(line.get('xMax', 0))
            yMax = float(line.get('yMax', 0))
            words = [w.text for w in line.iter() if w.tag.split('}')[-1] == 'word' and w.text]
            text = ' '.join(words)
            lines.append({
                'xMin': xMin, 'yMin': yMin, 'xMax': xMax, 'yMax': yMax,
                'text': text, 'words': words
            })

    header_lines = [l for l in lines if l['yMin'] < 60]
    body_lines = [l for l in lines if 60 <= l['yMin'] <= 860]

    meta = dict(prev_meta)
    for h in header_lines:
        txt = h['text'].strip()
        if _sem_acento(txt).startswith('capitulo'):
            parts = txt.split(None, 2)
            if len(parts) >= 3:
                try:
                    meta['capitulo_num'] = int(parts[1])
                except ValueError:
                    pass
                meta['capitulo'] = parts[2]
        elif _sem_acento(txt).startswith('secao'):
            parts = txt.split(None, 2)
            if len(parts) >= 3:
                meta['secao_num'] = parse_secao_num(parts[1])
                meta['secao'] = parts[2]

    left = []
    right = []
    full = []
    for l in body_lines:
        if l['xMin'] < 352 and l['xMax'] > 500:
            full.append(l)
        elif l['xMin'] < 352:
            left.append(l)
        else:
            right.append(l)

    left = [l for l in left if not is_noise(l['text'])]
    right = [l for l in right if not is_noise(l['text'])]
    full = [l for l in full if not is_noise(l['text'])]

    left.sort(key=lambda x: x['yMin'])
    right.sort(key=lambda x: x['yMin'])
    full.sort(key=lambda x: x['yMin'])

    # ORDEM DE LEITURA POR BANDAS.
    #
    # Ordenar tudo por faixa de y INTERCALA as duas colunas, e o resultado e prosa
    # picada: na pagina 96, a lista de rotulos de uma figura (coluna esquerda) entrava
    # no meio das frases do corpo (coluna direita) — "que acompanha a Coeficiente de
    # particao sangue-gas fase inicial da administracao".
    #
    # O certo: um bloco de largura total separa bandas verticais. Dentro de cada banda
    # le-se a coluna esquerda INTEIRA e depois a direita inteira, que e como um humano
    # le uma pagina de duas colunas.
    ordered = []
    limites = [l['yMin'] for l in full] + [float('inf')]
    anterior = float('-inf')
    for i, corte in enumerate(limites):
        na_banda = lambda ls: sorted(
            [l for l in ls if anterior <= l['yMin'] < corte], key=lambda l: l['yMin'])
        ordered.extend(na_banda(left))
        ordered.extend(na_banda(right))
        if i < len(full):
            ordered.append(full[i])
        anterior = corte

    heights = [l['yMax'] - l['yMin'] for l in ordered]
    median_h = statistics.median(heights) if heights else 0
    threshold = 1.35 * median_h if median_h > 0 else 20

    titulos = []
    final_texts = []
    for l in ordered:
        h = l['yMax'] - l['yMin']
        wc = len(l['words'])
        if h >= threshold and wc <= 12:
            titulos.append(l['text'].strip())
        final_texts.append(l['text'].strip())

    # Junta com QUEBRA DE LINHA, nao com espaco: (a) preserva o subtitulo numa
    # linha propria, que e o unico sinal de secao que sobrou neste scan, e (b) deixa
    # a hifenizacao de fim de linha ('recu-' + 'peracao') no formato que
    # normalizar() sabe juntar. Colapsar aqui destroi as duas coisas.
    texto = normalizar('\n'.join(final_texts))

    return {
        'pagina': page_num,
        'secao_num': meta.get('secao_num'),
        'secao': meta.get('secao'),
        'capitulo_num': meta.get('capitulo_num'),
        'capitulo': meta.get('capitulo'),
        'titulos': titulos,
        'texto': texto
    }, meta


def paginas(xml_path, start, end):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    pages = [e for e in root.iter() if e.tag.split('}')[-1] == 'page']
    result = {}
    prev_meta = {'secao_num': None, 'secao': None, 'capitulo_num': None, 'capitulo': None}

    for i, elem in enumerate(pages, 1):
        if elem.tag.split('}')[-1] != 'page':
            continue
        if i < start or i > end:
            continue
        page_res, prev_meta = process_page(i, elem, prev_meta)
        result[i] = page_res
    return result


def main():
    parser = argparse.ArgumentParser(description='Reconstroi ordem de leitura de livro escaneado.')
    parser.add_argument('xml_path', help='Caminho para o XML de coordenadas')
    parser.add_argument('--saida', default='p.jsonl', help='Arquivo de saida JSONL')
    parser.add_argument('--pagina-inicial', type=int, default=1, help='Pagina inicial (1-indexed)')
    parser.add_argument('--pagina-final', type=int, default=None, help='Pagina final (1-indexed)')
    args = parser.parse_args()

    if not os.path.exists(args.xml_path):
        print(f"Erro: arquivo {args.xml_path} nao encontrado.", file=sys.stderr)
        sys.exit(1)

    tree = ET.parse(args.xml_path)
    root = tree.getroot()
    pages = [e for e in root.iter() if e.tag.split('}')[-1] == 'page']
    total_pages = sum(1 for p in pages if p.tag.split('}')[-1] == 'page')
    end_page = args.pagina_final if args.pagina_final else total_pages

    res = paginas(args.xml_path, args.pagina_inicial, end_page)

    with open(args.saida, 'w', encoding='utf-8') as f:
        for p_num in sorted(res.keys()):
            f.write(json.dumps(res[p_num], ensure_ascii=False) + '\n')

    caps = sum(1 for p in res.values() if p['capitulo_num'] is not None)
    titulos = sum(len(p['titulos']) for p in res.values())
    print(f"Paginas processadas: {len(res)}, com capitulo: {caps}, titulos: {titulos}", file=sys.stderr)


if __name__ == '__main__':
    main()