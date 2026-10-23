from __future__ import annotations

from .data import GROUPS, MEMORY_GB, MODEL_ROWS, SHOTS, TRAINING, fmt


def _bold(text, flag):
    return f'\\textbf{{{text}}}' if flag else text


def _tex(text):
    return text.replace('_', '\\_')


def models_table(models):
    lines = [r'\begin{table}[t]', r'\centering',
             r'\setlength{\tabcolsep}{5pt}', r'\begin{tabular}{llrrll}', r'\toprule',
             r'Model & Size & Parameters & Memory & Quant. & Thinking \\', r'\midrule']
    previous = None
    for ollama, name, group, thinking in MODEL_ROWS:
        if previous and group != previous:
            lines.append(r'\addlinespace')
        details = models[ollama]['details']
        lines.append(f"{name} & {group} & {details['parameter_size']} & {MEMORY_GB[ollama]:.2f} & "
                     f"{_tex(details['quantization_level'])} & {thinking} \\\\")
        previous = group
    lines += [r'\botrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def classification_table(llms, baselines):
    best = {}
    for row in llms:
        for s in SHOTS:
            value = round(row['values'][s][0], 9)
            if value > best.get(row['group'], (-1,))[0]:
                best[row['group']] = (value, row['key'], s)
    lines = [r'\begin{table}', r'\centering',
             r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lcccccc@{}}', r'\toprule',
             r'\multicolumn{7}{@{}l}{\textbf{A. Large language models}} \\', r'\midrule',
             r' & \multicolumn{2}{c}{Zero-shot} & \multicolumn{2}{c}{One-shot} & \multicolumn{2}{c}{Three-shot} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}',
             r'Model & Accuracy & Macro-F1 & Accuracy & Macro-F1 & Accuracy & Macro-F1 \\', r'\midrule']
    previous = None
    for row in llms:
        if previous and row['group'] != previous:
            lines.append(r'\addlinespace')
        cells = []
        for s in SHOTS:
            flag = best[row['group']][1:] == (row['key'], s)
            cells += [_bold(fmt(row['values'][s][0]), flag), _bold(fmt(row['values'][s][1]), flag)]
        lines.append(f"{row['name']} & " + ' & '.join(cells) + r' \\')
        previous = row['group']
    best_baseline = max(((round(b['values'][n][0], 9), b['key'], n) for b in baselines for _, n in TRAINING))
    lines += [r'\midrule', r'\multicolumn{7}{@{}l}{\textbf{B. NLP baselines}} \\', r'\midrule',
              r'\multicolumn{2}{@{}l}{Training and validation reports} & ' + ' & '.join(str(n) for _, n in TRAINING) + r' \\',
              r'\midrule']
    for index, b in enumerate(baselines):
        if index:
            lines.append(r'\addlinespace')
        for position, label in enumerate(('Accuracy', 'Macro-F1')):
            cells = [_bold(fmt(b['values'][n][position]), best_baseline[1:] == (b['key'], n)) for _, n in TRAINING]
            lines.append(f"{b['name'] if position == 0 else ''} & {label} & " + ' & '.join(cells) + r' \\')
    lines += [r'\botrule', r'\end{tabular*}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def extraction_table(llms, baselines):
    best = {}
    for row in llms:
        if round(row['best'], 9) > best.get(row['group'], (-1,))[0]:
            best[row['group']] = (round(row['best'], 9), row['key'])
    best_baseline = max(baselines, key=lambda b: b['best'])['key']
    col = r'>{\centering\arraybackslash}p{\dimexpr%s\textwidth-%s\tabcolsep\relax}'
    lines = [r'\begin{table}', r'\centering',
             r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{@{}p{\dimexpr0.30\textwidth-\tabcolsep\relax}*{3}{' + col % ('0.10667', '2') + r'}'
             r'p{\dimexpr0.03\textwidth-2\tabcolsep\relax}*{3}{' + col % ('0.0875', '2') + r'}' + col % ('0.0875', '1') + r'@{}}',
             r'\toprule', r'\multicolumn{9}{@{}l}{\textbf{A. Large language models}} \\', r'\midrule',
             r' & \multicolumn{3}{c}{Overall} & & \multicolumn{4}{c}{By difficulty} \\',
             r'\cmidrule(lr){2-4}\cmidrule(l){6-9}', r'Model & 0-shot & 1-shot & 3-shot & & D1 & D2 & D3 & D4 \\', r'\midrule']
    previous = None
    for row in llms:
        if previous and row['group'] != previous:
            lines.append(r'\addlinespace')
        flag = best[row['group']][1] == row['key']
        cells = [_bold(fmt(row['overall'][s]), flag and s == row['best_shots']) for s in SHOTS]
        lines.append(f"{row['name']} & " + ' & '.join(cells) + ' & & ' + ' & '.join(fmt(v) for v in row['levels'].values()) + r' \\')
        previous = row['group']
    lines += [r'\bottomrule', r'\end{tabular}', r'\par\medskip',
              r'\begin{tabular}{@{}p{\dimexpr0.30\textwidth-\tabcolsep\relax}*{5}{' + col % ('0.064', '2') + r'}'
              r'p{\dimexpr0.03\textwidth-2\tabcolsep\relax}*{3}{' + col % ('0.0875', '2') + r'}' + col % ('0.0875', '1') + r'@{}}',
              r'\toprule', r'\multicolumn{11}{@{}l}{\textbf{B. NLP baselines}} \\', r'\midrule',
              r' & \multicolumn{5}{c}{Training reports} & & \multicolumn{4}{c}{By difficulty} \\',
              r'\cmidrule(lr){2-6}\cmidrule(l){8-11}', r'Method & ' + ' & '.join(str(n) for _, n in TRAINING) + r' & & D1 & D2 & D3 & D4 \\',
              r'\midrule']
    for b in baselines:
        cells = [_bold(fmt(b['training'][n]), b['key'] == best_baseline and n == 600) for _, n in TRAINING]
        lines.append(f"{b['name']} & " + ' & '.join(cells) + ' & & ' + ' & '.join(fmt(v) for v in b['levels'].values()) + r' \\')
    lines += [r'\botrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def pipeline_table(rows):
    best = max(range(len(rows)), key=lambda i: round(rows[i]['pipeline'], 9))
    lines = [r'\begin{table}[h]', r'\centering', r'\small',
             r'\setlength{\tabcolsep}{2pt}', r'\renewcommand{\arraystretch}{1.13}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllccc@{}}', r'\toprule',
             r' & \multicolumn{2}{c}{Methods} & \multicolumn{3}{c}{Accuracy} \\', r'\cmidrule(lr){2-3}\cmidrule(l){4-6}',
             r'Size group & Classification & Extraction \& matching & Classif. & Matching & Pipeline \\', r'\midrule']
    for index, row in enumerate(rows):
        if row['group'] not in GROUPS:
            lines.append(r'\addlinespace[2pt]')
        pipeline = fmt(row['pipeline'])
        pipeline = f'\\mathbf{{{pipeline}}}' if index == best else pipeline
        lines.append(f"{row['group']} & {row['classification']} & {row['extraction']} & ${fmt(row['cls'])}$ & "
                     f"${fmt(row['match'])}$ & ${pipeline}$ \\\\")
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    return '\n'.join(lines) + '\n'
