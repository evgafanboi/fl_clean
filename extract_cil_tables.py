#!/usr/bin/env python3
"""
Extract CIL results from Excel and generate per-task LaTeX tables.
Usage: python extract_cil_tables.py
"""

import pandas as pd
from pathlib import Path
import re
import sys

def fmt(value, decimals=4):
    if pd.isna(value) or value is None or (isinstance(value, str) and value.strip() == ""):
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return "-"

def safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8', errors='replace'))

SHEET_MAP = {
    'Finetune': 'FedAvg',
    'Lwf': 'FedAvg-LwF',
    'Ewc': 'FedAvg-EWC',
    'Mas': 'FedAvg-MAS',
    'Icarl': 'FedAvg-iCaRL',
    'Bic': 'FedAvg-BiC',
    'Feat': 'FedAvg-FEAT',
    'Pass': 'FedAvg-PASS',
    'Foster': 'FedAvg-FOSTER',
    'Cbkd': 'FedAvg-CBKD EFCIL',
    'Ours3': 'Đề xuất (single)',
    'Ours3 2': 'Đề xuất (multiple)',
    'Ours 3 2': 'Ours3 2',
}

def normalize_text(value):
    if value is None or pd.isna(value):
        return ''
    return re.sub(r'\s+', ' ', str(value)).strip()

def parse_task_round(task_value, round_value):
    task_text = normalize_text(task_value)
    round_text = normalize_text(round_value)

    task_match = re.search(r'\bT\s*([0-9]+)\b', task_text, flags=re.IGNORECASE)
    round_match = re.search(r'\bR\s*([0-9]+)\b', round_text, flags=re.IGNORECASE)

    if not task_match or not round_match:
        return None, None

    return int(task_match.group(1)), int(round_match.group(1))

REG_METHODS = ['FedAvg-EWC', 'FedAvg-MAS', 'FedAvg-LwF', 'FedAvg-PASS', 'FedAvg-CBKD EFCIL']
REPLAY_METHODS = ['FedAvg-iCaRL', 'FedAvg-BiC']
DYNAMIC_METHODS = ['FedAvg-FOSTER']
FEDERATED_METHODS = ['FedAvg-FEAT']
OURS = ['Đề xuất (single)', 'Đề xuất (multiple)']

def parse_sheet(df):
    result = {}
    if df is None or df.empty:
        return result

    for i in range(1, len(df)):
        row = df.iloc[i]
        task_id, round_num = parse_task_round(row.iloc[0], row.iloc[1])
        if task_id is None or round_num is None:
            continue
        metrics = {
            'acc': row.iloc[2],
            'f1_mac': row.iloc[4],
            'f1_mic': row.iloc[5],
            'f1_w': row.iloc[6],
            'prec_mac': row.iloc[7],
            'prec_mic': row.iloc[8],
            'prec_w': row.iloc[9],
            'rec_mac': row.iloc[10],
            'rec_mic': row.iloc[11],
            'rec_w': row.iloc[12],
        }
        result.setdefault(task_id, {})[round_num] = metrics
    return result

def row_line(name, m, bold=False):
    cells = [fmt(m['acc']), fmt(m['prec_mac']), fmt(m['rec_mac']), fmt(m['f1_mac']),
             fmt(m['prec_w']), fmt(m['rec_w']), fmt(m['f1_w']),
             fmt(m['prec_mic']), fmt(m['rec_mic']), fmt(m['f1_mic'])]
    if bold:
        cells = [f"\\textbf{{{c}}}" for c in cells]
    return f"        {name} & " + " & ".join(cells) + " \\\\"

def _color_cell(text, rank):
    if rank == 1:
        return f"\\cellcolor{{green!60!black}}{text}"
    if rank == 2:
        return f"\\cellcolor{{green!90!black}}{text}"
    if rank == 3:
        return f"\\cellcolor{{green!30}}{text}"
    return text

def rank_metrics(methods_data):
    metric_keys = ['acc', 'prec_mac', 'rec_mac', 'f1_mac', 'prec_w', 'rec_w', 'f1_w', 'prec_mic', 'rec_mic', 'f1_mic']
    ranks = {key: {} for key in metric_keys}
    for key in metric_keys:
        vals = []
        for method_name, metrics in methods_data.items():
            value = metrics.get(key)
            if pd.isna(value) or value is None:
                continue
            try:
                vals.append((method_name, float(value)))
            except (ValueError, TypeError):
                continue
        vals.sort(key=lambda x: x[1], reverse=True)
        for idx, (method_name, _) in enumerate(vals, start=1):
            ranks[key][method_name] = idx
    return ranks

def row_line_ranked(name, m, ranks):
    keys = ['acc', 'prec_mac', 'rec_mac', 'f1_mac', 'prec_w', 'rec_w', 'f1_w', 'prec_mic', 'rec_mic', 'f1_mic']
    cells = []
    for key in keys:
        cell = fmt(m[key])
        cell = _color_cell(cell, ranks.get(key, {}).get(name))
        cells.append(cell)
    return f"        {name} & " + " & ".join(cells) + " \\\\"

def generate_task_table(task_id, methods_data, output_file=None):
    ranks = rank_metrics(methods_data)
    lines = [
        "\\begin{table}[H]",
        "    \\centering",
        f"    \\caption{{Kết quả thí nghiệm học tiệm tiến - Task {task_id}.}}",
        f"    \\label{{tab:cil_results_task{task_id}}}",
        "    \\resizebox{\\linewidth}{!}{",
        "    \\begin{tabular}{lcccccccccc}",
        "        \\hline",
        "        \\multirow{2}{*}{Thuật toán} & \\multirow{2}{*}{Accuracy} & \\multicolumn{3}{c}{Macro} & \\multicolumn{3}{c}{Weighted} & \\multicolumn{3}{c}{Micro} \\\\",
        "        \\cline{3-5} \\cline{6-8} \\cline{9-11}",
        "        & & Prec & Rec & F1 & Prec & Rec & F1 & Prec & Rec & F1 \\\\",
        "        \\hline",
    ]
    if 'FedAvg' in methods_data:
        lines.append(row_line_ranked('FedAvg', methods_data['FedAvg'], ranks))
        lines.append("        \\hline")
    reg_found = [m for m in REG_METHODS if m in methods_data]
    if reg_found:
        lines.append("        \\multicolumn{11}{l}{\\textit{Regularization-based}} \\\\")
        for m in reg_found:
            lines.append(row_line_ranked(m, methods_data[m], ranks))
        lines.append("        \\hline")
    replay_found = [m for m in REPLAY_METHODS if m in methods_data]
    if replay_found:
        lines.append("        \\multicolumn{11}{l}{\\textit{Replay-based}} \\\\")
        for m in replay_found:
            lines.append(row_line_ranked(m, methods_data[m], ranks))
        lines.append("        \\hline")
    dyn_found = [m for m in DYNAMIC_METHODS if m in methods_data]
    if dyn_found:
        lines.append("        \\multicolumn{11}{l}{\\textit{Dynamic architecture-based}} \\\\")
        for m in dyn_found:
            lines.append(row_line_ranked(m, methods_data[m], ranks))
        lines.append("        \\hline")
    fed_found = [m for m in FEDERATED_METHODS if m in methods_data]
    if fed_found:
        lines.append("        \\multicolumn{11}{l}{\\textit{Federated CIL}} \\\\")
        for m in fed_found:
            lines.append(row_line_ranked(m, methods_data[m], ranks))
        lines.append("        \\hline")
    ours_found = [m for m in OURS if m in methods_data]
    if ours_found:
        lines.append("        \\multicolumn{11}{l}{\\textbf{Đề xuất}} \\\\")
        for m in ours_found:
            lines.append(row_line_ranked(m, methods_data[m], ranks))
        lines.append("        \\hline")
    lines.extend(["    \\end{tabular}}", "\\end{table}", ""])
    content = "\n".join(lines)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  -> {output_file}")
    return content

def main():
    excel_path = Path("results_cil/cil_comparison.xlsx")
    excel_data = pd.read_excel(excel_path, sheet_name=None)
    print(f"Found {len(excel_data)} sheets")
    all_methods = {}
    for sheet_name, df in excel_data.items():
        if sheet_name not in SHEET_MAP:
            continue
        display_name = SHEET_MAP[sheet_name]
        parsed = parse_sheet(df)
        if not parsed:
            safe_print(f"  {sheet_name} -> {display_name}: no task/round rows detected")
            continue
        all_methods[display_name] = parsed
        safe_print(f"  {sheet_name} -> {display_name}: tasks {sorted(parsed.keys())}")
    if not all_methods:
        raise RuntimeError("No usable method sheets were parsed from the Excel workbook.")
    all_tables = []
    for task_id in range(7):
        safe_print(f"\n=== Task {task_id} ===")
        methods_data = {}
        for method_name, tasks in all_methods.items():
            if task_id in tasks:
                best_round = sorted(tasks[task_id].keys(), reverse=True)[0]
                methods_data[method_name] = tasks[task_id][best_round]
                safe_print(f"  {method_name}: R{best_round} acc={fmt(tasks[task_id][best_round]['acc'])}")
        if not methods_data:
            safe_print(f"  No data for Task {task_id}")
            continue
        latex = generate_task_table(task_id, methods_data, f"cil_task{task_id}_table.tex")
        all_tables.append(latex)
    with open("cil_all_tables.tex", 'w', encoding='utf-8') as f:
        f.write("\n".join(all_tables))
    safe_print(f"\n=== Done: {len(all_tables)} tables + cil_all_tables.tex ===")

if __name__ == "__main__":
    main()
