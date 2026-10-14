"""Provenance check for the manuscript.

1. Every result macro used in main.tex / supplement.tex must be defined in numbers.tex
   (generated from tables/*.csv).
2. Every literal number left in the prose (outside macros, labels, citations, equations of the
   method, and LaTeX lengths) must be listed in notes/literal_numbers.csv with its source.
3. Writes notes/provenance.csv: one row per number shown in the text (macro or literal) with
   its value and source.

    python paper/scripts/check_numbers.py
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (ROOT / "numbers.tex").read_text()))
lit_file = ROOT / "notes" / "literal_numbers.csv"
literals = {}
if lit_file.exists():
    for row in csv.DictReader(lit_file.open()):
        literals.setdefault(row["number"], []).append(row["source"])

problems, used, seen_literals = [], set(), {}
for name in ["main.tex", "supplement.tex"]:
    p = ROOT / name
    if not p.exists():
        continue
    text = p.read_text()
    body = text.split(r"\begin{document}", 1)[-1]
    body = re.sub(r"(?<!\\)%.*", "", body)                     # comments
    for m in re.findall(r"\\([A-Za-z]+)", body):
        if m in macros:
            used.add(m)
    # macros referenced but undefined (heuristic: CamelCase names with result-like prefixes)
    for m in set(re.findall(r"\\((?:Sec|Scr|Inc|F(?:Zero|One|Two|Three)|L(?:Prim|Sec|k)|Ratio|Student|Teacher|Rnd|Corr|Compute|Sem|Ext|Vmb)\w*)", body)):
        if m not in macros:
            problems.append(f"{name}: undefined result macro \\{m}")
    clean = re.sub(r"\\(label|ref|cite|eqref|input|includegraphics|url|href|bibliography\w*)\*?(\[[^\]]*\])?\{[^}]*\}", " ", body)
    clean = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", clean, flags=re.S)
    clean = re.sub(r"(\d)\\,(\d)", r"\1\2", clean)          # thin-space thousands separators
    clean = re.sub(r"\\[A-Za-z]+", " ", clean)                 # remaining command names
    clean = re.sub(r"\[[^\]]*(width|height|pt|cm|in)[^\]]*\]", " ", clean)
    for n in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)", clean):
        if not n or n in {"0", "1", "2"} or re.fullmatch(r"(19|20)\d\d", n):
            continue
        seen_literals.setdefault(n, name)

for n, f in sorted(seen_literals.items()):
    if n not in literals:
        problems.append(f"{f}: literal number {n} has no source in notes/literal_numbers.csv")

with (ROOT / "notes" / "provenance.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["kind", "name_or_number", "value", "source"])
    for m in sorted(used):
        w.writerow(["macro", m, macros[m], "numbers.tex <- tables/*.csv (scripts/make_numbers.py)"])
    for n in sorted(seen_literals):
        w.writerow(["literal", n, n, " | ".join(literals.get(n, ["MISSING"]))])

print(f"{len(used)} macros used, {len(seen_literals)} distinct literals, {len(problems)} problems")
for p in problems:
    print("  -", p)
sys.exit(1 if problems else 0)
