"""Paper output helpers for experiments/: LaTeX tables and the output folder.

Tables and PDF figures go to <results>/all/<config>/paper/.
"""
from __future__ import annotations

from pathlib import Path

from confledger.utility.config import results_dir


def paper_dir(cfg: dict) -> Path:
    p = results_dir(cfg) / "paper"
    p.mkdir(parents=True, exist_ok=True)
    return p


def f(x, d: int = 3) -> str:
    return "--" if x is None or x != x else f"{x:.{d}f}"


def latex_table(header: list[str], rows: list[list[str]], caption: str, label: str,
                path: Path, align: str | None = None) -> None:
    align = align or "l" + "r" * (len(header) - 1)
    esc = lambda c: c if ("\\" in c or "$" in c) else c.replace("_", "\\_")
    body = "\n".join(" & ".join(esc(str(c)) for c in r) + r" \\" for r in rows)
    path.write_text(
        "\\begin{table}[t]\n\\centering\n\\small\n"
        f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{' & '.join(header)} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")
    print(f"wrote {path}")
