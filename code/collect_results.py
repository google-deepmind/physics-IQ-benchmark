"""Collect PDF figures and LaTeX tables from an output directory.

PDF figures: the ``figures/`` path component is stripped and files are copied
to the mirrored location under <output-dir>.

LaTeX tables: the ``tables/`` path component is stripped and files are copied
to a ``result_tables/`` folder placed next to <output-dir>.

Examples
--------
    uv run code/collect_results.py --input-dir code/output --output-dir ./plots

    code/output/figures/ranking_bump.pdf
        → plots/ranking_bump.pdf

    code/output/original_vs_verified/figures/bootstrap_ranking_scatter.pdf
        → plots/original_vs_verified/bootstrap_ranking_scatter.pdf

    code/output/tables/full_model_table_verified-score.tex
        → result_tables/full_model_table_verified-score.tex
"""

import argparse
import shutil
from pathlib import Path

INPUT_PATH = Path("./output")
OUTPUT_PATH = Path("./plots")


def _copy_files(input_dir: Path, output_dir: Path, glob: str, strip_component: str, label: str) -> None:
    files = sorted(input_dir.rglob(glob))
    if not files:
        print(f"No {glob} files found under {input_dir}")
        return

    copied = 0
    for src in files:
        rel = src.relative_to(input_dir)
        parts = [p for p in rel.parts if p != strip_component]
        dest = output_dir / Path(*parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"  {rel}  →  {dest.relative_to(output_dir)}")
        copied += 1

    print(f"Copied {copied} {label} to {output_dir}\n")


def collect_results(input_dir: Path, figures_dir: Path) -> None:
    tables_dir = figures_dir.parent / "result_tables"
    print(f"--- Figures → {figures_dir} ---")
    _copy_files(input_dir, figures_dir, "*.pdf", "figures", "PDF(s)")
    print(f"--- Tables → {tables_dir} ---")
    _copy_files(input_dir, tables_dir, "*.tex", "tables", ".tex file(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=INPUT_PATH, help="Root output directory to search")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_PATH, help="Destination directory for PDF figures")
    args = parser.parse_args()

    if not args.input_dir.exists():
        parser.error(f"--input-dir does not exist: {args.input_dir}")

    collect_results(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
