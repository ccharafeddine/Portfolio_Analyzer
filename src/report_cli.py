"""Headless report generation for automation / OS schedulers.

Runs the analysis pipeline for one or all saved portfolios and writes HTML/PDF reports —
no GUI. Invoked directly, via ``python -m src.report_cli``, or through the installed app
as ``PortfolioAnalyzer.exe --generate-report ...`` (see main_desktop.py).

Examples:
  python -m src.report_cli --portfolio "C:/.../Classic 60-40.json" --format pdf,html
  python -m src.report_cli --all --out "C:/Reports" --format pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def scheduler_command(formats, out_dir: str) -> str:
    """The command an OS scheduler (Task Scheduler / cron) runs for headless reports."""
    fmt = ",".join(formats) or "pdf"
    py = sys.executable
    if getattr(sys, "frozen", False):
        return f'"{py}" --generate-report --all --format {fmt} --out "{out_dir}"'
    root = Path(__file__).resolve().parents[1]  # src/ -> repo root
    return (f'"{py}" "{root / "main_desktop.py"}" --generate-report --all '
            f'--format {fmt} --out "{out_dir}"')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="report_cli",
        description="Generate Portfolio Analyzer reports without the GUI.",
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--portfolio", help="Path to a saved PortfolioConfig JSON.")
    group.add_argument("--all", action="store_true",
                       help="Every saved portfolio in the portfolios directory.")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: the Documents export folder).")
    ap.add_argument("--format", default="pdf",
                    help="Comma-separated: pdf,html (default: pdf).")
    args = ap.parse_args(argv)

    from src.config.models import PortfolioConfig
    from src.reports.generate import generate_report
    from src.ui import paths

    formats = tuple(f.strip().lower() for f in args.format.split(",") if f.strip())
    if not formats:
        formats = ("pdf",)
    out = Path(args.out) if args.out else paths.documents_export_dir()

    if args.all:
        pdir = paths.portfolios_dir()
        items = [(p.stem, p) for p in sorted(pdir.glob("*.json"))]
        if not items:
            print(f"No saved portfolios in {pdir}", file=sys.stderr)
            return 1
    else:
        p = Path(args.portfolio)
        if not p.exists():
            print(f"Portfolio not found: {p}", file=sys.stderr)
            return 1
        items = [(p.stem, p)]

    rc = 0
    for name, path in items:
        try:
            cfg = PortfolioConfig.load(str(path))
            for w in generate_report(cfg, out, formats=formats, name=name):
                print(f"wrote {w}")
        except Exception as e:  # keep going through the rest
            print(f"FAILED {name}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
