"""Tests for headless report generation + scheduler command."""

from __future__ import annotations

import src.pipeline as pipeline_mod
from src.report_cli import scheduler_command
from src.reports import generate as gen


class _FakePipeline:
    def __init__(self, config, output_dir=None):
        pass

    def run(self, progress=None):
        return object()  # report_figures/builders are mocked, so contents don't matter


def test_generate_report_writes_requested_formats(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "AnalysisPipeline", _FakePipeline)
    monkeypatch.setattr(gen, "report_figures", lambda r: {})
    monkeypatch.setattr(gen, "build_html_report", lambda r, chart_figures=None: "<html>ok</html>")
    monkeypatch.setattr(gen, "build_pdf_report", lambda r, chart_figures=None: b"%PDF-1.4 fake")

    written = gen.generate_report(
        object(), tmp_path, formats=("pdf", "html"), name="My Portfolio",
        stamp="20260101_0900",
    )
    names = sorted(p.name for p in written)
    assert names == ["My_Portfolio_20260101_0900.html", "My_Portfolio_20260101_0900.pdf"]
    assert all(p.exists() for p in written)
    assert (tmp_path / "My_Portfolio_20260101_0900.html").read_text() == "<html>ok</html>"


def test_generate_report_pdf_only(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "AnalysisPipeline", _FakePipeline)
    monkeypatch.setattr(gen, "report_figures", lambda r: {})
    monkeypatch.setattr(gen, "build_pdf_report", lambda r, chart_figures=None: b"%PDF")
    written = gen.generate_report(object(), tmp_path, formats=("pdf",), name="X", stamp="s")
    assert [p.name for p in written] == ["X_s.pdf"]


def test_safe_name():
    assert gen._safe("Classic 60-40") == "Classic_60-40"
    assert gen._safe("a/b c:d") == "a_b_c_d"
    assert gen._safe("") == "portfolio"


def test_scheduler_command_contains_cli_invocation():
    cmd = scheduler_command(["pdf", "html"], "C:/Reports")
    assert "--generate-report" in cmd
    assert "--all" in cmd
    assert "--format pdf,html" in cmd
    assert "C:/Reports" in cmd
