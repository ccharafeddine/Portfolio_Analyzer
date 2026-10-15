"""Dense HTML table helpers (sorting hooks, totals, copy-to-Excel text, heatmap)."""

import numpy as np
import pandas as pd

from src.ui.widgets.html_table import (
    Col, fmt_value, heatmap_table_html, numeric_class, rich_table_html, table_to_tsv,
)

DF = pd.DataFrame({"Ticker": ["AAA", "BBB"], "Weight": [0.6, 0.4], "Return": [0.12, -0.05]})
COLS = [Col("Ticker", "Security", "sym"), Col("Weight", "Weight", "pct", 1, bar=True),
        Col("Return", "Return", "spct", 1)]
TOTAL = {"Ticker": "Portfolio", "Weight": 1.0, "Return": 0.052}


def test_numeric_detection():
    assert numeric_class(["1.2%", "−3.4%", "$1,200", "—"])
    assert not numeric_class(["AAPL", "1.0"])
    assert not numeric_class(["—", ""])


def test_formatting_uses_true_minus_and_signs():
    assert fmt_value(-0.05, "spct", 1) == "−5.0%"
    assert fmt_value(0.12, "spct", 1) == "+12.0%"
    assert fmt_value(np.nan, "pct") == "—"


def test_rich_table_structure():
    html = rich_table_html(DF, COLS, total=TOTAL, table_id="h")
    assert "class='tbl sortable'" in html and "id='h'" in html
    assert "<tfoot><tr class='tot'>" in html  # totals stay out of the sorted body
    assert "data-v='0.6'" in html            # numeric sort key
    assert "class='n pos'" in html and "class='n neg'" in html
    assert "copyTable('h'" in html
    # the total row gets no weight bar
    foot = html.split("<tfoot>")[1]
    assert "width:0.0px" in foot


def test_tsv_for_excel():
    tsv = table_to_tsv(DF, COLS, TOTAL).splitlines()
    assert tsv[0] == "Security\tWeight\tReturn"
    assert tsv[-1] == "Portfolio\t100.0%\t+5.2%"


def test_heatmap_tints_by_sign():
    cal = pd.DataFrame({"Jan": [0.05], "Feb": [-0.02], "Year total": [0.029]}, index=[2024])
    html = heatmap_table_html(cal, "#3FB950", "#F47067")
    assert "rgba(63,185,80" in html and "rgba(244,112,103" in html
    assert "+2.9%" in html
