"""A QAbstractTableModel that renders a pandas DataFrame in a QTableView.

Supports optional per-column formatters (e.g. percent / dollar) and right-aligns
numeric cells. Mirrors how the Streamlit app displayed dataframes, minus the
web framework.
"""

from __future__ import annotations

from typing import Callable, Optional

import math

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


def _default_format(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        return f"{value:,.4f}" if abs(value) < 1 else f"{value:,.2f}"
    return str(value)


class PandasTableModel(QAbstractTableModel):
    def __init__(
        self,
        df: pd.DataFrame,
        formatters: Optional[dict[str, Callable]] = None,
        index_label: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._df = df if df is not None else pd.DataFrame()
        self._formatters = formatters or {}
        self._index_label = index_label
        self._show_index = not isinstance(self._df.index, pd.RangeIndex)

    # ── Qt model interface ──
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._df.columns) + (1 if self._show_index else 0)

    def _col_offset(self) -> int:
        return 1 if self._show_index else 0

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()

        if self._show_index and col == 0:
            if role == Qt.DisplayRole:
                return str(self._df.index[row])
            if role == Qt.TextAlignmentRole:
                return int(Qt.AlignLeft | Qt.AlignVCenter)
            return None

        df_col = col - self._col_offset()
        col_name = self._df.columns[df_col]
        value = self._df.iat[row, df_col]

        if role == Qt.DisplayRole:
            fmt = self._formatters.get(col_name)
            return fmt(value) if fmt else _default_format(value)
        if role == Qt.TextAlignmentRole:
            align = Qt.AlignLeft if isinstance(value, str) else Qt.AlignRight
            return int(align | Qt.AlignVCenter)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if self._show_index and section == 0:
                return self._index_label or ""
            return str(self._df.columns[section - self._col_offset()])
        return str(section + 1)
