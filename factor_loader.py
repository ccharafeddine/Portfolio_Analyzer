import io
import zipfile
import requests
import pandas as pd

from data_io import download_prices, monthly_returns_from_prices


# -------------------------------------------------------------
# URLs for Kenneth French factor datasets
# -------------------------------------------------------------
FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
CARHART_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"


# -------------------------------------------------------------
# Generic FF3 / FF5 parser
# -------------------------------------------------------------
def _extract_monthly_table_from_text(text: str) -> pd.DataFrame:
    """
    Generic parser for Kenneth French multi-factor CSVs (FF3, FF5).

    Steps:
    - Find the header line (contains 'Mkt-RF').
    - Take all lines until a blank or 'Annual' line.
    - Parse that slice with pandas.
    - Standardize first column as 'Date', parse YYYYMM -> datetime index.
    - Convert percentage columns to decimal returns.
    """
    lines = text.splitlines()

    # 1) Locate header line
    header_idx = None
    for i, line in enumerate(lines):
        if "Mkt-RF" in line:
            header_idx = i
            break

    if header_idx is None:
        raise RuntimeError("Could not locate FF3/FF5 header row in Kenneth French CSV")

    # 2) Locate end of monthly table (blank line or 'Annual' marker)
    end_idx = len(lines)
    for j in range(header_idx + 1, len(lines)):
        stripped = lines[j].strip()
        if not stripped:
            end_idx = j
            break
        if "Annual" in stripped:
            end_idx = j
            break

    table_lines = lines[header_idx:end_idx]
    csv_str = "\n".join(table_lines)

    df = pd.read_csv(io.StringIO(csv_str))

    # Standardize first column as Date
    df = df.rename(columns={df.columns[0]: "Date"})
    date_num = pd.to_numeric(df["Date"], errors="coerce")

    # Keep only rows that look like YYYYMM codes
    mask_monthly = date_num >= 100000
    df = df[mask_monthly].copy()

    date_str = date_num[mask_monthly].astype(int).astype(str)
    df["Date"] = pd.to_datetime(date_str, format="%Y%m")
    df = df.set_index("Date")

    # Convert factor values from percentages to decimals
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    df = df.dropna(how="all")
    return df


def _load_ff_zip(url: str) -> pd.DataFrame:
    """
    Download and parse a Kenneth French FF3/FF5 ZIP into a clean
    monthly factor DataFrame.
    """
    resp = requests.get(url)
    resp.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    csv_name = [name for name in zf.namelist() if name.endswith(".csv")][0]

    with zf.open(csv_name) as fh:
        text = fh.read().decode("latin1")

    return _extract_monthly_table_from_text(text)


# -------------------------------------------------------------
# Dedicated Carhart momentum parser
# -------------------------------------------------------------
def _load_momentum_zip(url: str) -> pd.DataFrame:
    """
    Dedicated parser for the Carhart momentum CSV.

    We ignore the original header and comments, and construct our own
    mini-CSV with:

        Date,MOM
        192701, 0.35
        ...

    Only lines that look like 'YYYYMM,<value>' are kept.
    """
    resp = requests.get(url)
    resp.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    csv_name = [name for name in zf.namelist() if name.endswith(".csv")][0]

    with zf.open(csv_name) as fh:
        text = fh.read().decode("latin1")

    lines = text.splitlines()
    data_rows = []

    for line in lines:
        s = line.strip()
        if not s:
            continue

        parts = [p.strip() for p in s.split(",")]
        if len(parts) < 2:
            continue

        date_part = parts[0]
        # Keep only rows whose first field is a numeric YYYYMM code
        if not date_part.isdigit():
            continue
        if len(date_part) < 6:  # e.g., '1927' -> annual row, skip
            continue

        value_part = parts[1]
        data_rows.append(f"{date_part},{value_part}")

    if not data_rows:
        raise RuntimeError(
            "Could not extract any monthly momentum rows from Carhart CSV."
        )

    csv_str = "Date,MOM\n" + "\n".join(data_rows)
    df = pd.read_csv(io.StringIO(csv_str))

    # Parse Date (YYYYMM) -> datetime index
    date_num = pd.to_numeric(df["Date"], errors="coerce")
    date_str = date_num.astype(int).astype(str)
    df["Date"] = pd.to_datetime(date_str, format="%Y%m")
    df = df.set_index("Date")

    # Convert MOM from % to decimal
    df["MOM"] = pd.to_numeric(df["MOM"], errors="coerce") / 100.0
    df = df.dropna(how="all")
    return df


# -------------------------------------------------------------
# Quality / Low-Vol proxy using QUAL & SPLV
# -------------------------------------------------------------
def _load_quality_lowvol(start, end) -> pd.DataFrame:
    """
    Quality & Low-Vol proxies using QUAL and SPLV monthly returns.
    Returns columns:
        QUAL, SPLV, RF=0 (no risk-free needed for regression)
    """
    tickers = ["QUAL", "SPLV"]
    prices = download_prices(tickers, start=start, end=end, interval="1d")
    rets = monthly_returns_from_prices(prices)

    rets["RF"] = 0.0
    return rets


# -------------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------------
def load_factors(model: str, start, end) -> pd.DataFrame:
    """
    model ∈ {"ff3", "carhart4", "ff5", "quality_lowvol"}

    Returns a monthly factor dataframe with a DatetimeIndex,
    trimmed to the requested [start, end] window.
    """
    model = model.lower()

    if model == "ff3":
        df = _load_ff_zip(FF3_URL)[["Mkt-RF", "SMB", "HML", "RF"]]

    elif model == "carhart4":
        # Base FF3 factors
        base = _load_ff_zip(FF3_URL)[["Mkt-RF", "SMB", "HML", "RF"]]

        # Dedicated momentum parser
        mom_raw = _load_momentum_zip(CARHART_URL)
        # Already has a 'MOM' column
        mom = mom_raw[["MOM"]]

        df = base.join(mom, how="inner")

    elif model == "ff5":
        df = _load_ff_zip(FF5_URL)[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]]

    elif model == "quality_lowvol":
        df = _load_quality_lowvol(start, end)
        df = df.rename(columns={"QUAL": "QUAL", "SPLV": "SPLV"})

    else:
        raise ValueError(f"Unknown factor model: {model}")

    # Trim to requested date range
    df = df.loc[
        (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))
    ]

    return df
