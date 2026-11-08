import csv
import io
from collections import Counter

import pandas as pd

MIN_ROWS = 30
MAX_COLUMNS = 500


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")  # also strips the BOM Excel adds
    except UnicodeDecodeError:
        return raw.decode("latin-1")  # never fails; good enough fallback


def read_csv(raw: bytes) -> pd.DataFrame:
    if not raw or not raw.strip():
        raise ValueError("File is empty")

    text = _decode(raw)

    try:
        header = [h.strip() for h in next(csv.reader(io.StringIO(text)))]
    except (StopIteration, csv.Error):
        raise ValueError("Could not read a header row from the file")

    if len(header) > MAX_COLUMNS:
        raise ValueError(f"Too many columns ({len(header)}); the limit is {MAX_COLUMNS}")
    if len(header) < 2:
        raise ValueError("File needs at least 2 columns (is it comma-separated?)")

    blanks = [i + 1 for i, h in enumerate(header) if not h]
    if blanks:
        raise ValueError(f"Column(s) at position(s) {blanks} have no name")

    dupes = sorted(h for h, n in Counter(header).items() if n > 1)
    if dupes:
        raise ValueError(f"Duplicate column names: {dupes}")

    try:
        df = pd.read_csv(io.StringIO(text), low_memory=False, skipinitialspace=True)
    except pd.errors.EmptyDataError:
        raise ValueError("File is empty")
    except (pd.errors.ParserError, ValueError) as e:
        raise ValueError(f"Could not parse CSV: {e}")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)

    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Dataset has {len(df)} rows; at least {MIN_ROWS} are needed "
            "for drift statistics to mean anything"
        )
    return df