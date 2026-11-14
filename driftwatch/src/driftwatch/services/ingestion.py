import csv
import io
from collections import Counter

import pandas as pd

MIN_ROWS = 30
MAX_COLUMNS = 500


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")  # Also strips the BOM Excel adds.
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def read_csv(raw: bytes, *, min_rows: int = MIN_ROWS) -> pd.DataFrame:
    if not raw or not raw.strip():
        raise ValueError("File is empty")

    text = _decode(raw)

    try:
        header = [name.strip() for name in next(csv.reader(io.StringIO(text)))]
    except (StopIteration, csv.Error):
        raise ValueError("Could not read a header row from the file")

    if len(header) > MAX_COLUMNS:
        raise ValueError(
            f"Too many columns ({len(header)}); the limit is {MAX_COLUMNS}"
        )
    if len(header) < 2:
        raise ValueError("File needs at least 2 columns (is it comma-separated?)")

    blanks = [index + 1 for index, name in enumerate(header) if not name]
    if blanks:
        raise ValueError(f"Column(s) at position(s) {blanks} have no name")

    duplicates = sorted(
        name for name, count in Counter(header).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate column names: {duplicates}")

    try:
        df = pd.read_csv(
            io.StringIO(text),
            low_memory=False,
            skipinitialspace=True,
        )
    except pd.errors.EmptyDataError:
        raise ValueError("File is empty")
    except (pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"Could not parse CSV: {error}")

    df.columns = [str(name).strip() for name in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)

    if len(df) < min_rows:
        raise ValueError(
            f"Dataset has {len(df)} rows; at least {min_rows} are needed "
            "for drift statistics to mean anything"
        )

    return df