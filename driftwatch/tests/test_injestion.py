import pytest

from driftwatch.services.ingestion import MIN_ROWS, read_csv


def csv_bytes(header="a,b,c", n=MIN_ROWS, extra_rows=""):
    rows = "\n".join(f"{i},{i * 2},x{i % 3}" for i in range(n))
    return f"{header}\n{rows}\n{extra_rows}".encode()


def test_valid_csv():
    df = read_csv(csv_bytes())
    assert list(df.columns) == ["a", "b", "c"]
    assert len(df) == MIN_ROWS


def test_header_whitespace_is_stripped():
    df = read_csv(csv_bytes(header=" a , b , c "))
    assert list(df.columns) == ["a", "b", "c"]


def test_utf8_bom_is_stripped():
    df = read_csv(b"\xef\xbb\xbf" + csv_bytes())
    assert list(df.columns)[0] == "a"


def test_latin1_fallback():
    rows = "\n".join(f"{i},caf\xe9" for i in range(MIN_ROWS))
    df = read_csv(f"id,name\n{rows}".encode("latin-1"))
    assert df["name"].iloc[0] == "café"


@pytest.mark.parametrize("raw", [b"", b"   \n  \n"])
def test_empty_file(raw):
    with pytest.raises(ValueError, match="empty"):
        read_csv(raw)


def test_duplicate_column_names():
    with pytest.raises(ValueError, match="Duplicate"):
        read_csv(csv_bytes(header="a,b,a"))


def test_blank_column_name():
    with pytest.raises(ValueError, match="no name"):
        read_csv(csv_bytes(header="a,,c"))


def test_single_column():
    with pytest.raises(ValueError, match="at least 2 columns"):
        read_csv(b"a\n" + b"\n".join(str(i).encode() for i in range(MIN_ROWS)))


def test_header_only():
    with pytest.raises(ValueError, match="0 rows"):
        read_csv(b"a,b,c\n")


def test_row_count_boundary():
    with pytest.raises(ValueError, match="rows"):
        read_csv(csv_bytes(n=MIN_ROWS - 1))
    assert len(read_csv(csv_bytes(n=MIN_ROWS))) == MIN_ROWS


def test_too_many_columns():
    header = ",".join(f"c{i}" for i in range(501))
    with pytest.raises(ValueError, match="Too many columns"):
        read_csv((header + "\n" + ",".join("1" * 501)).encode())


def test_fully_empty_rows_are_dropped():
    df = read_csv(csv_bytes(extra_rows=",,\n,,\n"))
    assert len(df) == MIN_ROWS