import numpy as np
import pandas as pd

from driftwatch.db.models import ColumnDtype as D
from driftwatch.services.schema_infer import infer_schema

N = 200
rng = np.random.default_rng(0)


def infer(**columns) -> dict:
    df = pd.DataFrame(columns)
    return {c.name: c for c in infer_schema(df)}


# ---- ids ----
def test_unique_integers_with_id_name_are_id():
    out = infer(user_id=rng.permutation(np.arange(1000, 1000 + N)))
    assert out["user_id"].dtype == D.id


def test_monotonic_unique_integers_are_id_even_without_a_name_hint():
    assert infer(row=np.arange(1, N + 1))["row"].dtype == D.id


def test_unique_integers_without_hint_are_numeric_not_id():
    # e.g. whole-dollar prices: nearly unique, but not an identifier
    out = infer(price=rng.permutation(np.arange(100, 100 + N) * 10))
    assert out["price"].dtype == D.numeric


def test_unique_short_strings_without_spaces_are_id():
    assert infer(code=[f"ab{i:05d}" for i in range(N)])["code"].dtype == D.id


def test_unique_strings_with_id_name_are_id_even_with_spaces():
    assert infer(customer_id=[f"A {i}" for i in range(N)])["customer_id"].dtype == D.id


def test_unique_free_text_is_text():
    out = infer(feedback=[f"Customer feedback about product {i} and service" for i in range(N)])
    assert out["feedback"].dtype == D.text


# ---- ignore ----
def test_constant_column_is_ignored():
    out = infer(x=["a"] * N)
    assert out["x"].dtype == D.ignore and "constant" in out["x"].reason


def test_all_missing_column_is_ignored():
    out = infer(x=[np.nan] * N)
    assert out["x"].dtype == D.ignore and "missing" in out["x"].reason


# ---- numeric vs categorical ----
def test_floats_are_numeric():
    assert infer(x=rng.normal(size=N))["x"].dtype == D.numeric


def test_numeric_with_missing_values_is_still_numeric():
    x = rng.normal(size=N)
    x[:20] = np.nan
    assert infer(x=x)["x"].dtype == D.numeric


def test_few_distinct_integers_are_categorical():
    assert infer(x=rng.integers(0, 3, N))["x"].dtype == D.categorical


def test_categorical_threshold_boundary():
    fifteen = np.tile(np.arange(15), N // 15 + 1)[:N]
    sixteen = np.tile(np.arange(16), N // 16 + 1)[:N]
    out = infer(a=fifteen, b=sixteen)
    assert out["a"].dtype == D.categorical
    assert out["b"].dtype == D.numeric


def test_whole_number_floats_with_nans_are_categorical():
    # pandas reads an int column with blanks as float; it should still count as integer-valued
    x = np.tile([1.0, 2.0, 3.0, np.nan], N // 4)
    assert infer(x=x)["x"].dtype == D.categorical


def test_booleans_are_categorical():
    out = infer(flag=rng.integers(0, 2, N).astype(bool))
    assert out["flag"].dtype == D.categorical and "boolean" in out["flag"].reason


def test_low_cardinality_strings_are_categorical():
    assert infer(color=rng.choice(["red", "green", "blue"], N))["color"].dtype == D.categorical


# ---- datetime ----
def test_datetime_dtype():
    assert infer(ts=pd.date_range("2024-01-01", periods=N, freq="D"))["ts"].dtype == D.datetime


def test_iso_date_strings_are_datetime():
    dates = pd.date_range("2024-01-01", periods=N, freq="D").strftime("%Y-%m-%d")
    assert infer(day=list(dates))["day"].dtype == D.datetime


def test_dashed_codes_are_not_mistaken_for_dates():
    out = infer(sku=[f"SKU-{i % 7}" for i in range(N)])
    assert out["sku"].dtype == D.categorical


# ---- text ----
def test_repeated_long_strings_are_text():
    out = infer(note=[f"This is a fairly long free text comment, number {i % 5}" for i in range(N)])
    assert out["note"].dtype == D.text


# ---- general ----
def test_mixed_types_do_not_crash():
    out = infer(x=[1, "a", 2.5, None] * (N // 4))
    assert out["x"].dtype in (D.categorical, D.text)


def test_every_column_has_a_reason_and_order_is_preserved():
    df = pd.DataFrame(
        {"b": rng.normal(size=N), "a": ["x"] * N, "c": rng.integers(0, 3, N)}
    )
    result = infer_schema(df)
    assert [c.name for c in result] == ["b", "a", "c"]
    assert all(c.reason for c in result)