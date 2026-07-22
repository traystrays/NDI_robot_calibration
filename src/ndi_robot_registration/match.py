import pandas as pd


def match_by_timestamp(
    data_1: pd.DataFrame,
    data_2: pd.DataFrame,
    tolerance: str | pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Match two transform DataFrames using the nearest timestamps.

    Each input must contain exactly two columns: ``timestamp`` and
    ``Transforms``. The timestamps must be parseable by pandas. The timestamps
    from ``data_1`` define the rows in the returned DataFrame. 

    Convert time stamp to nanoseconds UTC.

    Args:
        data_1: First DataFrame containing ``timestamp`` and ``Transforms``.
        data_2: Second DataFrame containing ``timestamp`` and ``Transforms``.
        tolerance: Maximum permitted difference between matched timestamps,
            such as ``"25ms"``. If omitted, the nearest timestamp is always
            selected.

    Returns:
        A DataFrame with ``timestamp`` in UTC, ``Transform 1``, and ``Transform 2``.
        ``Transform 2`` is missing when no timestamp falls within tolerance.
    """
    required_columns = {"timestamp", "Transforms"}

    for name, data in (("data_1", data_1), ("data_2", data_2)):
        if set(data.columns) != required_columns:
            raise ValueError(
                f"{name} must contain exactly the columns "
                f"{sorted(required_columns)}; got {list(data.columns)}"
            )

    left = data_1.rename(columns={"Transforms": "Transform 1"}).copy()
    right = data_2.rename(columns={"Transforms": "Transform 2"}).copy()

    # Convert to UTC 
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce").dt.as_unit("us").dt.tz_convert("America/Vancouver")
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True, errors="coerce").dt.as_unit("us").dt.tz_convert("America/Vancouver")

    if left["timestamp"].isna().any() or right["timestamp"].isna().any():
        raise ValueError("All timestamps must be valid datetimes.")

    parsed_tolerance = pd.Timedelta(tolerance) if tolerance is not None else None
    if parsed_tolerance is not None and parsed_tolerance < pd.Timedelta(0):
        raise ValueError("tolerance must not be negative.")

    left = left.sort_values("timestamp").reset_index(drop=True)
    right = right.sort_values("timestamp").reset_index(drop=True)

    return pd.merge_asof(
        left,
        right,
        on="timestamp",
        direction="nearest",
        tolerance=parsed_tolerance,
    )[["timestamp", "Transform 1", "Transform 2"]]

def remove_invalid(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that are not fully filled.
    Return a DataFrame with valid rows and drop count.
    """
    rows_before = len(data)
    cleaned_data = data.dropna().reset_index(drop=True)
    rows_after = len(cleaned_data)
    dropped_count = rows_before - rows_after

    return cleaned_data, dropped_count

def match(data_1: pd.DataFrame,
    data_2: pd.DataFrame,
    tolerance: str | pd.Timedelta | None = None,):
    data = match_by_timestamp(data_1, data_2, tolerance=tolerance)
    original_match = data.copy()
    cleaned_data, dropped_count = remove_invalid(data)
    length = len(cleaned_data)
    return cleaned_data, original_match, length, dropped_count
