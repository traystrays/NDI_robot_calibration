from pathlib import Path

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

def remove_invalid(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
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
    tolerance: str | pd.Timedelta | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    data = match_by_timestamp(data_1, data_2, tolerance=tolerance)
    original_match = data.copy()
    cleaned_data, dropped_count = remove_invalid(data)
    length = len(cleaned_data)
    return cleaned_data, original_match, length, dropped_count


def load_video_timestamps(
    timestamp_path: str | Path,
    *,
    timezone: str = "America/Vancouver",
    first_frame_number: int = 0,
) -> pd.DataFrame:
    """
    Load one video timestamp per line and assign sequential frame numbers.

    The timestamp files contain local times without a UTC offset, so they are
    localized to ``timezone``. Frame numbers are zero-based by default to
    match OpenCV indexing.
    """
    timestamp_path = Path(timestamp_path)
    if not timestamp_path.is_file():
        raise FileNotFoundError(
            f"Video timestamp file not found: {timestamp_path}"
        )
    if first_frame_number < 0:
        raise ValueError("first_frame_number cannot be negative.")

    timestamp_strings = [
        line.strip()
        for line in timestamp_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not timestamp_strings:
        raise ValueError(f"No timestamps found in {timestamp_path}.")

    timestamps = pd.to_datetime(
        pd.Series(timestamp_strings), errors="coerce"
    )
    if timestamps.isna().any():
        invalid_lines = [
            index + 1
            for index, value in enumerate(timestamps.isna())
            if value
        ]
        raise ValueError(
            "Invalid video timestamps on line(s): "
            + ", ".join(map(str, invalid_lines))
        )

    timestamps = timestamps.dt.tz_localize(
        timezone,
        ambiguous="raise",
        nonexistent="raise",
    ).dt.as_unit("us")

    return pd.DataFrame(
        {
            "frame_number": range(
                first_frame_number,
                first_frame_number + len(timestamps),
            ),
            "video_timestamp": timestamps,
        }
    )


def match_video(
    matched_data: pd.DataFrame,
    timestamp_path: str | Path,
    tolerance: str | pd.Timedelta | None = None,
    *,
    timezone: str = "America/Vancouver",
    first_frame_number: int = 0,
) -> pd.DataFrame:
    """
    Match each video frame to the nearest already-matched NDI/SI observation.

    ``matched_data`` must contain a ``timestamp`` column and may contain any
    additional sensor columns, such as ``NDI Transform`` and ``SI Transform``.
    Every video frame is preserved. Sensor columns are missing when no sensor
    timestamp falls within ``tolerance``.
    """
    if "timestamp" not in matched_data.columns:
        raise ValueError("matched_data must contain a 'timestamp' column.")
    if matched_data.columns.duplicated().any():
        raise ValueError("matched_data cannot contain duplicate column names.")

    parsed_tolerance = (
        pd.Timedelta(tolerance) if tolerance is not None else None
    )
    if parsed_tolerance is not None and parsed_tolerance < pd.Timedelta(0):
        raise ValueError("tolerance must not be negative.")

    frames = load_video_timestamps(
        timestamp_path,
        timezone=timezone,
        first_frame_number=first_frame_number,
    ).sort_values("video_timestamp")

    sensor_data = matched_data.copy()
    sensor_data["matched_timestamp"] = pd.to_datetime(
        sensor_data.pop("timestamp"),
        utc=True,
        errors="coerce",
    ).dt.tz_convert(timezone).dt.as_unit("us")
    if sensor_data["matched_timestamp"].isna().any():
        raise ValueError("All matched-data timestamps must be valid.")
    sensor_data = sensor_data.sort_values("matched_timestamp")

    result = pd.merge_asof(
        frames,
        sensor_data,
        left_on="video_timestamp",
        right_on="matched_timestamp",
        direction="nearest",
        tolerance=parsed_tolerance,
    )
    result["time_difference"] = (
        result["video_timestamp"] - result["matched_timestamp"]
    ).abs()

    sensor_columns = [
        column
        for column in matched_data.columns
        if column != "timestamp"
    ]
    return result[
        [
            "frame_number",
            "video_timestamp",
            "matched_timestamp",
            "time_difference",
            *sensor_columns,
        ]
    ].reset_index(drop=True)
