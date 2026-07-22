import numpy as np
import pandas as pd
from pathlib import Path
from typing import TypeAlias
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


# accepts Path or string
PathLike: TypeAlias = str | Path

def load_ndi_data(csv_path: PathLike) -> pd.DataFrame:
    """
    Load the SI data from a CSV file, returns a pd.DataFrame.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"NDI data file not found: {csv_path}")

    return pd.read_csv(csv_path)


def parse_all_tools(data: pd.DataFrame) -> dict:
    """
    Separate data into individual dataframes by Tool ID, preserving timestamps.
    Returns a dictionary mapping Tool ID to its DataFrame.
    """
    return {tool_id: group.reset_index(drop=True) 
            for tool_id, group in data.groupby('Tool ID')}

def parse_timestamps(data: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """
    Change timestamp from microsecond timestamps to Vancouver datetimes.
    Returns the same DataFrame with updated timestamp.
    """
    if timestamp_column not in data.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in data.")

    data = data.copy()
    data[timestamp_column] = pd.to_datetime(
        data[timestamp_column], unit="s", utc=True, errors="coerce"
    ).dt.tz_convert("America/Vancouver")

    return data

def extract_position(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract position of arm and return Transform, Rotation, and Translation in DataFrame.
    """
    data = data.copy()
    start = data.columns.get_loc("Tx")
    end = start + 12

    value = data.iloc[:, start:end].copy()
    if value.shape[1]!=12:
        raise ValueError(f"Expected 12 columns for position data, but got {value.shape[1]} columns.")
    
    # Rename each column to match 
        # Rename values to match transformation structure
    value.columns = [
        "pos_x",
        "pos_y",
        "pos_z",
        "r00",
        "r01",
        "r02",
        "r10",
        "r11",
        "r12",
        "r20",
        "r21",
        "r22",
    ]
    
    return value

def pose_to_transform(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a row of pose data into 4x4 transform matrix, 3x3 rotational matrix, and 3x1 translation vector.
    """
    t = np.array([row["pos_x"], row["pos_y"], row["pos_z"]], dtype=float)

    R = np.array(
        [
            [row["r00"], row["r01"], row["r02"]],
            [row["r10"], row["r11"], row["r12"]],
            [row["r20"], row["r21"], row["r22"]],
        ],
        dtype=float,
    )

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t

    return T, R, t

def clean_ndi_data(csv_path: PathLike, timestamp_column: str, toolkey: int) -> pd.DataFrame:
    """
    Load and clean NDI data from a CSV file.
    Returns a cleaned DataFrame with UTC timestamps in Vancouver time and extracted position data.
    """
    data = load_ndi_data(csv_path)
    tools_data = parse_all_tools(data)
    if toolkey not in tools_data:
        raise ValueError(f"Tool ID {toolkey} not found in data. Available Tools: {list(tools_data.keys())}")
    data = tools_data[toolkey]
    data = parse_timestamps(data, timestamp_column)
    position_data = extract_position(data)

    transforms = []
    for _, row in position_data.iterrows():
        T, _, _ = pose_to_transform(row)
        transforms.append(T)

    cleaned_data = pd.DataFrame(
        {"timestamp": data[timestamp_column], "Transforms": transforms}  
    )
    return cleaned_data
