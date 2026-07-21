import numpy as np
import pandas as pd
from pathlib import Path

def load_si_data(csv_path: Path) -> pd.DataFrame:
    """
    Load the SI data from a CSV file, returns a pd.DataFrame.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"SI data file not found:{csv_path}")

    return pd.read_csv(csv_path)

def parse_timestamps(data: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """
    Change timestamp from microsecond timestamps to Vancouver datetimes.
    Returns the same DataFrame with updated timestamp.
    """
    if timestamp_column not in data.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in data.")

    data[timestamp_column] = pd.to_datetime(data[timestamp_column], unit='us', utc=True, errors='coerce').dt.tz_convert('America/Vancouver')
    return data

def pose_to_transform(row:pd.Series):
    """
    Convert a row of pose data into 4x4 transform matrix, 3x3 rotational matrix, and 3x1 translation vector.
    """
    t = np.array([row["pos x"], row["pos y"], row["pos z"]], dtype=float)

    R = np.array([
                [row["o0"], row["o1"], row["o2"]],
                [row["o3"], row["o4"], row["o5"]],
                [row["o6"], row["o7"], row["o8"]],
            ], dtype=float)

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t

    return T, R, t

def extract_position(data: pd.DataFrame, arm_column: str) -> pd.DataFrame:
    """
    Extract position of arm and return Transform, Rotation, and Translation in DataFrame.
    """
    if arm_column not in data.columns:
        raise ValueError(f"Arm column '{arm_column}' not found in data.")
    
    start = data.columns.get_loc(arm_column)
    val_start = start + 1
    val_end = val_start + 12
    value = data.iloc[:, val_start:val_end].copy()

    if value.shape[1] != 12:
        raise ValueError(
            f"Expected 12 pose columns after {arm_column!r}; "
            f"found {value.shape[1]}"
          )
    
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

def clean_si_data(csv_path: Path, timestamp_column: str, arm_column: str) -> pd.DataFrame:
    """
    Load SI data from CSV, parse timestamps, and extract position data.
    Returns a cleaned DataFrame with timestamps and Transforms.
    """
    # Load the data
    raw_data = load_si_data(csv_path)

    # Parse timestamps
    time_data = parse_timestamps(raw_data, timestamp_column)

    # Extract position data
    position_data = extract_position(raw_data, arm_column)

    Transforms = []

    for idx, row in position_data.iterrows():
        T, _, _ = pose_to_transform(row)
        Transforms.append(T)

    cleaned_data = pd.DataFrame({"timestamp":time_data[timestamp_column], 
                                 "Transforms": Transforms})
    
    return cleaned_data

