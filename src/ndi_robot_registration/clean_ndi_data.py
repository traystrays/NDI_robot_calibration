import numpy as np
import pandas as pd
from pathlib import Path
from typing import TypeAlias

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