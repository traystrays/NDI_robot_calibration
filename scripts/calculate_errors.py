"""Evaluate fixed NDI/base and gripper/marker calibrations on synchronized data.

This script never fits or modifies either calibration transform.  Point it at
an evaluation configuration whose ``inputs`` identify the held-out NDI and SI
recordings, then it calculates:

    ndi_T_marker_predicted =
        ndi_T_base @ base_T_gripper @ gripper_T_marker

against every synchronized measured ``ndi_T_marker``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ndi_robot_registration.clean_ndi_data import clean_ndi_data
from ndi_robot_registration.clean_si_data import clean_si_data
from ndi_robot_registration.match import match
from ndi_robot_registration.transforms import (
    as_transform,
    average_transforms,
    invert_transform,
    is_valid_transform,
    rotation_angle_degrees,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scripts" / "calib_config.json"


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_transform(path: Path, keys: tuple[str, ...]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with np.load(path, allow_pickle=False) as calibration:
        for key in keys:
            if key in calibration:
                transform = as_transform(calibration[key])
                if not is_valid_transform(transform):
                    raise ValueError(
                        f"{key} in {path} is not a valid rigid transform"
                    )
                return transform
    raise KeyError(f"None of {keys} found in {path}")


def calculate_errors(
    matched_data: pd.DataFrame,
    ndi_T_base: np.ndarray,
    gripper_T_marker: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return per-observation errors and the average residual transform."""
    records = []
    residuals = []
    for matched_index, row in matched_data.iterrows():
        measured = as_transform(row["NDI Transform"])
        base_T_gripper = as_transform(row["SI Transform"])
        if not (
            is_valid_transform(measured)
            and is_valid_transform(base_T_gripper)
        ):
            continue

        predicted = ndi_T_base @ base_T_gripper @ gripper_T_marker
        residual = invert_transform(predicted) @ measured
        residuals.append(residual)
        translation_vector_ndi = measured[:3, 3] - predicted[:3, 3]
        records.append(
            {
                "matched_index": int(matched_index),
                "timestamp": row["timestamp"],
                "translation_error_mm": float(
                    np.linalg.norm(translation_vector_ndi) * 1000.0
                ),
                "translation_residual_x_ndi_mm": float(
                    translation_vector_ndi[0] * 1000.0
                ),
                "translation_residual_y_ndi_mm": float(
                    translation_vector_ndi[1] * 1000.0
                ),
                "translation_residual_z_ndi_mm": float(
                    translation_vector_ndi[2] * 1000.0
                ),
                "rotation_error_deg": rotation_angle_degrees(
                    residual[:3, :3]
                ),
            }
        )
    if not records:
        raise ValueError("No valid synchronized observations were available.")
    return pd.DataFrame.from_records(records), average_transforms(residuals)


def print_summary(name: str, values: np.ndarray, unit: str) -> None:
    print(f"\n{name}:")
    print(f"  count:  {len(values)}")
    print(f"  mean:   {values.mean():.6f} {unit}")
    print(f"  median: {np.median(values):.6f} {unit}")
    print(f"  RMSE:   {np.sqrt(np.mean(values ** 2)):.6f} {unit}")
    print(f"  90th:   {np.quantile(values, 0.90):.6f} {unit}")
    print(f"  95th:   {np.quantile(values, 0.95):.6f} {unit}")
    print(f"  99th:   {np.quantile(values, 0.99):.6f} {unit}")
    print(f"  max:    {values.max():.6f} {unit}")


def print_translation_bias(errors: pd.DataFrame) -> None:
    """Print the signed mean translation residual in NDI coordinates."""
    columns = [
        "translation_residual_x_ndi_mm",
        "translation_residual_y_ndi_mm",
        "translation_residual_z_ndi_mm",
    ]
    mean_bias = errors[columns].to_numpy(dtype=float).mean(axis=0)

    print("\nTranslation bias in NDI frame:")
    print(f"  mean x: {mean_bias[0]:.6f} mm")
    print(f"  mean y: {mean_bias[1]:.6f} mm")
    print(f"  mean z: {mean_bias[2]:.6f} mm")
    print(f"  magnitude: {np.linalg.norm(mean_bias):.6f} mm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Evaluation-data configuration JSON.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="NPZ containing ndi_T_base (default: output.calib_file).",
    )
    parser.add_argument(
        "--marker-mount",
        type=Path,
        default=None,
        help="NPZ containing gripper_T_marker (default: output.marker_mount_file).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Per-observation error CSV output.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    inputs = config["inputs"]
    si_config = config["si_arm"]
    ndi_config = config["ndi"]
    matching_config = config["matching"]
    output_config = config.get("output", {})

    calibration_path = project_path(
        args.calibration
        if args.calibration is not None
        else output_config["calib_file"]
    )
    marker_mount_path = project_path(
        args.marker_mount
        if args.marker_mount is not None
        else output_config["marker_mount_file"]
    )
    output_path = project_path(
        args.output
        if args.output is not None
        else output_config.get(
            "marker_error_output",
            "./data/marker_prediction_errors.csv",
        )
    )

    ndi_T_base = load_transform(calibration_path, ("ndi_T_base",))
    gripper_T_marker = load_transform(
        marker_mount_path,
        ("gripper_T_marker", "end_effector_T_marker"),
    )

    si_data = clean_si_data(
        project_path(inputs["si_csv"]),
        timestamp_column=si_config["timestamp_column"],
        arm_column=si_config["arm_column"],
    )
    ndi_data = clean_ndi_data(
        project_path(inputs["ndi_csv"]),
        timestamp_column=ndi_config["timestamp_column"],
        toolkey=ndi_config["tool_id"],
    )
    matched_data, _, matched_count, dropped_count = match(
        ndi_data,
        si_data,
        tolerance=matching_config["time_tolerance"],
    )
    matched_data.columns = ["timestamp", "NDI Transform", "SI Transform"]

    errors, average_residual = calculate_errors(
        matched_data,
        ndi_T_base,
        gripper_T_marker,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(output_path, index=False)

    print(f"Evaluation configuration: {args.config.resolve()}")
    print(f"Fixed ndi_T_base: {calibration_path}")
    print(f"Fixed gripper_T_marker: {marker_mount_path}")
    print(f"Matched observations: {matched_count}")
    print(f"Dropped timestamp matches: {dropped_count}")
    print(f"Valid observations evaluated: {len(errors)}")
    np.set_printoptions(precision=8, suppress=True)
    print("\nAverage residual transform (systematic pose bias):")
    print(average_residual)
    print_summary(
        "3D translation error",
        errors["translation_error_mm"].to_numpy(),
        "mm",
    )
    print_translation_bias(errors)
    print_summary(
        "3D rotation error",
        errors["rotation_error_deg"].to_numpy(),
        "deg",
    )
    print(f"\nSaved per-observation errors to: {output_path}")


if __name__ == "__main__":
    main()
