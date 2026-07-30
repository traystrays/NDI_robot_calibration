"""Analyze whether the NDI marker is rigid relative to the robot end effector."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ndi_robot_registration.clean_ndi_data import clean_ndi_data
from ndi_robot_registration.clean_si_data import clean_si_data
from ndi_robot_registration.hand_eye_calibration import Calibration
from ndi_robot_registration.match import match
from ndi_robot_registration.transforms import (
    as_transform,
    invert_transform,
    proper_rotation,
    rotation_angle_degrees,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scripts" / "calib_config.json"


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a JSON calibration configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def project_path(path_value: str) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def calculate_marker_mounts(
    matched_data: pd.DataFrame,
    valid_indices: list[int],
    ndi_T_base: np.ndarray,
) -> list[np.ndarray]:
    """
    Calculate one end-effector-to-marker transform for each observation.

    From ``Q_i = X P_i C``:

    * ``Q_i`` is ``ndi_T_marker_i``.
    * ``X`` is ``ndi_T_base``.
    * ``P_i`` is ``base_T_end_effector_i``.
    * ``C`` is ``end_effector_T_marker``.

    Therefore ``C_i = inverse(P_i) inverse(X) Q_i``.
    """
    base_T_ndi = invert_transform(ndi_T_base)
    marker_mounts = []

    for index in valid_indices:
        ndi_T_marker = as_transform(
            matched_data.iloc[index]["NDI Transform"]
        )
        base_T_end_effector = as_transform(
            matched_data.iloc[index]["SI Transform"]
        )
        end_effector_T_marker = (
            invert_transform(base_T_end_effector)
            @ base_T_ndi
            @ ndi_T_marker
        )
        marker_mounts.append(end_effector_T_marker)

    return marker_mounts


def analyze_marker_mounts(
    marker_mounts: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return a representative mount and per-sample translation/rotation errors.

    Translation is centered using the component-wise median. Rotation is
    centered by projecting the mean rotation matrix onto SO(3).
    """
    if not marker_mounts:
        raise ValueError("At least one marker-mount transform is required.")

    transforms = np.stack(marker_mounts)
    translations = transforms[:, :3, 3]
    rotations = transforms[:, :3, :3]

    center_transform = np.eye(4, dtype=float)
    center_transform[:3, :3] = proper_rotation(rotations.mean(axis=0))
    center_transform[:3, 3] = np.median(translations, axis=0)

    translation_errors = np.linalg.norm(
        translations - center_transform[:3, 3],
        axis=1,
    )
    rotation_errors = np.asarray(
        [
            rotation_angle_degrees(
                center_transform[:3, :3].T @ rotation
            )
            for rotation in rotations
        ]
    )

    return center_transform, translation_errors, rotation_errors


def calculate_marker_prediction_errors(
    matched_data: pd.DataFrame,
    valid_indices: list[int],
    ndi_T_base: np.ndarray,
    end_effector_T_marker: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare ``X P_i C`` against each measured ``ndi_T_marker_i``."""
    translation_errors = []
    rotation_errors = []
    for index in valid_indices:
        measured = as_transform(matched_data.iloc[index]["NDI Transform"])
        base_T_end_effector = as_transform(
            matched_data.iloc[index]["SI Transform"]
        )
        predicted = (
            ndi_T_base
            @ base_T_end_effector
            @ end_effector_T_marker
        )
        residual = invert_transform(predicted) @ measured
        translation_errors.append(np.linalg.norm(residual[:3, 3]))
        rotation_errors.append(
            rotation_angle_degrees(residual[:3, :3])
        )
    return np.asarray(translation_errors), np.asarray(rotation_errors)


def print_error_summary(
    name: str,
    errors: np.ndarray,
    unit: str,
) -> None:
    """Print common percentiles for one error vector."""
    print(f"\n{name}:")
    print(f"  mean:   {errors.mean():.6f} {unit}")
    print(f"  median: {np.median(errors):.6f} {unit}")
    print(f"  90th:   {np.quantile(errors, 0.90):.6f} {unit}")
    print(f"  99th:   {np.quantile(errors, 0.99):.6f} {unit}")
    print(f"  max:    {errors.max():.6f} {unit}")


def main() -> None:
    """Run calibration and report marker-mount consistency."""
    parser = argparse.ArgumentParser(
        description="Analyze rigid NDI marker-mount consistency."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Calibration JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for per-observation errors.",
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    inputs = config["inputs"]
    si_config = config["si_arm"]
    ndi_config = config["ndi"]
    matching_config = config["matching"]
    calibration_config = config["calibration"]

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
    matched_data.columns = [
        "timestamp",
        "NDI Transform",
        "SI Transform",
    ]

    calibration = Calibration(
        ndi_transforms=matched_data["NDI Transform"],
        si_transforms=matched_data["SI Transform"],
        min_rotation_deg=calibration_config["min_rotation_deg"],
        min_translation=calibration_config["min_translation"],
        min_index_separation=calibration_config["min_index_separation"],
        max_rotation_disagreement_deg=calibration_config[
            "max_rotation_disagreement_deg"
        ],
        max_pairs=calibration_config["max_pairs"],
    )
    output_config = config.get("output", {})
    calibration_path_value = output_config.get("calib_file")
    calibration_path = (
        project_path(calibration_path_value)
        if calibration_path_value is not None
        else None
    )
    if calibration_path is not None and calibration_path.is_file():
        with np.load(calibration_path, allow_pickle=False) as calibration_file:
            ndi_T_base = as_transform(calibration_file["ndi_T_base"])
        calibration.validate_observations()
        print(f"Using saved ndi_T_base from: {calibration_path}")
    else:
        ndi_T_base = calibration.calibrate()

    marker_mounts = calculate_marker_mounts(
        matched_data,
        calibration.valid_indices,
        ndi_T_base,
    )
    (
        center_mount,
        translation_errors_m,
        rotation_errors_deg,
    ) = analyze_marker_mounts(marker_mounts)
    (
        prediction_translation_errors_m,
        prediction_rotation_errors_deg,
    ) = calculate_marker_prediction_errors(
        matched_data,
        calibration.valid_indices,
        ndi_T_base,
        center_mount,
    )

    print(f"\nMatched observations: {matched_count}")
    print(f"Dropped timestamp matches: {dropped_count}")
    print(f"Valid observations analyzed: {len(marker_mounts)}")
    print("\nRepresentative end_effector_T_marker:")
    print(center_mount)

    print_error_summary(
        "Translation deviation",
        translation_errors_m * 1000.0,
        "mm",
    )
    print_error_summary(
        "Rotation deviation",
        rotation_errors_deg,
        "deg",
    )
    print_error_summary(
        "Predicted-vs-measured NDI marker translation error",
        prediction_translation_errors_m * 1000.0,
        "mm",
    )
    print_error_summary(
        "Predicted-vs-measured NDI marker rotation error",
        prediction_rotation_errors_deg,
        "deg",
    )

    marker_mount_value = output_config.get("marker_mount_file")
    marker_mount_path = (
        project_path(marker_mount_value)
        if marker_mount_value is not None
        else (
            calibration_path.with_name("marker_mount.npz")
            if calibration_path is not None
            else PROJECT_ROOT / "marker_mount.npz"
        )
    )
    marker_mount_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        marker_mount_path,
        gripper_T_marker=center_mount,
        end_effector_T_marker=center_mount,
        translation_errors_mm=prediction_translation_errors_m * 1000.0,
        rotation_errors_deg=prediction_rotation_errors_deg,
        valid_indices=np.asarray(calibration.valid_indices, dtype=int),
    )
    print(f"\nSaved gripper_T_marker and errors to: {marker_mount_path}")

    if arguments.output is not None:
        output_path = (
            arguments.output
            if arguments.output.is_absolute()
            else PROJECT_ROOT / arguments.output
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        valid_rows = matched_data.iloc[calibration.valid_indices]
        diagnostics = pd.DataFrame(
            {
                "matched_index": calibration.valid_indices,
                "timestamp": valid_rows["timestamp"].to_numpy(),
                "translation_error_mm": translation_errors_m * 1000.0,
                "rotation_error_deg": rotation_errors_deg,
                "prediction_translation_error_mm":
                    prediction_translation_errors_m * 1000.0,
                "prediction_rotation_error_deg":
                    prediction_rotation_errors_deg,
            }
        )
        diagnostics.to_csv(output_path, index=False)
        print(f"\nSaved per-observation errors to: {output_path}")


if __name__ == "__main__":
    main()
