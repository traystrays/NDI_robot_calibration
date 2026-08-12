"""Example config-driven workflow for NDI-to-robot calibration."""

import json
from pathlib import Path
import pandas as pd

import numpy as np

from ndi_robot_registration.clean_ndi_data import clean_ndi_data
from ndi_robot_registration.clean_si_data import clean_si_data, apply_transform
from ndi_robot_registration.hand_eye_calibration import Calibration
from ndi_robot_registration.match import match, match_video
from ndi_robot_registration.transforms import (
    as_transform,
    average_transforms,
    invert_transform,
    is_valid_transform,
    rotation_angle_degrees,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT/"scripts"/"calib_config.json"



def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load the calibration configuration from JSON."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def project_path(path_value: str) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def calculate_error(
    ndi_T_base: np.ndarray,
    ndi_data: np.ndarray,
    robot_data: np.ndarray,
) -> np.ndarray:
    """
    NDI marker position in NDI coordinate frame (ndi_data) is treated as ground truth.
    """

    ndi_T_robot_marker = ndi_T_base @ robot_data
    residual = invert_transform(ndi_T_robot_marker) @ ndi_data
    return residual

def calculate_df_errors(
    ndi_T_base: np.ndarray,
    matched_data: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float | int]]:
    """
    Calculate per-observation errors and a representative residual transform.
    """
    matched_data["Residual"] = None  # Initialize the new column
    numeric_columns = [
        "translation_error_mm",
        "translation_residual_x_ndi_mm",
        "translation_residual_y_ndi_mm",
        "translation_residual_z_ndi_mm",
        "rotation_error_deg",
    ]
    for column in numeric_columns:
        matched_data[column] = np.nan

    residual_list = []
    for idx, row in matched_data.iterrows():
        ndi_data = as_transform(row["NDI Transform"])
        robot_data = as_transform(row["SI with marker Transform"])
        if not (
            is_valid_transform(ndi_data, rotation_atol=2e-5)
            and is_valid_transform(robot_data, rotation_atol=2e-5)
        ):
            continue

        residual = calculate_error(ndi_T_base, ndi_data, robot_data)
        predicted = as_transform(ndi_T_base) @ robot_data
        translation_residual_ndi = ndi_data[:3, 3] - predicted[:3, 3]

        matched_data.at[idx, "Residual"] = residual
        matched_data.at[idx, "translation_error_mm"] = (
            np.linalg.norm(translation_residual_ndi) * 1000.0
        )
        matched_data.at[idx, "translation_residual_x_ndi_mm"] = (
            translation_residual_ndi[0] * 1000.0
        )
        matched_data.at[idx, "translation_residual_y_ndi_mm"] = (
            translation_residual_ndi[1] * 1000.0
        )
        matched_data.at[idx, "translation_residual_z_ndi_mm"] = (
            translation_residual_ndi[2] * 1000.0
        )
        matched_data.at[idx, "rotation_error_deg"] = (
            rotation_angle_degrees(residual[:3, :3])
        )
        residual_list.append(residual)

    if not residual_list:
        raise ValueError("No valid observations were available for error calculation.")

    average_residual = average_transforms(residual_list)
    valid_errors = matched_data.dropna(subset=numeric_columns)
    translation_errors = valid_errors["translation_error_mm"].to_numpy()
    rotation_errors = valid_errors["rotation_error_deg"].to_numpy()
    bias = valid_errors[
        [
            "translation_residual_x_ndi_mm",
            "translation_residual_y_ndi_mm",
            "translation_residual_z_ndi_mm",
        ]
    ].to_numpy().mean(axis=0)

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "rmse": float(np.sqrt(np.mean(values ** 2))),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max()),
        }

    metrics: dict[str, float | int] = {
        "valid_error_count": len(valid_errors),
        **{
            f"translation_{name}_mm": value
            for name, value in summarize(translation_errors).items()
        },
        "translation_bias_x_ndi_mm": float(bias[0]),
        "translation_bias_y_ndi_mm": float(bias[1]),
        "translation_bias_z_ndi_mm": float(bias[2]),
        "translation_bias_magnitude_mm": float(np.linalg.norm(bias)),
        **{
            f"rotation_{name}_deg": value
            for name, value in summarize(rotation_errors).items()
        },
    }
    return matched_data, average_residual, metrics


def print_error_metrics(metrics: dict[str, float | int]) -> None:
    """Print translation, directional bias, and rotation error summaries."""
    print(f"\nValid error observations: {metrics['valid_error_count']}")
    print("\n3D translation error:")
    for name in ("mean", "median", "rmse", "p90", "p95", "p99", "max"):
        print(f"  {name}: {metrics[f'translation_{name}_mm']:.6f} mm")

    print("\nTranslation bias in NDI frame:")
    for axis in "xyz":
        print(
            f"  mean {axis}: "
            f"{metrics[f'translation_bias_{axis}_ndi_mm']:.6f} mm"
        )
    print(
        "  magnitude: "
        f"{metrics['translation_bias_magnitude_mm']:.6f} mm"
    )

    print("\n3D rotation error:")
    for name in ("mean", "median", "rmse", "p90", "p95", "p99", "max"):
        print(f"  {name}: {metrics[f'rotation_{name}_deg']:.6f} deg")
    
    

    

def main() :
    """Clean, match video frames, calibrate, and save the results."""
    print(CONFIG_PATH)
    config = load_config()
    inputs = config["inputs"]
    si_config = config["si_arm"]
    ndi_config = config["ndi"]
    matching_config = config["matching"]
    calibration_config = config["calibration"]
    video_config = config["video"] # need key
    output_config = config.get("output", {})
    ndi_marker = config.get("ndi_marker", {}) # key may not be present

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

    # NDI is the slower stream, so we match nearest SI to NDI observation
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

    if matched_count < 3:
        raise RuntimeError(
            "Fewer than three timestamp matches remain after filtering."
        )

    video_matched_data = match_video(
        matched_data,
        project_path(video_config["timestamp"]),
        tolerance=video_config.get(
            "time_tolerance",
            matching_config["time_tolerance"],
        ),
    )

    valid_video_rows = (
        video_matched_data["NDI Transform"].map(
            lambda value: isinstance(value, np.ndarray)
        )
        & video_matched_data["SI Transform"].map(
            lambda value: isinstance(value, np.ndarray)
        )
    )
    video_matched_data = video_matched_data.loc[valid_video_rows].copy()


    ndi_translation = ndi_marker.get("translation")

    if ndi_marker.get("units") == "mm":
        ndi_translation = np.array(ndi_translation) / 1000.0  # Convert mm to meters
    ndi_transform = np.eye(4)
    ndi_transform[:3,3] = ndi_translation
    print(ndi_transform)

    apply_transform(
        matched_data,
        "SI Transform",
        "SI with marker Transform",
        ndi_transform,
    )
    apply_transform(
        video_matched_data,
        "SI Transform",
        "SI with marker Transform",
        ndi_transform,
    )


    matched_frame_count = int(
        video_matched_data["matched_timestamp"].notna().sum()
    )

    match_output_value = output_config.get("match_output")
    if match_output_value is not None:
        match_output_path = project_path(match_output_value)
        match_output_path.parent.mkdir(parents=True, exist_ok=True)
        video_matched_data.to_csv(match_output_path, index=False)
        print(f"Saved video-frame matches to: {match_output_path}")

    hand_pick_frame = calibration_config.get("hand-pick-frame", False)
    if not isinstance(hand_pick_frame, bool):
        raise TypeError(
            "calibration.hand-pick-frame must be true or false."
        )

    # Frame selection happens only after NDI, SI, and video are synchronized.
    if hand_pick_frame:
        from ndi_robot_registration.frame_selection import (
            select_calibration_frames,
        )

        calibration_data = select_calibration_frames(
            project_path(video_config["input"]),
            video_matched_data,
            selection_path=(
                project_path(output_config["selected_frames"])
                if output_config.get("selected_frames") is not None
                else None
            ),
        )
    else:
        calibration_data = matched_data

    calibration = Calibration(
        ndi_transforms=calibration_data["NDI Transform"],
        si_transforms=calibration_data["SI Transform"],
        min_rotation_deg=calibration_config["min_rotation_deg"],
        min_translation=calibration_config["min_translation"],
        min_index_separation=calibration_config["min_index_separation"],
        max_rotation_disagreement_deg=calibration_config[
            "max_rotation_disagreement_deg"
        ],
        max_pairs=calibration_config["max_pairs"],
    )
    ndi_T_base = (
        calibration.calibrate_selected()
        if hand_pick_frame
        else calibration.calibrate()
    )

    matched_data, average_residual, error_metrics = calculate_df_errors(
        ndi_T_base, matched_data
    )

    
    np.set_printoptions(precision=8, suppress=True)
    print("Average residual transform (systematic pose bias):")
    print(average_residual)
    print_error_metrics(error_metrics)
    print(f"Matched observations: {matched_count}")
    print(f"Unmatched observations removed: {dropped_count}")
    print(
        "Video frames matched to NDI/SI observations: "
        f"{matched_frame_count}/{len(video_matched_data)}"
    )
    print(f"Accepted motion pairs: {len(calibration.motion_pairs)}")
    if hand_pick_frame:
        print(f"Hand-selected calibration frames: {len(calibration_data)}")
    print("\nndi_T_base:")
    print(ndi_T_base)
    print("\nCalibration metrics:")
    for name, value in calibration.metrics.items():
        print(f"  {name}: {value}")


    output_value = output_config.get("calib_file")
    if output_value is not None:
        output_path = project_path(output_value)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            ndi_T_base=ndi_T_base,
            metric_names=np.asarray(list(calibration.metrics)),
            metric_values=np.asarray(list(calibration.metrics.values())),
        )
        print(f"\nSaved calibration to: {output_path}")


if __name__ == "__main__":
    main()
