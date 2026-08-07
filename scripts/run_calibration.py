"""Example config-driven workflow for NDI-to-robot calibration."""

import json
from os import remove
from pathlib import Path
import pandas as pd

import numpy as np

from ndi_robot_registration.clean_ndi_data import clean_ndi_data
from ndi_robot_registration.clean_si_data import clean_si_data, apply_transform
from ndi_robot_registration.hand_eye_calibration import Calibration
from ndi_robot_registration.match import match, match_video
from ndi_robot_registration.transforms import invert_transform, average_transforms


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

def calculate_error(ndi_T_base: np.ndarray, ndi_data: np.ndarray, robot_data: np.ndarray):
    """
    NDI marker position in NDI coordinate frame (ndi_data) is treated as ground truth.
    """

    ndi_T_robot_marker = ndi_T_base @ robot_data
    residual = invert_transform(ndi_T_robot_marker) @ ndi_data
    return residual

def calculate_df_errors(ndi_T_base: np.ndarray, matched_data: pd.DataFrame):
    """
    Return average error between ndi marker and ndi marker held by SI robot, calculated
    """
    matched_data["Residual"] = None  # Initialize the new column

    residual_list = []
    for idx, row in matched_data.iterrows():
        ndi_data = row["NDI Transform"]
        robot_data = row["SI with marker Transform"]
        residual = calculate_error(ndi_T_base, ndi_data, robot_data)
        matched_data.at[idx, "Residual"] = residual
        residual_list.append(residual)

    average_residual = average_transforms(residual_list)
    return matched_data, average_residual
    
    

    

def main() :
    """Clean, match video frames, calibrate, and save the results."""
    print(CONFIG_PATH)
    config = load_config()
    inputs = config["inputs"]
    si_config = config["si_arm"]
    ndi_config = config["ndi"]
    matching_config = config["matching"]
    calibration_config = config["calibration"]
    video_config = config["video"] # needed key
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

    for idx, row in video_matched_data.iterrows():

        if video_matched_data["NDI Transform"].isna().any() or video_matched_data["SI Transform"].isna().any():
            video_matched_data.drop(idx, inplace=True)


    ndi_translation = ndi_marker.get("translation")

    if ndi_marker.get("units") == "mm":
        ndi_translation = np.array(ndi_translation) / 1000.0  # Convert mm to meters
    ndi_transform = np.eye(4)
    ndi_transform[:3,3] = ndi_translation
    print(ndi_transform)

    apply_transform(video_matched_data, "SI Transform", "SI with marker Transform", ndi_transform)


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

    video_matched_data, errors = calculate_df_errors(
        ndi_T_base, video_matched_data
    )

    
    np.set_printoptions(precision=8, suppress=True)
    print(f"Average residual error:")
    print(errors)
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
