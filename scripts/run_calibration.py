"""Example config-driven workflow for NDI-to-robot calibration."""

import json
from pathlib import Path

import numpy as np

from ndi_robot_registration.clean_ndi_data import clean_ndi_data
from ndi_robot_registration.clean_si_data import clean_si_data
from ndi_robot_registration.hand_eye_calibration import Calibration
from ndi_robot_registration.match import match, match_video


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


def main() :
    """Clean, match video frames, calibrate, and save the results."""
    print(CONFIG_PATH)
    config = load_config()
    inputs = config["inputs"]
    si_config = config["si_arm"]
    ndi_config = config["ndi"]
    matching_config = config["matching"]
    calibration_config = config["calibration"]
    video_config = config["video"]
    output_config = config.get("output", {})

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
    matched_frame_count = int(
        video_matched_data["matched_timestamp"].notna().sum()
    )

    match_output_value = output_config.get("match_output")
    if match_output_value is not None:
        match_output_path = project_path(match_output_value)
        match_output_path.parent.mkdir(parents=True, exist_ok=True)
        video_matched_data.to_csv(match_output_path, index=False)
        print(f"Saved video-frame matches to: {match_output_path}")

    # initialize this calibration object
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
    ndi_T_base = calibration.calibrate()

    np.set_printoptions(precision=8, suppress=True)
    print(f"Matched observations: {matched_count}")
    print(f"Unmatched observations removed: {dropped_count}")
    print(
        "Video frames matched to NDI/SI observations: "
        f"{matched_frame_count}/{len(video_matched_data)}"
    )
    print(f"Accepted motion pairs: {len(calibration.motion_pairs)}")
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
