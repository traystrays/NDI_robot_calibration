"""Example config-driven workflow for NDI-to-robot calibration."""

import json
from pathlib import Path

import numpy as np

from ndi_robot_registration.clean_ndi_data import clean_ndi_data
from ndi_robot_registration.clean_si_data import clean_si_data
from ndi_robot_registration.hand_eye_calibration import Calibration
from ndi_robot_registration.match import match


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


def main() -> None:
    """Clean, match, calibrate, and report the base-to-NDI transform."""
    print(CONFIG_PATH)
    config = load_config()
    inputs = config["inputs"]
    si_config = config["si"]
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

    # NDI is the slower stream, so preserve one row per NDI observation and
    # attach the nearest SI observation. This avoids reusing an NDI transform
    # for several higher-frequency SI rows.
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
    print(f"Accepted motion pairs: {len(calibration.motion_pairs)}")
    print("\nndi_T_base:")
    print(ndi_T_base)
    print("\nCalibration metrics:")
    for name, value in calibration.metrics.items():
        print(f"  {name}: {value}")

    output_value = config.get("output", {}).get("result_file")
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
