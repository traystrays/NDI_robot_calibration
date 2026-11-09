"""Config-driven position-only AX=YB calibration.

Uses the same CSVs, timestamps, frame picker and metre conventions as
run_calibration_axxb.py. Estimates ndi_T_base and marker_offset_m in the end-effector
frame. NDI orientation, ndi_marker.translation and motion-pair thresholds are
unused. This script does not detect/average stationary windows: select settled
poses using the existing frame picker, or fit all timestamp matches.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from ndi_robot_registration.clean_ndi_data import clean_ndi_data
from ndi_robot_registration.clean_si_data import clean_si_data
from ndi_robot_registration.match import match, match_video
from ndi_robot_registration.position_calibration import (
    observation_arrays, predict_positions, solve_position_axyb,
)
from run_calibration_axxb import load_config, project_path


def main():
    config = load_config()
    inputs, si, ndi = config["inputs"], config["si_arm"], config["ndi"]
    output = config.get("output", {})
    tolerance = config["matching"]["time_tolerance"]
    robot = clean_si_data(project_path(inputs["si_csv"]), si["timestamp_column"], si["arm_column"])
    tracker = clean_ndi_data(project_path(inputs["ndi_csv"]), ndi["timestamp_column"], ndi["tool_id"])
    matched, _, count, dropped = match(tracker, robot, tolerance=tolerance)
    matched.columns = ["timestamp", "NDI Transform", "SI Transform"]
    hand_pick = config["calibration"].get("hand-pick-frame", False)
    if not isinstance(hand_pick, bool):
        raise TypeError("calibration.hand-pick-frame must be true or false.")
    # Retain video matching/output when configured; video is optional without picking.
    video_data = None
    video = config.get("video", {})
    if video.get("timestamp"):
        video_data = match_video(matched, project_path(video["timestamp"]),
                                 tolerance=video.get("time_tolerance", tolerance))
        if output.get("match_output"):
            path = project_path(output["match_output"])
            path.parent.mkdir(parents=True, exist_ok=True)
            video_data.to_csv(path, index=False)
    if hand_pick:
        from ndi_robot_registration.frame_selection import select_calibration_frames
        if video_data is None:
            raise ValueError("Frame selection requires video.timestamp and video.input.")
        calibration_data = select_calibration_frames(
            project_path(video["input"]), video_data,
            selection_path=project_path(output["selected_frames"]) if output.get("selected_frames") else None,
        )
        # Multiple video frames can correspond to the same sensor observation.
        calibration_data = calibration_data.drop_duplicates("matched_timestamp")
    else:
        calibration_data = matched
    t, c, metrics = solve_position_axyb(
        calibration_data["NDI Transform"], calibration_data["SI Transform"])
    indices, p, q = observation_arrays(matched["NDI Transform"], matched["SI Transform"])
    errors = q - predict_positions(t, c, p)
    distances = np.linalg.norm(errors, axis=1) * 1000
    metrics.update({
        "evaluation_observation_count": len(indices),
        "evaluation_translation_rmse_mm": float(np.sqrt(np.mean(distances ** 2))),
        "evaluation_translation_p95_mm": float(np.quantile(distances, 0.95)),
        "evaluation_translation_max_mm": float(distances.max()),
    })
    np.set_printoptions(precision=8, suppress=True)
    print(f"Matched observations: {count}; unmatched removed: {dropped}")
    print("ndi_T_base (translation in metres):\n", t)
    print("Marker-origin offset in end-effector frame (mm):", c * 1000)
    print("Position metrics (evaluation includes fitting observations; not held-out accuracy):")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
    destination = output.get("calib_file")
    if destination:
        path = project_path(str(destination))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ndi_T_base=t, marker_offset_m=c, marker_offset_mm=c * 1000,
                 metric_names=np.asarray(list(metrics)), metric_values=np.asarray(list(metrics.values())))
        print(f"Saved position calibration to: {path}")


if __name__ == "__main__":
    main()
