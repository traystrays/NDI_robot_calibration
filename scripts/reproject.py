"""Reproject the NDI-tracked marker position onto the recorded camera video."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from ndi_robot_registration.clean_ndi_data import clean_ndi_data
from ndi_robot_registration.clean_si_data import clean_si_data
from ndi_robot_registration.match import match, match_video
from ndi_robot_registration.transforms import (
    as_transform,
    invert_transform,
    is_valid_transform,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "scripts" / "calib_config.json"


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the reprojection and calibration configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def project_path(path_value: str) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_npz_transform(file_path: Path, key: str) -> np.ndarray:
    """Load and validate one 4x4 transform from an NPZ archive."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Parameter file not found: {file_path}")

    with np.load(file_path, allow_pickle=False) as parameters: #allow pickle -> save as bytes
        if key not in parameters:
            raise KeyError(
                f"{file_path} does not contain {key!r}. "
                f"Available keys: {parameters.files}"
            )
        transform = as_transform(parameters[key]).copy()

    if not is_valid_transform(transform):
        raise ValueError(f"{key!r} in {file_path} is not a rigid transform.")
    return transform


def load_camera_parameters(
    file_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load camera intrinsics and distortion coefficients from an NPZ archive.

    The intrinsic matrix may be stored as ``camera_matrix``, ``K``, or
    ``intrinsic_matrix``. Distortion is optional and may be stored as
    ``distortion_coefficients``, ``dist_coeffs``, or ``distortion``.
    """
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Camera-parameter file not found: {file_path}"
        )

    with np.load(file_path, allow_pickle=False) as parameters:
        intrinsic_key = next(
            (
                key
                for key in ("camera_matrix", "K", "intrinsic_matrix")
                if key in parameters
            ),
            None,
        )
        if intrinsic_key is None:
            raise KeyError(
                f"{file_path} has no camera intrinsic matrix. Add one under "
                "'camera_matrix', 'K', or 'intrinsic_matrix'. Available keys: "
                f"{parameters.files}"
            )

        camera_matrix = np.asarray(parameters[intrinsic_key], dtype=float).copy()
        if camera_matrix.shape != (3, 3):
            raise ValueError(
                f"Camera matrix must be 3x3; got {camera_matrix.shape}."
            )

        distortion_key = next(
            (
                key
                for key in (
                    "distortion_coefficients",
                    "dist_coeffs",
                    "distortion",
                )
                if key in parameters
            ),
            None,
        )
        distortion = (
            np.asarray(parameters[distortion_key], dtype=float).reshape(-1)
            if distortion_key is not None
            else np.zeros(5, dtype=float)
        )

    return camera_matrix, distortion


def NDI_in_cam(
    base_T_ecm: np.ndarray,
    ndi_T_base: np.ndarray,
    ecm_T_cam: np.ndarray,
    ndi_transform: np.ndarray,
) -> np.ndarray:
    """
    Return the tracked marker pose ``cam_T_marker`` in camera coordinates.

    ``inverse(ndi_T_base @ base_T_ecm @ ecm_T_cam)`` is ``cam_T_ndi``.
    Multiplying it by ``ndi_T_marker`` produces ``cam_T_marker``.
    """
    ndi_T_cam = (
        as_transform(ndi_T_base)
        @ as_transform(base_T_ecm)
        @ as_transform(ecm_T_cam)
    )
    cam_T_ndi = invert_transform(ndi_T_cam)
    cam_T_marker = cam_T_ndi @ as_transform(ndi_transform)
    return cam_T_marker


def project_camera_position(
    camera_position: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[float, float] | None:
    """
    Project a camera-frame 3D position into distorted pixel coordinates.
    Turn 3D coordinates into 2D pixel coordinates.
    """
    position = np.asarray(camera_position, dtype=float).reshape(3)
    if not np.isfinite(position).all() or position[2] <= 0:
        return None

    image_points, _ = cv2.projectPoints(
        position.reshape(1, 1, 3),
        np.zeros(3, dtype=float),
        np.zeros(3, dtype=float),
        camera_matrix,
        distortion,
    )
    pixel_x, pixel_y = image_points.reshape(2)
    if not np.isfinite([pixel_x, pixel_y]).all():
        return None
    return float(pixel_x), float(pixel_y)


def get_video_properties(
    video_path: Path,
) -> tuple[int, int, float, int]:
    """Return video width, height, frames per second, and frame count."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError(
            f"Invalid video properties: {width}x{height} at {fps} FPS."
        )
    return width, height, fps, frame_count


def prepare_frame_poses(
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Match NDI marker and time-varying ECM poses to every video frame.
    Using the reprojection configuration.
    """
    reprojection = config["reprojection"]
    ndi_config = config["ndi"]
    ecm_config = config["si_ecm"]
    matching_config = config["matching"]
    video_config = config["video"]

    ndi_data = clean_ndi_data(
        project_path(reprojection["ndi_csv"]),
        timestamp_column=ndi_config["timestamp_column"],
        toolkey=ndi_config["tool_id"],
    )
    ecm_data = clean_si_data(
        project_path(reprojection["si_csv"]),
        timestamp_column=ecm_config["timestamp_column"],
        arm_column=ecm_config["ecm_column"],
    )

    matched_data, _, matched_count, _ = match(
        ndi_data,
        ecm_data,
        tolerance=matching_config["time_tolerance"],
    )
    if matched_count == 0:
        raise RuntimeError("No NDI and ECM observations could be matched.")

    matched_data.columns = [
        "timestamp",
        "NDI Transform",
        "ECM Transform",
    ]
    return match_video(
        matched_data,
        project_path(reprojection["timestamp"]),
        tolerance=reprojection.get(
            "time_tolerance",
            matching_config["time_tolerance"],
        ),
    )


def write_reprojected_video(
    video_path: Path,
    output_path: Path,
    frame_poses: pd.DataFrame,
    ndi_T_base: np.ndarray,
    ecm_T_cam: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    marker_radius: int = 10,
) -> dict[str, int]:
    """Draw the NDI marker position on each frame and write an MP4 video."""
    width, height, fps, video_frame_count = get_video_properties(video_path)
    timestamp_frame_count = len(frame_poses)
    if timestamp_frame_count != video_frame_count:
        raise ValueError(
            "Video/timestamp frame-count mismatch: "
            f"{video_frame_count} video frames versus "
            f"{timestamp_frame_count} timestamps."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError(
            f"Could not initialize video input/output for {output_path}."
        )

    projected_count = 0
    outside_image_count = 0
    behind_camera_count = 0
    unmatched_count = 0
    processed_count = 0

    try:
        for frame_number in range(video_frame_count):
            success, frame = capture.read()
            if not success:
                raise RuntimeError(
                    f"Could not read video frame {frame_number}."
                )

            row = frame_poses.iloc[frame_number]
            ndi_transform = row["NDI Transform"]
            ecm_transform = row["ECM Transform"]

            if not (
                isinstance(ndi_transform, np.ndarray) # must be a np.ndarray
                and isinstance(ecm_transform, np.ndarray) # must be a np.ndarray
                and is_valid_transform(
                    ndi_transform,
                    rotation_atol=2e-5,
                )
                and is_valid_transform(
                    ecm_transform,
                    rotation_atol=2e-5,
                )
            ):
                unmatched_count += 1
                writer.write(frame)
                processed_count += 1
                continue

            cam_T_marker = NDI_in_cam(
                ecm_transform,
                ndi_T_base,
                ecm_T_cam,
                ndi_transform,
            )
            camera_position = cam_T_marker[:3, 3]
            pixel = project_camera_position(
                camera_position,
                camera_matrix,
                distortion,
            )

            if pixel is None:
                behind_camera_count += 1
            else:
                pixel_x, pixel_y = pixel
                # check it is within bounds of the image
                if 0 <= pixel_x < width and 0 <= pixel_y < height:
                    center = (int(round(pixel_x)), int(round(pixel_y)))
                    cv2.circle(
                        frame,
                        center,
                        marker_radius,
                        (0, 255, 0),
                        thickness=-1,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.circle(
                        frame,
                        center,
                        marker_radius + 3,
                        (255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        "NDI marker",
                        (center[0] + 15, center[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    projected_count += 1
                else:
                    outside_image_count += 1

            writer.write(frame)
            processed_count += 1
    finally:
        capture.release()
        writer.release()

    return {
        "processed_frames": processed_count,
        "projected_frames": projected_count,
        "unmatched_frames": unmatched_count,
        "behind_camera_frames": behind_camera_count,
        "outside_image_frames": outside_image_count,
    }


def main() -> None:
    """Load parameters, match poses to frames, and create the overlay video."""
    config = load_config()
    inputs = config["inputs"]
    reprojection = config["reprojection"]
    output_config = config["output"]
    camera_config = config["camera"]

    calibration_path = project_path(output_config["calib_file"])
    si_robot_parameter_path = project_path(inputs["si_robot_params"])
    camera_parameter_path = project_path(camera_config["parameters_file"])
    video_path = project_path(reprojection["video_input"])
    output_path = project_path(reprojection["output"])

    ndi_T_base = load_npz_transform(calibration_path, "ndi_T_base")
    ecm_T_cam = load_npz_transform(si_robot_parameter_path, "X")
    camera_matrix, distortion = load_camera_parameters(
        camera_parameter_path
    )

    width, height, fps, frame_count = get_video_properties(video_path)
    print(
        f"Video: {width}x{height}, {fps:.3f} FPS, "
        f"{frame_count} frames"
    )

    frame_poses = prepare_frame_poses(config)
    statistics = write_reprojected_video(
        video_path,
        output_path,
        frame_poses,
        ndi_T_base,
        ecm_T_cam,
        camera_matrix,
        distortion,
        marker_radius=int(reprojection.get("marker_radius", 10)),
    )

    print(f"Saved reprojected video to: {output_path}")
    for name, value in statistics.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
