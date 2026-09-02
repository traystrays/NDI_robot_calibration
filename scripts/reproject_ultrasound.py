"""Synchronize and overlay tracked BK ultrasound on the left ECM video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ndi_robot_registration.clean_ndi_data import clean_ndi_data
from ndi_robot_registration.clean_si_data import clean_si_data
from ndi_robot_registration.match import load_video_timestamps
from ndi_robot_registration.transforms import as_transform, invert_transform

try:
    from scripts.reproject_ndi import (
        load_camera_parameters,
        load_npz_transform,
    )
except ModuleNotFoundError:
    from reproject_ndi import load_camera_parameters, load_npz_transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DATA_DIR = Path(r"D:\20260831_reproj")
OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "20260831_reproj"
ECM_VIDEO = INPUT_DATA_DIR / "ecm_left_20260831_115144.mp4"
ECM_TIMESTAMPS = INPUT_DATA_DIR / "ecm_left_20260831_115144_timestamps.txt"
ULTRASOUND_VIDEO = INPUT_DATA_DIR / "us_20260831_115144.mp4"
ULTRASOUND_TIMESTAMPS = INPUT_DATA_DIR / "us_20260831_115144_timestamps.txt"
SI_CSV = INPUT_DATA_DIR / "data_local_part_1_31_8_2026_11_52_4.csv"
NDI_CSV = INPUT_DATA_DIR / "ParticipantData31-08-2026_11-51-56.csv"
IMAGE_TO_PROBE_FILE = OUTPUT_DATA_DIR / "image_to_probe_transform.npz"
NDI_TO_BASE_FILE = PROJECT_ROOT / "data" / "20260831_calib" / "calib.npz"
ECM_TO_CAMERA_FILE = PROJECT_ROOT / "data" / "si_robot" / "hand_eye_0727_python.npz"
CAMERA_PARAMETERS_FILE = PROJECT_ROOT / "data" / "si_robot" / "calib_intrinsics.npz"

# Exact raw-video rectangle from the fCal XML:
# ClipRectangleOrigin="180 169 0", ClipRectangleSize="558 727 1".
PLUS_RAW_CLIP_ORIGIN = (180, 169)
PLUS_RAW_CLIP_SIZE = (558, 727)

ULTRASOUND_PHYSICAL_SIZE_MM = (43.0, 50.0)
ULTRASOUND_ROLL_DEG = 0
ULTRASOUND_SCREEN_OFFSET_PX = (0.0, 0.0)

# Full-screen BK recording. This is deliberately configurable at the CLI because
# it is not the raw 558x727 Plus video stream described by ClipRectangleOrigin.
DEFAULT_SCREEN_ROI = (500, 0, 620, 580)  # x, y, width, height


def crop_plus_ultrasound_frame(frame: np.ndarray) -> np.ndarray:
    """Apply the exact raw BK crop configured in the supplied Plus XML."""
    image = np.asarray(frame)
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D frame; got {image.shape}.")
    x, y = PLUS_RAW_CLIP_ORIGIN
    width, height = PLUS_RAW_CLIP_SIZE
    if x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError(
            "Frame is too small for the Plus raw crop: need at least "
            f"{x + width}x{y + height}, got "
            f"{image.shape[1]}x{image.shape[0]}."
        )
    return image[y : y + height, x : x + width].copy()


def crop_recorded_ultrasound_frame(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] = DEFAULT_SCREEN_ROI,
) -> np.ndarray:
    """Return the scan area from a full-screen BK video frame."""
    image = np.asarray(frame)
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D frame; got {image.shape}.")

    x, y, width, height = roi
    if min(x, y) < 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid ultrasound ROI: {roi}.")
    if x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError(
            f"ROI {roi} exceeds frame size {image.shape[1]}x{image.shape[0]}."
        )
    return image[y : y + height, x : x + width].copy()


def _nearest_join(
    frames: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    sample_time: str,
    value_column: str,
    output_column: str,
    tolerance: str | pd.Timedelta,
) -> pd.DataFrame:
    """Attach the nearest sample and its signed timing error to frame rows."""
    frames = frames.copy()
    frames["ecm_video_timestamp"] = pd.to_datetime(
        frames["ecm_video_timestamp"], utc=True
    ).dt.tz_convert("America/Vancouver").dt.as_unit("us")
    right_time = f"{output_column}_timestamp"
    right = samples[[sample_time, value_column]].rename(
        columns={sample_time: right_time, value_column: output_column}
    )
    right[right_time] = pd.to_datetime(
        right[right_time], utc=True
    ).dt.tz_convert("America/Vancouver").dt.as_unit("us")
    result = pd.merge_asof(
        frames.sort_values("ecm_video_timestamp"),
        right.sort_values(right_time),
        left_on="ecm_video_timestamp",
        right_on=right_time,
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    result[f"{output_column}_time_error_ms"] = (
        result[right_time] - result["ecm_video_timestamp"]
    ).dt.total_seconds() * 1000.0
    return result


def synchronize_streams(
    *,
    ecm_tolerance: str = "20ms",
    ndi_tolerance: str = "40ms",
    ultrasound_tolerance: str = "20ms",
) -> pd.DataFrame:
    """Match ECM pose, NDI probe pose, and ultrasound frame to each ECM frame."""
    frames = load_video_timestamps(ECM_TIMESTAMPS).rename(
        columns={
            "frame_number": "ecm_frame_number",
            "video_timestamp": "ecm_video_timestamp",
        }
    )
    ecm = clean_si_data(SI_CSV, "Time_stamp", "mip Index (ECM)")
    ndi = clean_ndi_data(NDI_CSV, "Timestamp", 1)
    ultrasound = load_video_timestamps(ULTRASOUND_TIMESTAMPS).rename(
        columns={
            "frame_number": "ultrasound_frame_number",
            "video_timestamp": "ultrasound_timestamp",
        }
    )

    matched = _nearest_join(
        frames,
        ecm,
        sample_time="timestamp",
        value_column="Transforms",
        output_column="base_T_ecm",
        tolerance=ecm_tolerance,
    )
    matched = _nearest_join(
        matched,
        ndi,
        sample_time="timestamp",
        value_column="Transforms",
        output_column="ndi_T_probe",
        tolerance=ndi_tolerance,
    )
    matched = _nearest_join(
        matched,
        ultrasound,
        sample_time="ultrasound_timestamp",
        value_column="ultrasound_frame_number",
        output_column="ultrasound_frame_number",
        tolerance=ultrasound_tolerance,
    )
    return matched.sort_values("ecm_frame_number").reset_index(drop=True)


def load_image_to_probe(
    npz_path: Path = IMAGE_TO_PROBE_FILE,
    key: str | None = None,
) -> np.ndarray:
    """Load the calibrated pixel-to-probe affine matrix from NPZ.

    The matrix maps homogeneous image pixel coordinates to probe coordinates
    in millimetres. It is intentionally not validated as a rigid transform:
    its first two columns contain calibrated millimetres-per-pixel scale and
    may include shear.
    """
    if not npz_path.is_file():
        raise FileNotFoundError(f"Image-to-probe file not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as parameters:
        if key is None:
            key = next(
                (
                    candidate
                    for candidate in ("image_to_probe", "image_to_probe_transform")
                    if candidate in parameters
                ),
                None,
            )
        if key is None or key not in parameters:
            raise KeyError(
                f"{npz_path} has no {key!r}; available keys: {parameters.files}"
            )
        image_to_probe = np.asarray(parameters[key], dtype=float).copy()
    if image_to_probe.shape != (4, 4) or not np.isfinite(image_to_probe).all():
        raise ValueError("image_to_probe must be a finite 4x4 matrix.")
    if not np.allclose(image_to_probe[3], [0, 0, 0, 1]):
        raise ValueError("image_to_probe has an invalid homogeneous last row.")
    return image_to_probe


def ultrasound_corners_in_probe(
    probe_T_image: np.ndarray,
    physical_size_mm: tuple[float, float] = ULTRASOUND_PHYSICAL_SIZE_MM,
) -> np.ndarray:
    """Return TL, TR, BR, BL slice corners in probe coordinates (metres).

    The calibration supplies the image origin and lateral/depth directions.
    The independently known physical slice size supplies their magnitudes.
    """
    matrix = np.asarray(probe_T_image, dtype=float)
    width_mm, height_mm = physical_size_mm
    lateral = matrix[:3, 0]
    depth = matrix[:3, 1]
    lateral_norm = np.linalg.norm(lateral)
    depth_norm = np.linalg.norm(depth)
    if lateral_norm <= 0 or depth_norm <= 0:
        raise ValueError("Image-to-probe lateral/depth axes must be nonzero.")
    origin = matrix[:3, 3]
    width_vector = lateral / lateral_norm * width_mm
    height_vector = depth / depth_norm * height_mm
    corners_mm = np.array(
        [
            origin,
            origin + width_vector,
            origin + width_vector + height_vector,
            origin + height_vector,
        ]
    )
    return corners_mm / 1000.0


def roll_slice_about_depth_axis(
    corners: np.ndarray,
    angle_degrees: float = ULTRASOUND_ROLL_DEG,
) -> np.ndarray:
    """Roll a slice around its own depth axis, keeping its origin fixed."""
    points = np.asarray(corners, dtype=float)
    if points.shape != (4, 3):
        raise ValueError(f"Slice corners must have shape (4, 3); got {points.shape}.")
    origin = points[0]
    depth_axis = points[3] - origin
    norm = np.linalg.norm(depth_axis)
    if norm <= 0:
        raise ValueError("Slice depth axis must be nonzero.")
    x, y, z = depth_axis / norm
    angle = np.deg2rad(angle_degrees)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    one_minus_cosine = 1.0 - cosine
    rotation = np.array(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ]
    )
    return origin + (rotation @ (points - origin).T).T


def apply_screen_offset(
    image_points: np.ndarray,
    offset_px: tuple[float, float] = ULTRASOUND_SCREEN_OFFSET_PX,
) -> np.ndarray:
    """Apply the configured final ECM screen-space translation."""
    return np.asarray(image_points, dtype=np.float32) + np.asarray(
        offset_px, dtype=np.float32
    )


def project_ultrasound_corners(
    base_T_ecm: np.ndarray,
    ndi_T_probe: np.ndarray,
    ndi_T_base: np.ndarray,
    ecm_T_camera: np.ndarray,
    probe_corners: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray | None:
    """Project the four probe-frame ultrasound corners into the ECM image."""
    ndi_T_camera = (
        as_transform(ndi_T_base)
        @ as_transform(base_T_ecm)
        @ as_transform(ecm_T_camera)
    )
    camera_T_probe = invert_transform(ndi_T_camera) @ as_transform(ndi_T_probe)
    camera_points = (
        camera_T_probe[:3, :3] @ np.asarray(probe_corners).T
    ).T + camera_T_probe[:3, 3]
    if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
        return None
    pixels, _ = cv2.projectPoints(
        camera_points,
        np.zeros(3),
        np.zeros(3),
        camera_matrix,
        distortion,
    )
    return pixels.reshape(4, 2).astype(np.float32)


def overlay_slice(
    ecm_frame: np.ndarray,
    ultrasound_frame: np.ndarray,
    destination_corners: np.ndarray,
    *,
    opacity: float = 0.65,
) -> np.ndarray:
    """Perspective-warp one ultrasound frame onto the projected slice plane."""
    scan = np.asarray(ultrasound_frame)
    height, width = scan.shape[:2]
    source = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, destination_corners)
    output_size = (ecm_frame.shape[1], ecm_frame.shape[0])
    warped = cv2.warpPerspective(scan, homography, output_size)
    mask = cv2.warpPerspective(
        np.full((height, width), 255, dtype=np.uint8),
        homography,
        output_size,
    ).astype(np.float32) / 255.0
    alpha = np.clip(mask * opacity, 0.0, 1.0)[:, :, None]
    return np.clip(
        warped.astype(np.float32) * alpha
        + ecm_frame.astype(np.float32) * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)


def _read_frame(capture: cv2.VideoCapture, frame_number: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    success, frame = capture.read()
    if not success:
        raise RuntimeError(f"Could not read frame {frame_number}.")
    return frame


def render_preview(
    matched: pd.DataFrame,
    ecm_frame_number: int,
    output_path: Path,
    *,
    roi: tuple[int, int, int, int] = DEFAULT_SCREEN_ROI,
    opacity: float = 0.65,
) -> str:
    """Render one synchronized ultrasound overlay for geometry verification."""
    rows = matched[matched["ecm_frame_number"] == ecm_frame_number]
    if rows.empty:
        raise ValueError(f"No ECM frame {ecm_frame_number} in timestamp data.")
    row = rows.iloc[0]
    required = ("base_T_ecm", "ndi_T_probe", "ultrasound_frame_number")
    if any(
        row[name] is None
        or (not isinstance(row[name], np.ndarray) and pd.isna(row[name]))
        for name in required
    ):
        raise ValueError(f"ECM frame {ecm_frame_number} has unmatched data.")

    ecm_capture = cv2.VideoCapture(str(ECM_VIDEO))
    ultrasound_capture = cv2.VideoCapture(str(ULTRASOUND_VIDEO))
    try:
        ecm_frame = _read_frame(ecm_capture, ecm_frame_number)
        ultrasound_frame = crop_recorded_ultrasound_frame(
            _read_frame(ultrasound_capture, int(row["ultrasound_frame_number"])),
            roi,
        )
    finally:
        ecm_capture.release()
        ultrasound_capture.release()

    probe_T_image = load_image_to_probe()
    corners = roll_slice_about_depth_axis(
        ultrasound_corners_in_probe(probe_T_image)
    )
    destination = project_ultrasound_corners(
        row["base_T_ecm"],
        row["ndi_T_probe"],
        load_npz_transform(NDI_TO_BASE_FILE, "ndi_T_base"),
        load_npz_transform(ECM_TO_CAMERA_FILE, "X"),
        corners,
        *load_camera_parameters(CAMERA_PARAMETERS_FILE),
    )
    if destination is not None:
        destination = apply_screen_offset(destination)
    status = "behind_camera"
    if destination is not None:
        overlaps = (
            destination[:, 0].max() >= 0
            and destination[:, 1].max() >= 0
            and destination[:, 0].min() < ecm_frame.shape[1]
            and destination[:, 1].min() < ecm_frame.shape[0]
        )
        status = "outside_image"
        if overlaps:
            ecm_frame = overlay_slice(
                ecm_frame, ultrasound_frame, destination, opacity=opacity
            )
            cv2.polylines(
                ecm_frame,
                [np.rint(destination).astype(np.int32)],
                True,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            status = "rendered"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), ecm_frame):
        raise RuntimeError(f"Could not save {output_path}.")
    return status


def write_overlay_video(
    matched: pd.DataFrame,
    output_path: Path,
    *,
    roi: tuple[int, int, int, int] = DEFAULT_SCREEN_ROI,
    opacity: float = 0.65,
) -> dict[str, int]:
    """Write the synchronized tracked-ultrasound overlay on the ECM video."""
    ecm_capture = cv2.VideoCapture(str(ECM_VIDEO))
    ultrasound_capture = cv2.VideoCapture(str(ULTRASOUND_VIDEO))
    width = int(ecm_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(ecm_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(ecm_capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(ecm_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not ecm_capture.isOpened() or not ultrasound_capture.isOpened():
        ecm_capture.release()
        ultrasound_capture.release()
        raise RuntimeError("Could not open the ECM or ultrasound video.")
    if frame_count != len(matched):
        ecm_capture.release()
        ultrasound_capture.release()
        raise ValueError(
            f"ECM video has {frame_count} frames but matching has {len(matched)}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        ecm_capture.release()
        ultrasound_capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    ndi_T_base = load_npz_transform(NDI_TO_BASE_FILE, "ndi_T_base")
    ecm_T_camera = load_npz_transform(ECM_TO_CAMERA_FILE, "X")
    camera_matrix, distortion = load_camera_parameters(CAMERA_PARAMETERS_FILE)
    probe_corners = roll_slice_about_depth_axis(
        ultrasound_corners_in_probe(load_image_to_probe())
    )
    statistics = {
        "processed": 0,
        "rendered": 0,
        "unmatched": 0,
        "behind_camera": 0,
        "outside_image": 0,
    }
    ultrasound_current_number = -1
    ultrasound_current_frame: np.ndarray | None = None

    try:
        for frame_index, row in matched.iterrows():
            success, ecm_frame = ecm_capture.read()
            if not success:
                raise RuntimeError(f"Could not read ECM frame {frame_index}.")

            pose_valid = isinstance(row["base_T_ecm"], np.ndarray) and isinstance(
                row["ndi_T_probe"], np.ndarray
            )
            ultrasound_number = row["ultrasound_frame_number"]
            if not pose_valid or pd.isna(ultrasound_number):
                statistics["unmatched"] += 1
                writer.write(ecm_frame)
                statistics["processed"] += 1
                continue

            destination = project_ultrasound_corners(
                row["base_T_ecm"],
                row["ndi_T_probe"],
                ndi_T_base,
                ecm_T_camera,
                probe_corners,
                camera_matrix,
                distortion,
            )
            if destination is not None:
                destination = apply_screen_offset(destination)
            if destination is None:
                statistics["behind_camera"] += 1
            else:
                overlaps = (
                    destination[:, 0].max() >= 0
                    and destination[:, 1].max() >= 0
                    and destination[:, 0].min() < width
                    and destination[:, 1].min() < height
                )
                if not overlaps:
                    statistics["outside_image"] += 1
                else:
                    target_number = int(ultrasound_number)
                    if target_number < ultrasound_current_number:
                        ultrasound_capture.set(
                            cv2.CAP_PROP_POS_FRAMES, target_number
                        )
                        ultrasound_current_number = target_number - 1
                    while ultrasound_current_number < target_number:
                        success, decoded_frame = ultrasound_capture.read()
                        if not success:
                            raise RuntimeError(
                                "Could not decode ultrasound frame "
                                f"{ultrasound_current_number + 1}."
                            )
                        ultrasound_current_number += 1
                        ultrasound_current_frame = decoded_frame
                    if ultrasound_current_frame is None:
                        raise RuntimeError("No ultrasound frame was decoded.")
                    ultrasound_frame = crop_recorded_ultrasound_frame(
                        ultrasound_current_frame,
                        roi,
                    )
                    
                    ultrasound_frame = cv2.rotate(
                        ultrasound_frame,
                        cv2.ROTATE_180,
                    )
                    ecm_frame = overlay_slice(
                        ecm_frame,
                        ultrasound_frame,
                        destination,
                        opacity=opacity,
                    )
                    cv2.polylines(
                        ecm_frame,
                        [np.rint(destination).astype(np.int32)],
                        True,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    statistics["rendered"] += 1
            writer.write(ecm_frame)
            statistics["processed"] += 1
    finally:
        ecm_capture.release()
        ultrasound_capture.release()
        writer.release()
    return statistics


def synchronization_summary(matched: pd.DataFrame) -> str:
    """Return a compact coverage report for the matched streams."""
    total = len(matched)
    lines = [f"ECM video frames: {total}"]
    for column, label in (
        ("base_T_ecm", "ECM pose"),
        ("ndi_T_probe", "NDI probe pose"),
        ("ultrasound_frame_number", "ultrasound frame"),
    ):
        count = int(matched[column].notna().sum())
        lines.append(f"{label}: {count}/{total} matched")
    all_matched = matched[
        ["base_T_ecm", "ndi_T_probe", "ultrasound_frame_number"]
    ].notna().all(axis=1)
    lines.append(f"all required streams: {int(all_matched.sum())}/{total} matched")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-frame", type=int, default=None)
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument(
        "--video-output",
        type=Path,
        default=OUTPUT_DATA_DIR / "ultrasound_on_ecm_overlay.mp4",
    )
    parser.add_argument(
        "--preview-output",
        type=Path,
        default=OUTPUT_DATA_DIR / "ultrasound_overlay_preview.png",
    )
    parser.add_argument(
        "--matches-output",
        type=Path,
        default=OUTPUT_DATA_DIR / "ultrasound_ecm_matches.csv",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        default=DEFAULT_SCREEN_ROI,
    )
    args = parser.parse_args()

    matched = synchronize_streams()
    print(synchronization_summary(matched))

    export = matched.drop(columns=["base_T_ecm", "ndi_T_probe"])
    args.matches_output.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(args.matches_output, index=False)
    print(f"Saved matches: {args.matches_output}")

    if args.preview_frame is not None:
        status = render_preview(
            matched,
            args.preview_frame,
            args.preview_output,
            roi=tuple(args.roi),
        )
        print(f"Preview {status}: {args.preview_output}")

    if args.write_video:
        statistics = write_overlay_video(
            matched,
            args.video_output,
            roi=tuple(args.roi),
        )
        print(f"Saved overlay video: {args.video_output}")
        for name, value in statistics.items():
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
