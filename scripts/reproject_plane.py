"""Attach a textured plane to the robot end effector and render it in video."""

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from ndi_robot_registration.clean_si_data import clean_si_data
from ndi_robot_registration.match import match, match_video
from ndi_robot_registration.transforms import (
    as_transform,
    invert_transform,
    is_valid_transform,
)

from toolbox.projection import Projector

try:
    from scripts.reproject import (
        CONFIG_PATH,
        get_video_properties,
        load_camera_parameters,
        load_config,
        load_npz_transform,
        project_path,
    )
except ModuleNotFoundError:
    # Support direct execution with ``python scripts/reproject_plane.py``.
    from reproject import (
        CONFIG_PATH,
        get_video_properties,
        load_camera_parameters,
        load_config,
        load_npz_transform,
        project_path,
    )


# Plane dimensions [x, y, z] in metres in the end-effector frame.
# A plane has exactly two nonzero dimensions.
SIZE = np.array([0.050, 0.050, 0.0], dtype=float)

# Translation from the end-effector origin to the plane center, expressed in
# end-effector coordinates and metres.
# Realistically this should be a transform matrix
PLANE_OFFSET = np.array([0.0, 0.0, 0.0], dtype=float)


def plane_corners_from_size(
    size: np.ndarray,
    *,
    centered: bool = True,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return four plane corners expressed in end-effector coordinates.

    ``size`` is ``[x, y, z]`` and must contain exactly two positive values and
    one zero value. The zero dimension determines the plane normal. With
    ``centered=True``, the end-effector origin is at the plane center.
    """
    dimensions = np.asarray(size, dtype=float).reshape(3)
    if not np.isfinite(dimensions).all() or np.any(dimensions < 0):
        raise ValueError("SIZE must contain three finite, nonnegative values.")

    active_axes = np.flatnonzero(dimensions > 0)
    if len(active_axes) != 2:
        raise ValueError(
            "SIZE must have exactly two positive dimensions and one zero "
            "dimension, for example [0.05, 0.05, 0.0]."
        )

    axis_u, axis_v = (int(axis) for axis in active_axes)
    length_u = dimensions[axis_u]
    length_v = dimensions[axis_v]
    if centered:
        coordinates = np.array(
            [
                [-length_u / 2, -length_v / 2],
                [length_u / 2, -length_v / 2],
                [length_u / 2, length_v / 2],
                [-length_u / 2, length_v / 2],
            ],
            dtype=float,
        )
    else:
        coordinates = np.array(
            [
                [0.0, 0.0],
                [length_u, 0.0],
                [length_u, length_v],
                [0.0, length_v],
            ],
            dtype=float,
        )

    corners = np.zeros((4, 3), dtype=float)
    corners[:, axis_u] = coordinates[:, 0]
    corners[:, axis_v] = coordinates[:, 1]

    plane_offset = (
        np.zeros(3, dtype=float)
        if offset is None
        else np.asarray(offset, dtype=float).reshape(3)
    )
    if not np.isfinite(plane_offset).all():
        raise ValueError("Plane offset must contain three finite values.")
    return corners + plane_offset


def transform_points(
    transform: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Apply a rigid transform to an Nx3 array of points."""
    matrix = as_transform(transform)
    points_array = np.asarray(points, dtype=float)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    return (
        (matrix[:3, :3] @ points_array.T).T
        + matrix[:3, 3]
    )


def coordinate_axis_points(
    length: float,
    *,
    origin: np.ndarray | None = None,
) -> np.ndarray:
    """Return the origin and positive X/Y/Z endpoints in one frame."""
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Axis length must be a positive finite value.")
    axis_origin = (
        np.zeros(3, dtype=float)
        if origin is None
        else np.asarray(origin, dtype=float).reshape(3)
    )
    return np.vstack(
        [
            axis_origin,
            axis_origin + [length, 0.0, 0.0],
            axis_origin + [0.0, length, 0.0],
            axis_origin + [0.0, 0.0, length],
        ]
    )


def plane_in_camera(
    base_T_end_effector: np.ndarray,
    base_T_ecm: np.ndarray,
    ecm_T_cam: np.ndarray,
    plane_corners_end_effector: np.ndarray,
) -> np.ndarray:
    """Return the end-effector-attached plane corners in camera coordinates."""
    base_T_cam = as_transform(base_T_ecm) @ as_transform(ecm_T_cam)
    cam_T_end_effector = (
        invert_transform(base_T_cam)
        @ as_transform(base_T_end_effector)
    )
    return transform_points(
        cam_T_end_effector,
        plane_corners_end_effector,
    )


def project_camera_points(
    camera_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray | None:
    """Project camera-frame 3D points into image coordinates."""
    points = np.asarray(camera_points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) == 0
        or not np.isfinite(points).all()
    ):
        raise ValueError("Camera points must have shape (N, 3).")
    if np.any(points[:, 2] <= 0):
        return None

    image_points, _ = cv2.projectPoints(
        points.reshape(-1, 1, 3),
        np.zeros(3, dtype=float),
        np.zeros(3, dtype=float),
        camera_matrix,
        distortion,
    )
    pixels = image_points.reshape(-1, 2)
    return pixels if np.isfinite(pixels).all() else None


def draw_coordinate_axes(
    frame: np.ndarray,
    image_points: np.ndarray,
) -> None:
    """Draw RGB X/Y/Z arrows from four projected axis points."""
    pixels = np.round(np.asarray(image_points, dtype=float)).astype(int)
    if pixels.shape != (4, 2):
        raise ValueError("Axis image points must have shape (4, 2).")

    origin = tuple(pixels[0])
    axes = (
        ("X", tuple(pixels[1]), (0, 0, 255)),
        ("Y", tuple(pixels[2]), (0, 255, 0)),
        ("Z", tuple(pixels[3]), (255, 0, 0)),
    )
    for label, endpoint, colour in axes:
        cv2.arrowedLine(
            frame,
            origin,
            endpoint,
            colour,
            thickness=3,
            line_type=cv2.LINE_AA,
            tipLength=0.18,
        )
        cv2.putText(
            frame,
            label,
            (endpoint[0] + 5, endpoint[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2,
            cv2.LINE_AA,
        )
    cv2.circle(frame, origin, 4, (255, 255, 255), thickness=-1)


def create_default_plane_image(
    width: int = 640,
    height: int = 640,
) -> np.ndarray:
    """Create a visible checkerboard texture when no image is configured."""
    image = np.full((height, width, 3), 235, dtype=np.uint8)
    cells = 8
    cell_width = width // cells
    cell_height = height // cells
    for row in range(cells):
        for column in range(cells):
            if (row + column) % 2:
                top_left = (column * cell_width, row * cell_height)
                bottom_right = (
                    min((column + 1) * cell_width, width) - 1,
                    min((row + 1) * cell_height, height) - 1,
                )
                cv2.rectangle(
                    image,
                    top_left,
                    bottom_right,
                    (60, 120, 220),
                    thickness=-1,
                )
    cv2.putText(
        image,
        "END EFFECTOR PLANE",
        (35, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    return image


def load_plane_image(image_path: Path | None) -> np.ndarray:
    """Load a configured BGR/BGRA image or create a default texture."""
    if image_path is None:
        return create_default_plane_image()
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load plane image: {image_path}")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Plane image must contain BGR or BGRA pixels.")
    return image


def overlay_plane_image(
    frame: np.ndarray,
    plane_image: np.ndarray,
    destination_corners: np.ndarray,
    *,
    opacity: float = 0.75,
) -> np.ndarray:
    """Warp a source image onto four projected plane corners and composite it."""
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("Plane opacity must be between 0 and 1.")

    destination = np.asarray(destination_corners, dtype=np.float32)
    if destination.shape != (4, 2) or not np.isfinite(destination).all():
        raise ValueError("destination_corners must have shape (4, 2).")
    if abs(cv2.contourArea(destination)) < 1.0:
        return frame

    image_height, image_width = plane_image.shape[:2]
    source = np.array(
        [
            [0.0, 0.0],
            [image_width - 1.0, 0.0],
            [image_width - 1.0, image_height - 1.0],
            [0.0, image_height - 1.0],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, destination)
    frame_size = (frame.shape[1], frame.shape[0])

    source_bgr = (
        plane_image[:, :, :3]
        if plane_image.shape[2] == 4
        else plane_image
    )
    warped_image = cv2.warpPerspective(
        source_bgr,
        homography,
        frame_size,
        flags=cv2.INTER_LINEAR,
    )

    if plane_image.shape[2] == 4:
        source_alpha = plane_image[:, :, 3]
    else:
        source_alpha = np.full(
            (image_height, image_width),
            255,
            dtype=np.uint8,
        )
    warped_alpha = cv2.warpPerspective(
        source_alpha,
        homography,
        frame_size,
        flags=cv2.INTER_LINEAR,
    ).astype(np.float32) / 255.0
    warped_alpha *= opacity

    alpha = warped_alpha[:, :, None]
    composited = (
        warped_image.astype(np.float32) * alpha
        + frame.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(composited, 0, 255).astype(np.uint8)


def size_from_config(config: dict[str, Any]) -> np.ndarray:
    """Return configured plane size in metres, falling back to ``SIZE``."""
    plane_config = config.get("plane", {})
    size = np.asarray(plane_config.get("size", SIZE), dtype=float)
    units = plane_config.get("units", "m")
    if units == "mm":
        size = size / 1000.0
    elif units != "m":
        raise ValueError("plane.units must be 'm' or 'mm'.")
    return size


def offset_from_config(config: dict[str, Any]) -> np.ndarray:
    """Return configured plane-center offset in metres."""
    plane_config = config.get("plane", {})
    offset = np.asarray(
        plane_config.get("offset", PLANE_OFFSET),
        dtype=float,
    )
    units = plane_config.get("units", "m")
    return offset / 1000.0 if units == "mm" else offset


def axis_length_from_config(config: dict[str, Any]) -> float:
    """Return the coordinate-axis display length in metres."""
    plane_config = config.get("plane", {})
    default_length = float(np.max(size_from_config(config)) * 0.6)
    if "axis_length" not in plane_config:
        return default_length
    length = float(plane_config["axis_length"])
    units = plane_config.get("units", "m")
    return length / 1000.0 if units == "mm" else length


def prepare_plane_frame_poses(config: dict[str, Any]) -> pd.DataFrame:
    """Match time-varying end-effector and ECM poses to video frames."""
    reprojection = config["reprojection"]
    arm_config = config["si_arm"]
    ecm_config = config["si_ecm"]
    matching_config = config["matching"]

    arm_data = clean_si_data(
        project_path(reprojection["si_csv"]),
        timestamp_column=arm_config["timestamp_column"],
        arm_column=arm_config["arm_column"],
    )
    ecm_data = clean_si_data(
        project_path(reprojection["si_csv"]),
        timestamp_column=ecm_config["timestamp_column"],
        arm_column=ecm_config["ecm_column"],
    )
    matched_data, _, matched_count, _ = match(
        arm_data,
        ecm_data,
        tolerance=matching_config["time_tolerance"],
    )
    if matched_count == 0:
        raise RuntimeError("No end-effector and ECM observations matched.")

    matched_data.columns = [
        "timestamp",
        "End Effector Transform",
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


def render_plane_frame(
    frame: np.ndarray,
    frame_pose: pd.Series,
    ecm_T_cam: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    plane_corners_end_effector: np.ndarray,
    axis_points_end_effector: np.ndarray,
    plane_image: np.ndarray,
    *,
    opacity: float = 0.75,
) -> tuple[np.ndarray, str]:
    """Render the plane on one frame and return the frame and status."""
    end_effector_transform = frame_pose["End Effector Transform"]
    ecm_transform = frame_pose["ECM Transform"]
    valid = (
        isinstance(end_effector_transform, np.ndarray)
        and isinstance(ecm_transform, np.ndarray)
        and is_valid_transform(
            end_effector_transform,
            rotation_atol=2e-5,
        )
        and is_valid_transform(
            ecm_transform,
            rotation_atol=2e-5,
        )
    )
    if not valid:
        return frame, "unmatched"

    camera_corners = plane_in_camera(
        end_effector_transform,
        ecm_transform,
        ecm_T_cam,
        plane_corners_end_effector,
    )
    image_corners = project_camera_points(
        camera_corners,
        camera_matrix,
        distortion,
    )
    if image_corners is None:
        return frame, "behind_camera"

    height, width = frame.shape[:2]
    entirely_outside = (
        image_corners[:, 0].max() < 0
        or image_corners[:, 1].max() < 0
        or image_corners[:, 0].min() >= width
        or image_corners[:, 1].min() >= height
    )
    if entirely_outside:
        return frame, "outside_image"

    rendered_frame = overlay_plane_image(
        frame,
        plane_image,
        image_corners,
        opacity=opacity,
    )
    cv2.polylines(
        rendered_frame,
        [np.round(image_corners).astype(np.int32)],
        isClosed=True,
        color=(0, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    base_T_cam = as_transform(ecm_transform) @ as_transform(ecm_T_cam)
    cam_T_end_effector = (
        invert_transform(base_T_cam)
        @ as_transform(end_effector_transform)
    )
    camera_axis_points = transform_points(
        cam_T_end_effector,
        axis_points_end_effector,
    )
    image_axis_points = project_camera_points(
        camera_axis_points,
        camera_matrix,
        distortion,
    )
    if image_axis_points is not None:
        draw_coordinate_axes(rendered_frame, image_axis_points)
    return rendered_frame, "rendered"


def preview_plane_frame(
    video_path: Path,
    frame_number: int,
    frame_poses: pd.DataFrame,
    ecm_T_cam: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    plane_corners_end_effector: np.ndarray,
    axis_points_end_effector: np.ndarray,
    plane_image: np.ndarray,
    *,
    opacity: float = 0.75,
    preview_output: Path | None = None,
    display: bool = True,
) -> str:
    """Render one indexed video frame, optionally save it, and display it."""
    _, _, _, frame_count = get_video_properties(video_path)
    if not 0 <= frame_number < frame_count:
        raise ValueError(
            f"Frame number must be between 0 and {frame_count - 1}; "
            f"got {frame_number}."
        )
    if frame_number >= len(frame_poses):
        raise ValueError(
            f"No timestamp row exists for frame {frame_number}."
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f"Could not read frame {frame_number}.")

    rendered_frame, status = render_plane_frame(
        frame,
        frame_poses.iloc[frame_number],
        ecm_T_cam,
        camera_matrix,
        distortion,
        plane_corners_end_effector,
        axis_points_end_effector,
        plane_image,
        opacity=opacity,
    )

    cv2.putText(
        rendered_frame,
        f"Frame {frame_number} | {status}",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if preview_output is not None:
        preview_output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(preview_output), rendered_frame):
            raise RuntimeError(
                f"Could not save preview image: {preview_output}"
            )
        print(f"Saved frame preview to: {preview_output}")

    if display:
        cv2.imshow("End-effector plane preview", rendered_frame)
        print("Press any key in the preview window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return status


def write_plane_video(
    video_path: Path,
    output_path: Path,
    frame_poses: pd.DataFrame,
    ecm_T_cam: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    plane_corners_end_effector: np.ndarray,
    axis_points_end_effector: np.ndarray,
    plane_image: np.ndarray,
    *,
    opacity: float = 0.75,
) -> dict[str, int]:
    """Render an end-effector-attached plane into every matched video frame."""
    width, height, fps, frame_count = get_video_properties(video_path)
    if len(frame_poses) != frame_count:
        raise ValueError(
            f"Video has {frame_count} frames but timestamps have "
            f"{len(frame_poses)} rows."
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
        raise RuntimeError("Could not initialize plane video input/output.")

    rendered = 0
    unmatched = 0
    behind_camera = 0
    outside_image = 0

    try:
        for frame_number in range(frame_count):
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"Could not read frame {frame_number}.")

            frame, status = render_plane_frame(
                frame,
                frame_poses.iloc[frame_number],
                ecm_T_cam,
                camera_matrix,
                distortion,
                plane_corners_end_effector,
                axis_points_end_effector,
                plane_image,
                opacity=opacity,
            )
            if status == "unmatched":
                unmatched += 1
            elif status == "behind_camera":
                behind_camera += 1
            elif status == "outside_image":
                outside_image += 1
            else:
                rendered += 1
            writer.write(frame)
    finally:
        capture.release()
        writer.release()

    return {
        "processed_frames": frame_count,
        "rendered_frames": rendered,
        "unmatched_frames": unmatched,
        "behind_camera_frames": behind_camera,
        "outside_image_frames": outside_image,
    }


def main() -> None:
    """Preview one plane frame or write the full plane-overlay video."""
    parser = argparse.ArgumentParser(
        description="Reproject an end-effector-attached plane."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--frame",
        type=int,
        help="Render and preview one zero-based frame number.",
    )
    mode.add_argument(
        "--write-video",
        action="store_true",
        help="Render and save the complete configured output video.",
    )
    parser.add_argument(
        "--save-preview",
        type=Path,
        default=None,
        help="Optional path at which to save the selected frame as an image.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a GUI preview window (use with --save-preview).",
    )
    arguments = parser.parse_args()

    config = load_config(CONFIG_PATH)
    reprojection = config["reprojection"]
    plane_config = config.get("plane", {})

    video_path = project_path(reprojection["video_input"])
    output_path = project_path(
        plane_config.get(
            "output",
            str(Path(reprojection["output"]).with_name("plane_overlay.mp4")),
        )
    )
    parameter_path = project_path(config["inputs"]["si_robot_params"])
    camera_parameter_path = project_path(
        config["camera"]["parameters_file"]
    )
    image_value = plane_config.get("image")
    image_path = project_path(image_value) if image_value else None

    ecm_T_cam = load_npz_transform(parameter_path, "X")
    camera_matrix, distortion = load_camera_parameters(
        camera_parameter_path
    )
    plane_corners = plane_corners_from_size(
        size_from_config(config),
        centered=bool(plane_config.get("centered", True)),
        offset=offset_from_config(config),
    )
    axis_points = coordinate_axis_points(
        axis_length_from_config(config),
    )
    plane_image = load_plane_image(image_path)
    frame_poses = prepare_plane_frame_poses(config)

    print(f"Plane corners in end-effector frame (metres):\n{plane_corners}")

    opacity = float(plane_config.get("opacity", 0.75))
    if arguments.frame is not None:
        preview_output = arguments.save_preview
        if preview_output is not None and not preview_output.is_absolute():
            preview_output = project_path(str(preview_output))
        status = preview_plane_frame(
            video_path,
            arguments.frame,
            frame_poses,
            ecm_T_cam,
            camera_matrix,
            distortion,
            plane_corners,
            axis_points,
            plane_image,
            opacity=opacity,
            preview_output=preview_output,
            display=not arguments.no_display,
        )
        print(f"Frame {arguments.frame} projection status: {status}")
        return

    width, height, fps, frame_count = get_video_properties(video_path)
    print(
        f"Video: {width}x{height}, {fps:.3f} FPS, "
        f"{frame_count} frames"
    )
    statistics = write_plane_video(
        video_path,
        output_path,
        frame_poses,
        ecm_T_cam,
        camera_matrix,
        distortion,
        plane_corners,
        axis_points,
        plane_image,
        opacity=opacity,
    )
    print(f"Saved plane-overlay video to: {output_path}")
    for name, value in statistics.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
