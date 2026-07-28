"""Interactive selection of synchronized video frames for calibration."""

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "frame_number",
    "NDI Transform",
    "SI Transform",
}


def valid_calibration_frames(synced_data: pd.DataFrame) -> pd.DataFrame:
    """Return synchronized rows containing both calibration transforms."""
    missing = REQUIRED_COLUMNS.difference(synced_data.columns)
    if missing:
        raise ValueError(
            "Synchronized data is missing column(s): "
            + ", ".join(sorted(missing))
        )

    valid_mask = (
        synced_data["NDI Transform"].map(
            lambda value: isinstance(value, np.ndarray)
            and value.shape == (4, 4)
            and np.isfinite(value).all()
        )
        & synced_data["SI Transform"].map(
            lambda value: isinstance(value, np.ndarray)
            and value.shape == (4, 4)
            and np.isfinite(value).all()
        )
    )
    valid = synced_data.loc[valid_mask].copy()
    valid["frame_number"] = valid["frame_number"].astype(int)
    return valid.sort_values("frame_number").reset_index(drop=True)


def select_calibration_frames(
    video_path: str | Path,
    synced_data: pd.DataFrame,
    *,
    selection_path: str | Path | None = None,
) -> pd.DataFrame:
    """Open a frame picker and return the hand-selected synchronized rows.

    Only video frames with valid NDI and SI transforms are offered. Press S to
    toggle a frame, A/D to move, and Enter or Q to finish. At least four
    observations are required to produce three AX=XB motion pairs.
    """
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required for frame selection. Install the project "
            "dependencies with `python -m pip install -e .`."
        ) from error

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    valid = valid_calibration_frames(synced_data)
    if len(valid) < 4:
        raise ValueError(
            "Fewer than four video frames have valid synchronized NDI and SI "
            "transforms."
        )

    frame_numbers = valid["frame_number"].tolist()
    selected: set[int] = set()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    window = "Calibration Frame Selection"
    trackbar = "Valid frame"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1600, 900)
    cv2.createTrackbar(trackbar, window, 0, len(frame_numbers) - 1, lambda _: None)
    previous_position = -1
    display_frame = None

    print(
        "Frame selection: A/D = previous/next, S = select/unselect, "
        "Enter or Q = finish."
    )
    try:
        while True:
            try:
                position = cv2.getTrackbarPos(trackbar, window)
            except cv2.error:
                break

            if position != previous_position:
                previous_position = position
                frame_number = frame_numbers[position]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"Could not read video frame {frame_number}."
                    )
                display_frame = frame

            frame_number = frame_numbers[position]
            shown = display_frame.copy()
            is_selected = frame_number in selected
            colour = (0, 255, 0) if is_selected else (0, 200, 255)
            label = (
                f"Valid {position + 1}/{len(frame_numbers)} | "
                f"Frame {frame_number} | "
                f"{'SELECTED' if is_selected else 'not selected'} | "
                f"total {len(selected)}"
            )
            cv2.putText(
                shown, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, colour, 2, cv2.LINE_AA,
            )
            cv2.imshow(window, shown)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("a"), ord("A")):
                cv2.setTrackbarPos(trackbar, window, max(position - 1, 0))
            elif key in (ord("d"), ord("D")):
                cv2.setTrackbarPos(
                    trackbar, window, min(position + 1, len(frame_numbers) - 1)
                )
            elif key in (ord("s"), ord("S")):
                if frame_number in selected:
                    selected.remove(frame_number)
                else:
                    selected.add(frame_number)
            elif key in (13, ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(selected) < 4:
        raise ValueError(
            "At least four frames must be selected for calibration; "
            f"only {len(selected)} were selected."
        )

    result = valid[valid["frame_number"].isin(selected)].copy()
    result = result.sort_values("frame_number").reset_index(drop=True)

    if selection_path is not None:
        selection_path = Path(selection_path)
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        result[
            ["frame_number", "video_timestamp", "matched_timestamp",
             "time_difference"]
        ].to_csv(selection_path, index=False)
        print(f"Saved selected calibration frames to: {selection_path}")

    return result
