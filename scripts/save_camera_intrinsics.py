"""Save the calibrated camera intrinsics used for NDI reprojection."""

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "si_robot" / "camera_intrinsics.npz"

CAMERA_MATRIX = np.array(
    [
        [1.552570805831235e3, 0.0, 9.708891005219707e2],
        [0.0, 1.558233927349110e3, 6.895247852015215e2],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)

# OpenCV order: k1, k2, p1, p2, k3.
DISTORTION_COEFFICIENTS = np.array(
    [-0.297863212023482, 0.403564940306343, 0.0, 0.0, 0.0],
    dtype=float,
)


def main() -> None:
    """Write the intrinsic calibration to a numeric NPZ archive."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        camera_matrix=CAMERA_MATRIX,
        dist_coeffs=DISTORTION_COEFFICIENTS,
    )
    print(f"Saved camera intrinsics to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
