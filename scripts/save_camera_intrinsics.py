"""Save the calibrated camera intrinsics as npz file"""

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "si_robot" / "image_to_probe_transform.npz"

IMAGE_TO_PROBE = np.array(
    [
        [-0.0070738, 0.0648917, -6.3876e-05, 180.39],
        [ 0.065899, 0.00670635, 0.00420122, 34.7653],
        [0.00422142, 0.0003658, -0.0656911, 13.3835],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def main() -> None:
    """Write the intrinsic calibration to a numeric NPZ archive."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        image_to_probe = IMAGE_TO_PROBE,
    )
    print(f"Saved camera intrinsics to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
