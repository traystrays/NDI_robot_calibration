"""Save the image-to-probe transform in the reference NPZ format."""

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "si_robot" / "image_to_probe_transform.npz"

IMAGE_TO_PROBE = np.array(
    [
        [0.045, 0.998, 0.012, 110.713],
        [-0.999, 0.045, -0.003, 22.2082],
        [-0.003, -0.012, 0.999, -13.7571],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)


def main() -> None:
    """Write a 4x4 float64 array under the reference image_to_probe key."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        image_to_probe=IMAGE_TO_PROBE,
    )
    print(f"Saved image-to-probe transform to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
