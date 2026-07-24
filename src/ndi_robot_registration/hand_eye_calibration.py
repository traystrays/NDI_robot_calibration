"""Hand-eye and robot-base-to-NDI calibration workflow."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .transforms import (
    as_transform,
    invert_transform,
    is_valid_transform,
    proper_rotation,
    rotation_angle_degrees,
    translation_distance,
)


def solve_ax_xb(
    a_matrices: list[np.ndarray],
    b_matrices: list[np.ndarray],
) -> np.ndarray:
    """Solve ``A X = X B`` by linear least squares.

    Rotation is found from the Kronecker-product system and projected onto
    SO(3). Translation is then found from a second least-squares system.
    Motions about multiple, non-parallel axes are required.
    """
    if len(a_matrices) != len(b_matrices):
        raise ValueError("A and B must contain the same number of motions.")
    if len(a_matrices) < 3:
        raise ValueError("At least three motion pairs are required.")

    a_values = [as_transform(item) for item in a_matrices]
    b_values = [as_transform(item) for item in b_matrices]

    rotation_system = np.vstack(
        [
            np.kron(np.eye(3), a[:3, :3])
            - np.kron(b[:3, :3].T, np.eye(3))
            for a, b in zip(a_values, b_values)
        ]
    )
    _, singular_values, vt_matrix = np.linalg.svd(rotation_system)
    raw_rotation = vt_matrix[-1].reshape((3, 3), order="F")

    # The homogeneous solution has arbitrary sign and scale.
    if np.linalg.det(raw_rotation) < 0:
        raw_rotation *= -1
    rotation_x = proper_rotation(raw_rotation)

    translation_system = np.vstack(
        [a[:3, :3] - np.eye(3) for a in a_values]
    )
    translation_rhs = np.concatenate(
        [
            rotation_x @ b[:3, 3] - a[:3, 3]
            for a, b in zip(a_values, b_values)
        ]
    )

    if np.linalg.matrix_rank(translation_system) < 3:
        raise ValueError(
            "Selected motions do not contain enough translation/rotation "
            "diversity to determine the calibration."
        )

    translation_x, _, _, _ = np.linalg.lstsq(
        translation_system, translation_rhs, rcond=None
    )

    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation_x
    result[:3, 3] = translation_x

    scale = singular_values[0] if singular_values[0] > 0 else 1.0
    if singular_values[-2] / scale < 1e-10:
        raise ValueError(
            "Selected motions are rotationally degenerate; collect rotations "
            "about multiple axes."
        )

    return result


def ax_xb_errors(
    a_matrices: list[np.ndarray],
    b_matrices: list[np.ndarray],
    x_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pair AX=XB rotation and translation errors."""
    if len(a_matrices) != len(b_matrices):
        raise ValueError("A and B must contain the same number of motions.")

    x_transform = as_transform(x_transform)
    rotation_errors = []
    translation_errors = []

    for a_value, b_value in zip(a_matrices, b_matrices):
        left = as_transform(a_value) @ x_transform
        right = x_transform @ as_transform(b_value)
        residual = invert_transform(left) @ right

        rotation_errors.append(
            rotation_angle_degrees(residual[:3, :3])
        )
        translation_errors.append(
            np.linalg.norm(residual[:3, 3])
        )

    return np.asarray(rotation_errors), np.asarray(translation_errors)


@dataclass(frozen=True)
class MotionPair:
    """One accepted pair of relative SI and NDI motions."""

    index_i: int
    index_j: int
    a_matrix: np.ndarray
    b_matrix: np.ndarray
    si_rotation_deg: float
    ndi_rotation_deg: float
    si_translation: float
    ndi_translation: float


class Calibration:
    """
    Calibrate the robot base to the NDI coordinate system.
    See README.md for details on the Calibration. 

    Frame convention:

    * SI values are ``base_T_end_effector``.
    * NDI values are ``ndi_T_marker``.
    * EE to NDI marker rigid transform ``end_effector_T_marker`` is eliminated.
    * ``base_T_ndi`` is solved as ``X`` in ``A X = X B``.

    Given ``P_i = base_T_end_effector_i`` and ``Q_i = ndi_T_marker_i``, the
    absolute relationship is ``P_i C = X Q_i``, where ``C`` is the unknown
    attachment and ``X = base_T_ndi``. For observations ``i`` and ``j``:

    ``(P_j inverse(P_i)) X = X (Q_j inverse(Q_i))``.

    Thus the attachment ``C`` does not need to be known or estimated.

    SI and NDI Series must be row-aligned. NDI translations are assumed to be
    millimetres by default and are converted to metres with
    ``ndi_translation_scale=0.001``.
    """

    def __init__(
        self,
        ndi_transforms: pd.Series,
        si_transforms: pd.Series,
        *, # everything after this must be named
        min_rotation_deg: float = 5.0,
        min_translation: float = 0.005,
        min_index_separation: int = 1,
        max_rotation_disagreement_deg: float = 2.0,
        max_pairs: int | None = 500,
    ) -> None:
        if not isinstance(ndi_transforms, pd.Series):
            raise TypeError("ndi_transforms must be a pandas Series.")
        if not isinstance(si_transforms, pd.Series):
            raise TypeError("si_transforms must be a pandas Series.")
        if len(ndi_transforms) != len(si_transforms):
            raise ValueError("NDI and SI Series must have the same length.")
        if len(ndi_transforms) < 3:
            raise ValueError("At least three matched observations are required.")
        if min_rotation_deg < 0 or min_translation < 0:
            raise ValueError("Motion thresholds cannot be negative.")
        if min_index_separation < 1:
            raise ValueError("min_index_separation must be at least 1.")
        if max_pairs is not None and max_pairs < 3:
            raise ValueError("max_pairs must be at least 3 or None.")

        self.ndi_transforms = ndi_transforms.reset_index(drop=True)
        self.si_transforms = si_transforms.reset_index(drop=True)
        self.min_rotation_deg = float(min_rotation_deg)
        self.min_translation = float(min_translation)
        self.min_index_separation = min_index_separation
        self.max_rotation_disagreement_deg = float(
            max_rotation_disagreement_deg
        )
        self.max_pairs = max_pairs

        self.valid_indices: list[int] = []
        self.invalid_indices: list[int] = []
        self.motion_pairs: list[MotionPair] = []
        self.base_T_ndi: np.ndarray | None = None
        self.metrics: dict[str, float | int] = {}

        self._valid_si: dict[int, np.ndarray] = {}
        self._valid_ndi: dict[int, np.ndarray] = {}

    def validate_observations(self) -> list[int]:
        """Validate aligned transforms and return their positional indices."""
        self.valid_indices = []
        self.invalid_indices = []
        self._valid_si = {}
        self._valid_ndi = {}

        for index, (ndi_value, si_value) in enumerate(
            zip(self.ndi_transforms, self.si_transforms)
        ):
            if not (
                is_valid_transform(ndi_value)
                and is_valid_transform(si_value)
            ):
                self.invalid_indices.append(index)
                continue

            ndi_transform = as_transform(ndi_value).copy()
            si_transform = as_transform(si_value).copy()

            self.valid_indices.append(index)
            self._valid_ndi[index] = ndi_transform
            self._valid_si[index] = si_transform

        if len(self.valid_indices) < 3:
            raise ValueError(
                "Fewer than three valid matched observations remain."
            )

        return self.valid_indices.copy()

    def select_motion_pairs(self) -> list[MotionPair]:
        """Select valid pairs in which rotation and translation both changed."""
        if not self.valid_indices:
            self.validate_observations()

        candidates: list[tuple[float, MotionPair]] = []

        for position, index_i in enumerate(self.valid_indices):
            for index_j in self.valid_indices[position + 1 :]:
                if index_j - index_i < self.min_index_separation:
                    continue

                a_matrix = (
                    self._valid_si[index_j]
                    @ invert_transform(self._valid_si[index_i])
                )
                b_matrix = (
                    self._valid_ndi[index_j]
                    @ invert_transform(self._valid_ndi[index_i])
                )

                si_rotation = rotation_angle_degrees(a_matrix[:3, :3])
                ndi_rotation = rotation_angle_degrees(b_matrix[:3, :3])
                si_translation = translation_distance(a_matrix)
                ndi_translation = translation_distance(b_matrix)

                if (
                    si_rotation < self.min_rotation_deg
                    or ndi_rotation < self.min_rotation_deg
                    or si_translation < self.min_translation
                    or ndi_translation < self.min_translation
                ):
                    continue

                # Conjugate rotations in AX=XB have the same angle. A large
                # disagreement usually indicates a bad timestamp match.
                if (
                    abs(si_rotation - ndi_rotation)
                    > self.max_rotation_disagreement_deg
                ):
                    continue

                pair = MotionPair(
                    index_i=index_i,
                    index_j=index_j,
                    a_matrix=a_matrix,
                    b_matrix=b_matrix,
                    si_rotation_deg=si_rotation,
                    ndi_rotation_deg=ndi_rotation,
                    si_translation=si_translation,
                    ndi_translation=ndi_translation,
                )
                score = min(si_rotation, ndi_rotation) + 100.0 * min(
                    si_translation, ndi_translation
                )
                candidates.append((score, pair))

        candidates.sort(key=lambda item: item[0], reverse=True)
        if self.max_pairs is not None:
            candidates = candidates[: self.max_pairs]
        self.motion_pairs = [pair for _, pair in candidates]

        if len(self.motion_pairs) < 3:
            raise ValueError(
                "Fewer than three valid motion pairs remain. Reduce the motion "
                "thresholds or collect more varied poses."
            )

        return self.motion_pairs.copy()

    def solve_base_to_ndi(self) -> np.ndarray:
        """Solve ``A X = X B`` and return ``base_T_ndi`` directly."""
        if not self.motion_pairs:
            self.select_motion_pairs()

        self.base_T_ndi = solve_ax_xb(
            [pair.a_matrix for pair in self.motion_pairs],
            [pair.b_matrix for pair in self.motion_pairs],
        )
        return self.base_T_ndi.copy()

    def calculate_metrics(self) -> dict[str, float | int]:
        """Calculate AX=XB and absolute base-to-NDI consistency metrics."""
        if self.base_T_ndi is None:
            self.solve_base_to_ndi()

        a_matrices = [pair.a_matrix for pair in self.motion_pairs]
        b_matrices = [pair.b_matrix for pair in self.motion_pairs]
        rotation_errors, translation_errors = ax_xb_errors(
            a_matrices, b_matrices, self.base_T_ndi
        )

        self.metrics = {
            "observation_count": len(self.ndi_transforms),
            "valid_observation_count": len(self.valid_indices),
            "invalid_observation_count": len(self.invalid_indices),
            "motion_pair_count": len(self.motion_pairs),
            "mean_ax_xb_rotation_error_deg": float(rotation_errors.mean()),
            "max_ax_xb_rotation_error_deg": float(rotation_errors.max()),
            "mean_ax_xb_translation_error": float(translation_errors.mean()),
            "max_ax_xb_translation_error": float(translation_errors.max()),
        }
        return self.metrics.copy()

    def calibrate(self) -> np.ndarray:
        """Run the complete workflow and return ``base_T_ndi``."""
        self.validate_observations()
        self.select_motion_pairs()
        self.solve_base_to_ndi()
        self.calculate_metrics()
        return self.base_T_ndi.copy()
