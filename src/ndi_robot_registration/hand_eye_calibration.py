"""Classic AX=XB Hand-eye calibration workflow."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .transforms import (
    as_transform,
    invert_transform,
    is_valid_transform,
    proper_rotation,
    rotation_angle_degrees,
)


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
    Map robot-base coordinates into the NDI coordinate system.
    See README.md for details on the Calibration. 

    Frame convention:

    * SI values are ``base_T_end_effector``.
    * NDI values are ``ndi_T_marker``.
    * EE to NDI marker rigid transform ``end_effector_T_marker`` is eliminated.
    * ``ndi_T_base`` is solved as ``X`` in ``A X = X B``.

    Given ``P_i = base_T_end_effector_i`` and ``Q_i = ndi_T_marker_i``,
    the absolute relationship is ``Q_i = X P_i C``, where ``C`` is the
    unknown attachment and ``X = ndi_T_base``. For observations ``i`` and
    ``j``:

    ``(Q_j inverse(Q_i)) X = X (P_j inverse(P_i))``.

    Thus the attachment ``C`` does not need to be known or estimated.
    """

    def __init__(
        self,
        ndi_transforms: pd.Series,
        si_transforms: pd.Series,
        *, # everything after this must be named
        min_rotation_deg: float = 5.0,
        min_translation: float = 0.01,
        min_index_separation: int = 1,
        max_rotation_disagreement_deg: float = 2.0,
        max_pairs: int | None = 500, #default when omitted is 500
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
        self.pair_selection_progress: tuple[int, int] = (0, 0)
        self.ndi_T_base: np.ndarray | None = None
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
        """
        Select sequential, sufficiently separated, informative motion pairs.

        Start with the first valid observation as the anchor. Test candidates
        beginning ``min_index_separation`` positions later. If a candidate
        fails the motion thresholds, test the next valid observation against
        the same anchor. When a pair is accepted, its second observation
        becomes the next anchor.
        """
        if not self.valid_indices:
            self.validate_observations()

        observation_count = len(self.valid_indices)
        self.motion_pairs = []

        anchor_position = 0
        candidate_position = anchor_position + self.min_index_separation
        last_examined_position = anchor_position

        while candidate_position < observation_count:
            if (
                self.max_pairs is not None
                and len(self.motion_pairs) >= self.max_pairs
            ):
                break

            last_examined_position = candidate_position
            index_i = self.valid_indices[anchor_position]
            index_j = self.valid_indices[candidate_position]

            si_i = self._valid_si[index_i]
            si_j = self._valid_si[index_j]
            ndi_i = self._valid_ndi[index_i]
            ndi_j = self._valid_ndi[index_j]

            si_relative_rotation = si_j[:3, :3] @ si_i[:3, :3].T
            ndi_relative_rotation = ndi_j[:3, :3] @ ndi_i[:3, :3].T
            si_rotation = rotation_angle_degrees(si_relative_rotation)
            ndi_rotation = rotation_angle_degrees(ndi_relative_rotation)
            si_translation = float(
                np.linalg.norm(si_j[:3, 3] - si_i[:3, 3])
            )
            ndi_translation = float(
                np.linalg.norm(ndi_j[:3, 3] - ndi_i[:3, 3])
            )

            rotation_disagreement = abs(si_rotation - ndi_rotation)

            motion_is_valid = (
                si_rotation >= self.min_rotation_deg
                and ndi_rotation >= self.min_rotation_deg
                and si_translation >= self.min_translation
                and ndi_translation >= self.min_translation
                and rotation_disagreement <= self.max_rotation_disagreement_deg
            )

            if not motion_is_valid:
                candidate_position += 1
                continue

            self.motion_pairs.append(
                MotionPair(
                    index_i=index_i,
                    index_j=index_j,
                    a_matrix=ndi_j @ invert_transform(ndi_i),
                    b_matrix=si_j @ invert_transform(si_i),
                    si_rotation_deg=si_rotation,
                    ndi_rotation_deg=ndi_rotation,
                    si_translation=si_translation,
                    ndi_translation=ndi_translation,
                )
            )

            anchor_position = candidate_position
            candidate_position = (
                anchor_position + self.min_index_separation
            )

        reached_count = min(last_examined_position + 1, observation_count)
        self.pair_selection_progress = (reached_count, observation_count)
        print(
            "Motion pair selection reached valid index "
            f"{reached_count}/{observation_count}."
        )

        if len(self.motion_pairs) < 3:
            raise ValueError(
                "Fewer than three valid motion pairs remain. Reduce the motion "
                "thresholds or collect more varied poses."
            )

        return self.motion_pairs.copy()

    def use_selected_observations(self) -> list[MotionPair]:
        """Create motion pairs directly from consecutive selected observations.

        This intentionally does not apply the automatic motion thresholds:
        the caller's hand-picked observations are authoritative.
        """
        if not self.valid_indices:
            self.validate_observations()

        self.motion_pairs = []
        for index_i, index_j in zip(
            self.valid_indices[:-1], self.valid_indices[1:]
        ):
            si_i = self._valid_si[index_i]
            si_j = self._valid_si[index_j]
            ndi_i = self._valid_ndi[index_i]
            ndi_j = self._valid_ndi[index_j]
            self.motion_pairs.append(
                MotionPair(
                    index_i=index_i,
                    index_j=index_j,
                    a_matrix=ndi_j @ invert_transform(ndi_i),
                    b_matrix=si_j @ invert_transform(si_i),
                    si_rotation_deg=rotation_angle_degrees(
                        si_j[:3, :3] @ si_i[:3, :3].T
                    ),
                    ndi_rotation_deg=rotation_angle_degrees(
                        ndi_j[:3, :3] @ ndi_i[:3, :3].T
                    ),
                    si_translation=float(
                        np.linalg.norm(si_j[:3, 3] - si_i[:3, 3])
                    ),
                    ndi_translation=float(
                        np.linalg.norm(ndi_j[:3, 3] - ndi_i[:3, 3])
                    ),
                )
            )

        if len(self.motion_pairs) < 3:
            raise ValueError(
                "At least four valid selected observations are required."
            )
        self.pair_selection_progress = (
            len(self.valid_indices), len(self.valid_indices)
        )
        return self.motion_pairs.copy()

    def solve_ax_xb(self) -> np.ndarray:
        """Solve ``A X = X B`` and store ``X`` as ``ndi_T_base``."""
        if not self.motion_pairs:
            self.select_motion_pairs()

        a_values = [
            as_transform(pair.a_matrix) for pair in self.motion_pairs
        ]
        b_values = [
            as_transform(pair.b_matrix) for pair in self.motion_pairs
        ]

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

        scale = singular_values[0] if singular_values[0] > 0 else 1.0
        if singular_values[-2] / scale < 1e-10:
            raise ValueError(
                "Selected motions are rotationally degenerate; collect "
                "rotations about multiple axes."
            )

        self.ndi_T_base = np.eye(4, dtype=float)
        self.ndi_T_base[:3, :3] = rotation_x
        self.ndi_T_base[:3, 3] = translation_x
        return self.ndi_T_base.copy()

    def ax_xb_errors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-pair AX=XB rotation and translation errors."""
        if self.ndi_T_base is None:
            self.solve_ax_xb()

        rotation_errors = []
        translation_errors = []

        for pair in self.motion_pairs:
            left = pair.a_matrix @ self.ndi_T_base
            right = self.ndi_T_base @ pair.b_matrix
            residual = invert_transform(left) @ right

            rotation_errors.append(
                rotation_angle_degrees(residual[:3, :3])
            )
            translation_errors.append(
                np.linalg.norm(residual[:3, 3])
            )

        return (
            np.asarray(rotation_errors),
            np.asarray(translation_errors),
        )

    def calculate_metrics(self) -> dict[str, float | int]:
        """Calculate AX=XB and absolute base-to-NDI consistency metrics."""
        if self.ndi_T_base is None:
            self.solve_ax_xb()

        rotation_errors, translation_errors = self.ax_xb_errors()

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
        """Run the complete workflow and return ``ndi_T_base``."""
        self.validate_observations()
        self.select_motion_pairs()
        ndi_T_base = self.solve_ax_xb()
        self.calculate_metrics()
        return ndi_T_base

    def calibrate_selected(self) -> np.ndarray:
        """Calibrate using every hand-selected observation as supplied."""
        self.validate_observations()
        self.use_selected_observations()
        ndi_T_base = self.solve_ax_xb()
        self.calculate_metrics()
        return ndi_T_base
