"""Small, reusable operations for 3-D rigid transformations."""

from collections.abc import Sequence

import numpy as np


def as_transform(value: object) -> np.ndarray:
    """Return *value* as a floating-point 4x4 array."""
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform; got shape {transform.shape}.")
    return transform


def is_valid_transform(
    value: object,
    *, # everything after this must be named
    rotation_atol: float = 1e-5,
    bottom_row_atol: float = 1e-8,
) -> bool:
    """Return whether *value* is a finite homogeneous rigid transform."""
    try:
        transform = as_transform(value)
    except (TypeError, ValueError):
        return False

    if not np.isfinite(transform).all():
        return False
    if not np.allclose(
        transform[3], [0.0, 0.0, 0.0, 1.0], atol=bottom_row_atol
    ):
        return False

    rotation = transform[:3, :3]
    return bool(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=rotation_atol)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=rotation_atol)
    )


def invert_transform(value: object) -> np.ndarray:
    """Invert a rigid transform without using a general matrix inverse."""
    transform = as_transform(value)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def relative_transform(first: object, second: object) -> np.ndarray:
    """Return ``inverse(first) @ second``."""
    return invert_transform(first) @ as_transform(second)


def left_relative_transform(first: object, second: object) -> np.ndarray:
    """Return ``second @ inverse(first)``.

    Unlike :func:`relative_transform`, this motion is expressed in the fixed
    reference frame. It is the form needed to eliminate a rigid tool-marker
    attachment while solving directly for the transform between two fixed
    reference frames.
    """
    return as_transform(second) @ invert_transform(first)


def rotation_angle_degrees(rotation: object) -> float:
    """Return the unsigned angle of a 3x3 rotation matrix in degrees."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation; got shape {matrix.shape}.")
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def translation_distance(transform: object) -> float:
    """Return the magnitude of a transform's translation."""
    return float(np.linalg.norm(as_transform(transform)[:3, 3]))


def project_to_rotation(matrix: object) -> np.ndarray:
    """Project a 3x3 matrix onto the nearest proper rotation matrix."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix; got shape {matrix.shape}.")

    u_matrix, _, vt_matrix = np.linalg.svd(matrix)
    rotation = u_matrix @ vt_matrix
    if np.linalg.det(rotation) < 0:
        u_matrix[:, -1] *= -1
        rotation = u_matrix @ vt_matrix
    return rotation


def average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Average transform rotations by SVD and translations arithmetically."""
    if not transforms:
        raise ValueError("At least one transform is required.")

    matrices = np.stack([as_transform(item) for item in transforms])
    result = np.eye(4, dtype=float)
    result[:3, :3] = project_to_rotation(matrices[:, :3, :3].mean(axis=0))
    result[:3, 3] = matrices[:, :3, 3].mean(axis=0)
    return result


def solve_ax_xb(
    a_matrices: Sequence[np.ndarray],
    b_matrices: Sequence[np.ndarray],
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

    # The homogeneous solution has arbitrary sign and scale. Choose the sign
    # that permits projection to a proper rotation.
    if np.linalg.det(raw_rotation) < 0:
        raw_rotation *= -1
    rotation_x = project_to_rotation(raw_rotation)

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

    # A large second-smallest singular value is not required, but a second
    # near-zero value indicates an ambiguous rotation solution.
    scale = singular_values[0] if singular_values[0] > 0 else 1.0
    if singular_values[-2] / scale < 1e-10:
        raise ValueError(
            "Selected motions are rotationally degenerate; collect rotations "
            "about multiple axes."
        )

    return result


def ax_xb_errors(
    a_matrices: Sequence[np.ndarray],
    b_matrices: Sequence[np.ndarray],
    x_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pair rotation errors (degrees) and translation errors."""
    if len(a_matrices) != len(b_matrices):
        raise ValueError("A and B must contain the same number of motions.")

    x_transform = as_transform(x_transform)
    rotation_errors = []
    translation_errors = []

    for a_value, b_value in zip(a_matrices, b_matrices):
        left = as_transform(a_value) @ x_transform
        right = x_transform @ as_transform(b_value)
        residual = invert_transform(left) @ right
        rotation_errors.append(rotation_angle_degrees(residual[:3, :3]))
        translation_errors.append(np.linalg.norm(residual[:3, 3]))

    return np.asarray(rotation_errors), np.asarray(translation_errors)
