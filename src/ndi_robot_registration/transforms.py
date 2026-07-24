"""
Transform operations.
"""

from collections.abc import Sequence

import numpy as np


def as_transform(value: object) -> np.ndarray:
    """
    Input can be any Python object.
    Return *value* as a floating-point 4x4 array.
    """
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
    """
    Return whether *value* is a finite homogeneous rigid transform.
    """
    try:
        # needs to be transform
        transform = as_transform(value)
    except (TypeError, ValueError):
        return False

    # check all values are finite, remove Nan and infinity
    if not np.isfinite(transform).all():
        return False
    
    # check last row is 0,0,0,1
    if not np.allclose(
        transform[3], [0.0, 0.0, 0.0, 1.0], atol=bottom_row_atol, rtol=0.0
    ):
        return False

    # check roation 
    rotation = transform[:3, :3]

    orthonormal = np.allclose(rotation.T @ rotation, np.eye(3), atol=rotation_atol, rtol=0.0)

    determinant = np.isclose(np.linalg.det(rotation), 1.0, atol=rotation_atol, rtol=0.0)

    return bool(orthonormal and determinant)



def invert_transform(value: object) -> np.ndarray:
    """
    Invert a rigid transform.
    """
    transform = as_transform(value)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def rotation_angle_degrees(rotation: object) -> float:
    """
    Return the unsigned angle of a 3x3 rotation matrix in degrees.
    See README.md for math details
    """

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation; got shape {matrix.shape}.")
    
    # calculate (trace(matrix) - 1)/2
    raw_cosine = (np.trace(matrix) - 1.0) / 2.0

    # clip the cosine value
    cosine = np.clip(raw_cosine, -1.0, 1.0) 
    return float(np.degrees(np.arccos(cosine)))


def translation_distance(transform: object) -> float:
    """
    Return the magnitude of a transform's translation.
    """

    return float(np.linalg.norm(as_transform(transform)[:3, 3]))


def proper_rotation(matrix: object) -> np.ndarray:
    """
    Project a 3x3 matrix onto the nearest proper rotation matrix using SVD.
    Returns a 3x3 rotation matrix.
    """

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix; got shape {matrix.shape}.")

    # singular value decomposition
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
    result[:3, :3] = proper_rotation(matrices[:, :3, :3].mean(axis=0))
    result[:3, 3] = matrices[:, :3, 3].mean(axis=0)
    return result
