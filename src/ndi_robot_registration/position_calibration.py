"""Position-only robot-world/marker calibration, with all lengths in metres.

Q_i = T P_i C implies Q_i C^-1 = T P_i (AX=YB). We fit only
q_i = R_T (R_Pi c + t_Pi) + t_T: T has six parameters and c has three.
NDI orientations and the attachment rotation are deliberately unused.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .transforms import as_transform, is_valid_transform, proper_rotation


def observation_arrays(ndi_transforms, si_transforms):
    """Keep valid robot poses and finite NDI positions; ignore NDI rotation."""
    if len(ndi_transforms) != len(si_transforms):
        raise ValueError("NDI and SI observations must have equal lengths.")
    indices, poses, points = [], [], []
    for i, (q, p) in enumerate(zip(ndi_transforms, si_transforms)):
        try:
            q = as_transform(q)
        except (TypeError, ValueError):
            continue
        if is_valid_transform(p, rotation_atol=2e-5) and np.isfinite(q[:3, 3]).all():
            indices.append(i)
            poses.append(as_transform(p))
            points.append(q[:3, 3])
    return np.asarray(indices, dtype=int), np.asarray(poses), np.asarray(points)


def predict_positions(ndi_T_base, marker_offset_m, robot_poses):
    """Predict marker origins in NDI coordinates, in metres."""
    p = np.asarray(robot_poses)
    base_points = p[:, :3, :3] @ marker_offset_m + p[:, :3, 3]
    return base_points @ ndi_T_base[:3, :3].T + ndi_T_base[:3, 3]


def solve_position_axyb(ndi_transforms, si_transforms):
    """Return (ndi_T_base, marker_offset_m, metrics), rejecting rank loss.

    The offset is in the end-effector frame. Four observations are a minimum
    count, not an observability guarantee. Diverse orientations are required.
    """
    indices, p, q = observation_arrays(ndi_transforms, si_transforms)
    if len(p) < 4:
        raise ValueError("At least four valid observations are required.")
    rotations, origins = p[:, :3, :3], p[:, :3, 3]
    offset_system = (rotations[1:] - rotations[0]).reshape(-1, 3)
    if np.linalg.matrix_rank(offset_system, tol=1e-8) < 3:
        raise ValueError("Marker offset is unobservable; rotate about multiple axes.")

    def unpack(x):
        t = np.eye(4)
        t[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
        t[:3, 3] = x[3:6]
        return t, x[6:9]

    def residual(x):
        t, c = unpack(x)
        return (predict_positions(t, c, p) - q).ravel()

    # Alternating point registration and linear offset fitting supplies several
    # starts without ever using NDI orientation (even as an initializer).
    starts = []
    for initial_c in (np.zeros(3), np.array([0.05, 0, 0]),
                      np.array([0, 0.05, 0]), np.array([0, 0, 0.05])):
        c = initial_c.copy()
        for _ in range(30):
            source = rotations @ c + origins
            r = proper_rotation((q - q.mean(0)).T @ (source - source.mean(0)))
            translation = q.mean(0) - r @ source.mean(0)
            rhs = (q - translation) @ r - origins
            c = np.linalg.lstsq(rotations.reshape(-1, 3), rhs.ravel(), rcond=None)[0]
        starts.append(np.r_[Rotation.from_matrix(r).as_rotvec(), translation, c])
    results = [least_squares(residual, start, x_scale="jac", max_nfev=2000,
                             ftol=1e-12, xtol=1e-12, gtol=1e-12)
               for start in starts]
    converged = [result for result in results if result.success and np.isfinite(result.cost)]
    if not converged:
        raise ValueError("Position calibration did not converge.")
    best = min(converged, key=lambda result: result.cost)
    # Normalize columns so metres and radians do not determine the rank cutoff.
    norms = np.linalg.norm(best.jac, axis=0)
    singular = np.linalg.svd(best.jac / np.maximum(norms, 1e-15), compute_uv=False)
    if singular[-1] / singular[0] < 1e-6:
        raise ValueError("Position calibration is degenerate; collect more varied poses.")
    t, c = unpack(best.x)
    errors = np.linalg.norm(residual(best.x).reshape(-1, 3), axis=1) * 1000
    metrics = {
        "observation_count": len(ndi_transforms),
        "valid_observation_count": len(indices),
        "invalid_observation_count": len(ndi_transforms) - len(indices),
        "fit_translation_rmse_mm": float(np.sqrt(np.mean(errors ** 2))),
        "fit_translation_mean_mm": float(errors.mean()),
        "fit_translation_p95_mm": float(np.quantile(errors, 0.95)),
        "fit_translation_max_mm": float(errors.max()),
        "normalized_jacobian_condition": float(singular[0] / singular[-1]),
    }
    return t, c, metrics
