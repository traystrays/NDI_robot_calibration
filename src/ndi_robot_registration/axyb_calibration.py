"""Full robot-world/hand-eye AX=YB calibration; translations are in metres."""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .hand_eye_calibration import Calibration
from .transforms import average_transforms, invert_transform, is_valid_transform, proper_rotation


def valid_observations(ndi_transforms, si_transforms):
    """Return aligned indices, robot poses P, and tracker poses Q."""
    if len(ndi_transforms) != len(si_transforms):
        raise ValueError("NDI and SI observations must have equal lengths.")
    indices, p, q = [], [], []
    for index, (tracker, robot) in enumerate(zip(ndi_transforms, si_transforms)):
        if not (is_valid_transform(tracker, rotation_atol=2e-5)
                and is_valid_transform(robot, rotation_atol=2e-5)):
            continue
        # Remove small rounding errors in logged rotation matrices.
        robot, tracker = np.array(robot, dtype=float), np.array(tracker, dtype=float)
        robot[:3, :3] = proper_rotation(robot[:3, :3])
        tracker[:3, :3] = proper_rotation(tracker[:3, :3])
        indices.append(index)
        p.append(robot)
        q.append(tracker)
    return np.asarray(indices, dtype=int), np.asarray(p), np.asarray(q)


def pose_error_metrics(ndi_T_base, end_effector_T_marker, p, q, prefix="fit"):
    """Summarize absolute marker prediction errors (metres internally)."""
    predicted = ndi_T_base @ p @ end_effector_T_marker
    translation = np.linalg.norm(predicted[:, :3, 3] - q[:, :3, 3], axis=1) * 1000
    relative_rotation = np.swapaxes(q[:, :3, :3], 1, 2) @ predicted[:, :3, :3]
    rotation = np.degrees(np.linalg.norm(Rotation.from_matrix(relative_rotation).as_rotvec(), axis=1))
    metrics = {}
    for name, values, units in (("translation", translation, "mm"), ("rotation", rotation, "deg")):
        for statistic, value in (("mean", values.mean()), ("median", np.median(values)),
                                 ("rmse", np.sqrt(np.mean(values ** 2))),
                                 ("p95", np.quantile(values, 0.95)), ("max", values.max())):
            metrics[f"{prefix}_{name}_{statistic}_{units}"] = float(value)
    return metrics


def solve_axyb(ndi_transforms, si_transforms, *, translation_sigma_m=0.001,
               rotation_sigma_deg=1.0):
    """Return (ndi_T_base, end_effector_T_marker, metrics).

    P_i=base_T_ee, Q_i=ndi_T_marker, T=ndi_T_base, C=ee_T_marker.
    Q_i=T P_i C, or A_i X=Y B_i with A_i=Q_i, X=C^-1, Y=T, B_i=P_i.
    Both T and C are full rigid transforms (12 optimized parameters).

    Initialize with relative-motion AX=XB, then jointly refine absolute poses.
    Residuals are NDI-frame position differences and rotation vectors of
    R_Qi.T R_predicted, divided by their respective sigma settings. Defaults
    weight 1 mm equally with 1 degree; these are weights, not measured accuracy.
    Both streams must represent a rigid attachment and synchronized observations.
    """
    if not (np.isfinite(translation_sigma_m) and translation_sigma_m > 0
            and np.isfinite(rotation_sigma_deg) and rotation_sigma_deg > 0):
        raise ValueError("Translation and rotation sigma must be finite and positive.")
    indices, p, q = valid_observations(ndi_transforms, si_transforms)
    if len(p) < 4:
        raise ValueError("At least four valid matched poses are required.")
    initializer = Calibration(pd.Series(list(q)), pd.Series(list(p)))
    t0 = initializer.calibrate_selected()
    c0 = average_transforms([
        invert_transform(robot) @ invert_transform(t0) @ tracker
        for robot, tracker in zip(p, q)
    ])

    def pack(t):
        return np.r_[Rotation.from_matrix(t[:3, :3]).as_rotvec(), t[:3, 3]]

    def unpack(x):
        t = np.eye(4)
        t[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
        t[:3, 3] = x[3:6]
        return t

    def residual(x):
        predicted = unpack(x[:6]) @ p @ unpack(x[6:])
        dp = (predicted[:, :3, 3] - q[:, :3, 3]) / translation_sigma_m
        dr = Rotation.from_matrix(
            np.swapaxes(q[:, :3, :3], 1, 2) @ predicted[:, :3, :3]
        ).as_rotvec() / np.radians(rotation_sigma_deg)
        return np.column_stack((dp, dr)).ravel()

    result = least_squares(residual, np.r_[pack(t0), pack(c0)], x_scale="jac",
                           max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Full AX=YB optimization failed: {result.message}")
    norms = np.linalg.norm(result.jac, axis=0)
    singular = np.linalg.svd(result.jac / np.maximum(norms, 1e-15), compute_uv=False)
    if singular[-1] / singular[0] < 1e-6:
        raise ValueError("AX=YB calibration is degenerate; collect rotations about multiple axes.")
    t, c = unpack(result.x[:6]), unpack(result.x[6:])
    metrics = {
        "observation_count": len(ndi_transforms),
        "valid_observation_count": len(indices),
        "invalid_observation_count": len(ndi_transforms) - len(indices),
        "translation_sigma_m": float(translation_sigma_m),
        "rotation_sigma_deg": float(rotation_sigma_deg),
        "normalized_jacobian_condition": float(singular[0] / singular[-1]),
        **pose_error_metrics(t, c, p, q),
    }
    return t, c, metrics
