"""Full AX=YB recovery, orientation sensitivity and artifact checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from ndi_robot_registration.axyb_calibration import solve_axyb
from ndi_robot_registration.transforms import invert_transform
import run_calibration_axyb as runner


class AXYBTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.p = np.tile(np.eye(4), (25, 1, 1))
        self.p[:, :3, :3] = Rotation.random(25, random_state=rng).as_matrix()
        self.p[:, :3, 3] = rng.uniform(-0.2, 0.2, (25, 3))
        self.t = np.eye(4)
        self.t[:3, :3] = Rotation.from_rotvec([0.3, -0.6, 0.8]).as_matrix()
        self.t[:3, 3] = [0.7, -0.3, 1.2]
        self.c = np.eye(4)
        self.c[:3, :3] = Rotation.from_rotvec([-0.7, 0.4, 1.1]).as_matrix()
        self.c[:3, 3] = [0.03, -0.02, 0.08]
        self.q = self.t @ self.p @ self.c

    def test_exact_recovery_and_ax_equals_yb(self):
        t, c, metrics = solve_axyb(self.q, self.p)
        np.testing.assert_allclose(t, self.t, atol=1e-8)
        np.testing.assert_allclose(c, self.c, atol=1e-8)
        np.testing.assert_allclose(self.q @ invert_transform(c), t @ self.p, atol=1e-8)
        self.assertLess(metrics['fit_rotation_rmse_deg'], 1e-6)

    def test_ndi_orientation_changes_attachment_rotation(self):
        delta = np.eye(4)
        delta[:3, :3] = Rotation.from_rotvec([0.1, 0.2, -0.3]).as_matrix()
        t, c, _ = solve_axyb(self.q @ delta, self.p)
        np.testing.assert_allclose(t, self.t, atol=1e-8)
        np.testing.assert_allclose(c, self.c @ delta, atol=1e-8)

    def test_noise_and_invalid_rotation_filter(self):
        rng = np.random.default_rng(42)
        self.q[:, :3, 3] += rng.normal(0, 0.0002, (25, 3))
        self.q[:, :3, :3] = self.q[:, :3, :3] @ Rotation.from_rotvec(
            rng.normal(0, 0.001, (25, 3))).as_matrix()
        self.q[0, 0, 0] = np.nan
        t, c, metrics = solve_axyb(self.q, self.p)
        self.assertEqual(metrics['invalid_observation_count'], 1)
        np.testing.assert_allclose(t[:3, 3], self.t[:3, 3], atol=0.001)
        np.testing.assert_allclose(c[:3, 3], self.c[:3, 3], atol=0.001)

    def test_rejects_degenerate_data_and_bad_weights(self):
        p = self.p.copy()
        p[:, :3, :3] = np.eye(3)
        with self.assertRaises(ValueError):
            solve_axyb(self.t @ p @ self.c, p)
        for kwargs in ({'translation_sigma_m': 0}, {'rotation_sigma_deg': float('nan')}):
            with self.assertRaises(ValueError):
                solve_axyb(self.q, self.p, **kwargs)
        with self.assertRaises(ValueError):
            solve_axyb(self.q[:3], self.p[:3])
        with self.assertRaises(ValueError):
            solve_axyb(self.q[:4], self.p[:5])

    def test_script_saved_units_and_transform_keys(self):
        stamps = pd.date_range('2026-01-01', periods=25, freq='s', tz='UTC')
        robot = pd.DataFrame({'timestamp': stamps, 'Transforms': list(self.p)})
        tracker = pd.DataFrame({'timestamp': stamps, 'Transforms': list(self.q)})
        config = {
            'inputs': {'si_csv': 'unused', 'ndi_csv': 'unused'},
            'si_arm': {'timestamp_column': 'Time_stamp', 'arm_column': 'PSM1'},
            'ndi': {'timestamp_column': 'Timestamp', 'tool_id': 1},
            'matching': {'time_tolerance': '30ms'},
            'calibration': {'hand-pick-frame': False},
        }
        with tempfile.TemporaryDirectory() as folder:
            cfg, out = Path(folder) / 'config.json', Path(folder) / 'calib.npz'
            cfg.write_text(json.dumps(config))
            with patch.object(runner, 'clean_si_data', return_value=robot), \
                 patch.object(runner, 'clean_ndi_data', return_value=tracker):
                runner.main(['--config', str(cfg), '--out', str(out)])
            with np.load(out) as saved:
                np.testing.assert_allclose(saved['ndi_T_base'], self.t, atol=1e-8)
                np.testing.assert_allclose(saved['gripper_T_marker'], self.c, atol=1e-8)
                np.testing.assert_allclose(saved['axyb_X'], invert_transform(self.c), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
