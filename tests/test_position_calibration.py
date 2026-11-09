"""Synthetic recovery and degeneracy checks for position calibration."""
import sys
from pathlib import Path
import unittest
import tempfile
import json
from unittest.mock import patch
import pandas as pd

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ndi_robot_registration.position_calibration import solve_position_axyb


class PositionCalibrationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.p = np.tile(np.eye(4), (24, 1, 1))
        self.p[:, :3, :3] = Rotation.random(24, random_state=rng).as_matrix()
        self.p[:, :3, 3] = rng.uniform(-0.2, 0.2, (24, 3))
        self.t = np.eye(4)
        self.t[:3, :3] = Rotation.from_rotvec([0.4, -0.2, 0.7]).as_matrix()
        self.t[:3, 3] = [0.5, -0.3, 0.8]
        self.c = np.array([0.02, -0.01, 0.04])
        self.q = np.tile(np.eye(4), (24, 1, 1))
        # Generate truth independently with homogeneous point multiplication.
        self.q[:, :3, 3] = np.array([(self.t @ p @ np.r_[self.c, 1])[:3] for p in self.p])

    def test_recovers_transform_offset_and_ignores_ndi_rotation(self):
        self.q[:, :3, :3] = np.nan
        t, c, metrics = solve_position_axyb(self.q, self.p)
        np.testing.assert_allclose(t, self.t, atol=1e-7)
        np.testing.assert_allclose(c, self.c, atol=1e-7)
        self.assertLess(metrics['fit_translation_rmse_mm'], 1e-5)

    def test_noisy_data_and_invalid_observation(self):
        rng = np.random.default_rng(4)
        self.q[:, :3, 3] += rng.normal(0, 0.0002, (24, 3))
        self.q[0, 0, 3] = np.nan
        t, c, metrics = solve_position_axyb(self.q, self.p)
        self.assertEqual(metrics['invalid_observation_count'], 1)
        np.testing.assert_allclose(t[:3, 3], self.t[:3, 3], atol=0.001)
        np.testing.assert_allclose(c, self.c, atol=0.001)

    def test_rejects_constant_and_single_axis_orientations(self):
        for rotations in (np.tile(np.eye(3), (24, 1, 1)),
                          Rotation.from_rotvec(np.column_stack([np.zeros(24), np.zeros(24), np.linspace(0, 1, 24)])).as_matrix()):
            self.p[:, :3, :3] = rotations
            with self.assertRaisesRegex(ValueError, 'unobservable'):
                solve_position_axyb(self.q, self.p)

    def test_script_saves_compatible_artifact(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import run_calibration_pos_axyb as runner
        stamps = pd.date_range("2026-01-01", periods=len(self.p), freq="s", tz="UTC")
        robot = pd.DataFrame({"timestamp": stamps, "Transforms": list(self.p)})
        tracker = pd.DataFrame({"timestamp": stamps, "Transforms": list(self.q)})
        config = {
            "inputs": {"si_csv": "unused", "ndi_csv": "unused"},
            "si_arm": {"timestamp_column": "Time_stamp", "arm_column": "PSM1"},
            "ndi": {"timestamp_column": "Timestamp", "tool_id": 1},
            "matching": {"time_tolerance": "30ms"},
            "calibration": {"hand-pick-frame": False},
        }
        with tempfile.TemporaryDirectory() as folder:
            config_path = Path(folder) / "config.json"
            config_path.write_text(json.dumps(config))
            out = Path(folder) / "calib.npz"
            with patch.object(runner, "clean_si_data", return_value=robot), \
                 patch.object(runner, "clean_ndi_data", return_value=tracker):
                runner.main(["--config", str(config_path), "--out", str(out)])
            with np.load(out) as saved:
                np.testing.assert_allclose(saved["ndi_T_base"], self.t, atol=1e-7)
                np.testing.assert_allclose(saved["marker_offset_m"], self.c, atol=1e-7)
                np.testing.assert_allclose(saved["marker_offset_mm"], self.c * 1000, atol=1e-4)

    def test_rejects_too_few_or_unaligned_observations(self):
        with self.assertRaises(ValueError):
            solve_position_axyb(self.q[:3], self.p[:3])
        with self.assertRaises(ValueError):
            solve_position_axyb(self.q[:4], self.p[:5])


if __name__ == '__main__':
    unittest.main()
