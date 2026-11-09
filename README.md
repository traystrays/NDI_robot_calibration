# NDI to Robot Calibration
Using a NDI marker held by a robotic arm, we calibrate the Robot base to NDI coordinates.

Data is collected from davinci robots API and NDI polaris.

## Units
Using meters

## Data collection

Activate the Python 3.11 environment, install the package, and launch the
unified collector from the repository root:

```bash
conda activate ndi
python -m pip install -e .
python scripts/data_collection.py
```

In the GUI, select an output folder and NDI ROM, enter the NDI serial port,
refresh the camera list, and preview a frame from each camera before starting.
The ECM and ultrasound selections must be different camera indices.

Each run creates five synchronized files in the output folder:

- `ndi_<session>.csv`
- `ecm_<session>.mp4`
- `ecm_<session>_timestamps.txt`
- `us_<session>.mp4`
- `us_<session>_timestamps.txt`

The video timestamp files retain the one-timestamp-per-line format consumed by
the calibration and frame-selection tools.

Adding data collection for NDI as well

### Implementation
TODO: run matching videos twice, verify data pipeline
TODO: incorporate a NDI to pixel transform, and build reprojection pipline of that 
### Live synchronized ultrasound overlay

Run `python scripts/live_ultrasound_overlay.py --demo` for a hardware-free demo.
For device configuration, timestamp semantics, architecture, and step-by-step
hardware checks, see [the live overlay guide](scripts/LIVE_OVERLAY.md).

### Position-only robot-to-NDI calibration

Run the alternative solver with the same JSON configuration and CSV inputs as
`scripts/run_calibration.py`:

```bash
python scripts/run_calibration_pos_axyb.py --out data/20261012_calib/calib_pos_axyb.npz
```

Use `--config path/to/config.json` for another configuration. Relative paths
inside the configuration resolve against the repository root. Without `--out`,
this script writes to `output.calib_file`, replacing that file if it exists.
Video timestamp matching and `calibration.hand-pick-frame` use the existing frame
picker. With frame selection disabled, all valid timestamp matches are fitted;
video settings are optional. This workflow does not automatically detect or
average stationary intervals. Select settled, spatially varied poses with
rotations about multiple axes.

The position-only AX=YB model fits `q_i = T P_i c`, estimating the full rigid
`T = ndi_T_base` and the three-dimensional marker-origin offset `c` in the
end-effector frame. NDI orientation, `ndi_marker.translation`, and the AX=XB
motion-pair thresholds are unused. The marker origin must remain fixed relative
to the end effector. The solver rejects unobservable/degenerate pose sets.

The NPZ contains `ndi_T_base` (translation in metres), `marker_offset_m`,
`marker_offset_mm`, and numeric metrics. It does not estimate a marker attachment
rotation or save a full `gripper_T_marker`. Printed fit/evaluation errors include
training observations and must not be quoted as independent accuracy; validate
with separate poses. SciPy is required and is included in project dependencies.

### Full AX=YB robot-world and hand-eye calibration

```bash
python scripts/run_calibration_axyb.py --out data/20261012_calib/calib_axyb.npz
```

`solve_axyb()` in `src/ndi_robot_registration/axyb_calibration.py` estimates both
full rigid transforms in `Q_i = T P_i C`: `T = ndi_T_base` and
`C = end_effector_T_marker`. Equivalently, `A_i X = Y B_i` with `A_i = Q_i`,
`X = inverse(C)`, `Y = T`, and `B_i = P_i`. It initializes using relative-motion
AX=XB and then jointly optimizes all 12 parameters against absolute position
and orientation measurements. Both streams must contain valid rigid poses;
the marker attachment must stay rigid throughout collection.

The entry point retains the existing config, timestamp matching, optional video
matching and frame picker. `--config` selects another JSON file. Without manual
frame selection, all valid matched poses are fitted. Motion-pair thresholds and
`ndi_marker.translation` are unused. Stationary windows are not averaged.

Residual scales default to 1 mm and 1 degree. Adjust them with
`--translation-sigma-mm` and `--rotation-sigma-deg`, or JSON calibration keys
`translation_sigma_mm` and `rotation_sigma_deg`. A residual of one scale unit
in position has the same weight as one scale unit in orientation; these defaults
are not measured sensor uncertainties. The fit uses ordinary least squares.

The NPZ saves `ndi_T_base`, `end_effector_T_marker`, and its alias
`gripper_T_marker`, with all translations in metres; it also saves `axyb_X`,
`axyb_Y`, marker offsets and translation/rotation metrics. Pass the same NPZ to
`scripts/calculate_errors.py` as both `--calibration` and `--marker-mount` when
evaluating a separate recording. Without `--out`, the configured
`output.calib_file` is replaced. The separate `output.marker_mount_file` is not
written by this runner. Reported fit/evaluation metrics include training poses
and do not establish held-out accuracy.
