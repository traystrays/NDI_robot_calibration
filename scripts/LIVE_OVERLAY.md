# Live tracked-ultrasound overlay

Entry point: `scripts/live_ultrasound_overlay.py`. Acquisition, immutable packets,
and synchronization live in `src/LiveData`. This renders the tracked ultrasound
slice on **left ECM video**, using the geometry from `reproject_ultrasound.py`.
The right stereo camera is not needed by that projection and is not opened.

The window layout is defined in `src/LiveData/overlay_gui.py`. The large left
panel shows the time-matched overlay. The right column shows the latest raw ECM
and ultrasound feeds, ECM/PSM1/PSM2 positions in metres and row-major rotation
matrices, and the probe position in NDI tracker coordinates. Sidebar data is
latest available data, not the synchronized bundle. Missing, stale, or invalid
samples are labeled and their values hidden. Images retain their aspect ratio.
Press Q or Escape, or close the window, to stop. `--headless` skips the dashboard;
the optional MP4 still records only the overlay image.

## What was built

```text
ECM capture worker ----------> bounded frame history ----\
Ultrasound capture worker ---> bounded frame history -----\
Robot UDP worker ------------> robot history + adapter ----> FrameSynchronizer
NDI tracker worker ----------> bounded pose history ------/       |
                                                        matched bundle
                                                              |
                                              OverlayRenderer on main thread
                                                              |
                                               display + timing CSV + MP4
```

1. `RobotReceiver` registers its receiving UDP port with DVPControl and receives
   the existing three-manipulator, 156-byte packets. No GUI automation runs when
   imported. `RobotPoseHistory` extracts ECM ID 2 and converts the 12 values into
   `base_T_ecm`. Rotation order matches `clean_si_data`: row-major 3x3; translation
   is already metres. Verify that the publisher uses this convention.
2. Each `VideoReceiver` owns a capture device and calls `read()` on its own thread.
   It timestamps immediately after the read and publishes an owned BGR image.
   Capture resolution is checked against the requested dimensions; an unexpected
   resolution fails rather than silently misapplying camera intrinsics.
3. `NDIReceiver` uses the same `NDITracker` settings/API as the video logger, in
   matrix mode (`use_quaternions=False`). Tool index 0 is the one configured ROM,
   not NDI handle 0. Translation is converted from mm to metres. Missing,
   nonfinite, repeated device frames, or nonrigid transforms produce invalid
   packets rather than reusing a previous pose. Device frame number and quality
   are retained. No empirical quality-error cutoff is assumed.
4. Every stream keeps a bounded history. Defaults: 30 frames per camera and
   300 poses per tracker. Oldest data is evicted. At 1280x720, both frame histories
   together hold about 158 MiB of image data; account for higher resolutions.
5. `FrameSynchronizer.poll()` uses each ECM frame as the anchor. It waits until
   all streams cover their future tolerance windows, or 60 ms after ECM receipt.
   It then chooses the nearest sample in each history. Defaults allow US +/-20
   ms, robot +/-20 ms, and NDI +/-40 ms relative to ECM. Samples can be reused
   across ECM frames. These are per-anchor limits, not an all-pairs 20 ms bound.
6. A frozen `MatchedFrame` carries the chosen packets, signed timing errors, and
   match status. Missing or invalid inputs suppress the slice. An ECM frame over
   250 ms old is also rejected. Each anchor is finalized once; late arrivals do
   not revise an already displayed frame.
7. The renderer uses the existing chain:

   ```text
   ndi_T_camera = ndi_T_base @ base_T_ecm @ ecm_T_camera
   camera_T_probe = inverse(ndi_T_camera) @ ndi_T_probe
   ```

   It transforms the calibrated slice corners, projects with ECM intrinsics and
   distortion, crops the ultrasound ROI, optionally rotates it 180 degrees, and
   perspective-warps/blends it onto ECM. The physical slice size defaults to
   43x50 mm, as in the offline script. Behind-camera, offscreen, and degenerate
   projections are labelled and not drawn. The planar homography approximation
   is inherited from the offline implementation; it is not dense distortion-aware
   reprojection across every ultrasound pixel.
8. OpenCV display runs on the main thread. When overloaded, the app displays only
   the freshest finalized frame rather than accumulating a render queue. It
   counts skipped frames. If ECM stops, the old overlay is replaced by a waiting/
   stale screen. Acquisition exceptions terminate the session with an error.
9. Escape, q, window close, Ctrl+C, or an exception clean up through `ExitStack`.
   Workers release cameras/tracker in `finally`; robot reception uses its existing
   timeout and shutdown. Some native camera/serial drivers can block indefinitely:
   worker stop reports a timeout after five seconds, rather than claiming success.

## Timestamp contract: what is and is not synchronized

This implementation provides **bounded receipt-time matching on one computer**.
All streams use `time.monotonic_ns()`. Never mix wall-clock timestamps, a different
computer's monotonic clock, or unconverted device ticks into these histories.

OpenCV capture time is unavailable through the logger's current interface. The
robot packet has no device timestamp. The installed NDI driver's timestamp is a
host timestamp, not a synchronized capture timestamp; the receiver does not treat
it as one. Camera/USB/serial/network buffering can therefore create physical
misalignment even when the displayed deltas are small.

A positive per-source `latency_ms` means:

```text
matching timestamp = local receipt timestamp - measured latency
```

Keep zero until measured; this compensates a constant delay, not jitter or drift.
Increase `wait_ms` if independently measured source delays require additional
arrival time. Keep `max_age_ms` larger than the intended latency/wait budget.
There is no automatic delay estimation, pose interpolation, hardware trigger, or
clock-drift correction. Validate motion alignment on the actual rig before
claiming capture-time synchronization. NDI device frame numbers help diagnose
repeated measurements but are not themselves converted to capture timestamps.

Frozen dataclasses prohibit field reassignment. Images and transforms are copied
into arrays backed by immutable bytes, so their contents cannot be changed either.
History locks protect only append/snapshot operations; image decoding, matching,
and rendering do not hold acquisition locks. `FrameSynchronizer` is called by
one consumer (the main thread).

## Run the hardware-free demo

Use an environment with NumPy, OpenCV, and pandas. From the calibration repo:

```bash
python scripts/live_ultrasound_overlay.py --demo --seconds 10
```

Headless validation with artifacts:

```bash
python scripts/live_ultrasound_overlay.py --demo --headless --seconds 5 --output data/live_demo.mp4 --log data/live_demo_timing.csv
python -m unittest discover -s tests -v
```

The demo creates synthetic images and poses with known +5/-2/+3 ms matching
errors. It exercises packet histories, synchronization, projection, blending,
CSV logging, and optional video output. Tests additionally exercise fake camera/
NDI drivers and a real localhost UDP publisher. Synthetic success is not a test
of the physical camera, NDI calibration, or DVPControl installation.

## Configure the actual rig

Use the project's Python 3.11 hardware environment with `opencv-python`, `numpy`,
`pandas`, `scikit-surgerynditracker`, `ndicapi`, and the existing project source.
The entry script adds this repo's `src` directory to its import path. The existing
logger environment may be reused; no running logger process is needed. GUI mode
requires the desktop OpenCV package (not headless-only OpenCV).

Edit `scripts/live_overlay_config.json`:

1. Set `video.ecm_camera` and `video.ultrasound_camera` to different device indices.
   Set their actual resolutions, fps, and backend (`any`, `dshow`, `msmf`, or
   `avfoundation`). Those index/port defaults are examples, not discovered devices.
2. **Set `overlay.calibration_resolution` to the verified [width, height] used to
   obtain the intrinsics**, and use the same `video.ecm_resolution`. It is left
   `null` intentionally: the current calibration NPZ does not store image size.
   If the capture mode changes, supply compatible calibration instead of guessing.
3. Set `ndi.serial_port` and `ndi.rom_path` to the actual probe. The example points
   at the neighboring logger's `tooltip_marker.rom`; confirm that this is the
   rigid body used for `image_to_probe`, not a different marker tool.
4. Confirm the calibration file paths and physical ultrasound scan size/depth.
   Relative paths resolve from this repository root, including the ROM path.
5. Confirm the live ultrasound crop and orientation. The provided screen ROI
   `[500,0,620,580]` and 180-degree rotation reproduce the offline **video** path.
   The offline preview path differs in rotation. For a raw Plus stream, specify
   its actual crop (the old raw XML used `[180,169,558,727]`) and valid input size.
6. Set `robot.dvpcontrol_exe` to the DVPControl executable. The overlay script
   launches DVPControl, dismisses the `NO` dialog, selects `Save SUJ`, and waits
   for startup before opening the cameras and trackers. The robot receiver also
   retries registration while packets are absent. It binds to loopback; changing
   the publisher host does not enable remote-computer acquisition. The default
   publisher port is 60000.
7. Stop the video logger/preview before starting this app: each hardware device
   needs one owner. This implementation directly opens the same capture/tracker
   interfaces; it does not read the logger's files or consume its preview callbacks.

Run a startup check. It opens the hardware and requires at least one valid robot
and NDI pose within `synchronization.startup_timeout_s`; failures identify which
tracking source is absent or invalid:

```bash
python scripts/live_ultrasound_overlay.py --config scripts/live_overlay_config.json --check-config
```

Run the overlay:

```bash
python scripts/live_ultrasound_overlay.py --config scripts/live_overlay_config.json --log data/live_timing.csv
```

Optionally add `--output data/live_overlay.mp4`, `--seconds 30`, or `--headless`.
The MP4 is a constant-30-fps visual preview, **not a timing-preserving recording**.
The CSV contains actual chosen monotonic times, source sequences, signed errors,
display age, and rendering status. Only frames selected for display are logged;
console counters also report matched/unmatched anchors and skipped frames.
Log/output paths are overwritten on each run: supply new filenames per session.
No raw camera or raw pose recording is performed by this app.

## Hardware acceptance procedure

1. Check static alignment using a known probe target and the correct ROM/calibration.
   Confirm units, axes, slice orientation, crop, and ECM intrinsics independently.
2. Verify robot and NDI packet flow. The overlay must not appear until all four
   streams have matches. Review the CSV deltas and display age.
3. Cover the NDI marker: the overlay should disappear with `ndi_invalid` or
   `ndi_unmatched`, never freeze at the last valid position.
4. Stop DVPControl: after the matching window runs out, the overlay should be
   suppressed. Restart the overlay after restarting DVPControl to register again.
5. Disconnect a camera: an acquisition error should stop the session. If a driver
   blocks instead of returning, the ECM stale display should appear; investigate
   the driver if shutdown reports a timeout.
6. Move the probe and ECM separately while observing alignment. Estimate source
   delays against a common reference; document the measurements before configuring
   latency corrections. Small receipt-time errors alone do not prove good motion
   alignment.
7. Run longer than the history capacity, check bounded memory and drop counts,
   and verify that the UI stays responsive. Adjust resolutions and acquisition/
   display rates if rendering cannot keep up.

## Files

- `src/LiveData/packets.py`: owned packets, rigid-transform checks, bounded histories.
- `src/LiveData/receivers.py`: cameras, NDI tracker, robot pose adapter.
- `src/LiveData/synchronizer.py`: nearest matching, deadline/staleness policy, bundles.
- `src/LiveData/robot_receiver_class.py`: existing UDP receiver, reused unchanged.
- `scripts/live_ultrasound_overlay.py`: configuration, rendering, UI, logs, demo.
- `scripts/live_overlay_config.json`: explicit hardware/calibration/timing settings.
- `tests/test_live_overlay.py`: hardware-independent regression tests.
