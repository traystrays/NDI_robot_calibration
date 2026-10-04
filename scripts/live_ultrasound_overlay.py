"""Live time-matched ultrasound on ECM video. See scripts/LIVE_OVERLAY.md."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import faulthandler
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DVP_CONTROL_EXE = Path(
    r"c:\Users\rcl\Documents\Linghao\eye_gaze_epilogger_2\DVPControl_precompiled\DVPControl.exe"
)
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from LiveData.packets import History, VideoFrame, TransformSample
from LiveData.receivers import VideoReceiver, NDIReceiver, RobotPoseHistory
from LiveData.robot_receiver_class import RobotReceiver
from LiveData.synchronizer import FrameSynchronizer
from LiveData.overlay_gui import OverlayGUI
from scripts.reproject_ultrasound import (
    load_image_to_probe, load_npz_transform, load_camera_parameters,
    ultrasound_corners_in_probe, roll_slice_about_depth_axis,
    project_ultrasound_corners, crop_recorded_ultrasound_frame, overlay_slice,
)


class OverlayRenderer:
    """Reuse offline geometry; never draw a slice for an unmatched bundle."""
    def __init__(self, ndi_T_base, ecm_T_camera, probe_corners, camera_matrix,
                 distortion, *, roi, resolution, opacity=0.65, rotate_180=True):
        if not 0 <= opacity <= 1:
            raise ValueError("Opacity must be between 0 and 1")
        self.ndi_T_base, self.ecm_T_camera = ndi_T_base, ecm_T_camera
        self.corners, self.K, self.distortion = probe_corners, camera_matrix, distortion
        self.roi, self.resolution = tuple(roi), tuple(resolution)
        self.opacity, self.rotate_180 = opacity, rotate_180

    @classmethod
    def from_config(cls, config):
        resolution = config.get("calibration_resolution")
        if (not isinstance(resolution, (list, tuple)) or len(resolution) != 2
                or any(not isinstance(v, int) or v <= 0 for v in resolution)):
            raise ValueError("Set overlay.calibration_resolution to the verified [width, height] "
                             "used to calibrate the ECM camera; the NPZ does not record it")
        def path(name):
            result = Path(config[name])
            return result if result.is_absolute() else ROOT / result
        corners = ultrasound_corners_in_probe(
            load_image_to_probe(path("image_to_probe")), tuple(config["physical_size_mm"]))
        corners = roll_slice_about_depth_axis(corners, config.get("roll_deg", 0))
        return cls(load_npz_transform(path("ndi_to_base"), "ndi_T_base"),
                   load_npz_transform(path("ecm_to_camera"), "X"), corners,
                   *load_camera_parameters(path("camera_parameters")),
                   roi=config["roi"], resolution=config["calibration_resolution"],
                   opacity=config.get("opacity", .65), rotate_180=config.get("rotate_180", True))

    def render(self, bundle):
        frame = bundle.ecm.image.copy()
        if (frame.shape[1], frame.shape[0]) != self.resolution:
            raise ValueError("ECM dimensions differ from calibration_resolution; do not silently resize")
        status = bundle.status
        if status == "matched":
            scan = crop_recorded_ultrasound_frame(bundle.ultrasound.image, self.roi)
            if self.rotate_180:
                scan = cv2.rotate(scan, cv2.ROTATE_180)
            pixels = project_ultrasound_corners(
                bundle.robot.transform, bundle.ndi.transform, self.ndi_T_base,
                self.ecm_T_camera, self.corners, self.K, self.distortion)
            if pixels is None:
                status = "behind_camera"
            elif (not np.isfinite(pixels).all() or np.max(np.abs(pixels)) > 1e7
                  or abs(cv2.contourArea(pixels)) < 1):
                status = "degenerate_projection"
            elif (pixels[:, 0].max() < 0 or pixels[:, 1].max() < 0
                  or pixels[:, 0].min() >= frame.shape[1]
                  or pixels[:, 1].min() >= frame.shape[0]):
                status = "outside_image"
            else:
                frame = overlay_slice(frame, scan, pixels, opacity=self.opacity)
                cv2.polylines(frame, [np.rint(pixels).astype(np.int32)], True,
                              (0, 255, 255), 2, cv2.LINE_AA)
                status = "rendered"
        errors = "/".join("--" if d is None else f"{d:+.1f}" for d in bundle.errors_ms)
        cv2.putText(frame, f"{status} | US/robot/NDI dt ms: {errors}", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 255), 1, cv2.LINE_AA)
        return frame, status


def demo_setup():
    """Synthetic geometry only; never substitutes demo data in hardware mode."""
    K = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
    corners = np.array([[-.08, -.05, 0], [.08, -.05, 0], [.08, .05, 0], [-.08, .05, 0]])
    renderer = OverlayRenderer(np.eye(4), np.eye(4), corners, K, np.zeros(5),
                               roi=(0, 0, 160, 100), resolution=(640, 480), rotate_180=False)
    return renderer


def demo_feed(histories, sequence, now):
    ecm, us, robot, ndi = histories
    t = now - 80_000_000
    background = np.full((480, 640, 3), 35, np.uint8)
    scan = np.zeros((100, 160, 3), np.uint8)
    scan[:, :, :] = np.arange(160, dtype=np.uint8)[None, :, None]
    cv2.circle(scan, (80, 50), 25, (220, 220, 220), 3)
    ecm.append(VideoFrame(sequence, t, t, background))
    us.append(VideoFrame(sequence, t+5_000_000, t+5_000_000, scan))
    robot.append(TransformSample(sequence, t-2_000_000, t-2_000_000, np.eye(4)))
    pose = np.eye(4)
    pose[:3, 3] = [.04*np.sin(sequence/15), 0, .5]
    ndi.append(TransformSample(sequence, t+3_000_000, t+3_000_000, pose))


def wait_for_tracking(robot_history, ndi_history, robot_receiver, timeout_s):
    """Require valid live tracking before entering the display loop."""
    if not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("synchronization.startup_timeout_s must be finite and positive")
    deadline = time.monotonic() + timeout_s
    robot_samples = ndi_samples = ()
    while time.monotonic() < deadline:
        if robot_receiver.error is not None:
            raise RuntimeError(f"Robot receiver failed: {robot_receiver.error}")
        robot_samples = robot_history.snapshot()
        ndi_samples = ndi_history.snapshot()
        if any(sample.valid for sample in robot_samples) and any(
                sample.valid for sample in ndi_samples):
            return
        time.sleep(.01)

    failures = []
    if not robot_samples:
        failures.append(
            "no robot packets received from DVPControl at "
            f"{robot_receiver.publisher[0]}:{robot_receiver.publisher[1]} "
            f"(invalid UDP packets: {robot_receiver.invalid_packets})"
        )
    elif not any(sample.valid for sample in robot_samples):
        failures.append(f"robot pose invalid: {robot_samples[-1].reason}")
    if not ndi_samples:
        failures.append("no NDI measurements received")
    elif not any(sample.valid for sample in ndi_samples):
        failures.append(f"NDI pose invalid: {ndi_samples[-1].reason}")
    raise RuntimeError("Tracking startup failed: " + "; ".join(failures))


def start_dvpcontrol(executable=DVP_CONTROL_EXE):
    """Launch and prepare DVPControl before opening any overlay hardware."""
    try:
        from pywinauto.application import Application

        executable = Path(executable).resolve()
        if not executable.is_file():
            raise FileNotFoundError(executable)
        print("Starting DVPControl...", flush=True)
        app = Application(backend="win32").start(
            f'"{executable}"', work_dir=str(executable.parent),
            timeout=15, wait_for_idle=False)
        window = app.top_window()
        window["NO"].wait("visible enabled", timeout=15).click_input()
        print("Clicked 'NO' button to dismiss the dialog.", flush=True)

        time.sleep(1)
        window = app.top_window()
        window["Save SUJ"].wait("visible enabled", timeout=15).click_input()
        print("Clicked 'Save SUJ'; waiting for DVPControl startup.", flush=True)
        time.sleep(5)
        print("DVPControl startup step complete.", flush=True)
        return app
    except Exception as error:
        raise RuntimeError(f"DVPControl startup failed: {error}") from error


def run(args):
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        raise ValueError("--seconds must be finite and positive")
    display_scale = 3.0
    with ExitStack() as stack: # keeps track of the cleanup
        workers = []
        robot = None
        if args.demo:
            histories = [History(30) for _ in range(4)]
            renderer = demo_setup()
            timing = {}
        else:
            if args.config is None: # needs an input
                raise ValueError("Supply --config scripts/live_overlay_config.json or --demo")
            config = json.loads(args.config.read_text())
            robot_config = config.get("robot", {}) # if that key is missing use empty
            dvpcontrol = start_dvpcontrol(
                robot_config.get("dvpcontrol_exe", DVP_CONTROL_EXE))
            print("Loading overlay calibration...", flush=True)
            renderer = OverlayRenderer.from_config(config["overlay"]) # info for overlay
            video = config["video"]
            display_scale = float(video.get("display_scale", display_scale))
            if not np.isfinite(display_scale) or display_scale <= 0:
                raise ValueError("video.display_scale must be finite and positive")

            if video["ecm_camera"] == video["ultrasound_camera"]:
                raise ValueError("ECM and ultrasound must use different capture devices")
            
            if tuple(video["ecm_resolution"]) != renderer.resolution:
                raise ValueError("ECM capture resolution must equal calibration_resolution")
            
            backend = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW,
                       "msmf": cv2.CAP_MSMF, "avfoundation": cv2.CAP_AVFOUNDATION}[video.get("backend", "any")] # how openCV access the camera
            cameras = []

            for name in ("ecm", "ultrasound"):
                print(f"Opening {name} camera {video[name+'_camera']} "
                      f"(backend={video.get('backend', 'any')})...", flush=True)
                camera = VideoReceiver(video[name+"_camera"], resolution=video[name+"_resolution"],
                                       fps=video.get("fps", 30), backend=backend,
                                       latency_ms=video.get(name+"_latency_ms", 0))
                live_cam = stack.enter_context(camera)
                cameras.append(live_cam)
                print(f"{name} camera ready.", flush=True)

            ndi_config = dict(config["ndi"])
            rom = Path(ndi_config["rom_path"])
            ndi_config["rom_path"] = str(rom if rom.is_absolute() else ROOT / rom)

            print(f"Starting NDI tracker on {ndi_config['serial_port']}...", flush=True)
            ndi = stack.enter_context(NDIReceiver(**ndi_config)) # init ndi receiver

            print("NDI tracker ready. Starting robot UDP receiver...", flush=True)
            robot = stack.enter_context(RobotReceiver(
                publisher=(robot_config.get("host", "127.0.0.1"), robot_config.get("port", 60000))))
            robot_history = RobotPoseHistory(robot, robot_config.get("latency_ms", 0))
            workers = [*cameras, ndi, robot]
            histories = [cameras[0].history, cameras[1].history,
                         robot_history, ndi.history]
            timing = config.get("synchronization", {})
            startup_timeout_s = timing.pop("startup_timeout_s", 5)
            print("Waiting for valid robot and NDI measurements...", flush=True)
            wait_for_tracking(robot_history, ndi.history, robot, startup_timeout_s)
            print("Tracking ready.", flush=True)

        sync = FrameSynchronizer(*histories, **timing)

        if args.check_config:
            print("Configuration loaded and devices opened successfully")
            return
        args.log.parent.mkdir(parents=True, exist_ok=True)

        log = stack.enter_context(args.log.open("w", newline=""))

        csv_writer = csv.writer(log)
        csv_writer.writerow(["ecm_sequence", "ecm_time_ns", "us_sequence", "robot_sequence", "ndi_sequence",
                             "us_time_ns", "robot_time_ns", "ndi_time_ns", "us_dt_ms", "robot_dt_ms", "ndi_dt_ms",
                             "display_age_ms", "status"])
        writer = None
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                     30, renderer.resolution)
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open output {args.output}")
            stack.callback(writer.release)
        gui = None
        if not args.headless:
            print("Opening overlay dashboard...", flush=True)
            gui = OverlayGUI(renderer.resolution, display_scale)
            stack.callback(gui.close)
        faulthandler.cancel_dump_traceback_later()
        print("Overlay running.", flush=True)
        display_image = None
        last_gui_ns = 0
        started = time.monotonic()
        last_display_ns = 0
        last_report = started
        sequence = 0
        rendered = 0
        while args.seconds is None or time.monotonic() - started < args.seconds:
            now = time.monotonic_ns()
            for worker in workers:
                if worker.error is not None:
                    raise RuntimeError(f"Acquisition failed: {worker.error}")
            if args.demo:
                sequence += 1
                demo_feed(histories, sequence, now)
            bundles = sync.poll(now)
            # Prefer the freshest frame when rendering falls behind; keep a count.
            if len(bundles) > 1:
                sync.dropped += len(bundles) - 1
            for bundle in bundles[-1:]:
                image, status = renderer.render(bundle)
                rendered += status == "rendered"
                samples = (bundle.ultrasound, bundle.robot, bundle.ndi)

                csv_writer.writerow([bundle.ecm.sequence, bundle.ecm.monotonic_ns,
                    *[s.sequence if s else "" for s in samples],
                    *[s.monotonic_ns if s else "" for s in samples], *bundle.errors_ms,
                    (time.monotonic_ns()-bundle.ecm.monotonic_ns)/1e6, status])
                
                log.flush() # push buffered CSV from python to disk; OS may still buffer it
                if writer:
                    writer.write(image)
                display_image = image
                last_display_ns = now

            if gui is not None and now-last_gui_ns >= 33_000_000:
                current_overlay = display_image
                if not last_display_ns or now-last_display_ns > sync.max_age_ns:
                    current_overlay = None
                if not gui.update(
                        current_overlay, ecm=histories[0].latest(),
                        ultrasound=histories[1].latest(),
                        robot=robot.latest() if robot is not None else None,
                        ndi=histories[3].latest(), now_ns=now, max_age_ns=sync.max_age_ns):
                    break
                last_gui_ns = now
            if time.monotonic()-last_report >= 1:
                print(f"matched={sync.matched} unmatched={sync.unmatched} dropped={sync.dropped} rendered={rendered}", flush=True)
                last_report = time.monotonic()
            time.sleep(1/30 if args.demo else .002)
        print(f"Finished: matched={sync.matched}, unmatched={sync.unmatched}, dropped={sync.dropped}, rendered={rendered}")
        if args.demo and not rendered:
            raise RuntimeError("Demo produced no overlays")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--demo", action="store_true", help="Use synthetic streams, no hardware")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--output", type=Path, help="Optional constant-30-fps preview MP4; use CSV for actual timing")
    parser.add_argument("--log", type=Path, default=ROOT / "data/live_overlay_timing.csv")
    parser.add_argument("--check-config", action="store_true",
                        help="Load calibration and require valid robot/NDI input")
    args = parser.parse_args()
    # Report all worker stacks if a native call blocks hardware startup.
    faulthandler.dump_traceback_later(30, repeat=True)
    try:
        run(args)
    except KeyboardInterrupt:
        print("Stopped")
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as error:
        parser.exit(1, f"Overlay error: {error}\n")
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
