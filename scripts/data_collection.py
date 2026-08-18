"""
Unified GUI for synchronized NDI, ECM, and ultrasound data collection.

Run from the repository root with::

    python scripts/data_collection.py

Each session produces five files: NDI CSV, ECM video/timestamps, and ultrasound
video/timestamps. Camera previews are shown in this Tkinter window.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from toolbox.imaging import VideoRecorder, list_cameras, read_camera_frame
from toolbox.ndi_datalogger import NDIDataLogger


class DataCollectionApp:
    """Coordinate two cameras and an NDI tracker from one Tk event loop."""

    PREVIEW_SIZE = (480, 270)

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NDI / ECM / US Data Collection")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.output_dir = tk.StringVar(value=str(Path("data") / "collection"))
        self.rom_path = tk.StringVar()
        self.serial_port = tk.StringVar()
        self.ecm_camera = tk.StringVar()
        self.us_camera = tk.StringVar()
        self.use_quaternions = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Ready. Refresh cameras, then preview them.")

        self.recorders: list[VideoRecorder] = []
        self.ndi_logger: NDIDataLogger | None = None
        self._messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self._preview_images: dict[str, tk.PhotoImage] = {}
        self._stopping = False

        self._build_ui()
        self.root.after(50, self._poll_messages)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

        self._path_row(container, 0, "Output folder", self.output_dir, True)
        self._path_row(container, 1, "Tool ROM (.rom)", self.rom_path, False)

        ttk.Label(container, text="NDI serial port").grid(
            row=2, column=0, sticky="w", pady=3
        )
        ttk.Entry(container, textvariable=self.serial_port).grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=3
        )
        ttk.Checkbutton(
            container, text="Store NDI quaternions", variable=self.use_quaternions
        ).grid(row=3, column=1, sticky="w", pady=3)

        camera_box = ttk.LabelFrame(container, text="Camera/frame selection", padding=8)
        camera_box.grid(row=4, column=0, columnspan=3, sticky="ew", pady=8)
        camera_box.columnconfigure((1, 4), weight=1)
        ttk.Label(camera_box, text="ECM camera").grid(row=0, column=0, padx=3)
        self.ecm_combo = ttk.Combobox(
            camera_box, textvariable=self.ecm_camera, state="readonly", width=8
        )
        self.ecm_combo.grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(
            camera_box, text="Preview frame", command=lambda: self.preview("ECM")
        ).grid(row=0, column=2, padx=3)
        ttk.Label(camera_box, text="US camera").grid(row=0, column=3, padx=3)
        self.us_combo = ttk.Combobox(
            camera_box, textvariable=self.us_camera, state="readonly", width=8
        )
        self.us_combo.grid(row=0, column=4, sticky="ew", padx=3)
        ttk.Button(
            camera_box, text="Preview frame", command=lambda: self.preview("US")
        ).grid(row=0, column=5, padx=3)
        ttk.Button(camera_box, text="Refresh cameras", command=self.refresh_cameras).grid(
            row=1, column=0, columnspan=6, pady=(8, 0)
        )

        previews = ttk.Frame(container)
        previews.grid(row=5, column=0, columnspan=3, sticky="ew")
        previews.columnconfigure((0, 1), weight=1)
        self.ecm_preview = ttk.Label(previews, text="ECM preview", anchor="center")
        self.ecm_preview.grid(row=0, column=0, padx=4, sticky="nsew")
        self.us_preview = ttk.Label(previews, text="US preview", anchor="center")
        self.us_preview.grid(row=0, column=1, padx=4, sticky="nsew")

        controls = ttk.Frame(container)
        controls.grid(row=6, column=0, columnspan=3, pady=10)
        self.start_button = ttk.Button(controls, text="Start all", command=self.start)
        self.start_button.grid(row=0, column=0, padx=5)
        self.stop_button = ttk.Button(
            controls, text="Stop all", command=self.stop, state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=5)
        ttk.Label(container, textvariable=self.status, wraplength=900).grid(
            row=7, column=0, columnspan=3, sticky="w"
        )

    def _path_row(
        self, parent: ttk.Frame, row: int, label: str,
        variable: tk.StringVar, directory: bool,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        if directory:
            command = lambda: self._choose_directory(variable)
        else:
            command = lambda: self._choose_rom(variable)
        ttk.Button(parent, text="Browse", command=command).grid(
            row=row, column=2, padx=(5, 0), pady=3
        )

    @staticmethod
    def _choose_directory(variable: tk.StringVar) -> None:
        selected = filedialog.askdirectory(initialdir=variable.get() or ".")
        if selected:
            variable.set(selected)

    @staticmethod
    def _choose_rom(variable: tk.StringVar) -> None:
        selected = filedialog.askopenfilename(filetypes=[("NDI ROM", "*.rom")])
        if selected:
            variable.set(selected)

    def refresh_cameras(self) -> None:
        self.status.set("Searching for cameras…")
        self.root.update_idletasks()
        cameras = list_cameras()
        values = [str(camera) for camera in cameras]
        self.ecm_combo["values"] = values
        self.us_combo["values"] = values
        if values:
            self.ecm_camera.set(values[0])
            self.us_camera.set(values[1] if len(values) > 1 else values[0])
            self.status.set(f"Found camera indices: {', '.join(values)}")
        else:
            self.status.set("No cameras were found.")

    def preview(self, source: str) -> None:
        camera = self.ecm_camera.get() if source == "ECM" else self.us_camera.get()
        if not camera:
            messagebox.showerror("Camera required", f"Select the {source} camera first.")
            return
        try:
            self._show_frame(source, read_camera_frame(int(camera)))
            self.status.set(f"{source} is showing a frame from camera {camera}.")
        except Exception as error:
            messagebox.showerror("Camera preview failed", str(error))

    def start(self) -> None:
        try:
            config = self._validated_config()
        except (ValueError, FileNotFoundError) as error:
            messagebox.showerror("Invalid configuration", str(error))
            return
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.status.set("Connecting devices and starting collection…")
        threading.Thread(
            target=self._start_workers, args=(config,), name="start-collection", daemon=True
        ).start()

    def _validated_config(self) -> dict[str, object]:
        output = Path(self.output_dir.get()).expanduser()
        rom = Path(self.rom_path.get()).expanduser()
        if not rom.is_file() or rom.suffix.lower() != ".rom":
            raise FileNotFoundError(f"Select an existing .rom file: {rom}")
        if not self.serial_port.get().strip():
            raise ValueError("Enter the NDI serial port.")
        if not self.ecm_camera.get() or not self.us_camera.get():
            raise ValueError("Select both ECM and US cameras.")
        ecm = int(self.ecm_camera.get())
        us = int(self.us_camera.get())
        if ecm == us:
            raise ValueError("ECM and US must use different camera indices.")
        return {
            "output": output, "rom": rom, "port": self.serial_port.get().strip(),
            "ecm": ecm, "us": us, "quaternions": self.use_quaternions.get(),
        }

    def _start_workers(self, config: dict[str, object]) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = config["output"]
        assert isinstance(output, Path)
        output.mkdir(parents=True, exist_ok=True)
        self.recorders = [
            VideoRecorder(
                int(config["ecm"]), output / f"ecm_{stamp}.mp4",
                output / f"ecm_{stamp}_timestamps.txt",
                frame_callback=lambda _, frame: self._messages.put(("ECM", frame)),
            ),
            VideoRecorder(
                int(config["us"]), output / f"us_{stamp}.mp4",
                output / f"us_{stamp}_timestamps.txt",
                frame_callback=lambda _, frame: self._messages.put(("US", frame)),
            ),
        ]
        self.ndi_logger = NDIDataLogger(
            output / f"ndi_{stamp}.csv", config["rom"], str(config["port"]),
            use_quaternions=bool(config["quaternions"]),
        )
        started: list[object] = []
        try:
            for recorder in self.recorders:
                recorder.start()
                started.append(recorder)
            self.ndi_logger.start()
            started.append(self.ndi_logger)
            self._messages.put(("started", output))
        except Exception as error:
            for worker in reversed(started):
                try:
                    worker.stop()
                except Exception:
                    pass
            self._messages.put(("error", error))

    def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self.stop_button.configure(state="disabled")
        self.status.set("Stopping collection and closing files…")
        threading.Thread(target=self._stop_workers, name="stop-collection", daemon=True).start()

    def _stop_workers(self) -> None:
        errors: list[str] = []
        if self.ndi_logger is not None:
            try:
                self.ndi_logger.stop()
            except Exception as error:
                errors.append(str(error))
        for recorder in self.recorders:
            try:
                recorder.stop()
            except Exception as error:
                errors.append(str(error))
        self._messages.put(("stopped", errors))

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, value = self._messages.get_nowait()
                if kind in {"ECM", "US"}:
                    assert isinstance(value, np.ndarray)
                    self._show_frame(kind, value)
                elif kind == "started":
                    self.stop_button.configure(state="normal")
                    self.status.set(f"Recording all streams in {value}")
                elif kind == "error":
                    self.start_button.configure(state="normal")
                    self.status.set("Collection failed to start.")
                    messagebox.showerror("Collection failed", str(value))
                elif kind == "stopped":
                    self._stopping = False
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    errors = value
                    self.status.set(
                        "Collection stopped." if not errors
                        else "Stopped with errors: " + "; ".join(errors)
                    )
        except queue.Empty:
            pass

        if self.ndi_logger and self.ndi_logger.error and not self._stopping:
            error = self.ndi_logger.error
            self.ndi_logger.error = None
            self.status.set(f"NDI error: {error}")
            self.stop()
        for recorder in self.recorders:
            if recorder.error and not self._stopping:
                error = recorder.error
                recorder.error = None
                self.status.set(f"Camera error: {error}")
                self.stop()
                break
        self.root.after(50, self._poll_messages)

    def _show_frame(self, source: str, frame: np.ndarray) -> None:
        width, height = self.PREVIEW_SIZE
        shown = frame.copy()
        scale = min(width / shown.shape[1], height / shown.shape[0])
        shown = cv2.resize(shown, None, fx=scale, fy=scale)
        shown = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
        ppm = f"P6 {shown.shape[1]} {shown.shape[0]} 255\n".encode() + shown.tobytes()
        image = tk.PhotoImage(data=ppm, format="PPM")
        self._preview_images[source] = image
        label = self.ecm_preview if source == "ECM" else self.us_preview
        label.configure(image=image, text="")

    def close(self) -> None:
        active = any(recorder.recording for recorder in self.recorders) or (
            self.ndi_logger is not None and self.ndi_logger.recording
        )
        if active:
            if not messagebox.askyesno(
                "Stop collection?", "Collection is active. Stop it and close the application?"
            ):
                return
            self.stop()
            self.root.after(100, self._close_when_stopped)
        else:
            self.root.destroy()

    def _close_when_stopped(self) -> None:
        if self._stopping:
            self.root.after(100, self._close_when_stopped)
        else:
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    DataCollectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
