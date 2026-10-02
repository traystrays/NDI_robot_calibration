"""OpenCV dashboard for the live overlay; acquisition stays in the receivers."""
import cv2
import numpy as np


class OverlayGUI:
    """Large overlay, two raw previews, and four telemetry panels."""
    title = "Live ultrasound overlay"
    size = (1440, 960)

    def __init__(self, resolution, display_scale=1):
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        width = min(1800, max(960, round(resolution[0] * display_scale)))
        cv2.resizeWindow(self.title, width, round(width * self.size[1] / self.size[0]))

    def close(self):
        cv2.destroyWindow(self.title)

    @staticmethod
    def _text(canvas, text, x, y, scale=.48, color=(210, 218, 225)):
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    @classmethod
    def _panel(cls, canvas, box, title, image=None, lines=()):
        x, y, w, h = box
        cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), (65, 73, 83), 1)
        cls._text(canvas, title, x+12, y+23, .52, (230, 220, 150))
        if image is not None:
            available_w, available_h = w-16, h-42
            scale = min(available_w/image.shape[1], available_h/image.shape[0])
            width, height = max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale))
            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            left, top = x+(w-width)//2, y+34+(available_h-height)//2
            canvas[top:top+height, left:left+width] = resized
        for index, line in enumerate(lines):
            cls._text(canvas, line, x+12, y+44+index*16, .42)

    @staticmethod
    def _state(sample, now_ns, max_age_ns):
        if sample is None:
            return "WAITING"
        if now_ns-sample.monotonic_ns > max_age_ns:
            return "STALE"
        if not getattr(sample, "valid", True):
            return "INVALID TRACKING"
        return "LIVE"

    @classmethod
    def compose(cls, overlay, *, ecm, ultrasound, robot, ndi, now_ns, max_age_ns):
        """Build a display image. Sidebar samples are latest, not time-matched."""
        canvas = np.full((cls.size[1], cls.size[0], 3), (23, 27, 33), np.uint8)
        cls._panel(canvas, (12, 12, 984, 936), "LIVE OVERLAY | time-matched",
                   overlay, () if overlay is not None else ("WAITING / STALE: no current synchronized video",))
        for y, name, sample in ((12, "ECM", ecm), (264, "ULTRASOUND", ultrasound)):
            state = cls._state(sample, now_ns, max_age_ns)
            cls._panel(canvas, (1008, y, 420, 240), f"{name} FEED | {state}",
                       sample.image if state == "LIVE" else None,
                       () if state == "LIVE" else ("No current video",))
        robot_state = cls._state(robot, now_ns, max_age_ns)
        poses = dict(robot.poses) if robot is not None and robot_state == "LIVE" else {}
        for y, identifier, name in ((516, 2, "ECM"), (628, 0, "PSM1"), (740, 1, "PSM2")):
            values = poses.get(identifier)
            lines = ("No current pose",)
            if values is not None:
                lines = ("XYZ (m): " + "  ".join(f"{v:+.4f}" for v in values[:3]),
                         *[f"R{row+1}: " + "  ".join(f"{v:+.4f}" for v in values[3+row*3:6+row*3])
                           for row in range(3)])
            cls._panel(canvas, (1008, y, 420, 100), f"{name} POSE | {robot_state}", lines=lines)
        state = cls._state(ndi, now_ns, max_age_ns)
        lines = ("No current valid position",)
        if state == "LIVE":
            lines = ("XYZ (m): " + "  ".join(f"{v:+.4f}" for v in ndi.transform[:3, 3]),
                     "Coordinates in NDI tracker frame")
        cls._panel(canvas, (1008, 852, 420, 96), f"NDI POSITION | {state}", lines=lines)
        return canvas

    def update(self, overlay, **samples):
        """Draw and process window events; return False when the user exits."""
        if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
            return False
        cv2.imshow(self.title, self.compose(overlay, **samples))
        return cv2.waitKey(1) & 0xff not in (27, ord("q"))
