"""Bounded-wait nearest matching, anchored on ECM video receipt time.

One consumer calls poll(). Producers only write their own histories. Each ECM
frame is finalized exactly once, including unmatched frames. No last-good pose
is substituted for an invalid tracking sample. Matching permits reuse of slower
stream samples; it is not interpolation or hardware synchronization.
"""
from dataclasses import dataclass
import math

from .packets import VideoFrame, TransformSample
from .clock import clock_ns


@dataclass(frozen=True)
class MatchedFrame:
    ecm: VideoFrame
    ultrasound: VideoFrame | None
    robot: TransformSample | None
    ndi: TransformSample | None
    errors_ms: tuple[float | None, ...]  # ultrasound, robot, NDI minus ECM
    status: str


class FrameSynchronizer:
    def __init__(self, ecm, ultrasound, robot, ndi, *, tolerances_ms=(20, 20, 40),
                 wait_ms=60, max_age_ms=250):
        values = (*tolerances_ms, wait_ms, max_age_ms)
        if len(tolerances_ms) != 3 or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Timing limits must be finite and nonnegative")
        self.ecm = ecm
        self.sources = (ultrasound, robot, ndi)
        self.tolerances = tuple(v * 1_000_000 for v in tolerances_ms)
        self.wait_ns = wait_ms * 1_000_000
        self.max_age_ns = max_age_ms * 1_000_000
        self.last_sequence = 0
        self.matched = self.unmatched = self.dropped = 0

    def poll(self, now_ns=None):
        now = clock_ns() if now_ns is None else now_ns
        frames = [f for f in self.ecm.snapshot() if f.sequence > self.last_sequence]
        if frames and frames[0].sequence > self.last_sequence + 1:
            self.dropped += frames[0].sequence - self.last_sequence - 1
        histories = [source.snapshot() for source in self.sources]
        output = []
        for frame in frames:
            t = frame.monotonic_ns
            # Wait for full candidate windows, or a fixed receipt-time deadline.
            ready = all(h and h[-1].monotonic_ns >= t + tol
                        for h, tol in zip(histories, self.tolerances))
            if not ready and now - frame.received_ns < self.wait_ns:
                break
            chosen, errors, failures = [], [], []
            for name, history, tolerance in zip(
                    ("ultrasound", "robot", "ndi"), histories, self.tolerances):
                sample = min(history, key=lambda s: abs(s.monotonic_ns - t)) if history else None
                delta = sample.monotonic_ns - t if sample is not None else None
                errors.append(None if delta is None else delta / 1_000_000)
                if sample is None or abs(delta) > tolerance:
                    failures.append(name + "_unmatched")
                    sample = None
                elif not getattr(sample, "valid", True):
                    failures.append(name + "_invalid")
                chosen.append(sample)
            if now - t > self.max_age_ns:
                failures.append("stale")
            status = ",".join(failures) if failures else "matched"
            output.append(MatchedFrame(frame, *chosen, tuple(errors), status))
            self.last_sequence = frame.sequence
            if failures:
                self.unmatched += 1
            else:
                self.matched += 1
        return output
