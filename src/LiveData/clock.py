"""One high-resolution host clock for acquisition, matching, and display ages."""
import time


def clock_ns():
    # Python 3.11's time.monotonic_ns() uses coarse GetTickCount64 on Windows.
    # perf_counter_ns uses the system-wide monotonic QueryPerformanceCounter,
    # including in the NDI child process; do not mix the two clock epochs.
    return time.perf_counter_ns()
