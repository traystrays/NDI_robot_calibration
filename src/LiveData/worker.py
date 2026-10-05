"""
Data acquisition worker class for UDP robot pose packets. 
Shared lifecycle for live camera, tracking, and UDP acquisition workers.
"""

import threading
import time


class Worker:
    """One-use acquisition worker with visible failures and bounded shutdown."""
    def __init__(self, name):
        self.name = name
        self.error = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None

    def start(self, timeout=15):
        if self._thread is not None:
            raise RuntimeError("Create a new receiver to restart acquisition")
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._ready.wait(min(1, max(0, deadline - time.monotonic()))):
            if time.monotonic() >= deadline:
                self._stop.set()
                raise TimeoutError(f"{self.name}: startup timed out after {timeout:g}s; "
                                   "device driver may be blocked or device is in use")
            print(f"{self.name}: waiting for device startup...", flush=True)
        if self.error:
            self.stop()
            raise RuntimeError(f"{self.name}: {self.error}") from self.error
        return self

    def _run(self):
        # starts data acquisition. each subclass supplies acquire()
        try:
            self.acquire() 
        except Exception as error:
            self.error = error
        finally:
            self._ready.set() # set ready

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(5)
            if self._thread.is_alive():
                raise RuntimeError(f"{self.name}: driver blocked; worker did not stop")

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()

