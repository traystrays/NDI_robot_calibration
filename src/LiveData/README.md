# LiveData

Live acquisition components for the tracked ultrasound overlay. The application
entry point is `scripts/live_ultrasound_overlay.py`; the complete architecture,
clock contract, setup, and hardware acceptance steps are in
[`scripts/LIVE_OVERLAY.md`](../../scripts/LIVE_OVERLAY.md).

The acquisition layer has no GUI. Each device owns a worker thread and a bounded
history. Main-thread code snapshots those histories and uses `FrameSynchronizer`
to produce frozen `MatchedFrame` bundles. Start/stop on the application thread;
do not call `poll()` concurrently. Frozen image packets own their immutable pixel
storage. Robot samples remain in the existing receiver's own bounded buffer.
