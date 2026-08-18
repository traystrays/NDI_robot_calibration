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
