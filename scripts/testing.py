import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from ndi_robot_registration.clean_si_data import clean_si_data
from ndi_robot_registration.match import match, match_video
from ndi_robot_registration.transforms import (
    as_transform,
    invert_transform,
    is_valid_transform,
)

