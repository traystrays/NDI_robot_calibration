from pathlib import Path
from toolbox.imaging import ImageCapture


input_path = Path(r"data\20260804_test_data\plane_overlay.mp4")
output_path = Path(r"data\20260804_test_data\plane_overlay.mp4")

camera = ImageCapture(input_path)

# index = camera.find_camera()

# camera.record_video(index)



