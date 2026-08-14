"""
Opening and reading image live feed from cameras, saving timestamp and video frames.
TODO: live feed open, time synconize frames
Save data to file
live-stream real time process 
"""

import cv2
from pathlib import Path
from datetime import datetime

class ImageCapture:

    def __init__(self, output_folder:Path):

        self.output_folder = output_folder
        self.cap = None
        self.video_writer = None
        self.timestamp_file = None

        if self.output_folder.exists() and not self.output_folder.is_dir():
            raise ValueError(
                f"{self.output_folder} needs to be a folder, not a file"
            )

        
    def find_camera(self):

        # let max cameras index be 10
        max = 10

        cams = []
        for i in range(max):

            cap = cv2.VideoCapture(i)
              
            if not cap.isOpened():
                continue

            cams.append(i)

        if len(cams) == 0:
            print("No cameras found")

            return None
        
        print(f"found {cams}")
        print("Press ENTER to select camera, press Q to quit program, press SPACE for next")

        for i in cams:
            cap = cv2.VideoCapture(i)

            while True:

                ret, frame = cap.read()

                if not ret: break

                cv2.imshow(f"camera {i}", frame)
                key = cv2.waitKey(1) & 0xFF # mask for the 1st 8bits
                # Pressed ENTER
                if key == 13:
                    print(f"camera selected {i}")

                    return i

                if key == ord('q'):
                    print("Quitting program")
                    return None

                if key == 32:
                    print("Moving to next camera")
                    break

        return None

    def record_video(self, camera_index):
        """
        Record video and timestamps. Saved to provided output folder path.
        """

        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Camera{camera_index} did not open. Check camera connection")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # print(fps)

        if fps == 0 or fps > 100:
            print(fps)
            fps = 30

        self.video_path, self.timestamp_path = self.output_files()

        print(self.video_path, self.timestamp_path)

        exit()

        # initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter()



    def recording_loop():

        return None

    def output_files(self):
        """
        Check the folder and create it if necessary. 
        Produce folder for video and timestamp.
        """

        now = datetime.now().strftime("%H%M%S")
        video_path = self.output_folder + "_video_"+ now + ".mp4"
        timestamp_path = self.output_folder +   "_timestamp_"+ now + ".txt"

        return video_path, timestamp_path


