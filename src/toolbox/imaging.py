"""
Opening and reading image live feed from cameras, saving timestamp and video frames.
TODO: live feed open, time synconize frames
Save data to file
live-stream real time process 
"""

import cv2
from pathlib import Path
from datetime import datetime
import csv
import time

class VideoCapture:

    def __init__(self, output_folder:Path):

        self.output_folder = output_folder
        self.timestamp_file = None
        self.recording = False
        self.fps = None
        self.finish = False

        if self.output_folder.exists() and not self.output_folder.is_dir():
            raise ValueError(
                f"{self.output_folder} needs to be a folder, not a file"
            )
            
        self.output_folder.mkdir(parents=True, exist_ok=True)

        
    def find_camera(self):
        """
        Find camera index by checking 10 
        """
        # let max cameras index be 10
        max = 10

        cams = []
        for i in range(max):

            cap = cv2.VideoCapture(i)
        
            if not cap.isOpened():
                continue

            cams.append(i)
            cap.release()

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
                    
                    self.index = i

                    return i

                if key == ord('q'):
                    cv2.destroyAllWindows()
                    print("Quitting program")
                    
                    return None

                if key == 32:
                    print("Moving to next camera")
                    cv2.destroyWindow(f"camera {i}")
                    break

        return None

    def record_video(self, camera_index: int):
        """
        Record video and timestamps. Saved to provided output folder path.
        """

        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Camera{camera_index} did not open. Check camera connection")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # print(fps)

        if self.fps == 0 or self.fps > 100: # if fps weird reset to fps = 30
            print(self.fps)
            self.fps = 30

        now = datetime.now().strftime("%Y%m%d%H%M%S")
        self.video_path = self.output_folder / f"_video_{now}.mp4"
        self.timestamp_path = self.output_folder / f"_timestamp_{now}.txt"
        
        print(self.video_path, self.timestamp_path)
        
        # initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (width, height))
        
        print("Recording starting NOW")
        self.recording = True
        
        self.recording_loop()

        
        


    def recording_loop(self):
        
        with open(self.timestamp_path, "w",newline='') as timestamp_file:
            writer = csv.writer(timestamp_file)
        
        try:
            while not self.finish:
                self.ret, self.frame = self.cap.read()
                
                if not self.ret:
                    print("Missing frame")
                    break
                    
                else:
                    timestamp = datetime.now().isoformat(timespec="microseconds")
                    self.video_writer.write(self.frame)
                    
                    with open(self.timestamp_path, "a",newline='') as timestamp_file:
                        writer = csv.writer(timestamp_file)
                        writer.writerow([timestamp])

                    cv2.imshow("US", self.frame)
                    
                    if cv2.waitKey (1) & 0xFF == ord('q'): 
                        self.finish = True
                    
        finally:
            self.end_recording()    
                
        
    def end_recording(self):
        self.cap.release()
        self.video_writer.release()
        self.recording = False
        cv2.destroyAllWindows()

