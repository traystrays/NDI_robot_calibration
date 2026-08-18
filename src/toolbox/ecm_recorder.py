import cv2
import threading
import datetime
import os
import time
import argparse

# Global event to signal stopping of recording across all threads
stop_recording_event = threading.Event()

def list_cameras():
    """
    Attempts to list available video capture devices by trying common device indices.
    This helps the user identify which index corresponds to their DVI/USB streams.
    """
    print("Searching for available video capture devices...")
    available_cameras = []
    # Check up to 10 potential device indices.
    # On some systems, higher indices might be used for virtual cameras or specific hardware.
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"  Found camera at index {i}")
            available_cameras.append(i)
            cap.release() # Release the camera immediately after checking
        else:
            # If a camera at index i is not found, it doesn't necessarily mean
            # subsequent indices are also unavailable, so we continue checking.
            pass
    if not available_cameras:
        print("No cameras found. Please ensure your devices are connected and drivers are installed.")
        print("If cameras are connected, try increasing the range in list_cameras() function.")
    print("-" * 30)
    return available_cameras

def record_video(device_index, filename, resolution=(1280, 720), fps=30):
    """
    Records video from a specified device index to a file.
    It continuously captures frames until the global 'stop_recording_event' is set.

    Args:
        device_index (int): The index of the video capture device.
        filename (str): The path and filename for the output video file (e.g., "output.mp4").
        resolution (tuple): Desired resolution (width, height) for the video.
                            Note: The camera might not support the exact resolution.
        fps (int): Desired frames per second for the video.
                   Note: The camera might not support the exact FPS.
    """
    print(f"Attempting to open camera at index {device_index}...")
    cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        print(f"Error: Could not open video device at index {device_index}. "
              "Please check the index and ensure the device is not in use.")
        return

    # Try to set resolution and FPS. These might be overridden by the camera's capabilities.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Get the actual resolution and FPS the camera is providing
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Recording from device {device_index} with actual resolution: "
          f"{actual_width}x{actual_height} and FPS: {actual_fps:.2f}")

    # Define the codec and create VideoWriter object
    # 'mp4v' is a common codec for .mp4 files, generally well-supported on Windows.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(filename, fourcc, actual_fps, (actual_width, actual_height))

    ts_filename = filename.replace(".mp4", "_timestamps.txt")
    ts_file = open(ts_filename, "w")

    if not out.isOpened():
        print(f"Error: Could not create video writer for {filename}. "
              "Check codec, file path, or disk space.")
        cap.release()
        return

    print(f"Recording video from device {device_index} to {filename}. "
          "Press 'q' in any video window to stop all recordings.")

    # Loop to capture and write frames until the stop event is set
    while not stop_recording_event.is_set():
        ret, frame = cap.read()
        # # check frame
        # print(frame.dtype, frame.shape)
        # exit()
        if not ret:
            print(f"Warning: Failed to read frame from device {device_index}. "
                  "This stream might have disconnected or encountered an issue. Exiting recording for this stream.")
            break

        out.write(frame)
        ts_file.write(f"{datetime.datetime.now().isoformat()}\n")

        # resize the frame before displaying that
        # cv2.resize(frame, (640,480))

        # Display the frame in a named window for monitoring
        cv2.imshow(f'Device {device_index} - Stream Preview (Press Q to Stop)', frame)

        # Check for 'q' key press every 1 millisecond.
        # This check is global, so pressing 'q' in any video window will trigger the stop event.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n'q' pressed. Signaling all recordings to stop.")
            stop_recording_event.set() # Set the global event to stop all threads

    print(f"Stopping recording for device {device_index}...")
    cap.release() # Release the video capture object
    out.release() # Release the video writer object
    ts_file.close() # Close the timestamp file
    # Note: cv2.destroyAllWindows() is called once in the main thread after all recordings stop
    # to ensure all windows are closed together.

if __name__ == "__main__":
    print("--- Python Dual Video Stream Recorder for Windows 11 ---")
    print("This script will record two video streams simultaneously.")
    print("------------------------------------------------------")

    parser = argparse.ArgumentParser(description="Dual Video Stream Recorder")
    # parser.add_argument("--device1", type=int, default=0, help="Device index for the FIRST video stream")
    # parser.add_argument("--device2", type=int, default=1, help="Device index for the SECOND video stream")
    parser.add_argument("--output_dir", "-d", type=str, default="recorded_videos", help="Directory to save recorded videos")
    args = parser.parse_args()

    # Step 1: List available cameras to help the user find device indices
    list_cameras()
    # breakpoint()

    # Step 2: Get user input for device indices
    try:
        device_index_1 = int(input("Enter the device index for the FIRST video stream (e.g., 0): ")) # 1
        device_index_2 = int(input("Enter the device index for the SECOND video stream (e.g., 1): "))
    except ValueError:
        print("Invalid input. Please enter integer values for device indices.")
        exit()

    # Step 3: Define output directory and filenames with timestamps
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    filename_1 = os.path.join(output_dir, f"video_stream_1_{timestamp}.mp4")
    filename_2 = os.path.join(output_dir, f"video_stream_2_{timestamp}.mp4")

    # Step 4: Create and start recording threads for each stream
    print("\nStarting video recordings...")
    thread1 = threading.Thread(target=record_video, args=(device_index_1, filename_1))
    thread2 = threading.Thread(target=record_video, args=(device_index_2, filename_2))

    thread1.start()
    thread2.start()

    # Wait for both threads to complete.
    # They will complete when the 'q' key is pressed in any of the video preview windows,
    # which sets the global 'stop_recording_event'.
    thread1.join()
    thread2.join()

    # Close all OpenCV windows after all threads have finished
    cv2.destroyAllWindows()

    print("\nAll video recordings have stopped.")
    print(f"Videos saved to: {os.path.abspath(output_dir)}")
    print("You can find your recorded files in the 'recorded_videos' folder.")

