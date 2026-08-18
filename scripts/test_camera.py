"""Small manual camera-discovery smoke test."""

from toolbox.imaging import list_cameras, read_camera_frame


def main() -> None:
    cameras = list_cameras()
    print(f"Found cameras: {cameras}")
    if cameras:
        frame = read_camera_frame(cameras[0])
        print(f"Camera {cameras[0]} frame shape: {frame.shape}")


if __name__ == "__main__":
    main()
