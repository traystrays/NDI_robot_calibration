"""Receive and print DVPControl's local UDP pose stream.

Start DVPControl first, then run: python src/LiveData/robot_receiver.py
"""

import socket
import struct
import math


# EpiLogger's DVAPI_MIP: float pos[3], float orientation[9], int type.
# Three 52-byte records form the 156-byte pose packet on this Windows setup.
POSE_RECORD = struct.Struct("<12fi")
MANIPULATORS = {0: "PSM1", 1: "PSM2", 2: "ECM"}


def decode_poses(data: bytes) -> list[tuple[int, tuple[float, ...]]]:
    """Decode the layout expected by EpiLogger's callbackDVAPI()."""
    expected = 3 * POSE_RECORD.size
    if len(data) != expected:
        raise ValueError(f"Expected {expected} pose bytes, got {len(data)}")

    poses = []
    for values in POSE_RECORD.iter_unpack(data):
        manipulator_id = values[12]
        if manipulator_id not in MANIPULATORS:
            raise ValueError(f"Unexpected manipulator ID: {manipulator_id}")
        if not all(math.isfinite(value) for value in values[:12]):
            raise ValueError("Packet contains non-finite position/orientation values")
        poses.append((manipulator_id, values[:12]))
    if len({identifier for identifier, _ in poses}) != 3:
        raise ValueError("Packet contains duplicate manipulator IDs")
    return poses


def main() -> None:
    publisher = ("127.0.0.1", 60000)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        # Windows chooses an available port for receiving measurements.
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(5.0)
        receiver_port = receiver.getsockname()[1]

        # Match SlaveData::Start(): send the receiving port as a raw
        # little-endian unsigned short. Registration uses a separate socket.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as registration:
            registration.sendto(struct.pack("<H", receiver_port), publisher)

        print(
            f"Registration sent to {publisher[0]}:{publisher[1]}. "
            f"Listening on 127.0.0.1:{receiver_port}.",
            flush=True,
        )
        print("Waiting for UDP packets. Press Ctrl+C to stop.", flush=True)
        print(
            "Fields: manipulator, id, x, y, z, o0, o1, o2, o3, o4, o5, o6, o7, o8\n"
            "Positions are in metres; orientation values retain EpiLogger's "
            "wire order (not Euler angles).",
            flush=True,
        )
        packet_count = 0

        while True:
            try:
                data, address = receiver.recvfrom(65535)
            except socket.timeout:
                print(
                    "No packet received in the last 5 seconds. Check that "
                    "DVPControl is running and streaming robot data. If it "
                    "was restarted, restart this script to register again.",
                    flush=True,
                )
                continue

            try:
                poses = decode_poses(data)
            except ValueError as error:
                print(f"Skipping packet from {address}: {error}", flush=True)
                continue

            packet_count += 1
            lines = [f"received #{packet_count}: {len(data)} bytes from {address[0]}:{address[1]}"]
            for identifier, values in poses:
                numbers = ", ".join(f"{value:.5f}" for value in values)
                lines.append(f"{MANIPULATORS[identifier]}, {identifier}, {numbers}")
            print("\n".join(lines), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except OSError as error:
        raise SystemExit(f"UDP error: {error}") from error
