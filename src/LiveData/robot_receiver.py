"""Check DVPControl's local UDP pose stream without decoding robot data.

Start DVPControl first, then run: python src/LiveData/robot_receiver.py
"""

import socket
import struct


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

            print(
                f"received: {len(data)} bytes from {address[0]}:{address[1]}",
                flush=True,
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except OSError as error:
        raise SystemExit(f"UDP error: {error}") from error
