import socket
import cv2
import numpy as np
import struct

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 5000))
server.listen(1)

print("Waiting for Pi...")
conn, addr = server.accept()
print("Connected:", addr)

data = b""
payload_size = struct.calcsize("Q")

while True:
    try:
        # get size
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        packed_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_size)[0]

        # get frame
        while len(data) < msg_size:
            data += conn.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

        # 🔴 IMPORTANT FIX
        if frame is None:
            print("Frame decode failed")
            continue

        cv2.imshow("MICROSCOPE LIVE", frame)

        # 🔴 IMPORTANT FIX (MUST BE INSIDE LOOP)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("ERROR:", e)
        break

cv2.destroyAllWindows()