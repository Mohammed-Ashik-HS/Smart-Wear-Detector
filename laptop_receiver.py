import socket
import cv2
import numpy as np
import tensorflow as tf

# ===== LOAD MODEL =====
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = open("labels.txt").read().splitlines()

def predict(frame):
    img = cv2.resize(frame, (224,224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]

    idx = np.argmax(output)
    confidence = output[idx]

    return labels[idx], confidence


# ===== SOCKET =====
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 5000))
server.listen(1)

print("Waiting for Pi...")
conn, addr = server.accept()
print("Connected:", addr)

latest_frame = None

cv2.namedWindow("MICROSCOPE LIVE", cv2.WINDOW_NORMAL)

while True:
    try:
        size_data = conn.recv(4)
        if not size_data:
            break

        size = int.from_bytes(size_data, 'big')

        data = b""
        while len(data) < size:
            packet = conn.recv(size - len(data))
            if not packet:
                break
            data += packet

        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            continue

        latest_frame = frame.copy()

        display = frame.copy()
        cv2.putText(display, "LIVE STREAM", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("MICROSCOPE LIVE", display)

        key = cv2.waitKey(10) & 0xFF

        # ===== CAPTURE =====
        if key == ord('c') and latest_frame is not None:
            label, conf = predict(latest_frame)

            status = f"{label} ({conf:.2f})"

            print("RESULT:", status)

            color = (0,0,255) if label != "no_wear" else (0,255,0)

            result_img = latest_frame.copy()
            cv2.putText(result_img, status, (30,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("RESULT", result_img)

        if key == ord('q'):
            break

    except Exception as e:
        print("ERROR:", e)
        break

cv2.destroyAllWindows()