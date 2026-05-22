import cv2 
import socket 
LAPTOP_IP = "192.168.43.48" 
PORT = 5000 
cap = cv2.VideoCapture(0) 
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
client.connect((LAPTOP_IP, PORT)) 
while True: 
ret, frame = cap.read() 
if not ret: 
continue 
frame = cv2.resize(frame, (640, 480)) 
_, buffer = cv2.imencode('.jpg', frame, 
[int(cv2.IMWRITE_JPEG_QUALITY), 60]) 
data = buffer.tobytes() 
size = len(data).to_bytes(4, 'big') 
client.sendall(size) 
client.sendall(data)
