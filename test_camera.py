import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print("✅ Camera is working.")
else:
    print("❌ Failed to capture from camera.")

cap.release()