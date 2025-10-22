import cv2

cap = cv2.VideoCapture(0)  # Try 0, 1, or 2 if this doesn't work
if not cap.isOpened():
    print("Error: Cannot open camera")
else:
    print("Camera opened successfully! Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Test Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
