import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture(0) # use for webcam
cap = cv2.VideoCapture("test2_walking.mp4") # use for video file

CONF = 0.35

# 🟩 Define Region Of Interest (x1, y1, x2, y2)
ROI = (100, 100, 400, 450)

def draw_hud(frame, text):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 70), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    cv2.putText(frame, text, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame, classes=[0], conf=CONF, verbose=False)
    boxes = results[0].boxes

    # Draw ROI box
    x1r, y1r, x2r, y2r = ROI
    cv2.rectangle(frame, (x1r, y1r), (x2r, y2r), (255, 0, 0), 3)

    count = 0

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # center of detection
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # check if center is inside ROI
            inside = (x1r <= cx <= x2r) and (y1r <= cy <= y2r)

            color = (0, 255, 0) if inside else (0, 0, 255)

            # draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

            if inside:
                count += 1

    frame = draw_hud(frame, f"People in zone: {count}")

    cv2.imshow("People Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()





