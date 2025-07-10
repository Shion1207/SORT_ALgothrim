import cv2
import torch
import numpy as np
from sort import Sort

# Load YOLOv5 model từ torch hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # ngưỡng confidence

# Khởi tạo SORT
tracker = Sort()

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame)

    # Detections: [x1, y1, x2, y2, conf, class]
    detections = results.xyxy[0].cpu().numpy()

    # Lọc người (class == 0)
    person_detections = []
    for *xyxy, conf, cls in detections:
        if int(cls) == 0:
            person_detections.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])

    # Nếu không có người thì tracker.update([]), tránh lỗi
    dets_np = np.array(person_detections) if len(person_detections) > 0 else np.empty((0, 4))

    # Gửi vào SORT
    tracked = tracker.update(dets_np)

    # Vẽ bounding boxes
    for track in tracked:
        x1, y1, x2, y2 = map(int, track[0])
        track_id = int(track[1])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Tracking People with YOLOv5 + SORT", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
