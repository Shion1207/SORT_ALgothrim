import cv2
import torch
import numpy as np
from sort import Sort

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # confidence threshold

# Khởi tạo SORT tracker
tracker = Sort()

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize nhỏ để YOLO xử lý nhanh
    img = cv2.resize(frame, (640, 480))

    # YOLO inference
    results = model(img)

    # Chuyển kết quả sang định dạng numpy
    detections = results.xyxy[0].cpu().numpy()

    # Lọc chỉ lấy người (class_id == 0 trong COCO)
    person_detections = []
    for *xyxy, conf, cls in detections:
        if int(cls) == 0:  # 0 là người trong YOLOv5 COCO
            person_detections.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])

    # Chuyển sang numpy array (N, 4)
    dets_np = np.array(person_detections)

    # Cập nhật SORT
    tracks = tracker.update(dets_np)

    # Vẽ các bounding box và ID
    for track in tracks:
        bbox, track_id = track[0], int(track[1])
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hiển thị kết quả
    cv2.imshow("YOLOv5 + SORT", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
