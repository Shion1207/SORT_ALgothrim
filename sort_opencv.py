import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# ========== 1. Kalman Filter cho từng đối tượng ==========
class KalmanBoxTracker:
    count = 0  # Đếm ID cho các đối tượng

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)  # Trạng thái gồm: x, y, w, h, vx, vy, vw
        self.kf.F = np.eye(7)  # Ma trận chuyển trạng thái
        self.kf.F[0,4] = 1  # x += vx
        self.kf.F[1,5] = 1  # y += vy
        self.kf.F[2,6] = 1  # w += vw
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        self.kf.P *= 10   # Ma trận hiệp phương sai ban đầu
        self.kf.R *= 10   # Nhiễu đo lường
        self.kf.Q *= 0.01 # Nhiễu chuyển động
        self.kf.x[:4] = np.array(bbox).reshape((4,1))  # Gán vị trí ban đầu
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.kf.update(np.array(bbox).reshape((4,1)))

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x[:4].reshape(-1)

    def get_state(self):
        return self.kf.x[:4].reshape(-1)

# ========== 2. SORT tracker quản lý nhiều đối tượng ==========
class Sort:
    def __init__(self, max_age=5, iou_threshold=0.3):
        self.trackers = []
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
        yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2]*boxA[3]
        boxBArea = boxB[2]*boxB[3]
        return interArea / (boxAArea + boxBArea - interArea + 1e-6)

    def associate(self, dets, trks):
        if len(trks) == 0:
            return [], np.arange(len(dets)), []
        iou_matrix = np.zeros((len(dets), len(trks)))
        for d in range(len(dets)):
            for t in range(len(trks)):
                iou_matrix[d, t] = self.iou(dets[d], trks[t])
        matched = linear_sum_assignment(-iou_matrix)
        matched = np.array(list(zip(*matched)))

        unmatched_dets = [d for d in range(len(dets)) if d not in matched[:,0]]
        unmatched_trks = [t for t in range(len(trks)) if t not in matched[:,1]]

        good_matches = []
        for m in matched:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                good_matches.append(m)
        return good_matches, unmatched_dets, unmatched_trks

    def update(self, detections):
        trks = [trk.predict() for trk in self.trackers]
        matches, unmatched_dets, unmatched_trks = self.associate(detections, trks)

        for m in matches:
            self.trackers[m[1]].update(detections[m[0]])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i]))

        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]

        results = []
        for t in self.trackers:
            pos = t.get_state()
            results.append(np.append(pos, t.id))
        return results

# ========== 3. Main: Webcam + Giả lập đối tượng ==========
cap = cv2.VideoCapture(0)
sort_tracker = Sort()

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]

    # Giả lập các object di chuyển
    detections = []
    for i in range(3):
        x = int(100 + 50 * np.sin(cv2.getTickCount()/1e5 + i))
        y = 100 + i * 80
        detections.append([x, y, 60, 60])
    detections = np.array(detections)

    # Cập nhật SORT
    tracks = sort_tracker.update(detections)

    # Hiển thị kết quả
    for d in tracks:
        x, y, w_, h_, id_ = d.astype(int)
        cv2.rectangle(frame, (x, y), (x + w_, y + h_), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {id_}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("SORT Tracker", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
