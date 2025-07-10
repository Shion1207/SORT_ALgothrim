import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        for i in range(3): self.kf.F[i, i+4] = 1  # motion model
        self.kf.H = np.zeros((4, 7))
        self.kf.H[:4, :4] = np.eye(4)
        self.kf.R[2:, 2:] *= 10.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1)
        self.kf.x[:4] = np.array([[cx], [cy], [s], [r]])
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1)
        self.kf.update(np.array([cx, cy, s, r]))

    def get_state(self):
        cx, cy, s, r = self.kf.x[:4].flatten()
        w = np.sqrt(s * r)
        h = s / w
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

class Sort:
    def __init__(self, max_age=5, iou_threshold=0.3):
        self.trackers = []
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def update(self, detections):
        trks = []
        for t in self.trackers:
            trks.append(t.predict().flatten())
        matched, unmatched_dets, unmatched_trks = [], list(range(len(detections))), list(range(len(trks)))

        if len(trks) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(detections), len(trks)))
            for d, det in enumerate(detections):
                for t, trk in enumerate(trks):
                    iou_matrix[d, t] = iou(det, self.trackers[t].get_state())
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched.append((r, c))
                    unmatched_dets.remove(r)
                    unmatched_trks.remove(c)

        for r, c in matched:
            self.trackers[c].update(detections[r])
            self.trackers[c].time_since_update = 0

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i]))

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return [(t.get_state(), t.id) for t in self.trackers]
