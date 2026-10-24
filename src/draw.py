import cv2

# COCO-17 keypoint order used by YOLOv8-pose, and the bone pairs connecting them.
_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]

_COLORS = [
    (66, 135, 245),
    (245, 66, 93),
    (66, 245, 135),
    (245, 188, 66),
    (188, 66, 245),
    (66, 245, 230),
]


def color_for_id(track_id: int) -> tuple[int, int, int]:
    return _COLORS[track_id % len(_COLORS)]


def draw_tracked_boxes(frame, boxes_xyxy, track_ids, confidences, names=None):
    for idx, ((x1, y1, x2, y2), track_id, conf) in enumerate(zip(boxes_xyxy, track_ids, confidences)):
        color = color_for_id(track_id)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, 2)

        name = names[idx] if names else f"ID {track_id}"
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 8), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(
            frame, label, (p1[0] + 2, p1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return frame


def draw_pose(frame, keypoints_xy, keypoints_conf, track_ids, kpt_conf_thresh: float = 0.5):
    for person_kpts, person_conf, track_id in zip(keypoints_xy, keypoints_conf, track_ids):
        color = color_for_id(track_id)

        for i, j in _SKELETON:
            if person_conf[i] < kpt_conf_thresh or person_conf[j] < kpt_conf_thresh:
                continue
            pt1 = (int(person_kpts[i][0]), int(person_kpts[i][1]))
            pt2 = (int(person_kpts[j][0]), int(person_kpts[j][1]))
            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

        for (x, y), c in zip(person_kpts, person_conf):
            if c < kpt_conf_thresh:
                continue
            cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
    return frame
