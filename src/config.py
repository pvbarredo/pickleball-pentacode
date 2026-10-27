from dataclasses import dataclass


@dataclass
class Config:
    input_path: str = "picklecam.mp4"
    output_path: str = "output/picklecam_tracked.mp4"
    model_name: str = "yolov8n.pt"
    pose_model_name: str = "yolov8n-pose.pt"
    tracker: str = "src/trackers/botsort_pickleball.yaml"
    conf: float = 0.4
    person_class_id: int = 0
    show: bool = True
    max_frames: int | None = None
    pose: bool = False
    kpt_conf: float = 0.5
    min_box_height: int = 0
    court_only: bool = False
    seed_gallery: str | None = None
    confirm_seconds: float = 1.0
    match_thresh: float = 0.75
    detect_lines: bool = False
    court_profile: str = "picklecam_vertical"
    court_calibration: str | None = "output/calibration/picklecam_vertical.json"
    track_ball: bool = False
    ball_conf: float = 0.3
    ball_imgsz: int = 1280
    detect_violations: bool = False
