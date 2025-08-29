# app/extractor.py
from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import mediapipe as mp

# แมป Mediapipe Pose (33 จุด) -> MoveNet 17 จุด (index)
# Mediapipe indices: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#pose-landmarks
MP = mp.solutions.pose.PoseLandmark
MP_IDX = {lm.name.lower(): lm.value for lm in MP}

# MoveNet 17 ชื่อคีย์เรียงตามสคีมาเราใช้
MV_ORDER = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

# แมปชื่อ -> index ของ Mediapipe (ถ้าไม่มี ให้คาดเคลื่อนเล็กน้อยจากจุดใกล้เคียง)
MP_MAP = {
    "nose": MP_IDX["nose"],
    "left_eye": MP_IDX["left_eye"],
    "right_eye": MP_IDX["right_eye"],
    "left_ear": MP_IDX["left_ear"],
    "right_ear": MP_IDX["right_ear"],
    "left_shoulder": MP_IDX["left_shoulder"],
    "right_shoulder": MP_IDX["right_shoulder"],
    "left_elbow": MP_IDX["left_elbow"],
    "right_elbow": MP_IDX["right_elbow"],
    "left_wrist": MP_IDX["left_wrist"],
    "right_wrist": MP_IDX["right_wrist"],
    "left_hip": MP_IDX["left_hip"],
    "right_hip": MP_IDX["right_hip"],
    "left_knee": MP_IDX["left_knee"],
    "right_knee": MP_IDX["right_knee"],
    "left_ankle": MP_IDX["left_ankle"],
    "right_ankle": MP_IDX["right_ankle"],
}

def _to_movenet17(landmarks, image_w: int, image_h: int) -> List[List[float]]:
    """
    แปลง 33 แลนด์มาร์กของ Mediapipe -> 17 จุด MoveNet: [x,y,conf]
    ค่าพิกัด normalize เป็น 0..1 ด้วยขนาดภาพ
    conf ใช้ visibility ของ mediapipe (0..1)
    """
    pts = []
    for name in MV_ORDER:
        idx = MP_MAP[name]
        lm = landmarks[idx]
        x = float(np.clip(lm.x, 0.0, 1.0))
        y = float(np.clip(lm.y, 0.0, 1.0))
        c = float(np.clip(lm.visibility, 0.0, 1.0))
        pts.append([x, y, c])
    return pts

def extract_keypoints_from_video(
    video_path: str,
    target_fps: Optional[int] = 30,
    static_image_mode: bool = False,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """
    อ่านวิดีโอและคืน JSON keypoints:
    {
      "fps": <int>,
      "frames": [ [ [x,y,conf]*17 ], ... ],
      "meta": {"src":"path", "frame_count": N, "orig_fps": f}
    }
    - ถ้า target_fps < orig_fps จะ skip เฟรมให้ใกล้เคียง
    - ค่าพิกัด x,y เป็น normalized 0..1 เพื่อให้เทียบต่างขนาดภาพได้
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"เปิดวิดีโอไม่ได้: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    skip = 1
    if target_fps and orig_fps > 1e-3:
        skip = max(int(round(orig_fps / float(target_fps))), 1)
    out_frames: List[List[List[float]]] = []

    with mp.solutions.pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        idx = 0
        grabbed = True
        while grabbed:
            grabbed, frame = cap.read()
            if not grabbed:
                break
            # เฟรมที่ไม่ต้องประมวลผล ให้ข้าม
            if (idx % skip) != 0:
                idx += 1
                continue

            h, w = frame.shape[:2]
            # BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                pts17 = _to_movenet17(res.pose_landmarks.landmark, w, h)
            else:
                # ถ้าตรวจไม่เจอ ให้เติม conf=0 ทั้งเฟรม
                pts17 = [[0.0, 0.0, 0.0] for _ in range(17)]

            out_frames.append(pts17)
            idx += 1
            if max_frames is not None and len(out_frames) >= max_frames:
                break

    cap.release()
    return {
        "fps": int(target_fps if target_fps else round(orig_fps)),
        "frames": out_frames,
        "meta": {
            "src": video_path,
            "frame_count": int(frame_count),
            "orig_fps": float(orig_fps),
            "skip": int(skip),
        }
    }
