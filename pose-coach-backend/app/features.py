# app/features.py
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple

# MoveNet 17 จุด (ดัชนี)
KP = {
    "nose":0, "left_eye":1, "right_eye":2, "left_ear":3, "right_ear":4,
    "left_shoulder":5, "right_shoulder":6, "left_elbow":7, "right_elbow":8,
    "left_wrist":9, "right_wrist":10, "left_hip":11, "right_hip":12,
    "left_knee":13, "right_knee":14, "left_ankle":15, "right_ankle":16,
}

def _v(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """เวกเตอร์ q - p (shape: (2,))"""
    return q - p

def _safe_norm(v: np.ndarray, eps: float = 1e-8) -> float:
    n = float(np.linalg.norm(v))
    return max(n, eps)

def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    มุมที่จุด b (องศา) จากสามจุด a-b-c (ใช้เฉพาะ x,y)
    0° = เหยียดตรงแนวเดียว, 180° = งอพับมาก
    """
    ab = _v(b[:2], a[:2])
    cb = _v(b[:2], c[:2])
    na = _safe_norm(ab)
    nc = _safe_norm(cb)
    cos_th = float(np.clip(np.dot(ab, cb) / (na * nc), -1.0, 1.0))
    th = np.degrees(np.arccos(cos_th))
    return th

def _mid(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (p[:2] + q[:2]) * 0.5

def _dist2d(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p[:2] - q[:2]))

# ------------------------------
# 1) มุมข้อหลักต่อเฟรม
# ------------------------------
def compute_angles(frames: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    คำนวณมุมหลักต่อเฟรม (องศา) ใช้ x,y หลังผ่าน preprocess แล้ว
    คืน dict ของ time-series (length = T)
      - knee_L, knee_R, hip_L, hip_R, ankle_L, ankle_R
      - elbow_L, elbow_R
      - trunk  (มุมลำตัวเทียบแกนตั้ง: 0 = ตรง, มากขึ้น = ก้ม)
      - knee   (ตัวแทน นำสัญญาณ: min(knee_L, knee_R))
    """
    T = len(frames)
    ang = {
        "knee_L": np.zeros(T, dtype=np.float32),
        "knee_R": np.zeros(T, dtype=np.float32),
        "hip_L":  np.zeros(T, dtype=np.float32),
        "hip_R":  np.zeros(T, dtype=np.float32),
        "ankle_L":np.zeros(T, dtype=np.float32),
        "ankle_R":np.zeros(T, dtype=np.float32),
        "elbow_L":np.zeros(T, dtype=np.float32),
        "elbow_R":np.zeros(T, dtype=np.float32),
        "trunk":  np.zeros(T, dtype=np.float32),
        "knee":   np.zeros(T, dtype=np.float32),
    }

    for t, f in enumerate(frames):
        # ขา
        lhip, lknee, lank = f[KP["left_hip"]], f[KP["left_knee"]], f[KP["left_ankle"]]
        rhip, rknee, rank = f[KP["right_hip"]], f[KP["right_knee"]], f[KP["right_ankle"]]

        ang["knee_L"][t]  = _angle_deg(lhip, lknee, lank)   # มุมเข่าซ้าย
        ang["knee_R"][t]  = _angle_deg(rhip, rknee, rank)   # มุมเข่าขวา
        ang["hip_L"][t]   = _angle_deg(lknee, lhip, f[KP["left_shoulder"]])  # มุมสะโพกซ้าย
        ang["hip_R"][t]   = _angle_deg(rknee, rhip, f[KP["right_shoulder"]]) # มุมสะโพกขวา
        ang["ankle_L"][t] = _angle_deg(lknee, lank, lank + (lank - lknee))   # มุมข้อเท้า (ประมาณ)
        ang["ankle_R"][t] = _angle_deg(rknee, rank, rank + (rank - rknee))

        # แขน (เผื่อใช้ใน push-up)
        lsho, lel, lw = f[KP["left_shoulder"]], f[KP["left_elbow"]], f[KP["left_wrist"]]
        rsho, rel, rw = f[KP["right_shoulder"]], f[KP["right_elbow"]], f[KP["right_wrist"]]
        ang["elbow_L"][t] = _angle_deg(lsho, lel, lw)
        ang["elbow_R"][t] = _angle_deg(rsho, rel, rw)

        # ลำตัว: มุมเวกเตอร์ hip_mid -> shoulder_mid กับแกนตั้ง (0° = ตรง)
        hip_mid = _mid(lhip, rhip)
        sho_mid = _mid(lsho, rsho)
        v_trunk = sho_mid - hip_mid
        # แกนตั้ง (0,1)
        vy = np.array([0.0, 1.0], dtype=np.float32)
        cos_t = float(np.clip(np.dot(v_trunk, vy) / (_safe_norm(v_trunk) * _safe_norm(vy)), -1.0, 1.0))
        ang["trunk"][t] = np.degrees(np.arccos(cos_t))

        # ตัวแทนสัญญาณเข่า (ใช้ min เพื่อจับช่วงงอสุดใน squat)
        ang["knee"][t] = min(ang["knee_L"][t], ang["knee_R"][t])

    return ang

# ------------------------------
# 2) เวกเตอร์กระดูก (shape features)
# ------------------------------
_BONES: List[Tuple[str, str]] = [
    # ลำตัวและสะโพก
    ("left_hip","left_shoulder"), ("right_hip","right_shoulder"),
    ("left_shoulder","right_shoulder"), ("left_hip","right_hip"),
    # ขาซ้าย
    ("left_hip","left_knee"), ("left_knee","left_ankle"),
    # ขาขวา
    ("right_hip","right_knee"), ("right_knee","right_ankle"),
    # แขนซ้าย
    ("left_shoulder","left_elbow"), ("left_elbow","left_wrist"),
    # แขนขวา
    ("right_shoulder","right_elbow"), ("right_elbow","right_wrist"),
]

def pairwise_bone_vectors(frames: List[np.ndarray], unit: bool = True) -> np.ndarray:
    """
    สร้างฟีเจอร์รูปร่างต่อเฟรมจากเวกเตอร์กระดูกที่เลือกไว้
    คืน array shape = (T, 2 * len(BONES)) ถ้า unit=True จะเป็นเวกเตอร์หน่วย
    """
    T = len(frames)
    M = len(_BONES)
    out = np.zeros((T, M * 2), dtype=np.float32)
    for t, f in enumerate(frames):
        vecs = []
        for a, b in _BONES:
            pa = f[KP[a], :2]; pb = f[KP[b], :2]
            v = pb - pa
            if unit:
                n = _safe_norm(v)
                v = v / n
            vecs.extend([v[0], v[1]])
        out[t] = np.array(vecs, dtype=np.float32)
    return out

# ------------------------------
# 3) เรขาคณิตประกอบการตรวจ issues
# ------------------------------
def compute_geom(frames: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    เก็บสเกลและระยะสำคัญต่อเฟรม:
      - shoulder_width, hip_width, knee_width, ankle_width
      - shank_L/R (เข่า→ข้อเท้า), thigh_L/R (สะโพก→เข่า)
      - mid_hip(x,y), mid_shoulder(x,y)
    """
    T = len(frames)
    geom: Dict[str, np.ndarray] = {
        "shoulder_width": np.zeros(T, dtype=np.float32),
        "hip_width": np.zeros(T, dtype=np.float32),
        "knee_width": np.zeros(T, dtype=np.float32),
        "ankle_width": np.zeros(T, dtype=np.float32),
        "shank_L": np.zeros(T, dtype=np.float32),
        "shank_R": np.zeros(T, dtype=np.float32),
        "thigh_L": np.zeros(T, dtype=np.float32),
        "thigh_R": np.zeros(T, dtype=np.float32),
        "mid_hip": np.zeros((T,2), dtype=np.float32),
        "mid_shoulder": np.zeros((T,2), dtype=np.float32),
    }
    for t, f in enumerate(frames):
        ls, rs = f[KP["left_shoulder"]], f[KP["right_shoulder"]]
        lh, rh = f[KP["left_hip"]], f[KP["right_hip"]]
        lk, rk = f[KP["left_knee"]], f[KP["right_knee"]]
        la, ra = f[KP["left_ankle"]], f[KP["right_ankle"]]

        geom["shoulder_width"][t] = _dist2d(ls, rs)
        geom["hip_width"][t]      = _dist2d(lh, rh)
        geom["knee_width"][t]     = _dist2d(lk, rk)
        geom["ankle_width"][t]    = _dist2d(la, ra)

        geom["shank_L"][t] = _dist2d(lk, la)
        geom["shank_R"][t] = _dist2d(rk, ra)
        geom["thigh_L"][t] = _dist2d(lh, lk)
        geom["thigh_R"][t] = _dist2d(rh, rk)

        geom["mid_hip"][t]      = _mid(lh, rh)
        geom["mid_shoulder"][t] = _mid(ls, rs)

    return geom

# ------------------------------
# ตัวอย่างใช้งาน
# ------------------------------
if __name__ == "__main__":
    # ทดสอบแบบ dummy
    T = 30
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(T):
        f = np.zeros((17,3), dtype=np.float32)
        f[:, :2] = rng.normal(0, 1, size=(17,2)).astype(np.float32)  # ต้องส่งเข้ามาหลัง preprocess แล้ว
        f[:, 2]  = rng.uniform(0.7, 0.99, size=(17,))
        frames.append(f)

    angles = compute_angles(frames)
    vecs   = pairwise_bone_vectors(frames)
    geom   = compute_geom(frames)

    print("angles keys:", list(angles.keys()))
    print("vecs shape:", vecs.shape)
    print("geom keys:", list(geom.keys()))
