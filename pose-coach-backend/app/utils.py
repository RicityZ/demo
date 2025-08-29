# app/utils.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple

"""
utils.py — รวมฟังก์ชันช่วยเหลือเกี่ยวกับเรขาคณิต, มุม, เวกเตอร์
ใช้ซ้ำในหลายโมดูล เช่น features.py, segmentation.py, scoring.py, feedback.py
"""

# -----------------------------
# 1) เวกเตอร์ และมุม
# -----------------------------
def vector(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    สร้างเวกเตอร์จาก p → q
    Args:
        p, q: np.ndarray ขนาด (2,) หรือ (3,)
    Returns:
        np.ndarray(2,) หรือ (3,) = q - p
    """
    return q - p

def norm(v: np.ndarray, eps: float = 1e-8) -> float:
    """
    คำนวณความยาวเวกเตอร์ |v| พร้อมกัน NaN/Inf
    """
    return float(np.maximum(np.linalg.norm(v), eps))

def unit_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    คืนเวกเตอร์หน่วย v / |v|
    """
    n = norm(v, eps)
    return v / n

def angle_between(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    คำนวณมุมที่ p2 จากจุด p1-p2-p3 (องศา)
    - 0° = เหยียดตรงแนวเดียวกัน
    - 180° = งอพับสุด
    """
    v1 = vector(p2[:2], p1[:2])
    v2 = vector(p2[:2], p3[:2])
    n1 = norm(v1)
    n2 = norm(v2)
    cos_theta = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    theta = np.degrees(np.arccos(cos_theta))
    return theta

# -----------------------------
# 2) ระยะทาง และจุดกึ่งกลาง
# -----------------------------
def distance_2d(p: np.ndarray, q: np.ndarray) -> float:
    """
    ระยะทาง Euclidean บนระนาบ XY
    """
    return float(np.linalg.norm(p[:2] - q[:2]))

def midpoint(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    จุดกึ่งกลางของ p และ q (เฉพาะ x,y)
    """
    return (p[:2] + q[:2]) * 0.5

# -----------------------------
# 3) Normalization และ Scaling
# -----------------------------
def center_points(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    ย้าย origin ของ keypoints ให้ center อยู่ที่ (0,0)
    Args:
        points: np.ndarray(N, 2 or 3)
        center: np.ndarray(2,) = จุดอ้างอิง เช่น mid-hip
    Returns:
        np.ndarray(N, 2 or 3)
    """
    pts = points.copy()
    pts[:, :2] -= center[:2]
    return pts

def scale_points(points: np.ndarray, scale_len: float) -> np.ndarray:
    """
    ปรับสเกลให้ skeleton มีระยะ anchor = 1.0
    Args:
        points: np.ndarray(N, 2 or 3)
        scale_len: ระยะ anchor เช่น ความกว้างไหล่
    Returns:
        np.ndarray(N, 2 or 3)
    """
    pts = points.copy()
    if scale_len <= 1e-6:
        return pts
    pts[:, :2] = pts[:, :2] / scale_len
    return pts

def rotate_points(points: np.ndarray, theta: float) -> np.ndarray:
    """
    หมุน keypoints ทั้งชุดบนระนาบ XY ด้วยมุม theta (เรเดียน)
    Args:
        points: np.ndarray(N, 2 or 3)
        theta: มุมเรเดียนบวก = หมุนทวนเข็ม
    Returns:
        np.ndarray(N, 2 or 3)
    """
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ], dtype=np.float32)
    pts = points.copy()
    pts[:, :2] = (R @ pts[:, :2].T).T
    return pts

# -----------------------------
# 4) Filtering และ Smoothing
# -----------------------------
def moving_average(data: np.ndarray, win: int = 5) -> np.ndarray:
    """
    Moving Average สำหรับ smoothing 1D signal
    Args:
        data: np.ndarray(T,)
        win: window size (ต้องเป็นคี่)
    Returns:
        np.ndarray(T,)
    """
    if win < 1:
        return data
    if win % 2 == 0:
        win += 1
    pad = win // 2
    data_pad = np.pad(data, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(data_pad, kernel, mode="valid")

def low_pass_filter(data: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    Exponential Low-pass Filter สำหรับ smoothing 1D signal
    Args:
        data: np.ndarray(T,)
        alpha: ค่าระหว่าง 0..1 (ยิ่งเล็กยิ่ง smooth)
    """
    out = np.zeros_like(data, dtype=np.float32)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out

# -----------------------------
# 5) ตรวจสอบความถูกต้องของ keypoints
# -----------------------------
def validate_keypoints_shape(frames: List[np.ndarray], expected_kp: int = 17) -> None:
    """
    ตรวจสอบว่า frames เป็น list[np.ndarray] และมี shape (expected_kp, 3)
    """
    if not isinstance(frames, list):
        raise TypeError("frames ต้องเป็น list")
    for i, f in enumerate(frames):
        if not isinstance(f, np.ndarray):
            raise TypeError(f"frame[{i}] ต้องเป็น numpy.ndarray")
        if f.shape != (expected_kp, 3):
            raise ValueError(f"frame[{i}] ต้องมี shape ({expected_kp}, 3) แต่ได้ {f.shape}")
        if not np.all(np.isfinite(f)):
            raise ValueError(f"frame[{i}] พบ NaN/Inf")

# -----------------------------
# ตัวอย่างการใช้งาน / ทดสอบ
# -----------------------------
if __name__ == "__main__":
    # ทดสอบการคำนวณมุม
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    theta = angle_between(a, b, c)
    print(f"มุมที่ b: {theta:.2f}°")

    # ทดสอบ rotate
    pts = np.array([
        [1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]
    ], dtype=np.float32)
    pts_rot = rotate_points(pts, np.pi / 2)
    print("ก่อนหมุน:\n", pts[:, :2])
    print("หลังหมุน 90°:\n", pts_rot[:, :2])
