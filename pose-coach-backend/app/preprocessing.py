# app/preprocessing.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict

# MoveNet 17 keypoints index map
KP = {
    "nose":0, "left_eye":1, "right_eye":2, "left_ear":3, "right_ear":4,
    "left_shoulder":5, "right_shoulder":6, "left_elbow":7, "right_elbow":8,
    "left_wrist":9, "right_wrist":10, "left_hip":11, "right_hip":12,
    "left_knee":13, "right_knee":14, "left_ankle":15, "right_ankle":16
}

def validate_packet(frames: List[np.ndarray]) -> None:
    """
    ตรวจโครงสร้าง input ให้ถูกต้อง
    - frames เป็น list ของ array ขนาด (17,3) [x,y,conf]
    - ค่าเป็นตัวเลข finite ไม่มี NaN/Inf
    """
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError("frames ต้องเป็น list และห้ามว่าง")
    for i, f in enumerate(frames):
        if not isinstance(f, np.ndarray):
            raise TypeError(f"frame[{i}] ต้องเป็น numpy.ndarray")
        if f.shape != (17, 3):
            raise ValueError(f"frame[{i}] shape ต้องเป็น (17,3) แต่ได้ {f.shape}")
        if not np.all(np.isfinite(f)):
            raise ValueError(f"frame[{i}] พบ NaN/Inf")

def smooth_keypoints(frames: List[np.ndarray], win: int = 5) -> List[np.ndarray]:
    """
    moving average แบบ window คี่ เพื่อให้ keypoint นิ่งขึ้น
    """
    if win < 1 or win % 2 == 0:
        raise ValueError("win ต้องเป็นจำนวนคี่ >= 1")
    arr = np.stack(frames, axis=0)  # (T,17,3)
    pad = win // 2
    arr_pad = np.pad(arr, ((pad,pad),(0,0),(0,0)), mode="edge")
    out = []
    for t in range(arr.shape[0]):
        seg = arr_pad[t:t+win]  # (win,17,3)
        out.append(seg.mean(axis=0))
    return [o.astype(np.float32) for o in out]

def filter_low_conf(frames: List[np.ndarray], thr: float = 0.5, mode: str = "interp") -> List[np.ndarray]:
    """
    จัดการจุดที่ความมั่นใจต่ำ
    - mode="interp": เติมค่าโดยอนุมานจากเฟรมก่อนหน้า
    - mode="mask": ลดน้ำหนัก conf ลงแต่คงค่า x,y เดิม
    """
    out = [frames[0].copy()]
    for i in range(1, len(frames)):
        prev = out[-1].copy()
        cur = frames[i].copy()
        low = cur[:,2] < thr
        if mode == "interp":
            cur[low,:2] = prev[low,:2]                   # ใช้ตำแหน่งก่อนหน้า
            cur[low,2] = (cur[low,2] + prev[low,2]) * .5 # ยก conf ขึ้นเล็กน้อย
        elif mode == "mask":
            cur[low,2] *= 0.5
        else:
            raise ValueError("mode ต้องเป็น 'interp' หรือ 'mask'")
        out.append(cur.astype(np.float32))
    return out

def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[:2] + b[:2]) * 0.5

def center_on_midhip(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    ย้าย origin ไปที่ mid-hip เพื่อลดผลการเลื่อนตำแหน่งทั้งตัวในเฟรม
    """
    out = []
    for f in frames:
        lhip = f[KP["left_hip"], :2]
        rhip = f[KP["right_hip"], :2]
        center = _mid(lhip, rhip)  # (2,)
        g = f.copy()
        g[:, :2] = g[:, :2] - center  # translate
        out.append(g.astype(np.float32))
    return out

def _safe_dist(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.maximum(np.linalg.norm(a[:2] - b[:2]), eps))

def scale_by_skeleton(
    frames: List[np.ndarray],
    anchor_pairs: List[Tuple[str, str]] = [("left_shoulder","right_shoulder")]
) -> List[np.ndarray]:
    """
    ปรับสเกลด้วยระยะกระดูกอ้างอิง เช่น ความกว้างไหล่
    ถ้าไหล่มองไม่เห็น ให้ fallback ใช้สะโพก
    """
    out = []
    for f in frames:
        scale_len = 0.0
        for a, b in anchor_pairs:
            pa = f[KP[a], :2]
            pb = f[KP[b], :2]
            scale_len = _safe_dist(pa, pb)
            if scale_len > 1e-5:
                break
        if scale_len <= 1e-5:
            pa = f[KP["left_hip"], :2]
            pb = f[KP["right_hip"], :2]
            scale_len = _safe_dist(pa, pb)
        g = f.copy()
        g[:, :2] = g[:, :2] / scale_len
        out.append(g.astype(np.float32))
    return out

def rotate_to_shoulders(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    หมุนแกนให้แนวไหล่ขนานแกน x ลดผลเอียงกล้อง
    หมุนจุดทั้งหมดด้วยมุมเดียวกันต่อเฟรม
    """
    out = []
    for f in frames:
        ls = f[KP["left_shoulder"], :2]
        rs = f[KP["right_shoulder"], :2]
        v = rs - ls  # แนวไหล่
        ang = np.arctan2(v[1], v[0])  # กับแกน x
        c, s = np.cos(-ang), np.sin(-ang)  # หมุนกลับให้อยู่ขนานแกน x
        R = np.array([[c, -s],[s, c]], dtype=np.float32)
        g = f.copy()
        g[:, :2] = (R @ g[:, :2].T).T
        out.append(g.astype(np.float32))
    return out

def preprocess_pipeline(
    frames: List[np.ndarray],
    smooth_win: int = 5,
    conf_thr: float = 0.5,
    conf_mode: str = "interp"
) -> List[np.ndarray]:
    """
    พรีโพรเซสครบลำดับ: validate → smooth → filter_low_conf → center → scale → rotate
    """
    validate_packet(frames)
    x = smooth_keypoints(frames, win=smooth_win)
    x = filter_low_conf(x, thr=conf_thr, mode=conf_mode)
    x = center_on_midhip(x)
    x = scale_by_skeleton(x)
    x = rotate_to_shoulders(x)
    return x

# ตัวอย่างรันเดี่ยว
if __name__ == "__main__":
    T = 30
    rng = np.random.default_rng(42)
    frames = []
    for _ in range(T):
        f = np.zeros((17,3), dtype=np.float32)
        f[:, :2] = rng.normal(0, 0.1, size=(17,2))
        f[:, 2]  = rng.uniform(0.6, 0.99, size=(17,))
        frames.append(f)
    pp = preprocess_pipeline(frames, smooth_win=5, conf_thr=0.5, conf_mode="interp")
    print("done preprocess", len(pp))
