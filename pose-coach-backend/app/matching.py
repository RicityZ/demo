# app/matching.py
from __future__ import annotations
import numpy as np
from typing import Callable, List, Sequence, Tuple, Optional

"""
Dynamic Time Warping (DTW) utilities
- dtw_align: จับคู่ลำดับเวลา user ↔ reference ด้วยหน้าต่าง Sakoe–Chiba (radius)
- รองรับสัญญาณ 1 มิติ (เช่น มุมเข่า) และหลายมิติ (เช่น มุมหลายข้อพร้อมกัน)
- frame_map_from_dtw: แปลงเส้นทาง DTW → mapping ดัชนีเฟรม
- warp_to_ref: บีบ/ยืดลำดับ user ให้มีความยาวเท่า reference ตาม path
"""

# -----------------------------
# ระยะทางพื้นฐาน
# -----------------------------
def _dist_l2(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance สำหรับเวกเตอร์ (รองรับสเกลาร์ด้วย)"""
    return float(np.linalg.norm(a - b))

def _dist_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))

# -----------------------------
# เตรียมสัญญาณ (ให้เป็น 2D: T x D)
# -----------------------------
def _as_2d(x: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]  # (T,) -> (T,1)
    elif arr.ndim != 2:
        raise ValueError("input sequence ต้องเป็น 1D หรือ 2D")
    return arr

# -----------------------------
# DTW core with Sakoe–Chiba band
# -----------------------------
def dtw_align(
    user_seq: Sequence[float] | np.ndarray,
    ref_seq:  Sequence[float] | np.ndarray,
    radius: int = 8,
    metric: str = "l2",
) -> Tuple[float, List[Tuple[int,int]]]:
    """
    จัดแนวเวลา user_seq ↔ ref_seq ด้วย DTW (Sakoe–Chiba band)
    Args:
        user_seq: (Tu,) หรือ (Tu,D)
        ref_seq : (Tr,) หรือ (Tr,D)
        radius  : ความกว้างแถบการจัดแนว (ยิ่งเล็กยิ่งเร็ว/เข้ม)
        metric  : 'l2' (ค่าเริ่มต้น) หรือ 'l1'
    Returns:
        (total_cost, path) โดย path เป็น list ของ (i,j) ไล่จากต้นจนจบ
    หมายเหตุ:
        - ใช้ cost accumulative แบบ float32 เพื่อความเร็ว
        - รองรับหลายมิติด้วยการคำนวณระยะทางของเวกเตอร์ต่อคู่เฟรม
    """
    A = _as_2d(user_seq)  # (Tu,D)
    B = _as_2d(ref_seq)   # (Tr,D)
    Tu, D = A.shape
    Tr, _ = B.shape
    if Tu == 0 or Tr == 0:
        return 0.0, []

    if metric == "l2":
        dist_fn: Callable[[np.ndarray, np.ndarray], float] = _dist_l2
    elif metric == "l1":
        dist_fn = _dist_l1
    else:
        raise ValueError("metric ต้องเป็น 'l2' หรือ 'l1'")

    radius = max(int(radius), 0)

    # ค่าใหญ่สำหรับช่องที่อยู่นอกหน้าต่าง
    INF = np.float32(1e15)

    # ตารางต้นทุนสะสม (Tu+1) x (Tr+1)
    # ใช้แบบปิดขอบเพื่อสะดวกในการ backtrace
    C = np.full((Tu + 1, Tr + 1), INF, dtype=np.float32)
    C[0, 0] = 0.0

    # ตาราง pointer สำหรับ backtrace: 0=diag, 1=up, 2=left
    ptr = np.full((Tu + 1, Tr + 1), -1, dtype=np.int8)

    # คำนวณภายในวงแหวน Sakoe–Chiba
    for i in range(1, Tu + 1):
        j_start = max(1, i - radius)
        j_end   = min(Tr, i + radius)
        ai = A[i - 1]
        for j in range(j_start, j_end + 1):
            bj = B[j - 1]
            d = dist_fn(ai, bj)

            # เลือกทิศที่ต้นทุนสะสมน้อยสุด
            c_diag = C[i - 1, j - 1]
            c_up   = C[i - 1, j]
            c_left = C[i, j - 1]

            # standard step pattern (1,1), (1,0), (0,1)
            if c_diag <= c_up and c_diag <= c_left:
                C[i, j] = d + c_diag
                ptr[i, j] = 0
            elif c_up <= c_left:
                C[i, j] = d + c_up
                ptr[i, j] = 1
            else:
                C[i, j] = d + c_left
                ptr[i, j] = 2

    total_cost = float(C[Tu, Tr])

    # backtrace
    path: List[Tuple[int,int]] = []
    i, j = Tu, Tr
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))  # index ของข้อมูลจริง
        p = ptr[i, j]
        if p == 0:
            i -= 1; j -= 1
        elif p == 1:
            i -= 1
        elif p == 2:
            j -= 1
        else:
            break

    # กรณีวิ่งจนชิดขอบด้านใดด้านหนึ่ง
    while i > 0:
        path.append((i - 1, j - 1))
        i -= 1
    while j > 0:
        path.append((i - 1, j - 1))
        j -= 1

    path.reverse()
    return total_cost, path

# -----------------------------
# แปลง path → mapping ดัชนีเฟรม
# -----------------------------
def frame_map_from_dtw(
    path: List[Tuple[int,int]],
    tu: Optional[int] = None,
    tr: Optional[int] = None,
    direction: str = "user_to_ref",
) -> List[int]:
    """
    แปลงเส้นทาง DTW ให้เป็น mapping แบบอาเรย์ของดัชนี
    Args:
        path: list (i,j)
        tu  : ความยาว user (ถ้าไม่ใส่จะอนุมานจาก path)
        tr  : ความยาว ref  (ถ้าไม่ใส่จะอนุมานจาก path)
        direction:
          - 'user_to_ref' → คืน list[L=tu] โดย map_u[i] = j ที่จับคู่
          - 'ref_to_user' → คืน list[L=tr] โดย map_r[j] = i ที่จับคู่
    """
    if not path:
        return []

    if tu is None:
        tu = max(i for i, _ in path) + 1
    if tr is None:
        tr = max(j for _, j in path) + 1

    if direction == "user_to_ref":
        out = np.zeros(tu, dtype=np.int32)
        # เก็บค่าล่าสุดของ j ที่จับคู่กับ i (ใน path อาจมีหลายคู่ต่อ i)
        last_j = 0
        k = 0
        for i in range(tu):
            # เดิน path เก็บค่าที่ตรงกับ i
            while k < len(path) and path[k][0] < i:
                last_j = path[k][1]
                k += 1
            if k < len(path) and path[k][0] == i:
                last_j = path[k][1]
                # ขยับคอยเก็บคู่ของ i เดียวกัน (กรณีแนวนอน)
                while k + 1 < len(path) and path[k + 1][0] == i:
                    k += 1
                    last_j = path[k][1]
                k += 1
            out[i] = last_j
        return out.tolist()

    elif direction == "ref_to_user":
        out = np.zeros(tr, dtype=np.int32)
        last_i = 0
        k = 0
        for j in range(tr):
            while k < len(path) and path[k][1] < j:
                last_i = path[k][0]
                k += 1
            if k < len(path) and path[k][1] == j:
                last_i = path[k][0]
                while k + 1 < len(path) and path[k + 1][1] == j:
                    k += 1
                    last_i = path[k][0]
                k += 1
            out[j] = last_i
        return out.tolist()

    else:
        raise ValueError("direction ต้องเป็น 'user_to_ref' หรือ 'ref_to_user'")

# -----------------------------
# ยืด/บีบสัญญาณ user ให้ยาวเท่า reference ตาม path
# -----------------------------
def warp_to_ref(
    user_seq: Sequence[float] | np.ndarray,
    path: List[Tuple[int,int]],
    target_len: Optional[int] = None,
) -> np.ndarray:
    """
    สร้างสัญญาณใหม่ของ user ตามลำดับเฟรมอ้างอิง (ref timeline)
    - หาก target_len ไม่ระบุ จะใช้ความยาวสูงสุดของ j ใน path + 1
    - รองรับทั้ง 1D และ 2D
    """
    U = _as_2d(user_seq)        # (Tu, D)
    if not path:
        return U.copy()

    if target_len is None:
        target_len = max(j for _, j in path) + 1

    Tu, D = U.shape
    out = np.zeros((target_len, D), dtype=np.float32)

    # รวมค่า user ของเฟรมที่ถูกแม็ปมาที่ ref เฟรมเดียวกัน (เฉลี่ย)
    counts = np.zeros(target_len, dtype=np.int32)
    for i, j in path:
        j_clamped = min(max(j, 0), target_len - 1)
        out[j_clamped] += U[min(max(i, 0), Tu - 1)]
        counts[j_clamped] += 1

    # หารด้วยจำนวน mapping เพื่อให้ได้ค่าเฉลี่ย
    mask = counts > 0
    out[mask] = out[mask] / counts[mask, None]

    # ช่องที่ไม่มี mapping (กรณี path ไม่ครอบทั้งหมด) → เติมด้วยการคาบเกี่ยวเชิงเส้น
    if not np.all(mask):
        # เติมค่าจากจุดที่มีข้อมูลใกล้สุด
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            # fallback: ทั้งหมดไม่มี mapping → คืนศูนย์
            return out
        for d in range(D):
            out[:, d] = np.interp(
                np.arange(target_len, dtype=np.float32),
                idxs.astype(np.float32),
                out[idxs, d]
            ).astype(np.float32)

    # คืน 1D เมื่อ input เป็น 1D
    return out.squeeze() if (out.shape[1] == 1) else out

# -----------------------------
# ตัวช่วยสำหรับหลายมิติ (เช่น ใช้มุมหลายตัวพร้อมกัน)
# -----------------------------
def dtw_align_multi(
    user_feats: np.ndarray,
    ref_feats:  np.ndarray,
    radius: int = 8,
    metric: str = "l2",
) -> Tuple[float, List[Tuple[int,int]]]:
    """
    DTW สำหรับฟีเจอร์หลายมิติ (T,D)
    - เช่น ใช้มุม [knee, hip, trunk] รวมกันเพื่อเพิ่มความนิ่ง
    """
    U = _as_2d(user_feats)  # (Tu,D)
    R = _as_2d(ref_feats)   # (Tr,D)
    return dtw_align(U, R, radius=radius, metric=metric)

# -----------------------------
# ตัวอย่างใช้งาน / ทดสอบเบื้องต้น
# -----------------------------
if __name__ == "__main__":
    # สร้างสัญญาณทดสอบ: user ช้ากว่า ref
    T_ref = 100
    T_user = 140
    t_ref  = np.linspace(0, 2*np.pi, T_ref)
    t_user = np.linspace(0, 2*np.pi, T_user)
    ref = 150 - 40*np.sin(t_ref)
    user = 150 - 40*np.sin(t_user + 0.1) + np.random.normal(0, 0.5, size=T_user)

    cost, path = dtw_align(user, ref, radius=10, metric="l2")
    print("cost:", cost, "path_len:", len(path))

    # map user→ref
    u2r = frame_map_from_dtw(path, tu=T_user, tr=T_ref, direction="user_to_ref")
    print("map len:", len(u2r), "first/last:", u2r[0], u2r[-1])

    # warp user ให้ยาวเท่า ref
    user_warp = warp_to_ref(user, path, target_len=T_ref)
    print("warped shape:", user_warp.shape)
