# app/segmentation.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

"""
Segmentation utilities
- นับจำนวน rep แบบสตรีมด้วย hysteresis (ทนสัญญาณสั่น)
- ติดป้ายเฟสต่อเฟรม: start, down, bottom, up

คำจำกัดความ (สำหรับท่าตระกูล squat/push-up ที่ "ลง" = มุมข้อหลักลดลง, "ขึ้น" = มุมเพิ่มขึ้น)
- ใช้สัญญาณนำเป็น ang_series เช่น knee (องศา)
- start: ยืน/เหยียดตรง ใกล้ปลายบนของช่วงมุม
- down : มุม "ลดลง" อย่างมีนัยสำคัญ
- bottom: ใกล้จุดต่ำสุด (local minimum)
- up   : มุม "เพิ่มขึ้น" อย่างมีนัยสำคัญ
"""

# -----------------------------
# Helpers
# -----------------------------
def _nan_to_num(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def _grad(x: np.ndarray) -> np.ndarray:
    """อนุพันธ์เชิงตัวเลข 1D"""
    if len(x) < 2: 
        return np.zeros_like(x)
    return np.gradient(x).astype(np.float32)

def _smooth_1d(x: np.ndarray, win: int = 5) -> np.ndarray:
    """moving average (win คี่)"""
    if win <= 1:
        return x.astype(np.float32)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(win, dtype=np.float32) / win
    y = np.convolve(xp, ker, mode="valid")
    return y.astype(np.float32)

def _adaptive_thresholds(x: np.ndarray) -> Tuple[float, float, float, float]:
    """
    คำนวณ threshold อัตโนมัติจากสถิติสัญญาณ:
      - high / low ของค่า (สำหรับนิยาม start/bottom โดยสัมพัทธ์)
      - th_down / th_up ของ "ความเร็ว" (gradient) เพื่อรู้ทิศทางลง/ขึ้น
    """
    x = x.astype(np.float32)
    x_min, x_max = float(np.percentile(x, 5)), float(np.percentile(x, 95))
    span = max(x_max - x_min, 1e-3)

    # เกณฑ์ค่ามุม: บริเวณบน (start) / ล่าง (bottom)
    high = x_max - 0.15 * span
    low  = x_min + 0.15 * span

    # เกณฑ์อนุพันธ์: เร็วพอที่จะนับเป็นลง/ขึ้น (สเกลตามช่วง)
    # สัญญาณหน่วย = องศา/เฟรม
    th_vel = 0.10 * span
    th_down = -th_vel     # ลง: derivative < th_down
    th_up   = +th_vel     # ขึ้น: derivative > th_up
    return high, low, th_down, th_up

# -----------------------------
# Phase labeling (batch)
# -----------------------------
def label_phases(angles: Dict[str, np.ndarray],
                 lead_key: str = "knee",
                 smooth_win: int = 5) -> List[str]:
    """
    ให้ labels ต่อเฟรม: ['start'|'down'|'bottom'|'up']
    - ใช้สัญญาณนำ angles[lead_key] (เช่น 'knee')
    - ปรับเกณฑ์อัตโนมัติจาก percentiles ของสัญญาณ
    """
    x = angles.get(lead_key, None)
    if x is None or len(x) == 0:
        return []
    x = _nan_to_num(np.asarray(x, dtype=np.float32))
    x_sm = _smooth_1d(x, win=smooth_win)
    dx = _grad(x_sm)

    high, low, th_down, th_up = _adaptive_thresholds(x_sm)

    T = len(x_sm)
    labels = ["start"] * T

    # กติกาง่าย: ใช้ทั้งค่าและอนุพันธ์
    for t in range(T):
        if x_sm[t] <= low:
            labels[t] = "bottom"
        else:
            if dx[t] <= th_down:
                labels[t] = "down"
            elif dx[t] >= th_up:
                labels[t] = "up"
            else:
                labels[t] = "start"
    # ทำให้ bottom เป็นกอเดียว (เลือกที่ local minima แท้จริง)
    labels = _consolidate_bottom(x_sm, labels)
    return labels

def _consolidate_bottom(x: np.ndarray, labels: List[str]) -> List[str]:
    """
    รวมช่วงที่ติด bottom ยาว ๆ ให้เน้นจุดกึ่งกลางของส่วนต่ำสุด:
    - หา segment ต่อเนื่องที่ label=='bottom'
    - ภายในแต่ละ segment กำหนดจุดค่าต่ำสุดเป็น 'bottom' ที่เหลือให้เป็น 'down' หรือ 'up' ตามทิศใกล้เคียง
    """
    T = len(x)
    lbl = labels[:]
    i = 0
    while i < T:
        if lbl[i] != "bottom":
            i += 1
            continue
        j = i
        while j < T and lbl[j] == "bottom":
            j += 1
        # ช่วง [i, j)
        seg = slice(i, j)
        k_rel = int(np.argmin(x[seg]))
        k = i + k_rel
        # ตั้งให้เฉพาะ k เป็น bottom เด่น
        for t in range(i, j):
            if t == k:
                lbl[t] = "bottom"
            else:
                # ดูทิศทางใกล้เคียง
                if t < k:
                    lbl[t] = "down"
                else:
                    lbl[t] = "up"
        i = j
    return lbl

# -----------------------------
# Rep counting (streaming)
# -----------------------------
@dataclass
class RepCounterConfig:
    """
    พารามิเตอร์แบบอัตโนมัติ + สารตั้งต้น
    - เราใช้ threshold ด้านความเร็วจากสถิติสัญญาณที่อัปเดตระหว่างสตรีม
    """
    min_frames_per_rep: int = 8      # กันการเด้งนับเร็วเกิน
    idle_frames_reset: int = 90      # ถ้าหยุดนานให้รีเซ็ตสถานะ
    ema_alpha: float = 0.1           # ความไวของ EMA ต่อ span

class RepState:
    IDLE = 0      # อยู่บน (start/ยืน) ยังไม่เริ่มลง
    DOWN = 1      # กำลังลง
    BOTTOM = 2    # ถึงจุดต่ำสุด
    UP = 3        # กำลังขึ้น

class RepCounterStream:
    """
    ตัวนับ rep แบบออนไลน์
    - feed(angle_value) ทีละเฟรม
    - นับเมื่อ state วนครบ DOWN -> BOTTOM -> UP -> IDLE (กลับสู่บน)
    """
    def __init__(self, cfg: RepCounterConfig | None = None):
        self.cfg = cfg or RepCounterConfig()
        self.state = RepState.IDLE
        self.rep = 0
        self.frames_since_state = 0
        self.idle_frames = 0

        # Adaptive thresholds จาก EMA span
        self.min_val = np.inf
        self.max_val = -np.inf
        self.span_ema = 30.0  # ค่าเริ่มต้น (องศา)
        self.last_x = None

    def _update_span(self, x: float):
        # อัปเดต min/max และ EMA ของ span
        if x < self.min_val: self.min_val = x
        if x > self.max_val: self.max_val = x
        span = max(self.max_val - self.min_val, 1e-3)
        self.span_ema = (1 - self.cfg.ema_alpha) * self.span_ema + self.cfg.ema_alpha * span

    def _thresholds(self) -> Tuple[float, float, float, float]:
        """
        คำนวณ threshold แบบไดนามิกตาม span_ema ปัจจุบัน
        """
        # ค่ามุม
        high = self.max_val - 0.15 * self.span_ema
        low  = self.min_val + 0.15 * self.span_ema
        # ความเร็ว (อนุพันธ์ประมาณจาก last_x)
        th_vel = 0.10 * self.span_ema
        return high, low, -th_vel, +th_vel

    def feed(self, x: float) -> int:
        """
        ป้อนค่ามุม (เฟรมละค่า) คืนค่า rep ปัจจุบัน (สะสม)
        """
        if self.last_x is None:
            self.last_x = x
            self._update_span(x)
            return self.rep

        dx = x - self.last_x
        self.last_x = x
        self._update_span(x)
        high, low, th_down, th_up = self._thresholds()

        self.frames_since_state += 1
        self.idle_frames = 0 if abs(dx) > 1e-6 else self.idle_frames + 1
        if self.idle_frames > self.cfg.idle_frames_reset and self.state != RepState.IDLE:
            # นิ่งนานเกิน รีเซ็ต
            self.state = RepState.IDLE
            self.frames_since_state = 0

        # fsm
        if self.state == RepState.IDLE:
            # เริ่มลงเมื่อมีทิศลงชัด + ค่าเริ่มตกจาก high
            if dx <= th_down * 0.5 and x < high:
                self.state = RepState.DOWN
                self.frames_since_state = 0

        elif self.state == RepState.DOWN:
            # ถึง bottom เมื่อค่าต่ำและความเร็วเริ่มช้าลง/เปลี่ยนทิศ
            if x <= low and dx >= 0:
                self.state = RepState.BOTTOM
                self.frames_since_state = 0

        elif self.state == RepState.BOTTOM:
            # ขึ้นเมื่อทิศทางเป็นบวกและออกจากโซนต่ำ
            if dx >= th_up * 0.5 and x > low:
                self.state = RepState.UP
                self.frames_since_state = 0

        elif self.state == RepState.UP:
            # จบ rep เมื่อกลับสู่บริเวณบนและความเร็วต่ำลง
            if x >= high and abs(dx) < abs(th_up) * 0.5:
                if self.frames_since_state >= self.cfg.min_frames_per_rep:
                    self.rep += 1
                self.state = RepState.IDLE
                self.frames_since_state = 0

        return self.rep

# -----------------------------
# Batch wrapper
# -----------------------------
def detect_reps(angle_series: np.ndarray,
                min_frames_per_rep: int = 8) -> int:
    """
    นับจำนวน reps จากสัญญาณมุม (batch)
    - ใช้ FSM เดียวกับแบบสตรีม วิ่งผ่านทั้งสัญญาณ
    """
    x = _nan_to_num(np.asarray(angle_series, dtype=np.float32))
    counter = RepCounterStream(RepCounterConfig(min_frames_per_rep=min_frames_per_rep))
    rep = 0
    for v in x:
        rep = counter.feed(float(v))
    return rep

# -----------------------------
# Demo (manual test)
# -----------------------------
if __name__ == "__main__":
    # สร้างสัญญาณจำลอง: 3 rep ของสควอต (มุมเข่าลดลงแล้วเพิ่มขึ้น)
    T = 300
    t = np.linspace(0, 6*np.pi, T)
    # สร้างคลื่นรูปไซน์กลับหัว (สูง=ยืน, ต่ำ=ลึก) + noise เล็กน้อย
    knee = 150 - 40 * np.sin(t) + np.random.normal(0, 1.0, size=T)

    # ติดป้ายเฟส (batch)
    phases = label_phases({"knee": knee})
    print("phase counts:", {p: phases.count(p) for p in set(phases)})

    # นับ rep (batch)
    total = detect_reps(knee, min_frames_per_rep=10)
    print("rep_total:", total)

    # นับแบบสตรีม
    counter = RepCounterStream(RepCounterConfig(min_frames_per_rep=10))
    for v in knee:
        cur = counter.feed(float(v))
    print("rep_total(stream):", counter.rep)
