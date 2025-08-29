# app/scoring.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional

from app.matching import dtw_align, warp_to_ref

"""
Scoring utilities
- score_angles: ให้คะแนนจาก "มุมข้อ" ต่อมุม × ต่อเฟส ด้วยสถิติ μ,σ ของ reference (ไม่ต้องตั้งค่าเอง)
- score_shape : ให้คะแนนความคล้ายรูปร่าง (bone vectors) แบบทั้งตัว
- aggregate_scores: รวมคะแนนมุม + รูปร่าง เป็นสเกล 0–100

แนวคิด:
1) จัดเวลาอัตโนมัติ (DTW) โดยใช้สัญญาณนำ 'knee' เพื่อ align user ↔ reference
2) สำหรับแต่ละ angle และแต่ละ phase:
   - คำนวณ z-score = |user - μ| / σ  (μ,σ มาจาก reference library)
   - แปลงเป็นคะแนนนุ่ม ๆ: score = 100 * exp(-0.5 * z^2)  (เฉลี่ยภายในเฟส)
3) รวมคะแนนทุกเฟสด้วย phase weights และรวมทุกมุมด้วย angle weights
"""

# -----------------------------
# ค่าเริ่มต้น
# -----------------------------
DEFAULT_PHASE_WEIGHTS: Dict[str, float] = {
    "down": 1.2,
    "bottom": 1.4,
    "up": 1.0,
    "start": 0.6,
}

# มุมที่ใช้สรุปเป็น "มุมรวม" (บางมุมต้องรวมซ้าย-ขวา)
CANONICAL_ANGLES: Dict[str, List[str]] = {
    "knee":  ["knee"],                # มีสัญญาณ 'knee' อยู่แล้ว (min(L,R))
    "hip":   ["hip_L", "hip_R"],
    "ankle": ["ankle_L", "ankle_R"],
    "trunk": ["trunk"],
}

# -----------------------------
# Utilities
# -----------------------------
def _to_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.squeeze()

def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(max(v, 0.0) for v in w.values()))
    if s <= 1e-8:
        # ถ้าไม่มีอะไรเลย ให้กระจายเท่ากัน
        n = len(w) if len(w) > 0 else 1
        return {k: 1.0 / n for k in w}
    return {k: max(v, 0.0) / s for k, v in w.items()}

def _phase_list() -> List[str]:
    return ["start", "down", "bottom", "up"]

def _ensure_same_len(user: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    ทำให้ความยาวเท่ากัน (ถ้าไม่เท่า: ตัดให้ยาวเท่าขั้นต่ำ)
    (หมายเหตุ: ปกติเราควร warp ด้วย DTW อยู่แล้ว)
    """
    L = min(len(user), len(ref))
    return user[:L], ref[:L]

# -----------------------------
# 1) Angle-based scoring
# -----------------------------
def _build_user_ref_aligned(
    angles_user: Dict[str, np.ndarray],
    angles_ref: Dict[str, np.ndarray],
    lead_key: str = "knee",
    dtw_radius: int = 8,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    สร้าง dict ของ (user_aligned, ref) ต่อ key ของ angle
    - ใช้ DTW กับสัญญาณนำ lead_key เพื่อหา path
    - warp มุมอื่น ๆ ของ user ให้ยาวเท่า reference ตาม path เดียวกัน
    """
    if lead_key not in angles_user or lead_key not in angles_ref:
        raise ValueError(f"ไม่พบสัญญาณนำ '{lead_key}' ใน angles")

    # DTW บนสัญญาณนำ
    _, path = dtw_align(angles_user[lead_key], angles_ref[lead_key], radius=dtw_radius, metric="l2")

    aligned: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for k, ref_series in angles_ref.items():
        u_series = angles_user.get(k, None)
        if u_series is None:
            continue
        u_warp = warp_to_ref(u_series, path, target_len=len(ref_series))
        u_warp = _to_1d(u_warp)
        r_fix  = _to_1d(ref_series)
        u_warp, r_fix = _ensure_same_len(u_warp, r_fix)
        aligned[k] = (u_warp, r_fix)

    return aligned

def _angle_group_reduce(values: Dict[str, float]) -> float:
    """
    รวมค่า (เช่น คะแนนของซ้าย-ขวา) เป็นหนึ่งค่า (ใช้ค่าเฉลี่ย)
    """
    if not values:
        return 0.0
    return float(np.mean(list(values.values())))

def score_angles(
    angles_user: Dict[str, np.ndarray],
    angles_ref:  Dict[str, np.ndarray],
    angle_weights: Dict[str, float],
    phase_masks: Dict[str, np.ndarray],
    stats: Dict[str, Dict],           # {"angles": {angle_name: {phase: {"mu","sigma"}}}}
    phase_weights: Optional[Dict[str, float]] = None,
    lead_key: str = "knee",
    dtw_radius: int = 8,
) -> Tuple[Dict, Dict]:
    """
    คำนวณคะแนนจาก "มุมข้อ" แบบ per-angle × per-phase (0–100)
    Returns:
        (angle_score_dict, per_angle_phase_matrix)
        - angle_score_dict:
            {
              "overall": float,
              "per_angle": {"knee": .., "hip": .., "ankle": .., "trunk": ..},
              "per_phase": {"start": .., "down": .., "bottom": .., "up": ..}
            }
        - per_angle_phase_matrix:
            { angle_key: { phase: {"score": float, "z_mean": float} } }
    """
    phase_weights = phase_weights or DEFAULT_PHASE_WEIGHTS
    phase_w = _normalize_weights(phase_weights)

    # 1) Align user ↔ reference (ใช้ knee เป็นสัญญาณนำ)
    aligned = _build_user_ref_aligned(angles_user, angles_ref, lead_key=lead_key, dtw_radius=dtw_radius)

    # 2) คำนวณคะแนน per-angle × per-phase
    per_angle_phase: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_phase_agg: Dict[str, List[float]] = {p: [] for p in _phase_list()}

    # เลือก canonical angles (รวมซ้าย-ขวาบางมุม)
    canonical_scores: Dict[str, float] = {}

    for canon_key, members in CANONICAL_ANGLES.items():
        # สมาชิก เช่น ["hip_L","hip_R"] หรือ ["knee"]
        member_scores: Dict[str, float] = {}
        per_angle_phase[canon_key] = {}

        for m in members:
            if m not in aligned:
                continue
            u, r = aligned[m]
            # ถ้าไม่มีสถิติสำหรับมุมนี้ ให้ข้าม
            if "angles" not in stats or m not in stats["angles"]:
                continue

            # คำนวณ per-phase
            per_phase_scores: Dict[str, float] = {}
            for phase in _phase_list():
                mask = phase_masks.get(phase, None)
                if mask is None or not np.any(mask):
                    # ไม่มีเฟสนี้ใน template → ข้าม
                    continue
                # ensure mask length fits
                L = min(len(u), len(r), len(mask))
                if L == 0:
                    continue
                mask_L = mask[:L]
                if not np.any(mask_L):
                    continue

                mu = float(stats["angles"][m].get(phase, {}).get("mu", 0.0))
                sigma = float(stats["angles"][m].get(phase, {}).get("sigma", 1.0))
                sigma = max(sigma, 1e-3)

                z = np.abs(u[:L][mask_L] - mu) / sigma
                # คะแนนนุ่ม ๆ (Gaussian)
                s = 100.0 * float(np.exp(-0.5 * (z**2)).mean())
                per_phase_scores[phase] = s

                # เก็บลงเมทริกซ์ (สำหรับ feedback)
                per_angle_phase.setdefault(canon_key, {})
                per_angle_phase[canon_key].setdefault(phase, {})
                per_angle_phase[canon_key][phase] = {
                    "score": s,
                    "z_mean": float(z.mean()),
                }

                # สำหรับสรุป per-phase รวมทุกมุม
                per_phase_agg[phase].append(s)

            # รวมคะแนนของสมาชิกมุมเดียวกัน (เช่น L/R) ด้วยน้ำหนักเฟส
            if per_phase_scores:
                wsum = 0.0
                ssum = 0.0
                for p, w in phase_w.items():
                    if p in per_phase_scores:
                        ssum += per_phase_scores[p] * w
                        wsum += w
                member_scores[m] = ssum / max(wsum, 1e-8)

        # รวมสมาชิกของ canon_key (เช่น hip_L+hip_R → hip)
        canonical_scores[canon_key] = _angle_group_reduce(member_scores)

    # 3) รวมคะแนน per-phase (รวมทุกมุม) เผื่อใช้แสดงผล
    per_phase_final: Dict[str, float] = {}
    for p, arr in per_phase_agg.items():
        per_phase_final[p] = float(np.mean(arr)) if len(arr) > 0 else 0.0

    # 4) รวมคะแนนทุกมุมด้วย angle weights
    angle_w = _normalize_weights(angle_weights)
    overall = 0.0
    for k, w in angle_w.items():
        # ใช้เฉพาะคีย์ที่มีใน canonical_scores
        if k in canonical_scores:
            overall += canonical_scores[k] * w

    angle_score_dict = {
        "overall": float(overall),
        "per_angle": {k: float(v) for k, v in canonical_scores.items()},
        "per_phase": per_phase_final,
    }
    return angle_score_dict, per_angle_phase

# -----------------------------
# 2) Shape-based scoring
# -----------------------------
def _cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    cosine similarity ต่อเฟรม (คืนค่าระหว่าง -1..1)
    a,b: shape (T,D)
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    # ปรับขนาดให้เท่ากันแบบพื้นฐาน
    L = min(len(a), len(b))
    a = a[:L]; b = b[:L]
    # นอร์มต่อเฟรม
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    na = np.maximum(na, eps); nb = np.maximum(nb, eps)
    sim = np.sum(a * b, axis=1, keepdims=True) / (na * nb)
    return np.clip(sim.squeeze(), -1.0, 1.0)

def score_shape(
    user_vecs: np.ndarray,  # (Tu, D)
    ref_vecs:  np.ndarray,  # (Tr, D)
    dtw_radius: int = 8
) -> float:
    """
    ให้คะแนนความคล้ายเชิงรูปร่าง (0–100)
    - จัดเวลาอัตโนมัติด้วย DTW บนเวกเตอร์กระดูกหลายมิติ
    - คำนวณ cosine similarity ต่อเฟรมหลัง warp แล้ว เฉลี่ย → สเกล 0–100
    """
    if user_vecs.ndim != 2 or ref_vecs.ndim != 2:
        raise ValueError("user_vecs และ ref_vecs ต้องเป็นอาเรย์ 2 มิติ (T,D)")

    # DTW บนฟีเจอร์หลายมิติ
    _, path = dtw_align(user_vecs, ref_vecs, radius=dtw_radius, metric="l2")
    user_warp = warp_to_ref(user_vecs, path, target_len=len(ref_vecs))

    # cosine similarity ต่อเฟรม แล้วเฉลี่ย
    sim = _cosine_sim(user_warp, ref_vecs)  # -1..1
    score = float(((sim + 1.0) * 0.5).mean() * 100.0)  # map เป็น 0..100
    # ปรับขอบเขต (กันค่าล้น)
    score = float(np.clip(score, 0.0, 100.0))
    return score

# -----------------------------
# 3) Aggregate
# -----------------------------
def aggregate_scores(
    angle_score: Dict,
    shape_score: float,
    alpha: float = 0.7
) -> Dict[str, float | Dict]:
    """
    รวมคะแนน angle vs shape เป็นคะแนนรวม
    overall = alpha * angle_overall + (1-alpha) * shape_score
    """
    angle_overall = float(angle_score.get("overall", 0.0))
    overall = float(alpha * angle_overall + (1.0 - alpha) * float(shape_score))
    overall = float(np.clip(overall, 0.0, 100.0))
    return {
        "overall": overall,
        "angles": angle_score,
        "shape": float(shape_score),
        "alpha": float(alpha),
    }

# -----------------------------
# Quick self-test
# -----------------------------
if __name__ == "__main__":
    # สร้างสัญญาณทดสอบสำหรับมุม 'knee'
    t_ref = np.linspace(0, 2*np.pi, 120)
    t_usr = np.linspace(0, 2*np.pi, 150)
    knee_ref = 150 - 40*np.sin(t_ref)
    knee_usr = 150 - 40*np.sin(t_usr + 0.15) + np.random.normal(0, 1.0, size=len(t_usr))

    angles_ref = {"knee": knee_ref, "trunk": 10 + 5*np.sin(t_ref)}
    angles_usr = {"knee": knee_usr, "trunk": 12 + 6*np.sin(t_usr + 0.1)}

    # ทำสถิติ μ,σ แบบหยาบ (สมมุติ: ใช้ค่าเฉลี่ย/ส่วนเบี่ยงเบนของ reference)
    stats = {"angles": {}}
    for k, s in angles_ref.items():
        stats["angles"][k] = {}
        for p in _phase_list():
            stats["angles"][k][p] = {"mu": float(np.mean(s)), "sigma": float(max(np.std(s), 1e-3))}

    # phase_masks สมมุติใช้ทั้งสัญญาณ
    phase_masks = {p: np.ones_like(knee_ref, dtype=bool) for p in _phase_list()}
    angle_weights = {"knee": 0.6, "trunk": 0.4}

    a_score, matrix = score_angles(
        angles_usr, angles_ref, angle_weights, phase_masks, stats,
        phase_weights=DEFAULT_PHASE_WEIGHTS, lead_key="knee", dtw_radius=8
    )
    print("angle score:", a_score["overall"])

    # shape score (ฟีเจอร์สุ่ม)
    D = 8
    vec_ref = np.stack([np.sin(t_ref + i*0.1) for i in range(D)], axis=1).astype(np.float32)
    vec_usr = np.stack([np.sin(t_usr + i*0.1 + 0.05) for i in range(D)], axis=1).astype(np.float32)
    s_score = score_shape(vec_usr, vec_ref)
    print("shape score:", s_score)

    final = aggregate_scores(a_score, s_score, alpha=0.7)
    print("overall:", final["overall"])
