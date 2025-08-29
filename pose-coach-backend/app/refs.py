# app/refs.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from functools import lru_cache

from app.config import DATA_DIR
from app.preprocessing import preprocess_pipeline
from app.features import compute_angles, pairwise_bone_vectors
from app.segmentation import label_phases

# -------------------------------
# ชื่อไฟล์/โฟลเดอร์อ้างอิง
# -------------------------------
def _exercise_dir(exercise: str) -> Path:
    d = (Path(DATA_DIR) / exercise).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d

# export helper ให้ api.py ใช้
def ensure_exercise_dir(exercise: str) -> Path:
    return _exercise_dir(exercise)

def _ref_path(exercise: str, name: str) -> Path:
    return _exercise_dir(exercise) / name

def _stats_path(exercise: str) -> Path:
    return _exercise_dir(exercise) / f"stats_{exercise}.json"

# -------------------------------
# อัปโหลด/บันทึก reference
# โครงสร้างไฟล์ JSON ที่รองรับอย่างน้อย 2 แบบ:
# A) Compact
# {
#   "fps": 30,
#   "frames": [  # T เฟรม
#     [[x,y,conf], ... 17 จุด ...],
#     ...
#   ],
#   "meta": {...}
# }
# B) Verbose
# {
#   "fps": 30,
#   "frames": [
#     {"points": [[x,y,conf], ... 17 จุด ...]},
#     ...
#   ],
#   "meta": {...}
# }
# -------------------------------
def _is_compact_frame(f: Any) -> bool:
    return isinstance(f, list) and len(f) == 17 and isinstance(f[0], list) and len(f[0]) == 3

def _is_verbose_frame(f: Any) -> bool:
    return isinstance(f, dict) and "points" in f and isinstance(f["points"], list) and len(f["points"]) == 17

def _validate_ref_json(data: Dict[str, Any]) -> None:
    if "frames" not in data or not isinstance(data["frames"], list) or len(data["frames"]) == 0:
        raise ValueError("reference JSON ต้องมี key 'frames' เป็น list และต้องไม่ว่าง")
    f0 = data["frames"][0]
    if not (_is_compact_frame(f0) or _is_verbose_frame(f0)):
        raise ValueError("รูปแบบ frames ต้องเป็น [[x,y,conf]*17] หรือ [{points:[[x,y,conf]*17]}]")

def _normalize_frames_list(frames: List[Any]) -> List[np.ndarray]:
    """
    รับ list ของ frame (compact หรือ verbose) → คืน list[np.ndarray(17,3)] แบบมาตรฐาน
    """
    out: List[np.ndarray] = []
    for f in frames:
        points = f if _is_compact_frame(f) else f["points"] if _is_verbose_frame(f) else None
        if points is None:
            # ข้ามเฟรมเสีย (กันล้ม)
            continue
        arr = np.asarray(points, dtype=np.float32)
        if arr.shape != (17, 3):
            # รูปทรงเพี้ยน ข้ามไป
            continue
        out.append(arr)
    if not out:
        raise ValueError("ไม่พบเฟรมที่ถูกต้องรูปแบบเลย")
    return out

def save_reference(exercise: str, data: Dict[str, Any], name: str | None = None) -> Path:
    """
    เซฟไฟล์ reference (.json) ลงใน references/<exercise>/ref_*.json
    คืน Path ของไฟล์ที่บันทึก
    """
    _validate_ref_json(data)
    ex_dir = _exercise_dir(exercise)
    if name is None:
        # ตั้งชื่ออัตโนมัติตามจำนวนไฟล์ที่มีอยู่
        existing = sorted(ex_dir.glob("ref_*.json"))
        idx = len(existing) + 1
        name = f"ref_{idx:03d}.json"
    out_path = ex_dir / name
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False)
    # เคลียร์ cache โหลดอ้างอิง เพื่อให้ไฟล์ใหม่ถูกมองเห็น
    load_reference.cache_clear()  # type: ignore
    return out_path

# -------------------------------
# การสร้างสถิติ μ, σ ต่อ “มุม × เฟส”
# -------------------------------
def _phase_masks_from_labels(phases: List[str]) -> Dict[str, np.ndarray]:
    """
    รับลิสต์เฟสต่อเฟรม → คืน mask ต่อเฟสเป็น boolean array
    เฟสมาตรฐาน: start, down, bottom, up (ถ้าบางเฟสไม่มี ก็ได้ mask ว่าง)
    """
    T = len(phases)
    uniq = ["start", "down", "bottom", "up"]
    masks: Dict[str, np.ndarray] = {}
    ph_array = np.array(phases)
    for p in uniq:
        masks[p] = (ph_array == p)
    # สำรอง all-true กรณีไม่มี label ใด ๆ (ใช้ทั้งสัญญาณ)
    if not any(masks[p].any() for p in uniq):
        masks = {p: np.ones(T, dtype=bool) for p in uniq}
    return masks

def _collect_ref_series(ref_angles_list: List[Dict[str, np.ndarray]]) -> Dict[str, List[np.ndarray]]:
    """
    รวม series มุมจากหลายไฟล์ reference ตามชื่อมุม
    คืน dict angle_name -> list of np.ndarray (ยาว T อาจต่างกัน)
    """
    bucket: Dict[str, List[np.ndarray]] = {}
    for angs in ref_angles_list:
        for k, v in angs.items():
            bucket.setdefault(k, []).append(v.astype(np.float32))
    return bucket

def _phase_stats_for_angle(series_list: List[np.ndarray], phase_labels_list: List[List[str]]) -> Dict[str, Dict[str, float]]:
    """
    สร้าง μ, σ สำหรับ angle หนึ่ง โดยดู per-phase ข้ามหลายไฟล์
    คืน: dict[phase] -> {"mu": float, "sigma": float}
    หมายเหตุ: รวมค่าจากเฟรมที่มี label ตรง phase นั้น ๆ ของแต่ละไฟล์
    """
    phases = ["start", "down", "bottom", "up"]
    stats: Dict[str, Dict[str, float]] = {p: {"mu": 0.0, "sigma": 1.0} for p in phases}
    for p in phases:
        values: List[float] = []
        for series, labels in zip(series_list, phase_labels_list):
            mask = (np.array(labels) == p)
            if mask.any():
                values.extend(series[mask].tolist())
        if len(values) >= 5:  # มีข้อมูลพอประมาณ
            arr = np.asarray(values, dtype=np.float32)
            stats[p]["mu"] = float(arr.mean())
            stats[p]["sigma"] = float(max(arr.std(ddof=1), 1e-3))
        else:
            # ข้อมูลน้อย ตั้งค่า fallback
            stats[p]["mu"] = 0.0
            stats[p]["sigma"] = 1.0
    return stats

def build_stats_from_refs(ref_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    รับรายการอ็อบเจกต์อ้างอิง (ที่คำนวณ angles, phases แล้ว)
    → สร้างสถิติ μ,σ ต่อ angle×phase
    Return JSON-serializable dict
    """
    # รวม series มุมและ labels จากทุกไฟล์
    angles_list = [r["angles"] for r in ref_objects]
    labels_list = [r["phases"] for r in ref_objects]

    bucket = _collect_ref_series(angles_list)  # angle_name -> [series...]
    angle_names = list(bucket.keys())

    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for ang in angle_names:
        series_list = bucket[ang]
        # จับคู่ labels ให้ยาวเท่ากับ series (truncate/pad ถ้าจำเป็น)
        fixed_labels_list: List[List[str]] = []
        for r, series in zip(ref_objects, series_list):
            labels = r["phases"]
            if len(labels) != len(series):
                # ทำให้ยาวเท่ากันด้วยการตัดให้สั้นสุด
                m = min(len(labels), len(series))
                labels = labels[:m]
                series = series[:m]
            fixed_labels_list.append(labels)
        stats[ang] = _phase_stats_for_angle(series_list, fixed_labels_list)

    return {"angles": stats}

# -------------------------------
# Mapping สำหรับ feedback แบบอธิบายได้ (ไม่ if-else แข็ง)
# ผูก angle+phase → group/label ข้อความ ที่ feedback.py จะใช้
# -------------------------------
def _default_description_map(exercise: str) -> Dict[str, Dict[str, str]]:
    """
    คืน mapping อย่างง่าย angle→phase→tag ที่ไปผูกกับข้อความโค้ชชิ่งใน feedback.py
    สามารถปรับเฉพาะท่าได้
    """
    if exercise.lower() == "squat":
        return {
            "knee":   {"down": "knee_forward", "bottom": "knee_depth"},
            "knee_L": {"bottom":"knee_valgus_L"}, "knee_R":{"bottom":"knee_valgus_R"},
            "trunk":  {"down":"trunk_lean", "bottom":"trunk_lean"},
            "ankle_L":{"down":"ankle_ctrl_L"}, "ankle_R":{"down":"ankle_ctrl_R"},
            "hip_L": {"bottom":"hip_depth"}, "hip_R":{"bottom":"hip_depth"},
        }
    # ค่าเริ่มต้นทั่วไป
    return {
        "knee": {"down":"knee_ctrl","bottom":"knee_ctrl"},
        "trunk":{"down":"trunk_ctrl","bottom":"trunk_ctrl"},
    }

def _default_angle_weights(exercise: str) -> Dict[str, float]:
    if exercise.lower() == "squat":
        return {"knee":0.35, "hip":0.35, "ankle":0.15, "trunk":0.15}
    return {"knee":0.4, "hip":0.3, "ankle":0.15, "trunk":0.15}

# -------------------------------
# โหลด reference ทั้งชุด (พร้อม preprocess+features)
# แล้วคำนวณ stats หากยังไม่มี หรือใช้ที่มีอยู่แล้ว
# -------------------------------
def _load_raw_ref_files(exercise: str) -> List[Dict[str, Any]]:
    files = sorted(_exercise_dir(exercise).glob("ref_*.json"))
    data_list: List[Dict[str, Any]] = []
    for p in files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            _validate_ref_json(obj)
            data_list.append(obj)
        except Exception:
            # ข้ามไฟล์ที่เสีย
            continue
    return data_list

def _prepare_ref_object(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    จาก raw JSON → preprocess → angles, vecs, phases
    คืน dict ที่พร้อมใช้สร้าง stats/เทียบ DTW
    """
    # รองรับทั้ง compact และ verbose
    frames_std = _normalize_frames_list(raw["frames"])
    frames_np = preprocess_pipeline(frames_std)
    angles = compute_angles(frames_np)
    vecs   = pairwise_bone_vectors(frames_np)
    phases = label_phases(angles)
    return {
        "frames": frames_np,
        "angles": angles,
        "vecs": vecs,
        "phases": phases,
        "meta": raw.get("meta", {}),
        "fps": raw.get("fps", None),
    }

def _save_stats_json(exercise: str, stats: Dict[str, Any]) -> None:
    p = _stats_path(exercise)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, ensure_ascii=False)

def _load_stats_json(exercise: str) -> Dict[str, Any] | None:
    p = _stats_path(exercise)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

@lru_cache(maxsize=16)
def load_reference(exercise: str) -> Dict[str, Any] | None:
    """
    โหลด reference ของท่าที่ระบุ:
    - อ่าน ref_*.json ทั้งหมด → preprocess → angles/vecs/phases
    - โหลดหรือสร้าง stats (μ,σ ต่อ angle×phase)
    - คืนอ็อบเจกต์พร้อมใช้: angles_ref (แม่แบบนำ), vecs_ref, stats, masks, weights, description_map
    หมายเหตุ: ใช้ไฟล์แรกเป็น "แม่แบบนำ" สำหรับ DTW เริ่มต้น (สามารถขยายเป็นเลือกอัตโนมัติได้)
    """
    raw_list = _load_raw_ref_files(exercise)
    if not raw_list:
        return None

    # เตรียมวัตถุอ้างอิงทั้งหมด
    ref_objs = [_prepare_ref_object(r) for r in raw_list]

    # ใช้ไฟล์แรกเป็น template นำทาง
    lead = ref_objs[0]
    angles_ref = lead["angles"]
    vecs_ref   = lead["vecs"]
    phases_ref = lead["phases"]

    # โหลดหรือคำนวณ stats
    stats_json = _load_stats_json(exercise)
    if stats_json is None:
        stats_json = build_stats_from_refs(ref_objs)
        _save_stats_json(exercise, stats_json)

    # phase masks สำหรับ template นำ
    phase_masks = _phase_masks_from_labels(phases_ref)

    # weights และ mapping คำอธิบาย
    angle_weights = _default_angle_weights(exercise)
    description_map = _default_description_map(exercise)

    return {
        "angles": angles_ref,           # dict[str] -> np.ndarray(T,)
        "vecs": vecs_ref,               # np.ndarray(T, 2*BONES)
        "phase_masks": phase_masks,     # dict[phase] -> mask (T,)
        "stats": stats_json,            # {"angles": {angle: {phase:{mu,sigma}}}}
        "weights": angle_weights,       # angle weights สำหรับ aggregate
        "description_map": description_map,
        "fps": lead.get("fps", None),
        "meta": lead.get("meta", {}),
        "num_refs": len(ref_objs),
    }
