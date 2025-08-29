# app/feedback.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np

"""
Feedback utilities
- รับ per_angle_phase (ผลจาก scoring.score_angles) และ description_map (จาก refs.load_reference)
- เลือก "จุดที่เพี้ยนที่สุด" แบบอัตโนมัติ (ไม่ต้องตั้ง threshold แข็ง ๆ)
- สร้างข้อความโค้ชชิ่งสั้น ๆ เข้าใจง่าย (ภาษาไทยค่าเริ่มต้น, มีอังกฤษด้วย)

อินพุตหลัก:
  per_angle_phase: {
    "<angle_key>": {
      "<phase>": {"score": float(0..100), "z_mean": float}
    }, ...
  }

  description_map: กำหนด tag ที่ผูก angle+phase → หมวดคำอธิบาย
    ตัวอย่าง (refs.py มี default ให้):
      {
        "knee": {"down": "knee_forward", "bottom": "knee_depth"},
        "knee_L": {"bottom":"knee_valgus_L"},
        "knee_R": {"bottom":"knee_valgus_R"},
        "trunk": {"down":"trunk_lean", "bottom":"trunk_lean"},
        ...
      }

หลักการ:
- จัดอันดับ deviation โดยอิง "คะแนนต่ำ" และ "z-mean สูง"
- รวมเป็น severity score = w1*(100-score) + w2*(z_mean_std)
- เลือก top-k และแมปเป็นข้อความผ่านตาราง template (ไม่ if-else แข็ง)

ฟังก์ชันหลัก:
- explain_from_deviation(...) → คืน list[str] ของ coaching 1–3 ข้อ
- summarize_deviation(...)   → คืนข้อมูลเชิงโครงสร้างสำหรับแสดง UI/ดีบั๊ก
"""

# ---------------------------------------------------------
# กำหนด template ภาษาไทย/อังกฤษสำหรับ tag → ข้อความโค้ชชิ่ง
# ---------------------------------------------------------
_TEMPLATES_TH: Dict[str, str] = {
    "knee_forward": "ช่วงลง พยายามอย่าดันเข่าเลยปลายเท้า — ดันสะโพกไปหลังและกดส้นเท้า",
    "knee_depth":   "ช่วงลึกสุด รักษามุมเข่าในช่วงที่ควบคุมได้ — อย่าทิ้งน้ำหนักไปหน้า",
    "knee_valgus_L":"เข่าซ้ายบีบเข้าด้านในที่ก้นต่ำสุด — ดันเข่าออกให้ขนานปลายเท้า",
    "knee_valgus_R":"เข่าขวาบีบเข้าด้านในที่ก้นต่ำสุด — ดันเข่าออกให้ขนานปลายเท้า",
    "trunk_lean":   "ลำตัวเอนไปหน้ามากในช่วงลง — เก็บอก เปิดหน้าอก และรักษาลำตัวให้มั่นคง",
    "ankle_ctrl_L": "ข้อเท้าซ้ายควบคุมยังไม่คงที่ — กดส้นเท้าให้มั่นและกระจายน้ำหนักเท้า",
    "ankle_ctrl_R": "ข้อเท้าขวาควบคุมยังไม่คงที่ — กดส้นเท้าให้มั่นและกระจายน้ำหนักเท้า",
    "hip_depth":    "ความลึกของสะโพกยังไม่สม่ำเสมอ — คุมจังหวะลงให้ลึกพอดีและขึ้นอย่างคงที่",
    # fallback ทั่วไป
    "knee_ctrl":    "ควบคุมเข่าให้ไปในแนวเดียวกับปลายเท้า และอย่าดันไปข้างหน้ามากไป",
    "trunk_ctrl":   "รักษาแกนลำตัวให้มั่นคง เก็บท้องและเปิดอกตลอดการเคลื่อนไหว",
}

_TEMPLATES_EN: Dict[str, str] = {
    "knee_forward": "On the way down, avoid pushing knees past toes—send hips back and keep heels grounded.",
    "knee_depth":   "At the bottom, maintain a controllable knee angle—avoid shifting weight forward.",
    "knee_valgus_L":"Left knee caves inward at the bottom—push it out to align with toes.",
    "knee_valgus_R":"Right knee caves inward at the bottom—push it out to align with toes.",
    "trunk_lean":   "Excessive trunk lean on descent—brace, lift the chest, and keep the torso stable.",
    "ankle_ctrl_L": "Left ankle control is unstable—drive through the heel and spread foot pressure.",
    "ankle_ctrl_R": "Right ankle control is unstable—drive through the heel and spread foot pressure.",
    "hip_depth":    "Hip depth consistency needs work—control the descent and rise smoothly.",
    "knee_ctrl":    "Keep knees tracking over toes; avoid pushing too far forward.",
    "trunk_ctrl":   "Maintain a strong trunk—brace and keep the chest open throughout.",
}

def _get_template(tag: str, language: str = "th") -> str:
    if language.lower().startswith("en"):
        return _TEMPLATES_EN.get(tag, "Form needs adjustment for this phase—control tempo and alignment.")
    return _TEMPLATES_TH.get(tag, "ปรับฟอร์มในช่วงนี้ให้คุมจังหวะและแนวระนาบให้ดีขึ้น")

# ---------------------------------------------------------
# จัดอันดับ deviation
# ---------------------------------------------------------
def _rank_deviation(
    per_angle_phase: Dict[str, Dict[str, Dict[str, float]]],
    score_weight: float = 0.7,
    z_weight: float = 0.3,
) -> List[Tuple[str, str, float, float, float]]:
    """
    จัดอันดับความเบี่ยงเบน:
      severity = score_weight*(100 - score) + z_weight*(z_norm)
    คืนลิสต์ของ (angle_key, phase, score, z_mean, severity) เรียงจากหนัก→เบา
    """
    items: List[Tuple[str, str, float, float, float]] = []
    # รวม z_mean ให้ normalize ด้วยสถิติภายในเอง
    z_vals: List[float] = []
    for a, phases in per_angle_phase.items():
        for p, obj in phases.items():
            z_vals.append(float(obj.get("z_mean", 0.0)))
    z_vals = np.asarray(z_vals, dtype=np.float32) if len(z_vals) else np.array([0.0], dtype=np.float32)
    z_mu, z_sigma = float(np.mean(z_vals)), float(max(np.std(z_vals), 1e-6))

    for a, phases in per_angle_phase.items():
        for p, obj in phases.items():
            s = float(obj.get("score", 0.0))
            z = float(obj.get("z_mean", 0.0))
            z_norm = abs((z - z_mu) / z_sigma)  # ทำให้เทียบกันได้
            sev = score_weight * (100.0 - s) + z_weight * z_norm * 10.0  # คูณ 10 ให้มีอิทธิพลพอประมาณ
            items.append((a, p, s, z, sev))

    items.sort(key=lambda x: x[4], reverse=True)
    return items

# ---------------------------------------------------------
# รวมประเด็นที่ซ้ำซ้อนซ้าย/ขวา
# ---------------------------------------------------------
def _merge_side_duplicates(ranked: List[Tuple[str, str, float, float, float]]) -> List[Tuple[str, str, float, float, float]]:
    """
    ถ้ามีทั้ง _L และ _R อยู่บนเฟสเดียวกันและ tag เดียวกัน ให้เก็บอัน severity สูงสุดอันเดียว
    (ลดการแสดงผลซ้ำซ้อน)
    """
    out: List[Tuple[str, str, float, float, float]] = []
    seen_keys = set()
    for a, p, s, z, sev in ranked:
        base = a.replace("_L", "").replace("_R", "")
        key = (base, p)
        if key in seen_keys:
            continue
        # ถ้าพบฝั่งตรงข้าม ให้เลือกอันที่ severity สูงกว่า
        opp = None
        if a.endswith("_L"):
            opp = a.replace("_L", "_R")
        elif a.endswith("_R"):
            opp = a.replace("_R", "_L")
        if opp:
            # หาใน ranked ว่ามี opp+p ไหม
            cand = [(aa, pp, ss, zz, sv) for (aa, pp, ss, zz, sv) in ranked if aa == opp and pp == p]
            if cand:
                # เปรียบเทียบ severity แล้วเลือก
                aa, pp, ss, zz, sv = cand[0]
                if sv > sev:
                    out.append((opp, p, ss, zz, sv))
                else:
                    out.append((a, p, s, z, sev))
                seen_keys.add(key)
                continue
        # ไม่มีฝั่งตรงข้าม
        out.append((a, p, s, z, sev))
        seen_keys.add(key)
    return out

# ---------------------------------------------------------
# แปลง angle+phase → tag จาก description_map
# ---------------------------------------------------------
def _angle_phase_to_tag(angle_key: str, phase: str, description_map: Dict[str, Dict[str, str]]) -> str | None:
    if angle_key in description_map:
        return description_map[angle_key].get(phase, None)
    # เผื่อกรณีใช้คีย์รวม เช่น 'knee' แต่ map อยู่ที่ 'knee_L/R'
    for k in (angle_key + "_L", angle_key + "_R"):
        if k in description_map and phase in description_map[k]:
            return description_map[k][phase]
    return None

# ---------------------------------------------------------
# สร้างข้อความโค้ชชิ่งจาก deviation
# ---------------------------------------------------------
def explain_from_deviation(
    per_angle_phase: Dict[str, Dict[str, Dict[str, float]]],
    description_map: Dict[str, Dict[str, str]],
    top_k: int = 3,
    language: str = "th",
) -> List[str]:
    """
    รับเมทริกซ์คะแนน per_angle_phase และ mapping → คืน coaching ข้อความ 1..top_k ข้อ
    - เลือกจาก deviation ที่หนักสุดก่อน
    - ใช้ template ตาม tag
    """
    if not per_angle_phase:
        return []  # ไม่มีข้อมูลพอ

    ranked = _rank_deviation(per_angle_phase)
    ranked = _merge_side_duplicates(ranked)

    coaching: List[str] = []
    used_tags = set()
    for angle_key, phase, score, z_mean, sev in ranked:
        tag = _angle_phase_to_tag(angle_key, phase, description_map)
        if not tag or tag in used_tags:
            continue
        text = _get_template(tag, language=language)
        coaching.append(text)
        used_tags.add(tag)
        if len(coaching) >= top_k:
            break
    return coaching

# ---------------------------------------------------------
# สรุป deviation สำหรับ UI/ดีบั๊ก
# ---------------------------------------------------------
def summarize_deviation(
    per_angle_phase: Dict[str, Dict[str, Dict[str, float]]],
    description_map: Dict[str, Dict[str, str]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    คืนรายการรายละเอียดของ deviation อันดับต้น ๆ
    รูปแบบ:
      [
        {
          "angle": "knee_L",
          "phase": "bottom",
          "score": 72.3,
          "z_mean": 1.85,
          "severity": 36.2,
          "tag": "knee_valgus_L"
        }, ...
      ]
    """
    ranked = _rank_deviation(per_angle_phase)
    out: List[Dict[str, Any]] = []
    for angle_key, phase, score, z_mean, sev in ranked[:top_k]:
        tag = _angle_phase_to_tag(angle_key, phase, description_map)
        out.append({
            "angle": angle_key,
            "phase": phase,
            "score": float(score),
            "z_mean": float(z_mean),
            "severity": float(sev),
            "tag": tag or "",
        })
    return out

# ---------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------
if __name__ == "__main__":
    # ตัวอย่าง per_angle_phase สมมุติ
    per = {
        "knee": {
            "down":   {"score": 70.0, "z_mean": 1.2},
            "bottom": {"score": 65.0, "z_mean": 1.5},
        },
        "knee_L": {"bottom": {"score": 62.0, "z_mean": 1.8}},
        "knee_R": {"bottom": {"score": 78.0, "z_mean": 0.9}},
        "trunk":  {"down":   {"score": 68.0, "z_mean": 1.1}},
    }
    desc = {
        "knee":   {"down":"knee_forward", "bottom":"knee_depth"},
        "knee_L": {"bottom":"knee_valgus_L"},
        "knee_R": {"bottom":"knee_valgus_R"},
        "trunk":  {"down":"trunk_lean", "bottom":"trunk_lean"},
    }

    tips_th = explain_from_deviation(per, desc, top_k=3, language="th")
    print("TH:", tips_th)
    tips_en = explain_from_deviation(per, desc, top_k=3, language="en")
    print("EN:", tips_en)

    dbg = summarize_deviation(per, desc, top_k=5)
    print("DBG:", dbg)
