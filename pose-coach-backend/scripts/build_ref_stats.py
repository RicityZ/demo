# scripts/build_ref_stats.py
from __future__ import annotations
import json
from pathlib import Path
from app.refs import build_stats_from_refs, save_stats

BASE_DIR = Path(__file__).resolve().parent.parent
REF_DIR = BASE_DIR / "references"

def main():
    print(f"[INFO] เริ่มสร้าง stats จาก references ทั้งหมดใน: {REF_DIR}")

    if not REF_DIR.exists():
        print("[ERROR] ไม่พบโฟลเดอร์ references/")
        return

    # loop exercises: squat/, pushup/, ...
    for exercise_dir in REF_DIR.iterdir():
        if not exercise_dir.is_dir():
            continue

        exercise = exercise_dir.name
        print(f"\n[INFO] === สร้าง stats สำหรับท่า: {exercise} ===")

        # หา ref ทั้งหมดของท่านี้ (*.json ที่ไม่ใช่ stats)
        ref_files = list(exercise_dir.glob("ref_*.json"))
        if not ref_files:
            print(f"[WARN] ไม่มีไฟล์ reference สำหรับ {exercise}, ข้าม...")
            continue

        # รวมข้อมูลจาก refs ทั้งหมด
        refs = []
        for rf in ref_files:
            try:
                with open(rf, "r", encoding="utf-8") as f:
                    refs.append(json.load(f))
            except Exception as e:
                print(f"[ERROR] อ่าน {rf.name} ไม่ได้: {e}")

        if not refs:
            print(f"[WARN] ไม่มีข้อมูล usable refs สำหรับ {exercise}, ข้าม...")
            continue

        # สร้าง stats จาก refs
        stats = build_stats_from_refs(refs)
        save_stats(exercise, stats)

        print(f"[OK] สร้าง stats สำหรับ {exercise} สำเร็จ")

    print("\n[INFO] เสร็จเรียบร้อย!")

if __name__ == "__main__":
    main()
