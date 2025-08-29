# scripts/mock_client.py
import requests
import json
from pathlib import Path

API_URL = "http://127.0.0.1:8000"

def upload_video(path: str, exercise="squat", ref=True):
    url = f"{API_URL}/upload_video?exercise={exercise}&as_reference={str(ref).lower()}"
    with open(path, "rb") as f:
        res = requests.post(url, files={"file": f})
    print("[UPLOAD VIDEO]", res.status_code)
    print(res.json())

def extract_keypoints(path: str):
    url = f"{API_URL}/extract_keypoints"
    with open(path, "rb") as f:
        res = requests.post(url, files={"file": f})
    print("[EXTRACT KEYS]", res.status_code)
    data = res.json()
    Path("keypoints.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("→ บันทึก keypoints.json เรียบร้อย")

def score_from_json(json_path: str, exercise="squat"):
    url = f"{API_URL}/score"
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["exercise"] = exercise
    res = requests.post(url, json=payload)
    print("[SCORE]", res.status_code)
    print(json.dumps(res.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # 1) ทดสอบอัปโหลดวิดีโอ (สร้าง reference ใหม่)
    upload_video("examples/squat_correct.mp4", exercise="squat")

    # 2) สกัด keypoints จากคลิปเดียวกัน
    extract_keypoints("examples/squat_correct.mp4")

    # 3) ยิง /score จาก keypoints.json ที่สกัดมา
    score_from_json("keypoints.json", exercise="squat")
