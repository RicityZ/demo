# app/api.py
from __future__ import annotations
import json
import hashlib
import time
import tempfile
import secrets
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response

from app.config import settings, DATA_DIR  # .env: DATA_DIR, CORS_ALLOW_ORIGINS
from app.extractor import extract_keypoints_from_video  # path -> dict {fps, frames:[{points:[[x,y,c],...]}]}

# =============================
# Helpers
# =============================
def _make_etag_from_file(path: Path) -> str:
    st = path.stat()
    raw = f"{path.name}:{st.st_mtime_ns}:{st.st_size}".encode()
    return hashlib.md5(raw).hexdigest()

def _json_cached(payload: dict, etag_seed: str, request: Request, max_age: int = 300) -> Response:
    etag = hashlib.md5((etag_seed + json.dumps(payload, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()
    inm = request.headers.get("if-none-match")
    headers = {"ETag": etag, "Cache-Control": f"public, max-age={max_age}"}
    if inm == etag:
        return Response(status_code=304, headers=headers)
    return JSONResponse(payload, headers=headers)

def _ensure_exercise_dir(exercise: str) -> Path:
    p = Path(DATA_DIR) / exercise
    p.mkdir(parents=True, exist_ok=True)
    return p

def _list_demo_videos(exercise: str) -> List[str]:
    """
    หาไฟล์วิดีโอเดโมได้ 2 ที่:
      1) references/{exercise}/*.mp4 (แนะนำจัดวางที่นี่)
      2) references/_videos/*.mp4 (โฟลเดอร์รวม)
    """
    results: List[str] = []
    ex_dir = Path(DATA_DIR) / exercise
    if ex_dir.exists():
        for p in sorted(ex_dir.glob("*.mp4")):
            # คืน path สัมพัทธ์สำหรับ /videos/{filename:path}
            results.append(f"{exercise}/{p.name}")
    common_dir = Path(DATA_DIR) / "_videos"
    if common_dir.exists():
        for p in sorted(common_dir.glob("*.mp4")):
            name = p.name.lower()
            if name.startswith(exercise.lower()) or name.startswith("demo_"):
                results.append(f"_videos/{p.name}")
    return results

# =============================
# FastAPI App & CORS
# =============================
app = FastAPI(
    title="Pose Coach Assets API",
    description="Read-only + Upload API สำหรับ reference/stats/video",
    version="2.2.0",
)

# CORS จาก ENV (รองรับ comma-separated). ถ้าเป็น "*" จะ allow ทุกโดเมน
orig = settings.CORS_ALLOW_ORIGINS
if isinstance(orig, str):
    allow_origins = ["*"] if orig.strip() == "*" else [o.strip() for o in orig.split(",") if o.strip()]
else:
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "HEAD", "OPTIONS", "POST"],  # รวม POST สำหรับอัปโหลด
    allow_headers=["*"],
)

# =============================
# Health check
# =============================
@app.get("/ping")
def ping():
    return {"ok": True, "role": "data-provider", "version": app.version if hasattr(app, "version") else "2.2.0"}

# =============================
# Read-only Asset Endpoints (JSON + Cache)
# =============================
@app.get("/exercises")
def list_exercises(request: Request):
    """
    รายชื่อท่าที่มีภายใต้ DATA_DIR (เช่น references/squat, references/pushup)
    """
    base = Path(DATA_DIR)
    exercises = sorted([p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")]) if base.exists() else []
    return _json_cached({"exercises": exercises}, etag_seed="exercises", request=request)

@app.get("/assets/{exercise}/refs")
def list_refs(exercise: str, request: Request):
    """
    รายชื่อ reference JSON ของท่าที่ระบุ และชื่อไฟล์ stats ถ้ามี
    - refs: ["ref_YYYYMMDD-HHMMSS.json", ...]
    - stats: "stats_{exercise}.json" หรือ null ถ้าไม่มี
    """
    ex_dir = Path(DATA_DIR) / exercise
    if not ex_dir.exists() or not ex_dir.is_dir():
        raise HTTPException(status_code=404, detail="exercise not found")

    refs = sorted([p.name for p in ex_dir.glob("ref_*.json")])
    stats_name = f"stats_{exercise}.json"
    payload = {
        "exercise": exercise,
        "refs": refs,
        "stats": stats_name if (ex_dir / stats_name).exists() else None,
    }
    stats_mtime = (ex_dir / stats_name).stat().st_mtime_ns if (ex_dir / stats_name).exists() else 0
    seed = f"refs:{exercise}:{len(refs)}:{stats_mtime}"
    return _json_cached(payload, etag_seed=seed, request=request)

@app.get("/assets/{exercise}/refs/{name}")
def get_ref(exercise: str, name: str, request: Request):
    """
    คืนเนื้อหา reference JSON (เช่น ref_YYYYMMDD-HHMMSS.json)
    """
    path = Path(DATA_DIR) / exercise / name
    if not path.exists() or path.suffix.lower() != ".json":
        raise HTTPException(status_code=404, detail="reference not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        seed = f"ref:{exercise}:{name}:{path.stat().st_mtime_ns}"
        return _json_cached(data, etag_seed=seed, request=request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"อ่านไฟล์อ้างอิงล้มเหลว: {e}")

@app.get("/assets/{exercise}/stats")
def get_stats(exercise: str, request: Request):
    """
    คืนเนื้อหา stats_{exercise}.json (μ, σ ต่อ angle×phase และเมทาดาต้า)
    """
    path = Path(DATA_DIR) / exercise / f"stats_{exercise}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="stats not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        seed = f"stats:{exercise}:{path.stat().st_mtime_ns}"
        return _json_cached(data, etag_seed=seed, request=request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"อ่านไฟล์สถิติล้มเหลว: {e}")

@app.get("/assets/{exercise}/meta")
def get_exercise_meta(exercise: str, request: Request):
    """
    รวมข้อมูลที่ frontend ต้องใช้ในครั้งเดียว:
    - refs list
    - stats object (ถ้ามี)
    - videos list (เดโม) จาก references/{exercise}/*.mp4 และ references/_videos/*.mp4
    """
    ex_dir = Path(DATA_DIR) / exercise
    if not ex_dir.exists() or not ex_dir.is_dir():
        raise HTTPException(status_code=404, detail="exercise not found")

    refs = sorted([p.name for p in ex_dir.glob("ref_*.json")])
    stats_path = ex_dir / f"stats_{exercise}.json"
    stats_obj = None
    stats_mtime = 0
    if stats_path.exists():
        try:
            stats_obj = json.loads(stats_path.read_text(encoding="utf-8"))
            stats_mtime = stats_path.stat().st_mtime_ns
        except Exception:
            stats_obj = None

    vids = _list_demo_videos(exercise)  # รายการ path สัมพัทธ์ เช่น "squat/demo_xxx.mp4", "_videos/demo_xxx.mp4"
    payload = {
        "exercise": exercise,
        "refs": refs,
        "stats": stats_obj,
        "videos": vids,
    }
    seed = f"meta:{exercise}:{len(refs)}:{stats_mtime}:{len(vids)}"
    return _json_cached(payload, etag_seed=seed, request=request, max_age=300)

# =============================
# Demo Video Static (ETag + Cache)
# =============================
@app.get("/videos/{filename:path}")
def get_video(filename: str, request: Request):
    """
    เสิร์ฟไฟล์วิดีโอเดโม:
      - /videos/squat/demo_squat_lv1.mp4        → references/squat/demo_squat_lv1.mp4
      - /videos/_videos/demo_squat_lv1.mp4      → references/_videos/demo_squat_lv1.mp4
    """
    path = Path(DATA_DIR) / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"video not found: {filename}")

    etag = _make_etag_from_file(path)
    headers = {"ETag": etag, "Cache-Control": "public, max-age=86400"}  # 1 วัน
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)

    return FileResponse(str(path), media_type="video/mp4", headers=headers)

# =============================
# Upload & Extract Endpoints
# =============================
@app.post("/upload_video")
async def upload_video(
    exercise: str = Query(..., description="ชื่อท่า เช่น squat/pushup"),
    as_reference: bool = Query(True, description="บันทึกเป็น reference ไหม"),
    target_fps: int = Query(30, ge=1, le=120, description="fps สำหรับ sample"),
    save_video: bool = Query(True, description="เก็บไฟล์วิดีโอต้นฉบับไว้ด้วยหรือไม่"),
    video_name: Optional[str] = Query(None, description="ตั้งชื่อไฟล์วิดีโอ (เช่น demo_squat_lv1.mp4)"),
    file: UploadFile = File(..., description="ไฟล์วิดีโอ .mp4/.mov"),
):
    """
    อัปโหลดวิดีโอ → สกัด keypoints → (ออปชัน) บันทึกเป็น reference JSON
    และ (ออปชัน) เก็บไฟล์วิดีโอต้นฉบับสำหรับพรีวิว
    """
    if not file.content_type or "video" not in file.content_type.lower():
        raise HTTPException(status_code=400, detail="ต้องเป็นไฟล์วิดีโอ (mp4/mov)")

    # 1) เซฟไฟล์ชั่วคราว
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = Path(tmp.name)
        content = await file.read()
        tmp.write(content)

    saved_json = None
    saved_video_rel = None
    try:
        # 2) สกัด keypoints
        kp = extract_keypoints_from_video(str(temp_path), target_fps=target_fps)

        # 3) บันทึก reference JSON ถ้าต้องการ
        if as_reference:
            ex_dir = _ensure_exercise_dir(exercise)
            ts = time.strftime("%Y%m%d-%H%M%S")
            json_path = ex_dir / f"ref_{exercise}_{ts}.json"
            json_path.write_text(json.dumps(kp, ensure_ascii=False), encoding="utf-8")
            saved_json = str(json_path)

        # 4) เก็บวิดีโอต้นฉบับ (ออปชัน)
        if save_video:
            ex_dir = _ensure_exercise_dir(exercise)
            if video_name:
                safe_name = Path(video_name).name  # กัน path traversal
            else:
                ts = time.strftime("%Y%m%d-%H%M%S")
                safe_name = f"demo_{exercise}_{ts}_{secrets.token_hex(3)}.mp4"
            final_path = ex_dir / safe_name
            final_path.write_bytes(temp_path.read_bytes())
            saved_video_rel = f"{exercise}/{safe_name}"  # relative path for /videos/{...}

        # 5) URL สำหรับวิดีโอ (ถ้าเก็บไว้)
        video_url = f"/videos/{saved_video_rel}" if saved_video_rel else None

        return {
            "exercise": exercise,
            "fps": kp.get("fps"),
            "frames": len(kp.get("frames", [])),
            "saved_reference": saved_json,
            "saved_video": saved_video_rel,
            "video_url": video_url,
        }
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

@app.post("/extract_keypoints")
async def extract_keypoints(
    target_fps: int = Query(30, ge=1, le=120),
    file: UploadFile = File(..., description="ไฟล์วิดีโอ .mp4/.mov"),
):
    """
    อัปโหลดวิดีโอ → คืน keypoints JSON ทันที (ไม่บันทึกไฟล์)
    """
    if not file.content_type or "video" not in file.content_type.lower():
        raise HTTPException(status_code=400, detail="ต้องเป็นไฟล์วิดีโอ (mp4/mov)")

    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = Path(tmp.name)
        content = await file.read()
        tmp.write(content)

    try:
        kp = extract_keypoints_from_video(str(temp_path), target_fps=target_fps)
        return kp
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

@app.post("/upload_demo_video")
async def upload_demo_video(
    exercise: str = Query(..., description="ชื่อท่า เช่น squat/pushup"),
    video_name: Optional[str] = Query(None, description="ตั้งชื่อไฟล์เดโม เช่น demo_squat_lv1.mp4"),
    file: UploadFile = File(..., description="ไฟล์วิดีโอ .mp4/.mov"),
):
    """
    อัปโหลดเฉพาะไฟล์เดโมวิดีโอ (ไม่สกัด keypoints)
    """
    if not file.content_type or "video" not in file.content_type.lower():
        raise HTTPException(status_code=400, detail="ต้องเป็นไฟล์วิดีโอ (mp4/mov)")

    ex_dir = _ensure_exercise_dir(exercise)
    if video_name:
        safe_name = Path(video_name).name
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        safe_name = f"demo_{exercise}_{ts}.mp4"

    final_path = ex_dir / safe_name
    data = await file.read()
    final_path.write_bytes(data)

    return {
        "exercise": exercise,
        "saved_video": f"{exercise}/{safe_name}",
        "video_url": f"/videos/{exercise}/{safe_name}",
    }
