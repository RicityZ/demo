import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import app.api as api_mod

@pytest.fixture()
def tmp_refs(tmp_path, monkeypatch):
    """
    สร้างโครง references ชั่วคราว:
      references/
        squat/
          ref_001.json
          stats_squat.json
        _videos/
          demo_squat.mp4 (ไฟล์ dummy)
    แล้วผูก app.api.DATA_DIR -> tmp_path / "references"
    """
    base = tmp_path / "references"
    squat = base / "squat"
    vids = base / "_videos"
    squat.mkdir(parents=True, exist_ok=True)
    vids.mkdir(parents=True, exist_ok=True)

    # ref file
    (squat / "ref_001.json").write_text(json.dumps({"frames": [], "fps": 30}), encoding="utf-8")
    # stats file
    (squat / "stats_squat.json").write_text(json.dumps({"angles": {}, "phase_masks": {}, "weights": {}}), encoding="utf-8")
    # video dummy (ไม่ต้องมีเนื้อหา mp4 จริงก็พอสำหรับ 200/404/304 flow)
    (vids / "demo_squat.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")

    # monkeypatch DATA_DIR ในโมดูล api (สำคัญ!)
    monkeypatch.setattr(api_mod, "DATA_DIR", base, raising=True)

    return base

@pytest.fixture()
def client(tmp_refs):
    return TestClient(api_mod.app)

def test_exercises(client, tmp_refs):
    r = client.get("/exercises")
    assert r.status_code == 200
    data = r.json()
    assert "exercises" in data
    assert "squat" in data["exercises"]

def test_list_refs(client):
    r = client.get("/assets/squat/refs")
    assert r.status_code == 200
    data = r.json()
    assert data["exercise"] == "squat"
    assert "ref_001.json" in data["refs"]
    assert data["stats"] == "stats_squat.json"

def test_get_ref(client):
    r = client.get("/assets/squat/refs/ref_001.json")
    assert r.status_code == 200
    data = r.json()
    assert "frames" in data
    assert "fps" in data

def test_get_stats(client):
    r = client.get("/assets/squat/stats")
    assert r.status_code == 200
    data = r.json()
    assert "angles" in data
    assert "phase_masks" in data
    assert "weights" in data

def test_meta(client):
    r = client.get("/assets/squat/meta")
    assert r.status_code == 200
    data = r.json()
    assert data["exercise"] == "squat"
    assert "refs" in data and "stats" in data and "videos" in data
    assert "ref_001.json" in data["refs"]
    # stats object should be inlined
    assert isinstance(data["stats"], dict)
    # video listed
    assert any(name.endswith(".mp4") for name in data["videos"])

def test_video_200_and_304(client):
    # first fetch -> 200 + ETag
    r1 = client.get("/videos/demo_squat.mp4")
    assert r1.status_code == 200
    etag = r1.headers.get("ETag")
    assert etag

    # second fetch with If-None-Match -> 304
    r2 = client.get("/videos/demo_squat.mp4", headers={"If-None-Match": etag})
    assert r2.status_code == 304

def test_404s(client):
    assert client.get("/assets/unknown/refs").status_code == 404
    assert client.get("/assets/squat/refs/nope.json").status_code == 404
    assert client.get("/assets/unknown/stats").status_code == 404
    assert client.get("/videos/nope.mp4").status_code == 404
