# tests/test_preprocessing.py
import numpy as np
import pytest
from app.preprocessing import preprocess_pipeline

def _make_dummy_frames(num_frames=5, num_kp=17):
    frames = []
    for _ in range(num_frames):
        # keypoints: x,y,conf
        frame = np.random.rand(num_kp, 3).astype(np.float32)
        frame[:, 2] = 1.0  # ให้ conf = 1 ทุกจุด
        frames.append(frame)
    return frames

def test_preprocess_pipeline_shape():
    frames = _make_dummy_frames(10)
    out = preprocess_pipeline(frames)
    assert isinstance(out, list)
    assert len(out) == len(frames)
    assert all(f.shape == (17, 3) for f in out)

def test_preprocess_pipeline_nan_conf():
    frames = _make_dummy_frames(5)
    # ปลอม conf = 0 เพื่อทดสอบ filter
    frames[0][0, 2] = 0.0
    out = preprocess_pipeline(frames)
    assert out[0][0, 2] >= 0.0  # ต้องไม่เกิด NaN
