# tests/test_segmentation.py
import numpy as np
import pytest
from app.segmentation import detect_reps, label_phases

def test_detect_reps_simple():
    # ทำสัญญาณเข่าขึ้น-ลง 2 รอบ
    t = np.linspace(0, 2*np.pi, 100)
    knee_angle = 90 + 30*np.sin(2*t)
    reps = detect_reps(knee_angle)
    assert isinstance(reps, int)
    assert reps >= 1  # ต้องนับได้อย่างน้อย 1 รอบ

def test_label_phases_output():
    frames = 50
    # สร้างมุมเข่าขึ้นลง
    t = np.linspace(0, 2*np.pi, frames)
    angles = {
        "knee": 90 + 30*np.sin(t)
    }
    phases = label_phases(angles)
    assert isinstance(phases, list)
    assert len(phases) == frames
    assert all(p in ["start", "down", "bottom", "up"] for p in phases)
