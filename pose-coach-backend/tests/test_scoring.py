# tests/test_scoring.py
import numpy as np
import pytest
from app.scoring import score_angles, score_shape, aggregate_scores

def _make_dummy_angles(num_frames=100):
    return {
        "knee": np.linspace(160, 90, num_frames),
        "hip": np.linspace(150, 100, num_frames),
        "trunk": np.linspace(90, 80, num_frames),
    }

def _make_dummy_phase_masks(num_frames=100):
    # ใช้ทั้งสัญญาณเป็น phase เดียว
    return {p: np.ones(num_frames, dtype=bool) for p in ["start", "down", "bottom", "up"]}

def test_score_angles_shape():
    frames = 100
    angles_user = _make_dummy_angles(frames)
    angles_ref = _make_dummy_angles(frames)
    stats = {
        "angles": {
            "knee": {p: {"mu": 120.0, "sigma": 10.0} for p in ["start","down","bottom","up"]},
            "hip": {p: {"mu": 120.0, "sigma": 10.0} for p in ["start","down","bottom","up"]},
            "trunk": {p: {"mu": 85.0, "sigma": 5.0} for p in ["start","down","bottom","up"]},
        }
    }
    phase_masks = _make_dummy_phase_masks(frames)
    angle_weights = {"knee":0.5, "hip":0.3, "trunk":0.2}

    a_score, per_angle_phase = score_angles(
        angles_user, angles_ref, angle_weights, phase_masks, stats,
        lead_key="knee"
    )
    assert "overall" in a_score
    assert 0 <= a_score["overall"] <= 100

    # shape score ทดสอบด้วย random vecs
    vec_ref = np.random.randn(frames, 8).astype(np.float32)
    vec_usr = vec_ref + np.random.normal(0, 0.1, vec_ref.shape)
    s_score = score_shape(vec_usr, vec_ref)
    assert 0 <= s_score <= 100

    # aggregate
    combined = aggregate_scores(a_score, s_score)
    assert "overall" in combined
    assert 0 <= combined["overall"] <= 100
