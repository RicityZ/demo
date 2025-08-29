# app/schemas.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator

"""
Pydantic models สำหรับ request/response ของ FastAPI
- ออกแบบให้รองรับการส่ง keypoints แบบเรียลไทม์จาก React หรือ Mobile
- รองรับผลลัพธ์เต็ม: คะแนน, feedback, phases, rep_count
"""

# ---------------------------------------------------------
# Models สำหรับ keypoints และ frames
# ---------------------------------------------------------
class Keypoint(BaseModel):
    x: float = Field(..., description="ตำแหน่ง x ของ keypoint (normalized)")
    y: float = Field(..., description="ตำแหน่ง y ของ keypoint (normalized)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="ความมั่นใจของ keypoint")

    @validator("x", "y", "confidence", pre=True)
    def ensure_float(cls, v):
        return float(v)

class Frame(BaseModel):
    points: List[List[float]] = Field(
        ..., 
        description="รายการ keypoints 17 จุด [x,y,conf] ต่อเฟรม"
    )

    @validator("points")
    def validate_points(cls, v):
        if len(v) != 17:
            raise ValueError("ต้องมี keypoints 17 จุดต่อเฟรม")
        for pt in v:
            if len(pt) != 3:
                raise ValueError("แต่ละ keypoint ต้องเป็น [x, y, conf]")
        return v

# ---------------------------------------------------------
# Request model สำหรับ /score
# ---------------------------------------------------------
class ScoreRequest(BaseModel):
    exercise: str = Field(..., description="ชื่อท่าออกกำลังกาย เช่น 'squat'")
    frames: List[Frame] = Field(..., description="ลิสต์เฟรม keypoints จากผู้ใช้")
    user_id: Optional[str] = Field(None, description="รหัสผู้ใช้ (ถ้ามี)")

    @validator("exercise")
    def normalize_exercise(cls, v: str):
        return v.strip().lower()

# ---------------------------------------------------------
# Response models สำหรับ /score
# ---------------------------------------------------------
class PhaseDetail(BaseModel):
    angle: str
    phase: str
    score: float
    z_mean: float

class ScoreBreakdown(BaseModel):
    overall: float = Field(..., description="คะแนนรวม 0–100")
    per_angle: Dict[str, float] = Field(..., description="คะแนนรายมุม")
    per_phase: Dict[str, float] = Field(..., description="คะแนนรายเฟส")

class ScoreResponse(BaseModel):
    rep_total: int = Field(..., description="จำนวน rep ที่ทำสำเร็จ")
    scores: Dict[str, Any] = Field(..., description="โครงสร้างคะแนน รวม angle+shape+overall")
    coaching: List[str] = Field(..., description="ข้อความ feedback สำหรับผู้ใช้")
    issues: List[str] = Field(..., description="จุดผิดท่าที่พบ (ถ้ามี)")
    phases: List[str] = Field(..., description="ป้ายเฟสของแต่ละเฟรม")

# ---------------------------------------------------------
# Response สำหรับ /upload_reference
# ---------------------------------------------------------
class UploadReferenceResponse(BaseModel):
    status: str
    exercise: str
    message: str

# ---------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------
if __name__ == "__main__":
    # ทดสอบการสร้าง request
    dummy_req = ScoreRequest(
        exercise="squat",
        frames=[
            Frame(points=[[0.5, 0.5, 0.9]]*17),
            Frame(points=[[0.55, 0.55, 0.95]]*17),
        ],
        user_id="user_123"
    )
    print(dummy_req.json(indent=2))

    # ทดสอบ response
    dummy_res = ScoreResponse(
        rep_total=10,
        scores={
            "overall": 85.5,
            "angles": {
                "overall": 83.2,
                "per_angle": {"knee": 90.1, "hip": 80.2},
                "per_phase": {"start": 85, "down": 82, "bottom": 78, "up": 88},
            },
            "shape": 87.0,
            "alpha": 0.7
        },
        coaching=[
            "ช่วงลง พยายามอย่าดันเข่าเลยปลายเท้า",
            "ลำตัวเอนไปหน้ามากในช่วงลง — เปิดอกและรักษาแกนลำตัวให้มั่นคง"
        ],
        issues=[],
        phases=["start", "down", "bottom", "up"]
    )
    print(dummy_res.json(indent=2))
