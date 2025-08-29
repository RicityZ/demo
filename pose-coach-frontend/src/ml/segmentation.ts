// src/ml/segmentation.ts
import { PhaseLabel } from "./types"

type SegmentationState = {
  phase: PhaseLabel
  reps: number
}

// กำหนด threshold มุมเข่าเพื่อแยกเฟส squat เป็นตัวอย่าง
const KNEE_DOWN = 80  // ถ้าต่ำกว่า 80° → ลง
const KNEE_UP = 160   // ถ้ามากกว่า 160° → ขึ้นสุด

export function labelPhase(
  kneeAngle: number,
  prevState?: SegmentationState
): SegmentationState {
  const prevPhase = prevState?.phase ?? "start"
  let reps = prevState?.reps ?? 0

  let phase: PhaseLabel = prevPhase

  if (prevPhase === "start" || prevPhase === "up") {
    if (kneeAngle < KNEE_DOWN) {
      phase = "down"
    }
  }
  if (prevPhase === "down") {
    if (kneeAngle <= KNEE_DOWN) {
      phase = "bottom"
    }
  }
  if (prevPhase === "bottom") {
    if (kneeAngle > KNEE_UP) {
      phase = "up"
      reps += 1 // นับ 1 rep เมื่อกลับมาขึ้นสุด
    }
  }

  return { phase, reps }
}
