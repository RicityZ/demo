// src/ml/features.ts
import { KPFrame } from "./types"

// Helper: คำนวณมุมจาก 3 จุด
function calcAngle(a: any, b: any, c: any): number {
  const ab = { x: a.x - b.x, y: a.y - b.y }
  const cb = { x: c.x - b.x, y: c.y - b.y }
  const dot = ab.x * cb.x + ab.y * cb.y
  const magAB = Math.hypot(ab.x, ab.y)
  const magCB = Math.hypot(cb.x, cb.y)
  const cosTheta = dot / (magAB * magCB)
  return (Math.acos(Math.min(Math.max(cosTheta, -1), 1)) * 180) / Math.PI
}

// คำนวณมุมหลัก ๆ
export function computeAngles(kp: KPFrame) {
  return {
    kneeL: calcAngle(kp[23], kp[25], kp[27]), // Hip-L, Knee-L, Ankle-L
    kneeR: calcAngle(kp[24], kp[26], kp[28]),
    hipL: calcAngle(kp[11], kp[23], kp[25]), // Shoulder-L, Hip-L, Knee-L
    hipR: calcAngle(kp[12], kp[24], kp[26]),
    ankleL: calcAngle(kp[25], kp[27], kp[29]),
    ankleR: calcAngle(kp[26], kp[28], kp[30]),
  }
}
