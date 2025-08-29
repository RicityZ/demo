// src/ml/preprocess.ts
import { KPFrame } from "./types"

// EMA smoothing เพื่อลด noise
export function smoothKeypoints(
  current: KPFrame,
  prev?: KPFrame,
  alpha = 0.7
): KPFrame {
  if (!prev) return current
  return current.map((kp, i) => ({
    x: alpha * kp.x + (1 - alpha) * prev[i].x,
    y: alpha * kp.y + (1 - alpha) * prev[i].y,
    score: kp.score,
  }))
}

// ปรับตำแหน่งให้ศูนย์กลาง (mid-hip) เป็น origin
export function centerKeypoints(kp: KPFrame): KPFrame {
  const midHip = {
    x: (kp[23]?.x + kp[24]?.x) / 2,
    y: (kp[23]?.y + kp[24]?.y) / 2,
  }
  return kp.map((p) => ({ ...p, x: p.x - midHip.x, y: p.y - midHip.y }))
}

// ปรับสเกลให้ขนาดโครงสร้างสัมพันธ์เท่ากันทุกเฟรม (normalize)
export function scaleKeypoints(kp: KPFrame): KPFrame {
  const shoulderDist = Math.hypot(kp[11].x - kp[12].x, kp[11].y - kp[12].y)
  return kp.map((p) => ({
    ...p,
    x: p.x / shoulderDist,
    y: p.y / shoulderDist,
  }))
}
