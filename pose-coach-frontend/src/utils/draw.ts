// src/utils/draw.ts
import type { KPFrame } from "../ml/types"

type DrawOpts = {
  pointRadius?: number
  minScore?: number
}

/**
 * วาด skeleton ลงบน 2D canvas
 * รองรับทั้ง MoveNet-17 และ BlazePose-33 (จะเลือก edges ตามจำนวนจุดอัตโนมัติ)
 */
export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  kp: KPFrame,
  opts: DrawOpts = {}
) {
  const r = opts.pointRadius ?? 4
  const minScore = opts.minScore ?? 0.2

  if (!kp || kp.length === 0) return

  const isMoveNet17 = kp.length <= 17
  const edges: [number, number][] = isMoveNet17
    ? // MoveNet 17 (tfjs pose-detection)
      [
        [5, 7], [7, 9],  // left shoulder->elbow->wrist
        [6, 8], [8, 10], // right shoulder->elbow->wrist
        [5, 6],          // shoulders
        [5, 11], [6, 12],// shoulders->hips
        [11, 12],        // hips
        [11, 13], [13, 15], // left hip->knee->ankle
        [12, 14], [14, 16], // right hip->knee->ankle
      ]
    : // BlazePose 33 (mediapipe) subset
      [
        [11, 13], [13, 15], // left shoulder->elbow->wrist
        [12, 14], [14, 16], // right shoulder->elbow->wrist
        [11, 12],           // shoulders
        [23, 24],           // hips
        [11, 23], [12, 24], // shoulders->hips
        [23, 25], [25, 27], // left hip->knee->ankle
        [24, 26], [26, 28], // right hip->knee->ankle
      ]

  // เส้นกระดูก
  ctx.save()
  ctx.lineWidth = 3
  ctx.strokeStyle = "rgba(255,255,255,0.85)"
  ctx.beginPath()
  for (const [a, b] of edges) {
    const pa = kp[a], pb = kp[b]
    if (!pa || !pb) continue
    if ((pa.score ?? 1) < minScore || (pb.score ?? 1) < minScore) continue
    ctx.moveTo(pa.x, pa.y)
    ctx.lineTo(pb.x, pb.y)
  }
  ctx.stroke()
  ctx.restore()

  // จุด keypoints
  ctx.save()
  for (const p of kp) {
    if (!p) continue
    if ((p.score ?? 1) < minScore) continue
    ctx.beginPath()
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2)
    ctx.fillStyle = "rgba(0,200,255,0.9)"
    ctx.fill()
  }
  ctx.restore()
}
