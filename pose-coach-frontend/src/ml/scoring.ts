// src/ml/scoring.ts
import { Stats } from "./types"

// คำนวณคะแนนต่อมุมแบบ z-score
export function scoreAngle(
  userValue: number,
  refStats: Stats,
  key: string
): number {
  const mu = refStats.mu[key] ?? 0
  const sigma = refStats.sigma[key] ?? 1
  const z = Math.abs(userValue - mu) / sigma
  const score = Math.max(0, 100 * Math.exp(-0.5 * z * z))
  return score
}

// คำนวณคะแนนรวมจากทุกมุม
export function aggregateScore(scores: Record<string, number>): number {
  const vals = Object.values(scores)
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
}

// คำนวณ shape similarity score (โครงสร้างร่างกาย)
export function shapeScore(
  userAngles: number[],
  refAngles: number[]
): number {
  if (!userAngles.length || !refAngles.length) return 0
  const sum = userAngles.reduce(
    (acc, v, i) => acc + Math.abs(v - refAngles[i]),
    0
  )
  const avgDiff = sum / userAngles.length
  return Math.max(0, 100 - avgDiff) // ยิ่งต่างน้อย ยิ่งได้คะแนนสูง
}
