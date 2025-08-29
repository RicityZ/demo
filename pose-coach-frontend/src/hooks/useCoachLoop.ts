// src/hooks/useCoachLoop.ts
import { useEffect, useState } from "react"
import { KPFrame, ScoreSnapshot, Stats } from "../ml/types"
import { smoothKeypoints, centerKeypoints, scaleKeypoints } from "../ml/preprocess"
import { computeAngles } from "../ml/features"
import { labelPhase } from "../ml/segmentation"
import { scoreAngle, aggregateScore } from "../ml/scoring"
import { makeFeedback } from "../ml/feedback"

type CoachLoopProps = {
  keypoints: KPFrame | null
  stats?: Stats
}

export function useCoachLoop({ keypoints, stats }: CoachLoopProps) {
  const [prevKeypoints, setPrevKeypoints] = useState<KPFrame | null>(null)
  const [snapshot, setSnapshot] = useState<ScoreSnapshot | null>(null)
  const [phase, setPhase] = useState<string>("start")
  const [reps, setReps] = useState(0)

  useEffect(() => {
    if (!keypoints || !stats) return

    // 1) Smooth
    const smooth = smoothKeypoints(keypoints, prevKeypoints || undefined)
    setPrevKeypoints(smooth)

    // 2) Normalize
    const normalized = scaleKeypoints(centerKeypoints(smooth))

    // 3) Features
    const angles = computeAngles(normalized)

    // 4) Segmentation + Reps
    const seg = labelPhase(angles.kneeL, { phase: phase as any, reps })
    setPhase(seg.phase)
    setReps(seg.reps)

    // 5) Scoring
    const angleScores: Record<string, number> = {}
    for (const [key, value] of Object.entries(angles)) {
      angleScores[key] = scoreAngle(value, stats, key)
    }
    const overall = aggregateScore(angleScores)

    // 6) Feedback
    const tips = makeFeedback({
      overall,
      angle: aggregateScore(angleScores),
      shape: 100,
      angleBreakdown: angleScores,
    })

    setSnapshot({
      overall,
      angle: aggregateScore(angleScores),
      shape: 100,
      angleBreakdown: angleScores,
      tips,
    })
  }, [keypoints, stats])

  return { snapshot, phase, reps }
}
