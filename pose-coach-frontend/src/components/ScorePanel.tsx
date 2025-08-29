import React from "react"
import { ScoreSnapshot } from "../ml/types"

type ScorePanelProps = {
  score?: ScoreSnapshot
}

export default function ScorePanel({ score }: ScorePanelProps) {
  if (!score) return null

  return (
    <div
      style={{
        padding: 12,
        border: "1px solid #222",
        borderRadius: 12,
        background: "#111",
      }}
    >
      <h3>คะแนน</h3>
      <div>รวม: {score.overall.toFixed(1)} / 100</div>
      <div>มุม: {score.angle.toFixed(1)} / 100</div>
      <div>โครงสร้าง: {score.shape.toFixed(1)} / 100</div>
    </div>
  )
}
