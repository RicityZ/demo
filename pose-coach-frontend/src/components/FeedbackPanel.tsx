import React from "react"
import { ScoreSnapshot } from "../ml/types"

type FeedbackPanelProps = {
  score?: ScoreSnapshot
}

export default function FeedbackPanel({ score }: FeedbackPanelProps) {
  if (!score || !score.tips || score.tips.length === 0) {
    return (
      <div
        style={{
          padding: 12,
          border: "1px solid #222",
          borderRadius: 12,
          background: "#111",
          opacity: 0.6,
        }}
      >
        ยังไม่มีคำแนะนำ
      </div>
    )
  }

  return (
    <div
      style={{
        padding: 12,
        border: "1px solid #222",
        borderRadius: 12,
        background: "#111",
      }}
    >
      <h3>คำแนะนำ</h3>
      <ul>
        {score.tips.map((tip, i) => (
          <li key={i}>{tip}</li>
        ))}
      </ul>
    </div>
  )
}
