import React from "react"

type ExercisePickerProps = {
  exercises: string[]
  value?: string
  onChange: (val: string) => void
}

export default function ExercisePicker({
  exercises,
  value,
  onChange,
}: ExercisePickerProps) {
  return (
    <div style={{ marginBottom: "16px" }}>
      <label style={{ fontWeight: "bold" }}>เลือกท่าออกกำลังกาย:</label>
      <select
        value={value || ""}
        onChange={(e) => onChange(e.target.value)}
        style={{
          display: "block",
          width: "100%",
          marginTop: 8,
          padding: "8px 12px",
          borderRadius: 6,
          border: "1px solid #333",
          backgroundColor: "#111",
          color: "#fff",
        }}
      >
        <option value="" disabled>
          — กรุณาเลือก —
        </option>
        {exercises.map((ex) => (
          <option key={ex} value={ex}>
            {ex}
          </option>
        ))}
      </select>
    </div>
  )
}
