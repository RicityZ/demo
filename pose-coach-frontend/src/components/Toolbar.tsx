import React from "react"

type ToolbarProps = {
  onStart?: () => void
  onStop?: () => void
  onReset?: () => void
  isRunning?: boolean
}

export default function Toolbar({
  onStart,
  onStop,
  onReset,
  isRunning,
}: ToolbarProps) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        gap: "12px",
        marginTop: "16px",
      }}
    >
      {!isRunning ? (
        <button
          onClick={onStart}
          style={{
            padding: "10px 20px",
            background: "#0f9d58",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
          }}
        >
          ▶ เริ่ม
        </button>
      ) : (
        <button
          onClick={onStop}
          style={{
            padding: "10px 20px",
            background: "#db4437",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
          }}
        >
          ⏸ หยุด
        </button>
      )}
      <button
        onClick={onReset}
        style={{
          padding: "10px 20px",
          background: "#4285f4",
          color: "#fff",
          border: "none",
          borderRadius: 6,
          cursor: "pointer",
        }}
      >
        🔄 รีเซ็ต
      </button>
    </div>
  )
}
