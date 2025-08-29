import React from "react"

type VideoDemoProps = {
  src?: string
}

export default function VideoDemo({ src }: VideoDemoProps) {
  if (!src) {
    return <div style={{ opacity: 0.6 }}>ยังไม่มีวิดีโอเดโม</div>
  }

  return (
    <video
      src={src}
      controls
      style={{
        width: "100%",
        borderRadius: 12,
        background: "#000"
      }}
    />
  )
}
