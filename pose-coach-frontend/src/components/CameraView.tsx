// src/components/CameraView.tsx
import React from "react"

// ใช้ forwardRef เพื่อส่ง video ref กลับไปใช้ใน hook/หน้า Coach
const CameraView = React.forwardRef<HTMLVideoElement, {}>((_props, ref) => {
  return (
    <video
      ref={ref}
      autoPlay
      playsInline
      muted
      style={{
        width: "100%",
        borderRadius: 12,
        background: "#000",
      }}
    />
  )
})

export default CameraView
