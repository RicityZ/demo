import React, { useEffect, useRef } from "react"
import { KPFrame } from "../ml/types"
import { drawSkeleton } from "../utils/draw"

type SkeletonOverlayProps = {
  keypoints?: KPFrame
  width?: number
  height?: number
}

export default function SkeletonOverlay({
  keypoints,
  width = 1280,
  height = 720,
}: SkeletonOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!canvasRef.current || !keypoints) return
    const ctx = canvasRef.current.getContext("2d")
    if (!ctx) return

    ctx.clearRect(0, 0, width, height)
    drawSkeleton(ctx, keypoints)
  }, [keypoints, width, height])

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        width: "100%",
        height: "auto",
        position: "absolute",
        top: 0,
        left: 0,
        pointerEvents: "none",
      }}
    />
  )
}
