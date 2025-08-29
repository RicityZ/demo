// src/hooks/usePoseStream.ts
import { useEffect, useRef, useState, type RefObject } from "react"
import { loadDetector, estimatePoses } from "../ml/detector"
import type { KPFrame } from "../ml/types"

export function usePoseStream(videoRef: RefObject<HTMLVideoElement>) {
  const [keypoints, setKeypoints] = useState<KPFrame | null>(null)
  const [running, setRunning] = useState(false)
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    let cancelled = false

    const loop = async () => {
      if (cancelled || !running || !videoRef.current) return
      const poses = await estimatePoses(videoRef.current)
      if (!cancelled && poses.length > 0) {
        const kp: KPFrame = poses[0].keypoints.map((p: any) => ({
          x: p.x,
          y: p.y,
          score: p.score,
        }))
        setKeypoints(kp)
      }
      rafRef.current = requestAnimationFrame(loop)
    }

    if (running && videoRef.current) {
      loadDetector().then(loop).catch(console.error)
    }

    return () => {
      cancelled = true
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }, [running, videoRef])

  return {
    keypoints,
    running,
    start: () => setRunning(true),
    stop: () => setRunning(false),
  }
}
