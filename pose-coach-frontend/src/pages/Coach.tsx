// src/pages/Coach.tsx
import React, { useEffect, useRef, useState } from "react"
import { useAppStore } from "../app/store"
import { getRef } from "../services/api"

import CameraView from "../components/CameraView"
import Toolbar from "../components/Toolbar"
import ScorePanel from "../components/ScorePanel"
import FeedbackPanel from "../components/FeedbackPanel"
import VideoDemo from "../components/VideoDemo"

import { usePoseStream } from "../hooks/usePoseStream"   // tracking (keypoints)
import { useCoachLoop } from "../hooks/useCoachLoop"     // analysis (score/feedback)
import { drawSkeleton } from "../utils/draw"

export default function Coach() {
  const { exercise, refName, videoURL, set } = useAppStore()

  // --- refs & states ---
  const videoRef  = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const [camReady, setCamReady]   = useState(false)
  const [camError, setCamError]   = useState<string | null>(null)
  const [refData, setRefData]     = useState<any>(null)      // reference + stats จาก backend
  const [isRunning, setIsRunning] = useState(false)          // สวิตช์เริ่ม/หยุด "analysis"

  // ---------- โหลด reference/stats/media จาก backend (คงระบบเดิม) ----------
  useEffect(() => {
    if (!exercise || !refName) return
    ;(async () => {
      try {
        const data = await getRef(exercise, refName)
        setRefData(data)
        set("ref", data)     // เก็บลง store ตามเดิมถ้าส่วนอื่นใช้
      } catch {
        // optional: โชว์ banner error ได้ตามต้องการ
      }
    })()
  }, [exercise, refName, set])

  // ---------- เปิดกล้อง ----------
  useEffect(() => {
    let stream: MediaStream | null = null
    const open = async () => {
      setCamError(null)
      try {
        const s = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720, frameRate: 30, facingMode: "user" },
          audio: false,
        })
        stream = s
        if (videoRef.current) {
          videoRef.current.muted = true
          ;(videoRef.current as any).srcObject = s
          await videoRef.current.play()
          setCamReady(true)
        }
      } catch (e: any) {
        setCamError(`${e?.name || "CameraError"} — ${e?.message || ""}`)
      }
    }
    open()
    return () => stream?.getTracks().forEach(t => t.stop())
  }, [])

  // ปรับขนาด canvas ให้เท่ากับสตรีมจริง (กันเพี้ยนสเกล)
  useEffect(() => {
    const v = videoRef.current
    const c = canvasRef.current
    if (!v || !c) return
    const setSize = () => {
      if (v.videoWidth && v.videoHeight) {
        c.width  = v.videoWidth
        c.height = v.videoHeight
      }
    }
    v.addEventListener("loadedmetadata", setSize)
    setSize()
    return () => v.removeEventListener("loadedmetadata", setSize)
  }, [camReady])

  // ---------- TRACKING: ให้ skeleton ขึ้นและ "ขยับ" ทันที ----------
  const { keypoints, start: startTrack, stop: stopTrack } = usePoseStream(videoRef as any)

  // เริ่ม tracking อัตโนมัติเมื่อกล้องพร้อม (เฉพาะโชว์โครงกระดูก)
  useEffect(() => {
    if (!camReady) return
    startTrack?.()
    return () => stopTrack?.()
  }, [camReady, startTrack, stopTrack])

  // วาดด้วย requestAnimationFrame เพื่อไม่ค้าง
  const kpRef = useRef<any>(null)
  useEffect(() => { kpRef.current = keypoints }, [keypoints])

  useEffect(() => {
    let raf = 0
    const loop = () => {
      const c = canvasRef.current
      const ctx = c?.getContext("2d")
      if (c && ctx) {
        ctx.clearRect(0, 0, c.width, c.height)
        const kp = kpRef.current
        if (kp) drawSkeleton(ctx, kp as any)
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [])

  // ---------- ANALYSIS: เริ่ม/หยุดด้วยปุ่ม ----------
  // ไม่บังคับว่าต้องมี stats ก่อนค่อยเริ่ม; ถ้ายังไม่มา useCoachLoop ควรรอเองได้
  const { snapshot, phase, reps } = useCoachLoop({
    keypoints: isRunning ? (kpRef.current as any) : null,  // ยังไม่เริ่ม → ส่ง null เพื่อไม่คำนวณ
    stats: refData?.stats ?? null,                         // ถ้ายังไม่มา → null
  } as any)

  // ปุ่มควบคุม
  const handleStart = () => {
    if (!camReady) return            // ต้องมีกล้องพร้อมอย่างน้อย
    setIsRunning(true)               // เปิดลูปวิเคราะห์
  }
  const handleStop  = () => setIsRunning(false)
  const handleReset = () => {
    setIsRunning(false)
    // TODO: ถ้าต้องล้างค่าใน store (คะแนน/เฟส/ครั้ง) ให้ทำที่นี่
  }

  return (
    <div style={{ maxWidth: 1100, margin: "24px auto", padding: "16px" }}>
      <h2>Coach</h2>

      {camError && (
        <div style={{ margin: "8px 0", color: "#ffb4b4" }}>
          เปิดกล้องไม่ได้: {camError} — ตรวจสิทธิ์กล้อง/ใช้ https หรือ localhost
        </div>
      )}

      <div style={{ display: "grid", gap: 16, gridTemplateColumns: "2fr 1fr" }}>
        {/* ซ้าย: กล้อง + โครงกระดูก + ปุ่ม */}
        <div style={{ border: "1px solid #222", borderRadius: 12, padding: 12, position: "relative", overflow: "hidden" }}>
          <div style={{ position: "relative" }}>
            <CameraView ref={videoRef} />
            <canvas
              ref={canvasRef}
              style={{
                width: "100%",
                height: "auto",
                position: "absolute",
                top: 0,
                left: 0,
                pointerEvents: "none",
              }}
            />
          </div>

          <div style={{ marginTop: 12 }}>
            <Toolbar
              onStart={handleStart}
              onStop={handleStop}
              onReset={handleReset}
              isRunning={isRunning}
            />
          </div>
        </div>

        {/* ขวา: คะแนน + Feedback + วิดีโอเดโม */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <ScorePanel    score={snapshot || undefined} />
          <FeedbackPanel score={snapshot || undefined} />
          <VideoDemo src={videoURL || undefined} />
          {(phase !== undefined || reps !== undefined) && (
            <div style={{ fontSize: 12, opacity: 0.75 }}>
              Phase: {String(phase)} | Reps: {String(reps)}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
