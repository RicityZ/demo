import { useEffect, useState } from "react"

export function useCamera(videoRef: React.RefObject<HTMLVideoElement>) {
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let stream: MediaStream | null = null
    let stopped = false

    async function init(constraints: MediaStreamConstraints) {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error("เบราว์เซอร์ไม่รองรับกล้อง (getUserMedia)")
        }

        stream = await navigator.mediaDevices.getUserMedia(constraints)

        const el = videoRef.current
        if (!el) return

        el.srcObject = stream
        el.muted = true // สำคัญสำหรับ autoplay
        try {
          await el.play()
        } catch (e) {
          // บางทีต้องโต้ตอบผู้ใช้ก่อน (แต่ส่วนใหญ่กับ stream จะเล่นได้)
          console.warn("video.play() needs user gesture:", e)
        }

        if (!stopped) setReady(true)
      } catch (err: any) {
        console.error("Camera error:", err)
        // ถ้าขอความละเอียดสูงเกินไป ให้รีทรายความละเอียดต่ำ
        if (
          err?.name === "OverconstrainedError" ||
          err?.message?.includes("Overconstrained")
        ) {
          try {
            console.log("Retry with 640x480")
            stream = await navigator.mediaDevices.getUserMedia({
              video: { width: 640, height: 480, facingMode: "user" },
              audio: false,
            })
            const el = videoRef.current
            if (el) {
              el.srcObject = stream
              el.muted = true
              await el.play()
            }
            if (!stopped) setReady(true)
            return
          } catch (e: any) {
            console.error("Retry failed:", e)
            setError(e.message || "ไม่สามารถเปิดกล้องได้")
            return
          }
        }

        setError(err?.message || "ไม่สามารถเปิดกล้องได้")
      }
    }

    // ขอที่ 1280x720 ก่อน
    init({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    })

    return () => {
      stopped = true
      if (stream) {
        stream.getTracks().forEach((track) => track.stop())
      }
    }
  }, [videoRef])

  return { ready, error }
}
