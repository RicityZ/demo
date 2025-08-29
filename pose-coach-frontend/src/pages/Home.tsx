import React, { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"
import { useAppStore } from "../app/store"
import { getExercises, getMeta } from "../services/api"

export default function Home() {
  const nav = useNavigate()
  const set = useAppStore((s) => s.set)
  const [list, setList] = useState<string[]>([])
  const [picked, setPicked] = useState("")
  const [loading, setLoading] = useState(false)

  // โหลดรายการท่าจาก backend
  useEffect(() => {
    getExercises().then(({ exercises }) => setList(exercises || []))
  }, [])

  const start = async () => {
    if (!picked) return
    setLoading(true)

    // โหลด meta ของท่าที่เลือก
    const meta = await getMeta(picked)
    set("exercise", picked)
    set("stats", meta.stats || undefined)
    set("refName", meta.refs?.[0])
    set(
      "videoURL",
      meta.videos?.[0]
        ? `${import.meta.env.VITE_API_BASE}/videos/${meta.videos[0]}`
        : undefined
    )

    setLoading(false)
    nav("/coach")
  }

  return (
    <div style={{ maxWidth: 720, margin: "40px auto", padding: "16px" }}>
      <h2>เลือกท่าออกกำลังกาย</h2>
      <select
        value={picked}
        onChange={(e) => setPicked(e.target.value)}
        style={{ padding: 8, width: "100%", margin: "12px 0" }}
      >
        <option value="" disabled>
          — เลือก —
        </option>
        {list.map((x) => (
          <option key={x} value={x}>
            {x}
          </option>
        ))}
      </select>
      <button
        onClick={start}
        disabled={!picked || loading}
        style={{ padding: "10px 16px" }}
      >
        {loading ? "กำลังโหลด…" : "เริ่มโค้ช"}
      </button>
    </div>
  )
}
