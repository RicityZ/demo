// src/app/store.ts
import { create } from "zustand"
import type { Stats, RefBundle, ScoreSnapshot, PhaseLabel } from "../ml/types"

// การตั้งค่าทั่วไปของระบบ
type Settings = {
  fps: number
  resolution: "720p" | "1080p"
  lang: "th" | "en"
}

// State หลักของแอป
type AppState = {
  exercise?: string
  refName?: string
  stats?: Stats
  ref?: RefBundle
  videoURL?: string
  phase?: PhaseLabel
  reps: number
  score?: ScoreSnapshot
  settings: Settings

  // ฟังก์ชันแก้ไข state
  set: <K extends keyof AppState>(k: K, v: AppState[K]) => void
}

// สร้าง global store
export const useAppStore = create<AppState>((set) => ({
  reps: 0,
  settings: { fps: 30, resolution: "720p", lang: "th" },
  set: (k, v) => set({ [k]: v } as any)
}))
