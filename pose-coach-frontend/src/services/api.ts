// src/services/api.ts

const BASE = import.meta.env.VITE_API_BASE

// Utility: เรียก API แล้วแปลงเป็น JSON
async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url, { cache: "no-cache" })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

// ดึงรายชื่อท่าออกกำลังกายทั้งหมด
export function getExercises() {
  return getJSON<{ exercises: string[] }>(`${BASE}/exercises`)
}

// ดึง meta ของท่าที่เลือก (refs, stats, videos)
export function getMeta(exercise: string) {
  return getJSON<{
    refs: string[]
    stats: any
    videos: string[]
    exercise: string
  }>(`${BASE}/assets/${exercise}/meta`)
}

// ดึง reference JSON ของท่าที่เลือก
export function getRef(exercise: string, name: string) {
  return getJSON<any>(`${BASE}/assets/${exercise}/refs/${name}`)
}
