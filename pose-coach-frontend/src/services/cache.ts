// src/services/cache.ts

// เก็บ cache ใน memory ชั่วคราว
const memoryCache = new Map<string, any>()

// ดึงข้อมูลจาก cache ถ้ามี
export function getCache<T>(key: string): T | undefined {
  return memoryCache.get(key)
}

// บันทึกข้อมูลลง cache
export function setCache<T>(key: string, value: T) {
  memoryCache.set(key, value)
}

// ลบข้อมูลจาก cache
export function clearCache(key: string) {
  memoryCache.delete(key)
}

// ลบ cache ทั้งหมด
export function clearAllCache() {
  memoryCache.clear()
}
