// src/ml/types.ts

// Keypoint หนึ่งจุด
export type KP = { x: number; y: number; score?: number }

// เฟรมของ keypoints (คาดหวัง 17 จุดต่อเฟรมสำหรับ MoveNet)
export type KPFrame = KP[]

// ป้ายเฟสการเคลื่อนไหว
export type PhaseLabel = "start" | "down" | "bottom" | "up"

// สถิติอ้างอิง (ใช้คำนวณ z-score)
export type Stats = {
  version?: string
  mu: Record<string, number>        // ค่าเฉลี่ยต่อคีย์ เช่น "knee"
  sigma: Record<string, number>     // ส่วนเบี่ยงเบนมาตรฐาน
  phaseWeights?: Record<PhaseLabel, number>
  angleWeights?: Record<string, number>
}

// ข้อมูล reference ที่ดึงจาก backend (ยืดหยุ่นตามไฟล์จริง)
export type RefBundle = {
  meta?: any
  // ตัวอย่าง: มุมต่อเฟรมของ reference, key เช่น "knee", "hip" เป็นต้น
  angles?: Record<string, number[]>
  // ป้ายเฟสของ reference ต่อเฟรม
  phases?: PhaseLabel[]
  // ฟิลด์อื่น ๆ สามารถเพิ่มภายหลังได้ (bone vectors ฯลฯ)
  [k: string]: any
}

// คะแนนสรุปหนึ่งสแน็ปช็อต (อัปเดตทุก ~100–300ms)
export type ScoreSnapshot = {
  overall: number        // 0–100
  angle: number          // 0–100 (เฉลี่ยรายมุมหลังถ่วงน้ำหนัก)
  shape: number          // 0–100 (โครงสร้าง/เวกเตอร์)
  // รายละเอียดคะแนนรายมุม เพื่อใช้ทำ feedback เชิงมุม
  angleBreakdown?: Record<string, number>  // เช่น { knee: 72, hip: 65, ... }
  tips?: string[]         // คำแนะนำที่สร้างแล้ว
}
