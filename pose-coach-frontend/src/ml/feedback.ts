// src/ml/feedback.ts
import type { ScoreSnapshot } from "./types"

/**
 * สร้างคำแนะนำจากคะแนนรายมุมแบบง่าย:
 * - ถ้าคะแนนของมุมนั้นต่ำกว่า threshold → สร้าง tip
 */
export function makeAngleTips(
  angleScores: Record<string, number>,
  locale: "th" | "en" = "th",
  threshold = 70
): string[] {
  const tips: string[] = []
  const t = (th: string, en: string) => (locale === "th" ? th : en)

  for (const [k, s] of Object.entries(angleScores || {})) {
    if (s >= threshold) continue
    switch (k.toLowerCase()) {
      case "kneel":
      case "kneer":
      case "knee":
        tips.push(
          t("งอเข่าให้ลึกขึ้นเล็กน้อยและคุมเข่าให้อยู่แนวปลายเท้า",
            "Bend knees a bit deeper and track them over toes")
        )
        break
      case "hipl":
      case "hipr":
      case "hip":
        tips.push(
          t("ดันสะโพกไปหลัง รักษาหลังตรงเพื่อลดการงอหลัง",
            "Push hips back and keep a neutral spine")
        )
        break
      case "anklel":
      case "ankler":
      case "ankle":
        tips.push(
          t("ถ่ายน้ำหนักให้สมดุล ระวังส้นเท้ายก",
            "Balance your weight and keep heels grounded")
        )
        break
      case "trunk":
      case "back":
        tips.push(
          t("ยืดอก คางถอยเล็กน้อย รักษาหลังตรง",
            "Open chest, tuck chin slightly, keep back neutral")
        )
        break
      default:
        tips.push(
          t(`ปรับมุม ${k} ให้ใกล้เคียงตัวอย่างมากขึ้น`,
            `Adjust ${k} angle closer to the reference`)
        )
    }
  }
  return tips
}

/**
 * รวมคำแนะนำระดับสรุปจากสแน็ปช็อตคะแนน
 * - ถ้า overall ต่ำ → ให้คำแนะนำภาพรวมเพิ่ม
 */
export function makeFeedback(
  snapshot: ScoreSnapshot,
  locale: "th" | "en" = "th"
): string[] {
  if (!snapshot) return []
  const tips = [...(snapshot.angleBreakdown ? makeAngleTips(snapshot.angleBreakdown, locale) : [])]

  const t = (th: string, en: string) => (locale === "th" ? th : en)

  if (snapshot.overall < 60) {
    tips.unshift(
      t("ชะลอจังหวะเล็กน้อย โฟกัสการควบคุมท่าพื้นฐานก่อน",
        "Slow down and focus on controlled fundamental form first")
    )
  } else if (snapshot.overall < 80) {
    tips.unshift(
      t("ดีแล้ว! ลองรักษาจังหวะให้สม่ำเสมอ เพื่อคะแนนที่เสถียรขึ้น",
        "Nice! Keep a steady tempo for more consistent scores")
    )
  }

  // ลบข้อความซ้ำ
  return Array.from(new Set(tips))
}
