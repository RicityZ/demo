// src/ml/dtw.ts

// DTW แบบง่าย ใช้สำหรับปรับ sequence สองชุดให้เทียบกันได้
export function dtwDistance(
  seqA: number[],
  seqB: number[],
  radius = 10
): number {
  const n = seqA.length
  const m = seqB.length
  const INF = 1e9

  const cost = Array.from({ length: n + 1 }, () =>
    Array(m + 1).fill(INF)
  )
  cost[0][0] = 0

  for (let i = 1; i <= n; i++) {
    const jStart = Math.max(1, i - radius)
    const jEnd = Math.min(m, i + radius)
    for (let j = jStart; j <= jEnd; j++) {
      const dist = Math.abs(seqA[i - 1] - seqB[j - 1])
      cost[i][j] = dist + Math.min(cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1])
    }
  }
  return cost[n][m]
}

// คืน mapping ว่าเฟรมไหนใน seqA ตรงกับ seqB
export function dtwPath(
  seqA: number[],
  seqB: number[],
  radius = 10
): [number, number][] {
  const n = seqA.length
  const m = seqB.length
  const INF = 1e9

  const cost = Array.from({ length: n + 1 }, () =>
    Array(m + 1).fill(INF)
  )
  const path: [number, number][] = []
  cost[0][0] = 0

  for (let i = 1; i <= n; i++) {
    const jStart = Math.max(1, i - radius)
    const jEnd = Math.min(m, i + radius)
    for (let j = jStart; j <= jEnd; j++) {
      const dist = Math.abs(seqA[i - 1] - seqB[j - 1])
      cost[i][j] = dist + Math.min(cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1])
    }
  }

  // ไล่ย้อนหา path
  let i = n
  let j = m
  while (i > 0 && j > 0) {
    path.push([i - 1, j - 1])
    if (cost[i - 1][j] < cost[i][j - 1] && cost[i - 1][j] < cost[i - 1][j - 1]) {
      i--
    } else if (cost[i][j - 1] < cost[i - 1][j - 1]) {
      j--
    } else {
      i--
      j--
    }
  }
  return path.reverse()
}
