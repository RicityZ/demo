// src/ml/detector.ts
import * as posedetection from "@tensorflow-models/pose-detection"
import * as tf from "@tensorflow/tfjs-core"
import "@tensorflow/tfjs-backend-webgl"

let detector: posedetection.PoseDetector | null = null

// โหลด MoveNet Lightning model
export async function loadDetector() {
  if (detector) return detector

  // ตั้งค่า backend เป็น WebGL
  await tf.setBackend("webgl")
  await tf.ready()

  detector = await posedetection.createDetector(
    posedetection.SupportedModels.MoveNet,
    {
      modelType: posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
      enableSmoothing: true,
    }
  )

  return detector
}

// ประเมิน keypoints จาก <video>
export async function estimatePoses(video: HTMLVideoElement) {
  if (!detector) await loadDetector()
  return detector!.estimatePoses(video, { flipHorizontal: false })
}
