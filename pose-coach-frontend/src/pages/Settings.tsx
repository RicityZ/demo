import React from "react"
import { useAppStore } from "../app/store"

export default function Settings() {
  const { settings, set } = useAppStore()

  return (
    <div style={{ maxWidth: 600, margin: "40px auto", padding: "16px" }}>
      <h2>การตั้งค่า</h2>
      <div style={{ margin: "16px 0" }}>
        <label>
          FPS:{" "}
          <input
            type="number"
            value={settings.fps}
            onChange={(e) =>
              set("settings", { ...settings, fps: +e.target.value })
            }
          />
        </label>
      </div>
      <div style={{ margin: "16px 0" }}>
        <label>
          ความละเอียด:{" "}
          <select
            value={settings.resolution}
            onChange={(e) =>
              set("settings", {
                ...settings,
                resolution: e.target.value as "720p" | "1080p",
              })
            }
          >
            <option value="720p">720p</option>
            <option value="1080p">1080p</option>
          </select>
        </label>
      </div>
    </div>
  )
}
