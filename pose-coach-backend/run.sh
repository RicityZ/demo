#!/usr/bin/env bash
set -euo pipefail

# โหลดค่า .env ถ้ามี
if [ -f ".env" ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs -I {} echo {})
fi

# ค่าเริ่มต้น
: "${API_PORT:=8000}"
: "${ENABLE_TEST_MODE:=false}"
: "${CORS_ALLOW_ORIGINS:=*}"

echo "[INFO] API_PORT=${API_PORT}"
echo "[INFO] ENABLE_TEST_MODE=${ENABLE_TEST_MODE}"
echo "[INFO] CORS_ALLOW_ORIGINS=${CORS_ALLOW_ORIGINS}"

# รัน uvicorn
exec uvicorn app.api:app --host 0.0.0.0 --port "${API_PORT}" --proxy-headers

