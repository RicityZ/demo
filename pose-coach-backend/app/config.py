# app/config.py
from __future__ import annotations
from pathlib import Path

# Pydantic v2: BaseSettings moved to pydantic-settings
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    # เผื่อกรณีเครื่องยังเป็น pydantic v1 (ไม่แนะนำ) — จะ fallback ให้
    from pydantic import BaseSettings  # type: ignore
    SettingsConfigDict = dict  # dummy เพื่อให้โค้ดรันได้

class Settings(BaseSettings):
    DATA_DIR: Path = Path("./references")
    LOG_LEVEL: str = "INFO"
    API_PORT: int = 8000
    CORS_ALLOW_ORIGINS: str = "*"
    NUM_KEYPOINTS: int = 17
    TARGET_FPS: int = 30
    ENABLE_TEST_MODE: bool = False

    # Pydantic v2 ใช้ model_config; v1 จะมองข้ามไปเอง
    try:
        model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    except Exception:
        # สำหรับ pydantic v1
        class Config:  # type: ignore
            env_file = ".env"
            env_file_encoding = "utf-8"

settings = Settings()

# เผื่อโค้ดเก่าที่ import DATA_DIR ตรงๆ (refs.py ใช้อยู่)
DATA_DIR: Path = settings.DATA_DIR

# สร้างโฟลเดอร์ถ้ายังไม่มี
DATA_DIR.mkdir(parents=True, exist_ok=True)
