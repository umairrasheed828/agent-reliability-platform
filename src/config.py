from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(override=True)  # this project's .env is authoritative


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""  # optional so imports never crash in CI
    judge_model: str = "gpt-4o-mini"  # the live judge (still on the API)
    judge_temperature: float = 1.25  # P4 calibration constant, applied in prod
    agent_url: str = "http://127.0.0.1:8001"  # the P2 agent service P5 drives


settings = Settings()
