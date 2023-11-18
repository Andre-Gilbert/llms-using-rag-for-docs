"""Global settings."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    AGENT_MAX_RETRIES: int = 3
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_MAX_RESPONSE_TOKENS: int = 3000
    AGENT_TOKEN_LIMIT: int = 32768

    REQUEST_TIMEOUT: int = 30
    UAA_TOKEN_EXPIRY_THRESHOLD_MINUTES: int = 60


settings = Settings()
