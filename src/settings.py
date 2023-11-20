"""Global settings."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Class that implements the global settings."""

    SERVICE_KEY: dict

    LLM_CONFIG: dict = {
        "deployment_id": "gpt-4-32k",
        "max_response_tokens": 3000,
        "temperature": 0.1,
    }

    AGENT_MAX_RETRIES: int = 3
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_TOKEN_LIMIT: int = 32768

    REQUEST_TIMEOUT: int = 30
    UAA_TOKEN_EXPIRY_THRESHOLD_MINUTES: int = 60

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
