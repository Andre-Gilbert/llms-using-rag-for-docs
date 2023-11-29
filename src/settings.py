"""Global settings."""
from pathlib import Path

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
    AGENT_MAX_RESPONSE_TOKENS: int = 2000

    REQUEST_TIMEOUT: int = 30
    UAA_TOKEN_EXPIRY_THRESHOLD_MINUTES: int = 60

    STANDARD_SYSTEM_INSTRUCTION: str = """
    You are an AI assistant who can write code using pandas.
    All necessary code that is part of the answer must be in a single python function called response_function.
    Do not write any text or code outside this function when constructing an answer or action. Assume that pandas
    has already been imported as pd. Pandas is the only non-standard package you are allowed to use.

    Always use the following JSON response format:
    {
        "Question": the input question you must answer
        "Thought": you should always think about what to do
        "Action": "def response_function(arguments as required by the user prompt):\ncode goes here\"
    }
    The users system that is interacting with you will then add an observation to the conversation by executing the action.
    {"Observation": the result of the action}
    ... (this Thought/Action/Observation can repeat N times)
    When you are confident that you found the final answer and the observation contains positive feedback, answer:
    {
        "Thought": I now know the final answer
        "Answer": the final Python code to the original input question
    }

    Do not write any text or code outside the given JSON framework. Also do not write any observations by yourself.
    Always escape special characters to enable parsing with json.loads().
    """

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
settings = Settings()
