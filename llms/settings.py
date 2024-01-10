"""Global settings."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Class that implements the global settings."""

    CLIENT_ID: str
    CLIENT_SECRET: str
    AUTH_URL: str
    API_BASE: str

    AGENT_MAX_ITERATIONS: int = 10
    AGENT_TOKEN_LIMIT: int = 32768
    AGENT_MAX_RESPONSE_TOKENS: int = 1000

    API_MAX_RETRIES: int = 3
    API_REQUEST_TIMEOUT_SECONDS: int = 90
    API_MIN_REQUEST_TIMEOUT_SECONDS: int = 4
    API_MAX_REQUEST_TIMEOUT_SECONDS: int = 10
    API_ACCESS_TOKEN_EXPIRY_MINUTES: int = 60

    TOOL_INSTRUCTION: str = """
    Beyond your own knowledge, you can use the following two tools:
    RAG: let's you access additional information from a documentation on pandas.
    CoALA: let's you access question & correct answer pairs and pandas docs as well.

    In order to use the tools, just name them in the Action. You can use one or both at a time, but never use the same
    tool multiple times in the same conversation.
    Bear in mind that in case the RAG tool does not help, the CoALA tool might still contain valuable information.
    """

    STANDARD_SYSTEM_INSTRUCTION: str = """
    You are an AI assistant who can write code using pandas.
    All necessary code that is part of the answer must be in a single python function called response_function.
    If you have one argument given to you in the user prompt, write your response function so that it takes one argument.
    If you have two arguments given in the user prompt, write your response function so that it takes two arguments.
    Do not write any text or code outside this function when constructing an answer or action.
    In the first line of code inside the function please always import pandas as pd or pyarrow as pa, depending on what you need.
    Pandas, numpy and pyarrow are the only non-standard packages you are allowed to use.

    Always use the following JSON response format:
    {
        "Question": the input question you must answer
        "Thought": you should always think about what to do
        "Action": "name of the tool OR for code use the following: def response_function(arguments as required by the user prompt):\ncode goes here\"
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
