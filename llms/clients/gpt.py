"""OpenAI GPT client."""
import logging
import time

import requests

from llms.clients.base import BaseLLMClient
from llms.settings import settings


class GPTClient(BaseLLMClient):
    """Class that implements the OpenAI models.

    Attributes:
        client_id: LLM service client id.
        client_secret: LLM service client secret.
        auth_url: LLM service authentication url.
        api_base: LLM service base url.
        access_token: LLM service access token
        access_token_expiry: LLM service access token expiry.
        headers: LLM service request headers.
        llm_deployment_id: GPT model id.
        llm_max_response_token: The maximum number of tokens to generate in the chat completion.
        llm_temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8
            will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        llm_embedding_model: GPT embedding model id. Defaults to text-embedding-ada-002-v2.
        llm_usage: Usage statistics for GPT.
    """

    deployment_id: str
    max_response_tokens: int
    temperature: float
    embedding_model: str = "text-embedding-ada-002-v2"
    chat_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    embeddings_usage: dict = {"prompt_tokens": 0, "total_tokens": 0}

    def get_completion(self, messages: list[dict]) -> requests.Response.json:
        """Creates a model response for the given chat conversation.

        Args:
            messages: A list of messages comprising the conversation so far.

        Returns:
            A chat completion object.
        """
        rate_limit_per_minute = (
            settings.GPT_4_REQUEST_LIMIT_MINUTES
            if self.deployment_id == "gpt-4-32k"
            else settings.GPT_35_REQUEST_LIMIT_MINUTES
        )
        delay = int(60.0 / rate_limit_per_minute) + 1
        logging.info("Waiting %ss to avoid rate limit", delay)
        time.sleep(delay)
        response = self._request_handler(
            api_url=f"{self.api_base}/api/v1/completions",
            data={
                "deployment_id": self.deployment_id,
                "messages": messages,
                "max_tokens": self.max_response_tokens,
                "temperature": self.temperature,
            },
        )
        self.chat_usage["prompt_tokens"] += response["usage"]["prompt_tokens"]
        self.chat_usage["completion_tokens"] += response["usage"]["completion_tokens"]
        self.chat_usage["total_tokens"] += response["usage"]["total_tokens"]
        return response

    def get_embedding(self, text: str) -> requests.Response.json:
        """Creates an embedding vector representing the input text.

        Args:
            text: Input text to embed, encoded as a string.

        Returns:
            A list of embedding objects.
        """
        rate_limit_per_minute = settings.TEXT_ADA_002_REQUEST_LIMIT_MINUTES
        delay = int(60.0 / rate_limit_per_minute) + 1
        logging.info("Waiting %ss to avoid rate limit", delay)
        time.sleep(delay)
        response = self._request_handler(
            api_url=f"{self.api_base}/api/v1/embeddings",
            data={
                "deployment_id": self.embedding_model,
                "input": text.replace("\n", " "),
            },
        )
        self.embeddings_usage["prompt_tokens"] += response["usage"]["prompt_tokens"]
        self.embeddings_usage["total_tokens"] += response["usage"]["total_tokens"]
        return response
