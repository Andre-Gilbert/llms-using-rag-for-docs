"""LLM clients."""
from datetime import datetime, timedelta, timezone

import requests
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from settings import settings


class LLMClient(BaseModel):
    """Class that implements the LLM client."""

    client_id: str
    client_secret: str
    auth_url: str
    api_base: str

    access_token: str or None = None
    access_token_expiry: str or None = None
    headers: dict = {
        "Content-Type": "application/json",
        "Authorization": None,
    }

    def _fetch_access_token(self) -> None:
        """Fetches the access token."""
        current_time = datetime.now(timezone.utc)
        response = requests.post(
            f"{self.auth_url}/oauth/token",
            auth=(self.client_id, self.client_secret),
            params={"grant_type": "client_credentials"},
            timeout=settings.API_MAX_REQUEST_TIMEOUT_SECONDS,
        )
        response = response.json()
        self.access_token = response.get("access_token", None)
        if self.access_token is None:
            error = response.get("error", None)
            raise ValueError(f"Error while getting access token: url={self.auth_url}, exception={error}")
        expiry = int(response.get("expires_in", None))
        self.access_token_expiry = current_time + timedelta(seconds=expiry)
        self.headers["Authorization"] = f"Bearer {self.access_token}"

    def _access_token_expired_or_missing(self) -> bool:
        """Checks if the access token exists or has expired."""
        current_time = datetime.now(timezone.utc)
        return (self.access_token is None) or (
            current_time - self.access_token_expiry < timedelta(minutes=settings.API_ACCESS_TOKEN_EXPIRY_MINUTES)
        )

    @retry(
        stop=stop_after_attempt(settings.API_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=settings.API_MIN_REQUEST_TIMEOUT_SECONDS,
            max=settings.API_MAX_REQUEST_TIMEOUT_SECONDS,
        ),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def _request_handler(self, api_url: str, data: dict) -> requests.Response.json:
        """Handles the request to the LLM service.

        Args:
            api_url: Completions or embeddings URL.
            data: JSON object to send to the specified URL.

        Returns:
            JSON response content.

        Raises:
            RequestException: An error that occurred while handling the API request.
        """
        if self._access_token_expired_or_missing():
            self._fetch_access_token()
        try:
            response = requests.post(
                api_url, headers=self.headers, json=data, timeout=settings.API_MAX_REQUEST_TIMEOUT_SECONDS
            )
            if response.status_code in (401, 403):
                self._fetch_access_token()
                response = requests.post(
                    api_url, headers=self.headers, json=data, timeout=settings.API_MAX_REQUEST_TIMEOUT_SECONDS
                )
        except requests.exceptions.RequestException as exception:
            raise exception
        return response.json()


class GPTClient(LLMClient):
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

    llm_deployment_id: str
    llm_max_response_tokens: int
    llm_temperature: float
    llm_embedding_model: str = "text-embedding-ada-002-v2"
    llm_usage: dict = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def get_completion(self, messages: list[dict]) -> requests.Response.json:
        """Creates a model response for the given chat conversation.

        Args:
            messages: A list of messages comprising the conversation so far.

        Returns:
            A chat completion object.
        """
        response = self._request_handler(
            api_url=f"{self.api_base}/api/v1/completions",
            data={
                "deployment_id": self.llm_deployment_id,
                "messages": messages,
                "max_tokens": self.llm_max_response_tokens,
                "temperature": self.llm_temperature,
            },
        )
        self.llm_usage["prompt_tokens"] += response["usage"]["prompt_tokens"]
        self.llm_usage["completion_tokens"] += response["usage"]["completion_tokens"]
        self.llm_usage["total_tokens"] += response["usage"]["total_tokens"]
        return response

    def get_embedding(self, text: str) -> requests.Response.json:
        """Creates an embedding vector representing the input text.

        Args:
            text: Input text to embed, encoded as a string.

        Returns:
            A list of embedding objects.
        """
        response = self._request_handler(
            api_url=f"{self.api_base}/api/v1/embeddings",
            data={
                "deployment_id": self.llm_embedding_model,
                "input": text.replace("\n", " "),
            },
        )
        self.llm_usage["prompt_tokens"] += response["usage"]["prompt_tokens"]
        self.llm_usage["total_tokens"] += response["usage"]["total_tokens"]
        return response
