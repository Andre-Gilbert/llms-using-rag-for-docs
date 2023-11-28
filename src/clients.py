"""LLM clients."""
from datetime import datetime, timedelta, timezone

import requests

from settings import settings


class LLMClient:
    """Class that implements the LLM service."""

    def __init__(self, service_key: dict):
        self.uaa_client_id = service_key["client_id"]
        self.uaa_client_secret = service_key["client_secret"]
        self.uaa_url = service_key["auth_url"]

        self.uaa_token = None
        self.uaa_token_expiry = None

        self.headers = {"Content-Type": "application/json", "Authorization": None}
        self.service_url = f"{service_key['url']}/api/v1"

    def _get_token(self) -> None:
        """Retrieves the UAA client credentials token."""
        current_time = datetime.now(timezone.utc)
        response = requests.post(
            f"{self.uaa_url}/oauth/token",
            auth=(self.uaa_client_id, self.uaa_client_secret),
            params={"grant_type": "client_credentials"},
            timeout=settings.REQUEST_TIMEOUT,
        )
        self.uaa_token = response.json().get("access_token", None)
        if self.uaa_token is None:
            error = response.json().get("error", None)
            raise ValueError(f"Error while getting UAA token: url={self.service_url}, exception={error}")
        expiry = int(response.json().get("expires_in", None))
        self.uaa_token_expiry = current_time + timedelta(seconds=expiry)
        self.headers["Authorization"] = f"Bearer {self.uaa_token}"


class OpenAIClient(LLMClient):
    """Class that implements the OpenAI models."""

    def __init__(self, service_key: dict, llm_config: dict):
        super().__init__(service_key)

        self.llm_deployment_id = llm_config["deployment_id"]
        self.llm_max_response_tokens = llm_config["max_response_tokens"]
        self.llm_temperature = llm_config["temperature"]
        self.llm_embedding_model = "text-embedding-ada-002-v2"
        self.llm_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_completion(self, messages: list[dict[str, str]]) -> requests.Response.json:
        """Sends a request to the specified LLM."""
        current_time = datetime.now(timezone.utc)
        if (self.uaa_token is None) or (
            current_time - self.uaa_token_expiry < timedelta(minutes=settings.UAA_TOKEN_EXPIRY_THRESHOLD_MINUTES)
        ):
            self._get_token()
        url = f"{self.service_url}/completions"
        data = {
            "deployment_id": self.llm_deployment_id,
            "messages": messages,
            "max_tokens": self.llm_max_response_tokens,
            "temperature": self.llm_temperature,
        }
        response = None
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=settings.REQUEST_TIMEOUT)
            if response.status_code in (401, 403):
                self._get_token()
                response = requests.post(url, headers=self.headers, json=data, timeout=settings.REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as exception:
            raise exception
        response = response.json()
        self.llm_usage["prompt_tokens"] += response["usage"]["prompt_tokens"]
        self.llm_usage["completion_tokens"] += response["usage"]["completion_tokens"]
        self.llm_usage["total_tokens"] += response["usage"]["total_tokens"]
        return response

    def get_embedding(self, text: str) -> requests.Response.json:
        """Gets the embedding for the given text."""
        current_time = datetime.now(timezone.utc)
        if (self.uaa_token is None) or (
            current_time - self.uaa_token_expiry < timedelta(minutes=settings.UAA_TOKEN_EXPIRY_THRESHOLD_MINUTES)
        ):
            self._get_token()
        url = f"{self.service_url}/embeddings"
        text = text.replace("\n", " ")
        data = {"deployment_id": self.llm_embedding_model, "input": text}
        response = None
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=settings.REQUEST_TIMEOUT)
            if response.status_code in (401, 403):
                self._get_token()
                response = requests.post(url, headers=self.headers, json=data, timeout=settings.REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as exception:
            raise exception
        response = response.json()
        self.llm_usage["prompt_tokens"] += response["usage"]["prompt_tokens"]
        self.llm_usage["total_tokens"] += response["usage"]["total_tokens"]
        return response
