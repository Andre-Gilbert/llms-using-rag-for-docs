"""LLM clients."""
from datetime import datetime, timedelta, timezone

import requests

from settings import settings


class OpenAIClient:
    """Class that implements the OpenAI models."""

    def __init__(self, service_key, llm_config):
        self.uaa_client_id = service_key["uaa"]["clientid"]
        self.uaa_client_secret = service_key["uaa"]["clientsecret"]
        self.uaa_url = service_key["uaa"]["url"]

        self.uaa_token = None
        self.uaa_token_expiry = None

        self.headers = {"Content-Type": "application/json", "Authorization": None}
        self.service_url = f"{service_key['url']}/api/v1"

        self.llm_deployment_id = llm_config["DEPLOYMENT_ID"]
        self.llm_max_tokens = llm_config["MAX_TOKENS"]
        self.llm_temperature = llm_config["TEMPERATURE"]

    def _get_token(self) -> str:
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
        return self.uaa_token

    def get_completion(self, messages: list[dict[str, str]]) -> requests.Response:
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
            "max_tokens": self.llm_max_tokens,
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
        return response

    def get_embedding(self, text: str, model: str = "text-embedding-ada-002") -> requests.Response:
        """Gets the embedding for the given text."""
        current_time = datetime.now(timezone.utc)
        if (self.uaa_token is None) or (
            current_time - self.uaa_token_expiry < timedelta(minutes=settings.UAA_TOKEN_EXPIRY_THRESHOLD_MINUTES)
        ):
            self._get_token()
        url = f"{self.service_url}/embeddings"
        data = {"deployment_id": model, "input": text}
        response = None
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=settings.REQUEST_TIMEOUT)
            if response.status_code in (401, 403):
                self._get_token()
                response = requests.post(url, headers=self.headers, json=data, timeout=settings.REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as exception:
            raise exception
        return response
