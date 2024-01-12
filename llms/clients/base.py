"""Base LLM client."""
import logging
from datetime import datetime, timedelta, timezone

import requests
from pydantic import BaseModel
from tenacity import (
    after_log,
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from llms.settings import settings

logger = logging.getLogger(__name__)


class BaseLLMClient(BaseModel):
    """Class that implements the LLM client."""

    client_id: str
    client_secret: str
    auth_url: str
    api_base: str

    access_token: str | None = None
    access_token_expiry: str | None = None
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
            timeout=settings.API_REQUEST_TIMEOUT_SECONDS,
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
        wait=wait_random_exponential(
            min=settings.API_MIN_RETRY_TIMEOUT_SECONDS,
            max=settings.API_MAX_RETRY_TIMEOUT_SECONDS,
        ),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        before=before_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
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
                api_url,
                headers=self.headers,
                json=data,
                timeout=settings.API_REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code in (401, 403):
                self._fetch_access_token()
                response = requests.post(
                    api_url,
                    headers=self.headers,
                    json=data,
                    timeout=settings.API_REQUEST_TIMEOUT_SECONDS,
                )
        except requests.exceptions.RequestException as e:
            raise e
        return response.json()
