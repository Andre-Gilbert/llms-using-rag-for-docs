"""AI agent implementation."""
import requests

from clients import OpenAIClient
from rag import RetrievalAugmentedGeneration
from settings import settings
from utils import num_tokens_from_messages


class AIAgent:
    """AI agent class."""

    def __init__(
        self,
        llm_client: OpenAIClient,
        tools: dict,
        system_prompt: str,
        rag: RetrievalAugmentedGeneration = None,
    ):
        self.llm_client = llm_client
        self.tools = tools
        self.rag = rag
        self.conversation = [{"role": "system", "content": system_prompt}]

    def _parse_response(self, response: requests.Response):
        pass

    def run(self, user_prompt: str) -> str:
        """Runs the AI agent given a user input."""
        if self.rag:
            # retrieve relevant content
            #
            # context = self.rag.search(user_prompt)
            # self.conversation.append({"role": "user", "content": f"{user_prompt} Context: {context}"})
            #
            pass
        else:
            self.conversation.append({"role": "user", "content": user_prompt})
        iterations = 0
        while iterations < settings.AGENT_MAX_ITERATIONS:
            iterations += 1
            self._trim_conversation()

    def reset(self) -> None:
        """Resets the conversation."""
        self.conversation = [self.conversation[0]]

    def _trim_conversation(self) -> None:
        """Trims the conversation length to prevent LLM context length overflow."""
        conversation_history_tokens = num_tokens_from_messages(self.conversation)
        while conversation_history_tokens + settings.AGENT_MAX_RESPONSE_TOKENS >= settings.AGENT_TOKEN_LIMIT:
            del self.conversation[1]
            conversation_history_tokens = num_tokens_from_messages(self.conversation)
