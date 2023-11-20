"""AI agent implementation."""
from client import OpenAIClient
from settings import settings
from utils import num_tokens_from_messages


class AIAgent:
    """AI agent class."""

    def __init__(self, model: OpenAIClient, tools: dict, system_prompt: str):
        self.model = model
        self.tools = tools
        self.conversation = [{"role": "system", "content": system_prompt}]

    def run(self, user_prompt: str) -> str:
        """Runs the AI agent given a user input."""
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
