"""AI agent implementation."""


class Agent:
    """AI agent class."""

    def __init__(self, model, tools, system_prompt):
        self.model = model
        self.tools = tools
        self.conversation = [{"role": "system", "content": system_prompt}]
        self.steps = []

    def run(self, user_prompt):
        """Runs the AI agent given a user request."""
        self.conversation.append({"role": "user", "content": user_prompt})

    def reset(self):
        """Resets the conversation."""
        self.conversation = [self.conversation[0]]
        self.steps = []
