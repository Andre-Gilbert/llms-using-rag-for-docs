"""Retrieval augmented generation implementation."""
from clients import OpenAIClient


class RetrievalAugmentedGeneration:
    """Class that implements RAG."""

    def __init__(self, llm_client: OpenAIClient, embeddings: list, num_results: int = 3):
        pass

    def search(self, query: str) -> str:
        """Gets relevant context based on the user's query."""
        pass
