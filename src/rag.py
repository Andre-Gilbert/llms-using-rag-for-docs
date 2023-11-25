"""Retrieval augmented generation implementation."""
from client import OpenAIClient


class RetrievalAugmentedGeneration:
    """Class that implements RAG."""

    def __init__(self, llm_client: OpenAIClient, embeddings: list, num_results: int):
        pass

    def search(self, query: str) -> str:
        """Gets relevant context based on the user's query."""
        pass