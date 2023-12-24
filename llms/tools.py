"""AI agent tools."""
from llms.rag import FAISS


class Tools:
    """Class that implements the tools of the AI agent.

    Attributes:
        documentation_vector_store: A FAISS index representing the vectors
        code_vector_store: A FAISS index representing
    """

    def __init__(self, documentation_vector_store: FAISS, code_vector_store: FAISS):
        self.documentation_vector_store = documentation_vector_store
        self.code_vector_store = code_vector_store

    def search_documentation(self, user_text: str, k: int) -> list[str]:
        """Returns relevant content from the documentation."""
        return self.documentation_vector_store.similarity_search(user_text, k)

    def search_code(self, user_text: str, k: int) -> list[str]:
        """Returns relevant code snippets."""
        return self.code_vector_store.similarity_search(user_text, k)
