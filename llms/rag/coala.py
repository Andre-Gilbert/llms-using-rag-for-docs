"""Cognitive Architecture for Language Agent (CoALA) implementation."""
from llms.rag.faiss import FAISS


class CoALA:
    """Class that implements the CoALA framework.

    Cognitive Architecture for Language Agent (CoALA) implementation as proposed in
    https://arxiv.org/pdf/2309.02427.pdf. This builds on the FAISS RAG implementation.

    Attributes:
        docs_vector_store: Vector store that stores the embedded documents (semantic memory).
        code_vector_store: Vector store that stores question & answer pairs (episodic memory).
    """

    def __init__(self, docs_vector_store: FAISS, code_vector_store: FAISS):
        self.docs_vector_store = docs_vector_store
        self.code_vector_store = code_vector_store

    def similarity_search(self, text: str) -> str:
        """Returns the similarity search results for both the docs storage and the code storage."""
        docs_result = self.docs_vector_store.similarity_search(text=text)
        result = f"Use the additional information to solve the user's question.\n\
        Relevant documentation, sorted by relevancy:\n{docs_result}"
        if self.code_vector_store.index is not None:
            code_result = self.code_vector_store.similarity_search(text=text)
            result += f"\n\nRelevant previous answers with code, sorted by \
        relevancy:\n{code_result}"
        return result

    def add_answer_to_code_storage(self, text: str) -> None:
        """Writes question & answer pairs into the vector store."""
        self.code_vector_store.add_texts([text])
