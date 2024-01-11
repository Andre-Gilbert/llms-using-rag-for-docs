"""Cognitive Architecture for Language Agent (CoALA) implementation."""
from llms.rag.faiss import FAISS


class CoALA:
    """
    Cognitive Architecture for Language Agent (CoALA) implementation as proposed in
    https://arxiv.org/pdf/2309.02427.pdf.
    This builds on the FAISS RAG implementation.
    """

    def __init__(self, docs_storage: FAISS, code_storage: FAISS):
        self.docs = docs_storage
        self.code = code_storage

    def similarity_search(self, text: str) -> str:
        "Returns the similarity search results for both the docs storage and the code storage as a tuple."
        docs_result = self.docs.similarity_search(text=text)
        result = (
            f"Relevant documentation, sorted by similarity of the embedding in descending order:\n{docs_result}\n\n"
        )
        if self.code.index is not None:
            code_result = self.code.similarity_search(text=text)
            result += f"Relevant previous answers with code, sorted by similarity of the embedding in descending order:\n{code_result}"
        return result

    def add_answer_to_code_storage(self, text: str) -> None:
        "Gets a new text of question and correct answer for the code storage."

        # TODO: When running the chat_executor with a while loop around the agent.run(), the storage is renewed every iteration for an unknown reason.
        self.code.add_texts([text])
