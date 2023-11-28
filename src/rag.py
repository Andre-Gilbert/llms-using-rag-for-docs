"""Retrieval augmented generation implementation."""
from enum import Enum
from typing import Callable, TypeAlias

import faiss
import numpy as np
import requests
import tiktoken

from utils import batched


class DistanceStrategy(str, Enum):
    """Enumerator of the distance strategies for calculating distances between vectors."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    DOT_PRODUCT = "DOT_PRODUCT"
    JACCARD = "JACCARD"
    COSINE = "COSINE"


EmbeddingFunction: TypeAlias = Callable[[str], requests.Response.json]


class RetrievalAugmentedGeneration:
    """Class that implements RAG."""

    def __init__(
        self,
        texts: list[str],
        embedding_function: EmbeddingFunction,
        chunk_size: int = 512,
        use_weighted_average_of_chunks: bool = True,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        k: int = 4,
        fetch_k: int = 20,
    ):
        self.index = self._create_search_index(texts, chunk_size, use_weighted_average_of_chunks)
        self.embedding_function = embedding_function
        self.distance_strategy = distance_strategy
        self.k = k
        self.fetch_k = fetch_k
        self.doc_store = {}

    def _chunked_tokens(self, text: str, chunk_size: int):
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        chunks_iterator = batched(tokens, chunk_size)
        yield from chunks_iterator

    def _len_safe_get_embedding(
        self,
        text: str,
        chunk_size: int,
        use_weighted_average_of_chunks: bool,
    ):
        chunk_texts = []
        chunk_embeddings = []
        chunk_lens = []
        for chunk in self._chunked_tokens(text, chunk_size):
            chunk_embedding = self.embedding_function(chunk)["data"][0]["embedding"]
            chunk_embeddings.append(chunk_embedding)
            chunk_lens.append(len(chunk))
            chunk_texts.append(chunk)
        if use_weighted_average_of_chunks:
            chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
            chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
            chunk_embeddings = chunk_embeddings.tolist()
        return (
            chunk_embeddings,
            [text] if use_weighted_average_of_chunks else chunk_texts,
        )

    def _create_search_index_from_texts(
        self,
        texts: list[str],
        chunk_size: int,
        use_weighted_average_of_chunks: bool = True,
    ):
        embeddings = []
        documents = []
        for text in texts:
            chunk_embeddings, document = self._len_safe_get_embedding(
                text,
                chunk_size,
                use_weighted_average_of_chunks,
            )
            embeddings.append(chunk_embeddings)
            documents.extend(document)
        self.doc_store = dict(enumerate(documents))
        vectors = np.array(embeddings, dtype=np.float32)
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        return index

    def search(self, query: str) -> str:
        """Gets relevant context based on the user's query."""
        embedding = self.embedding_function(query)["data"][0]["embeddings"]
        vector = np.array([embedding], dtype=np.float32)
        D, I = self.index.search(vector, self.fetch_k)
        results = sorted([{self.doc_store[idx]: score} for idx, score in zip(I[0], D[0]) if idx in self.doc_map])
        return results[: self.k]


class CoALA:
    """Class that implements a cognitive architecture."""

    def __init__(self):
        pass
