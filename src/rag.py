"""Retrieval augmented generation implementations."""
from __future__ import annotations

import logging
import operator
import pickle
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np
import requests
from pydantic import BaseModel

from utils import chunked_tokens, get_text_from_tokens


class DistanceMetric(str, Enum):
    """Enumerator of the distance metrics for calculating distances between vectors."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"


class FAISS(BaseModel):
    """Class that implements RAG using Meta FAISS.

    Attributes:
        embedding_function: Embeddings to use when generating queries.
        index: The FAISS index.
        documents: Mapping of indices to document.
        num_search_results: Number of documents to return per similarity search.
        similarity_search_score_threshold: The similarity score for a document to be included in the search results.
        distance_metric: The distance metric for calculating distances between vectors.
        text_chunk_size: Divides the input text into chunks of the specified size.
        use_weighted_average_of_text_chunks: Whether the weighted average of the chunk embeddings should be used.
            Defaults to False.
        _normalize_L2: Whether the vectors should be normalized before storing.
    """

    embedding_function: Any
    index: Any = None
    documents: dict = {}
    num_search_results: int = 4
    similarity_search_score_threshold: float = 0.0
    distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN_DISTANCE
    text_chunk_size: int = 512
    use_weighted_average_of_text_chunks: bool = False
    _normalize_L2: bool = False

    def _len_safe_get_embedding(self, text: str) -> tuple:
        """Embeds the given text.

        Please refer to https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb.

        Args:
            text: The text to embed.

        Returns:
            A tuple containing the text and embedding chunks.
        """
        chunk_texts = []
        chunk_embeddings = []
        chunk_lens = []
        for chunk in chunked_tokens(text, self.text_chunk_size):
            chunk_text = get_text_from_tokens(chunk)
            chunk_embedding = self.embedding_function(chunk_text)["data"][0]["embedding"]
            chunk_embeddings.append(chunk_embedding)
            chunk_lens.append(len(chunk))
            chunk_texts.append(chunk_text)
        if self.use_weighted_average_of_text_chunks:
            chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
            chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
            chunk_embeddings = chunk_embeddings.tolist()
        return (
            [text] if self.use_weighted_average_of_text_chunks else chunk_texts,
            chunk_embeddings,
        )

    def _embed_texts(self, texts: list[str]) -> tuple:
        """Embeds texts using the initialized embedding function.

        Args:
            texts: A list of texts to embed.

        Returns:
            A tuple containing the documents and embeddings.
        """
        documents = []
        embeddings = []
        for text in texts:
            chunk_texts, chunk_embeddings = self._len_safe_get_embedding(text)
            for text, embedding in zip(chunk_texts, chunk_embeddings):
                documents.append(text)
                embeddings.append(embedding)
        return documents, embeddings

    def add_texts(self, texts: list[str]) -> None:
        """Adds texts to the FAISS index."""
        documents, embeddings = self._embed_texts(texts)
        vectors = np.array(embeddings, dtype=np.float32)
        if self.distance_metric == DistanceMetric.MAX_INNER_PRODUCT:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        else:
            self.index = faiss.IndexFlatL2(vectors.shape[1])
        if self._normalize_L2:
            faiss.normalize_L2(vectors)
        self.index.add(vectors)
        if not self.documents:
            self.documents = dict(enumerate(documents))
        else:
            index = len(documents)
            for document in documents:
                self.documents[index] = document
                index += 1

    @classmethod
    def create_index_from_texts(
        cls,
        texts: list[str],
        embedding_function: Callable[[str], requests.Response.json],
        **kwargs: dict[str, Any],
    ) -> FAISS:
        """Creates a FAISS index from texts.

        Args:
            texts: A list of texts used for creating the FAISS index.
            embedding_function: Embeddings to use when generating queries.
            **kwargs:
                num_search_results: Number of documents to return per similarity search.
                similarity_search_score_threshold: The similarity score for a document
                    to be included in the search results.
                distance_metric: The distance metric for calculating distances between vectors.
                text_chunk_size: Divides the input text into chunks of the specified size.
                use_weighted_average_of_text_chunks: Whether the weighted average of the
                    chunk embeddings should be used. Defaults to False.
                normalize_L2: Whether the vectors should be normalized before storing.

        Returns:
            An instance of the FAISS index.
        """
        vector_store = cls(embedding_function=embedding_function, **kwargs)
        if vector_store.distance_metric != DistanceMetric.EUCLIDEAN_DISTANCE and vector_store._normalize_L2:
            logging.warning(
                "Normalizing L2 is not applicable for metric type: %s. Setting normalize L2 to False.",
                vector_store.distance_metric,
            )
            vector_store._normalize_L2 = False
        vector_store.add_texts(texts)
        return vector_store

    def save_local(self, folder_path: str, index_name: str) -> None:
        """Saves FAISS index and documents to disk.

        Args:
            folder_path: The folder path to save index and documents to.
            index_name: The index filename.
        """
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)
        faiss.write_index(self.index, str(path / f"{index_name}.faiss"))
        with open(path / f"{index_name}.pkl", "wb") as file:
            pickle.dump(
                (
                    self.documents,
                    self.num_search_results,
                    self.similarity_search_score_threshold,
                    self.distance_metric,
                    self.text_chunk_size,
                    self.use_weighted_average_of_text_chunks,
                    self._normalize_L2,
                ),
                file,
            )

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        index_name: str,
        embedding_function: Callable[[str], requests.Response.json],
    ) -> FAISS:
        """Loads FAISS index and documents from disk.

        Args:
            folder_path: The folder path to load index and documents from.
            index_name: The index filename.
            embedding_function: Embeddings to use when generating queries.

        Returns:
            An instance of the FAISS index.
        """
        path = Path(folder_path)
        index = faiss.read_index(str(path / f"{index_name}.faiss"))
        with open(path / f"{index_name}.pkl", "rb") as file:
            (
                documents,
                num_search_results,
                similarity_search_score_threshold,
                distance_metric,
                text_chunk_size,
                use_weighted_average_of_text_chunks,
                normalize_L2,
            ) = pickle.load(file)
        return cls(
            embedding_function=embedding_function,
            index=index,
            documents=documents,
            num_search_results=num_search_results,
            similarity_search_score_threshold=similarity_search_score_threshold,
            distance_metric=distance_metric,
            text_chunk_size=text_chunk_size,
            use_weighted_average_of_text_chunks=use_weighted_average_of_text_chunks,
            _normalize_L2=normalize_L2,
        )

    def similarity_search(self, query: str) -> list[tuple[str, float]]:
        """Gets relevant context.

        Args:
            query: The AI agent user input.

        Returns:
            A list of documents most similar to the query text and L2 distance in float for each.
        """
        embedding = self.embedding_function(query)["data"][0]["embedding"]
        vector = np.array([embedding], dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, self.num_search_results)
        documents = [
            (self.documents[index], score) for index, score in zip(indices[0], scores[0]) if index in self.documents
        ]
        if self.similarity_search_score_threshold:
            cmp = operator.ge if self.distance_metric == DistanceMetric.MAX_INNER_PRODUCT else operator.le
            documents = [
                (document, score) for document, score in documents if cmp(score, self.similarity_search_score_threshold)
            ]
        return documents


class CoALA:
    """Class that implements a cognitive architecture."""

    def __init__(self):
        pass
