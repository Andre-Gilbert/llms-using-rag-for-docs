"""Tools used for evaluating AI agents."""
from enum import Enum
from typing import Generator

from pydantic import BaseModel
from tqdm import tqdm
from tqdm.contrib.itertools import product

from llms.clients import GPTClient
from llms.rag import FAISS, DistanceMetric


class TestCase(BaseModel):
    prompt: str
    data: str
    correct_function: str


class RAGRetriever(str, Enum):
    NO_RAG = "NO_RAG"
    RAG = "RAG"
    CoALA = "CoALA"


class RAG(BaseModel):
    retrievers: list[RAGRetriever]
    distance_metrics: list[DistanceMetric]
    num_search_results: list[int]
    texts: list[str]
    text_chunk_size: list[int]


class ConfigGrid(BaseModel):
    llms: list[GPTClient]
    rag: RAG


class Config(BaseModel):
    llm: GPTClient
    retriever: RAGRetriever
    distance_metric: DistanceMetric
    num_search_result: int
    text_chunk_size: int


def _get_config_from_grid(config_grid: ConfigGrid) -> Generator[Config, None, None]:
    """Gets a config from the config grid using the cartesian product."""
    for (
        agent,
        retriever,
        distance_metric,
        num_search_result,
        text_chunk_size,
    ) in product(
        config_grid.agents,
        config_grid.rag.retrievers,
        config_grid.rag.distance_metrics,
        config_grid.rag.num_search_results,
        config_grid.rag.text_chunk_size,
    ):
        yield Config(
            agent=agent,
            retriever=retriever,
            distance_metric=distance_metric,
            num_search_result=num_search_result,
            text_chunk_size=text_chunk_size,
        )


def evaluate_code_generation(config_grid: ConfigGrid, test_cases: list[TestCase]):
    texts = config_grid.rag.texts
    for config in _get_config_from_grid(config_grid):
        if config.retriever == RAGRetriever.RAG:
            try:
                # load index file
                pass
            except FileNotFoundError:
                rag = FAISS.create_index_from_texts(texts, config.llm)
        elif config.retriever == RAGRetriever.CoALA:
            pass
        for test_case in tqdm(test_cases):
            answer = config.agent.run(test_case.prompt)
