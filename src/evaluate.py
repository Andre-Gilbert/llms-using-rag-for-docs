"""Tools used for evaluating AI agents."""
from typing import Generator

from pydantic import BaseModel
from tqdm import tqdm
from tqdm.contrib.itertools import product

from agent import AIAgent
from rag import FAISS, CoALA, DistanceMetric


class RAG(BaseModel):
    retrievers: list[FAISS | CoALA | None]
    distance_metrics: list[DistanceMetric]
    normalize_L2: list[bool]
    num_search_results: list[int]
    texts: list[str]
    text_chunk_size: list[int]
    use_weighted_average_of_text_chunks: list[bool]


class ConfigGrid(BaseModel):
    agents: list[AIAgent]
    rag: RAG


def get_config_from_grid(config_grid: ConfigGrid) -> Generator[tuple, None, None]:
    """Gets a config from the config grid using the cartesian product."""
    for (
        agent,
        retriever,
        distance_metric,
        normalize_L2,
        num_search_result,
        texts,
        text_chunk_size,
        use_weighted_average_of_text_chunk,
    ) in product(
        config_grid.agents,
        config_grid.rag.retrievers,
        config_grid.rag.distance_metrics,
        config_grid.rag.normalize_L2,
        config_grid.rag.num_search_results,
        config_grid.rag.texts,
        config_grid.rag.text_chunk_size,
        config_grid.rag.use_weighted_average_of_text_chunks,
    ):
        yield (
            agent,
            retriever,
            distance_metric,
            normalize_L2,
            num_search_result,
            texts,
            text_chunk_size,
            use_weighted_average_of_text_chunk,
        )


def evaluate_code_generation(config_grid: ConfigGrid, test_cases: list):
    for config in get_config_from_grid(config_grid):
        (
            agent,
            retriever,
            distance_metric,
            normalize_L2,
            num_search_result,
            texts,
            text_chunk_size,
            use_weighted_average_of_text_chunk,
        ) = config
        for test_case in tqdm(test_cases):
            pass
