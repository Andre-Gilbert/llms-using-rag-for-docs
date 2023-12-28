"""Tools used for evaluating AI agents."""
from enum import Enum
from pathlib import Path
from typing import Generator

from pydantic import BaseModel, Field
from tqdm import tqdm
from tqdm.contrib.itertools import product

from llms.agents.react import ReActAgent
from llms.clients.gpt import GPTClient
from llms.rag.coala import CoALA
from llms.rag.faiss import FAISS, DistanceMetric

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class CodeTestCase(BaseModel):
    """Class that represents a code generation test case.

    Attributes:
        prompt: The user question.
        data: The data provided as a function argument.
        correct_function: The correct Python function.
    """

    prompt: str
    data: str
    correct_function: str


class RAGRetriever(str, Enum):
    """Class that represents the types of RAG retrievers."""

    NONE = "NONE"
    RAG = "RAG"
    RAG_AS_TOOL = "RAG_AS_TOOL"
    CoALA = "CoALA"
    CoALA_AS_TOOL = "CoALA_AS_TOOL"


class RAG(BaseModel):
    retrievers: list[RAGRetriever]
    distance_metrics: list[DistanceMetric]
    num_search_results: list[int]
    similarity_search_score_thresholds: list[float]
    texts: list[str]
    text_chunk_sizes: list[int]
    use_weighted_average_of_text_chunks: list[bool] = Field(default=[False, True], frozen=True)


class ConfigGrid(BaseModel):
    llms: list[GPTClient]
    rag: RAG


class Config(BaseModel):
    llm: GPTClient
    retriever: RAGRetriever
    distance_metric: DistanceMetric
    num_search_result: int
    similarity_search_score_threshold: float
    text_chunk_size: int
    use_weighted_average_of_text_chunks: bool


def _get_config(config_grid: ConfigGrid) -> Generator[Config, None, None]:
    """Gets a config from the config grid using the cartesian product."""
    for (
        llm,
        retriever,
        distance_metric,
        num_search_result,
        similarity_search_score_threshold,
        text_chunk_size,
        use_weighted_average_of_text_chunks,
    ) in product(
        config_grid.llms,
        config_grid.rag.retrievers,
        config_grid.rag.distance_metrics,
        config_grid.rag.num_search_results,
        config_grid.rag.similarity_search_score_thresholds,
        config_grid.rag.text_chunk_sizes,
        config_grid.rag.use_weighted_average_of_text_chunks,
    ):
        yield Config(
            llm=llm,
            retriever=retriever,
            distance_metric=distance_metric,
            num_search_result=num_search_result,
            similarity_search_score_threshold=similarity_search_score_threshold,
            text_chunk_size=text_chunk_size,
            use_weighted_average_of_text_chunks=use_weighted_average_of_text_chunks,
        )


def _get_rag(folder_path, prefix, config, texts) -> FAISS:
    index_filename = (
        f"{prefix}_"
        + "embeddings_"
        + f"{config.distance_metric}_"
        + f"{config.similarity_search_score_threshold}_"
        + f"{config.text_chunk_size}_"
        + "use_weighted_average_of_text_chunks"
        if config.use_weighted_average_of_text_chunks
        else ""
    )
    try:
        rag = FAISS.load_local(folder_path, index_filename, config.llm)
    except RuntimeError:
        normalize_L2 = config.distance_metric == DistanceMetric.COSINE_SIMILARITY
        rag = FAISS.create_index_from_texts(
            texts,
            config.llm,
            similarity_search_score_threshold=config.similarity_search_score_threshold,
            distance_metric=config.distance_metric,
            text_chunk_size=config.text_chunk_size,
            use_weighted_average_of_text_chunks=config.use_weighted_average_of_text_chunks,
            _normalize_L2=normalize_L2,
        )
        rag.save_local(folder_path, index_filename)
    return rag


def _get_coala(config, texts) -> CoALA:
    docs_vector_store = _get_rag(_ROOT_DIR / "embeddings" / "semantic", "semantic", config, texts)
    index_filename = index_filename = (
        "episodic_"
        + "embeddings_"
        + f"{config.distance_metric}_"
        + f"{config.similarity_search_score_threshold}_"
        + f"{config.text_chunk_size}_"
        + "use_weighted_average_of_text_chunks"
        if config.use_weighted_average_of_text_chunks
        else ""
    )
    try:
        code_vector_store = FAISS.load_local(_ROOT_DIR / "embeddings" / "episodic", index_filename, config.llm)
    except RuntimeError:
        normalize_L2 = config.distance_metric == DistanceMetric.COSINE_SIMILARITY
        code_vector_store = FAISS(
            llm_client=config.llm,
            similarity_search_score_threshold=config.similarity_search_score_threshold,
            distance_metric=config.distance_metric,
            text_chunk_size=config.text_chunk_size,
            use_weighted_average_of_text_chunks=config.use_weighted_average_of_text_chunks,
            _normalize_L2=normalize_L2,
        )
    return CoALA(docs_vector_store, code_vector_store)


def _run_tests(agent, test_cases: list[CodeTestCase]):
    for test_case in tqdm(test_cases):
        pass
    return "result"


def evaluate_code_generation(config_grid: ConfigGrid, test_cases: list[CodeTestCase]):
    num_tests = len(test_cases)
    texts = config_grid.rag.texts
    for config in _get_config(config_grid):
        if config.retriever == RAGRetriever.RAG:
            rag = _get_rag(_ROOT_DIR / "embeddings" / "semantic", "semantic", config, texts)
        elif config.retriever == RAGRetriever.CoALA:
            rag = _get_coala(config, texts)
        agent = ReActAgent(llm_client=config.llm, tools=None, rag=rag, rag_num_search_results=config.num_search_result)
        result = _run_tests(agent, test_cases)
