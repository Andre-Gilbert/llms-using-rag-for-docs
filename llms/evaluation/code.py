"""Tools used for evaluating AI agents."""
from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Any, Generator

import pandas as pd
from pydantic import BaseModel
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
        id: ID of the test case.
        prompt: The user question.
        data: The data provided as a function argument.
        correct_function: The correct Python function.
    """

    id: int
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
    use_weighted_average_of_text_chunks: list[bool]


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


class TestCaseResult(BaseModel):
    correct: bool
    time_taken: float
    cost: float
    test_case: str
    test_case_output: Any
    test_case_input_data: str
    generated_code: str
    generated_code_output: Any
    agent_error: str | None = None


class Metrics(BaseModel):
    accuracy: float
    total_time: float
    total_cost: float


class Result(BaseModel):
    config: dict
    metrics: Metrics
    details_csv_filepath: str


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
        + f"similarity_search_score_threshold_{config.similarity_search_score_threshold}_"
        + f"text_chunk_size_{config.text_chunk_size}_"
        + f"use_weighted_average_of_text_chunks_{config.use_weighted_average_of_text_chunks}"
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
    index_filename = (
        "episodic_"
        + "embeddings_"
        + f"{config.distance_metric}_"
        + f"similarity_search_score_threshold_{config.similarity_search_score_threshold}_"
        + f"text_chunk_size_{config.text_chunk_size}_"
        + f"use_weighted_average_of_text_chunks_{config.use_weighted_average_of_text_chunks}"
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
    print(type(code_vector_store))
    return CoALA(docs_vector_store, code_vector_store)


def _is_output_equal(desired_result, agent_result):
    if isinstance(desired_result, (pd.DataFrame, pd.Series, pd.Index)):
        return desired_result.equals(agent_result)
    return desired_result == agent_result


def _run_tests(agent: ReActAgent, test_cases: list[CodeTestCase], config: Config):
    num_correct_code = 0
    test_results = []
    for test_case in tqdm(test_cases):
        agent_error = None  # variable to store errors that the agent code produces

        # get response function from agent
        start = time.time()
        final_answer = agent.run(test_case.prompt)
        end = time.time()
        namespace_agent = {}
        exec(final_answer, namespace_agent)
        response_function = namespace_agent["response_function"]

        # get desired result and save it in a variable called data
        data_string = test_case.data
        local_vars = {}
        exec(data_string, globals(), local_vars)

        # retrieve the correct function
        correct_function_string = test_case.correct_function
        namespace_correct = {}
        exec(correct_function_string, namespace_correct)
        correct_function = namespace_correct["correct_function"]

        # execute the correct function with the data as parameter and save it as desired result
        desired_result = correct_function(*[local_vars.get(arg, None) for arg in local_vars])

        # execute the agent function with the data as parameter and save it as agent_result, store error, if agent code produces an error
        try:
            agent_result = response_function(*[local_vars.get(arg, None) for arg in local_vars])
        except Exception as e:
            agent_result = None
            agent_error = e

        # this has to be extended, each time we expect another data type as the desired output
        if _is_output_equal(desired_result, agent_result):
            num_correct_code += 1
            correct = 1
        else:
            correct = 0
        if agent.llm_client.deployment_id == "gpt-4-32k":
            cost = (
                0.06 * (agent.llm_client.chat_usage["prompt_tokens"] / 1000)
                + 0.12 * (agent.llm_client.chat_usage["completion_tokens"] / 1000)
                + 0.0001 * (agent.llm_client.embeddings_usage["prompt_tokens"] / 1000)
            )
        else:
            cost = (
                0.001 * (agent.llm_client.chat_usage["prompt_tokens"] / 1000)
                + 0.002 * (agent.llm_client.chat_usage["completion_tokens"] / 1000)
                + 0.0001 * (agent.llm_client.embeddings_usage["prompt_tokens"] / 1000)
            )
        test_results.append(
            TestCaseResult(
                correct=correct,
                time_taken=end - start,
                cost=cost,
                test_case=test_case.prompt,
                test_case_input_data=test_case.data,
                generated_code=final_answer,
                test_case_output=desired_result,
                generated_code_output=agent_result,
                agent_error=agent_error,
            )
        )
        agent.reset_conversation()
    path = Path(_ROOT_DIR / "results" / "details")
    path.mkdir(exist_ok=True, parents=True)
    filename = filename = (
        f"{config.distance_metric}_"
        + f"similarity_search_score_threshold_{config.similarity_search_score_threshold}_"
        + f"text_chunk_size_{config.text_chunk_size}_"
        + f"use_weighted_average_of_text_chunks_{config.use_weighted_average_of_text_chunks}"
    )
    df = pd.DataFrame([test_result.model_dump() for test_result in test_results])
    df.to_csv(path / f"{filename}_details.csv", index=False)
    return Metrics(
        accuracy=num_correct_code / len(test_cases),
        total_time=df.time_taken.sum(),
        total_cost=df.cost.sum(),
    )


def _save_results_to_csv(results: list[Result]) -> None:
    path = Path(_ROOT_DIR / "results")
    path.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame([result.model_dump() for result in results])
    df.to_csv(path / "results.csv", index=False)


def _get_agent(config: Config, texts):
    if config.retriever == RAGRetriever.RAG:
        rag = _get_rag(_ROOT_DIR / "embeddings" / "semantic", "semantic", config, texts)
        return ReActAgent(
            llm_client=config.llm,
            tools=None,
            rag=rag,
            rag_num_search_results=config.num_search_result,
        )
    elif config.retriever == RAGRetriever.CoALA:
        rag = _get_coala(config, texts)
        return ReActAgent(
            llm_client=config.llm,
            tools=None,
            rag=rag,
            rag_num_search_results=config.num_search_result,
        )
    elif config.retriever == RAGRetriever.RAG_AS_TOOL:
        rag = _get_rag(_ROOT_DIR / "embeddings" / "semantic", "semantic", config, texts)
        tools = {"RAG": rag}
        return ReActAgent(
            llm_client=config.llm,
            tools=tools,
            rag=None,
            rag_num_search_results=config.num_search_result,
        )
    elif config.retriever == RAGRetriever.CoALA_AS_TOOL:
        rag = _get_coala(config, texts)
        tools = {"CoALA": rag}
        return ReActAgent(
            llm_client=config.llm,
            tools=tools,
            rag=None,
            rag_num_search_results=config.num_search_result,
        )
    elif config.retriever == RAGRetriever.NONE:
        return ReActAgent(llm_client=config.llm)


def evaluate_code_generation(config_grid: ConfigGrid, test_cases: list[CodeTestCase]) -> pd.DataFrame:
    results = []
    texts = config_grid.rag.texts
    for config in _get_config(config_grid):
        agent = _get_agent(config, texts)
        metrics = _run_tests(agent, test_cases, config)
        filename = (
            f"{config.distance_metric}_"
            + f"similarity_search_score_threshold_{config.similarity_search_score_threshold}_"
            + f"text_chunk_size_{config.text_chunk_size}_"
            + f"use_weighted_average_of_text_chunks_{config.use_weighted_average_of_text_chunks}"
        )
        results.append(
            Result(
                config={
                    "retriever": config.retriever,
                    "distance_metric": config.distance_metric,
                    "num_search_result": config.num_search_result,
                    "similarity_search_score_threshold": config.similarity_search_score_threshold,
                    "text_chunk_size": config.text_chunk_size,
                    "use_weighted_average_of_text_chunks": config.use_weighted_average_of_text_chunks,
                },
                metrics=metrics,
                details_csv_filepath=f"results/details/{filename}_details.csv",
            ),
        )
    _save_results_to_csv(results)
    return results
