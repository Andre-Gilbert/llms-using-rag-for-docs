"""Tools used for evaluating AI agents."""
from __future__ import annotations

import logging
import time
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from llms.agents.react import ReActAgent
from llms.clients.gpt import GPTClient
from llms.rag.coala import CoALA
from llms.rag.faiss import FAISS, DistanceMetric
from llms.settings import settings

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
    """Class that represents the configuration options for RAG."""

    retrievers: list[RAGRetriever]
    distance_metrics: list[DistanceMetric]
    num_search_results: list[int]
    similarity_search_score_thresholds: list[float]
    texts: list[str]
    text_chunk_sizes: list[int]
    use_weighted_average_of_text_chunks: list[bool]


class ConfigGrid(BaseModel):
    """Class that represents the configurations to use for evaluating code generation."""

    llms: list[GPTClient]
    rag: RAG


class Config(BaseModel):
    """Class that represents the current configuration to process during evaluation."""

    llm: GPTClient
    retriever: RAGRetriever
    distance_metric: DistanceMetric | None = None
    num_search_results: int | None = None
    similarity_search_score_threshold: float | None = None
    text_chunk_size: int | None = None
    use_weighted_average_of_text_chunks: bool | None = None


class TestCaseResult(BaseModel):
    """Class that represents the result of running a test case."""

    correct: bool
    time_taken: float
    cost: float
    test_case: str
    test_case_output: Any
    test_case_input_data: str
    generated_code: str
    generated_code_output: Any
    agent_error: str | None = None
    agent_reasoning: list


class Result(BaseModel):
    """Class that represents the results of the configurations."""

    config: dict
    accuracy: float
    total_time: float
    total_cost: float
    details_csv_filepath: str


def _get_configs_from_grid(config_grid: ConfigGrid) -> list[Config]:
    """Gets the configs from the config grid using cartesian product."""
    configs = []

    # Get configurations for no RAG
    index_rag_none_exists = RAGRetriever.NONE in config_grid.rag.retrievers
    if index_rag_none_exists:
        config_grid.rag.retrievers.remove(RAGRetriever.NONE)
        for llm in config_grid.llms:
            configs.append(Config(llm=llm, retriever=RAGRetriever.NONE))

    # Get configurations for RAG
    for (
        llm,
        retriever,
        distance_metric,
        num_search_results,
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
        configs.append(
            Config(
                llm=llm,
                retriever=retriever,
                distance_metric=distance_metric,
                num_search_results=num_search_results,
                similarity_search_score_threshold=similarity_search_score_threshold,
                text_chunk_size=text_chunk_size,
                use_weighted_average_of_text_chunks=use_weighted_average_of_text_chunks,
            )
        )

    return configs


def _get_filename_from_config(config: Config) -> str:
    """Gets the filename given the current configuration that is evaluated."""
    return (
        f"{config.distance_metric}_"
        + f"num_search_results_{config.num_search_results}_"
        + f"score_threshold_{config.similarity_search_score_threshold}_"
        + f"text_chunk_size_{config.text_chunk_size}_"
        + f"use_weighted_average_{config.use_weighted_average_of_text_chunks}"
    )


def _get_rag(folder_path: str, config: Config, texts: list[str]) -> FAISS:
    """Returns a FAISS vector store."""
    index_filename = f"{config.llm.deployment_id}_embeddings_" + _get_filename_from_config(config)
    try:
        logging.info("Loading existing FAISS docs vector store.")
        rag = FAISS.load_local(folder_path, index_filename, config.llm)
    except RuntimeError:
        logging.info("Creating new FAISS docs vector store.")
        normalize_L2 = config.distance_metric == DistanceMetric.COSINE_SIMILARITY
        rag = FAISS.create_index_from_texts(
            texts,
            config.llm,
            num_search_results=config.num_search_results,
            similarity_search_score_threshold=config.similarity_search_score_threshold,
            distance_metric=config.distance_metric,
            text_chunk_size=config.text_chunk_size,
            use_weighted_average_of_text_chunks=config.use_weighted_average_of_text_chunks,
            _normalize_L2=normalize_L2,
        )
        rag.save_local(folder_path, index_filename)
    return rag


def _get_coala(config, texts) -> CoALA:
    """Returns an instance of CoALA"""
    docs_vector_store = _get_rag(f"{_ROOT_DIR}/embeddings/semantic", config, texts)
    index_filename = f"{config.llm.deployment_id}_embeddings_" + _get_filename_from_config(config)
    try:
        logging.info("Loading existing FAISS code vector store")
        code_vector_store = FAISS.load_local(f"{_ROOT_DIR}/embeddings/episodic", index_filename, config.llm)
    except RuntimeError:
        logging.info("Creating new FAISS code vector store.")
        normalize_L2 = config.distance_metric == DistanceMetric.COSINE_SIMILARITY
        code_vector_store = FAISS(
            llm_client=config.llm,
            num_search_results=config.num_search_results,
            similarity_search_score_threshold=config.similarity_search_score_threshold,
            distance_metric=config.distance_metric,
            text_chunk_size=config.text_chunk_size,
            use_weighted_average_of_text_chunks=config.use_weighted_average_of_text_chunks,
            _normalize_L2=normalize_L2,
        )
    return CoALA(docs_vector_store, code_vector_store)


def _is_output_equal(desired_result, agent_result) -> bool:
    """Checks if the generated code output is equal to the correct function."""
    if isinstance(desired_result, (pd.DataFrame, pd.Series, pd.Index)):
        return desired_result.equals(agent_result)
    return desired_result == agent_result


def _calculate_cost(agent: ReActAgent) -> float:
    """Returns the cost in $ for the test case."""
    if agent.llm_client.deployment_id == "gpt-4-32k":
        return (
            0.06 * (agent.llm_client.chat_usage["prompt_tokens"] / 1000)
            + 0.12 * (agent.llm_client.chat_usage["completion_tokens"] / 1000)
            + 0.0001 * (agent.llm_client.embeddings_usage["prompt_tokens"] / 1000)
        )
    elif agent.llm_client.deployment_id == "gpt-35-turbo-16k":
        return (
            0.001 * (agent.llm_client.chat_usage["prompt_tokens"] / 1000)
            + 0.002 * (agent.llm_client.chat_usage["completion_tokens"] / 1000)
            + 0.0001 * (agent.llm_client.embeddings_usage["prompt_tokens"] / 1000)
        )


def _run_tests(agent: ReActAgent, test_cases: list[CodeTestCase], config: Config) -> tuple:
    """Runs the defined test cases."""
    num_correct_code = 0
    test_results = []
    for test_case in tqdm(test_cases, desc="Test cases"):
        logging.info("Running test: %s", test_case.model_dump())
        agent_error = None

        # Get response function from agent
        start = time.time()
        final_answer = agent.run(test_case.prompt)
        end = time.time()

        # Retrieve the generated response function
        namespace_agent = {}
        exec(final_answer, namespace_agent)
        response_function = namespace_agent["response_function"]

        # Get input data
        data_string = test_case.data
        local_vars = {}
        exec(data_string, globals(), local_vars)

        # Retrieve the correct function
        correct_function_string = test_case.correct_function
        namespace_correct = {}
        exec(correct_function_string, namespace_correct)
        correct_function = namespace_correct["correct_function"]

        # Execute correct function with input data
        desired_result = correct_function(*[local_vars.get(arg, None) for arg in local_vars])

        # Execute agent function with input data
        try:
            agent_result = response_function(*[local_vars.get(arg, None) for arg in local_vars])
        # pylint: disable=broad-exception-caught
        except Exception as e:
            agent_result = None
            agent_error = e

        # Check code output for equality
        if _is_output_equal(desired_result, agent_result):
            num_correct_code += 1
            correct = 1
        else:
            correct = 0

        time_taken = end - start
        cost = _calculate_cost(agent)

        logging.info(
            "Results of test case: %s (1=correct, 0=incorrect), %d time taken, %f cost in $",
            correct,
            time_taken,
            cost,
        )
        test_results.append(
            TestCaseResult(
                correct=correct,
                time_taken=time_taken,
                cost=cost,
                test_case=test_case.prompt,
                test_case_input_data=test_case.data,
                generated_code=final_answer,
                test_case_output=desired_result,
                generated_code_output=agent_result,
                agent_error=str(agent_error),
                agent_reasoning=agent.reasoning,
            )
        )

        # Reset conversation history to calculate cost per test case
        agent.reset_conversation()

        rate_limit_per_minute = (
            settings.GPT_4_REQUEST_LIMIT_MINUTES
            if agent.llm_client.deployment_id == "gpt-4-32k"
            else settings.GPT_35_REQUEST_LIMIT_MINUTES
        )
        delay = 60.0 / rate_limit_per_minute
        logging.info("Waiting %ss to avoid rate limit", delay)
        time.sleep(delay)

    # Store details of results
    path = Path(_ROOT_DIR / "results" / "details")
    path.mkdir(exist_ok=True, parents=True)
    filename = f"{config.llm.deployment_id}_{config.retriever}_" + _get_filename_from_config(config)
    logging.info("Saving details of results for test cases to file: %s_details.csv", filename)
    df = pd.DataFrame([test_result.model_dump() for test_result in test_results])
    df.reset_index()
    df.to_csv(path / f"{filename}_details.csv", index=False)

    return (
        num_correct_code / len(test_cases),
        df.time_taken.sum(),
        df.cost.sum(),
    )


def _get_agent(config: Config, texts: list[str]) -> ReActAgent:
    """Returns a ReAct agent based on the config and text to embed."""
    if config.retriever == RAGRetriever.RAG:
        rag = _get_rag(f"{_ROOT_DIR}/embeddings/semantic", config, texts)
        return ReActAgent(llm_client=config.llm, tools=None, rag=rag)
    elif config.retriever == RAGRetriever.CoALA:
        rag = _get_coala(config, texts)
        return ReActAgent(llm_client=config.llm, tools=None, rag=rag)
    elif config.retriever == RAGRetriever.RAG_AS_TOOL:
        rag = _get_rag(f"{_ROOT_DIR}/embeddings/semantic", config, texts)
        tools = {"RAG": rag}
        return ReActAgent(llm_client=config.llm, tools=tools, rag=None)
    elif config.retriever == RAGRetriever.CoALA_AS_TOOL:
        rag = _get_coala(config, texts)
        tools = {"CoALA": rag}
        return ReActAgent(llm_client=config.llm, tools=tools, rag=None)
    elif config.retriever == RAGRetriever.NONE:
        return ReActAgent(llm_client=config.llm)


def evaluate_code_generation(config_grid: ConfigGrid, test_cases: list[CodeTestCase]) -> pd.DataFrame:
    results = []
    texts = config_grid.rag.texts
    for config in tqdm(_get_configs_from_grid(config_grid), desc="Configurations"):
        current_config = {
            "llm": config.llm.deployment_id,
            "retriever": config.retriever,
            "distance_metric": config.distance_metric,
            "num_search_results": config.num_search_results,
            "similarity_search_score_threshold": config.similarity_search_score_threshold,
            "text_chunk_size": config.text_chunk_size,
            "use_weighted_average_of_text_chunks": config.use_weighted_average_of_text_chunks,
        }
        logging.info("Current configuration: %s", current_config)
        agent = _get_agent(config, texts)
        accuracy, total_time, total_cost = _run_tests(agent, test_cases, config)
        logging.info(
            "Configuration results: %s accuracy, %d total time taken, %f total cost in $",
            accuracy,
            total_time,
            total_cost,
        )
        filename = f"{config.llm.deployment_id}_{config.retriever}_" + _get_filename_from_config(config)
        results.append(
            Result(
                config=current_config,
                accuracy=accuracy,
                total_cost=total_cost,
                total_time=total_time,
                details_csv_filepath=f"results/details/{filename}_details.csv",
            ),
        )

        # Store episodic memory after run
        if isinstance(agent.rag, CoALA):
            agent.rag.code_vector_store.save_local(
                folder_path=f"{_ROOT_DIR}/embeddings/episodic",
                index_filename=f"{config.llm.deployment_id}_embeddings_" + _get_filename_from_config(config),
            )

    # Store results
    path = Path(_ROOT_DIR / "results")
    path.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame([result.model_dump() for result in results])
    df = df.sort_values(by=["accuracy", "total_cost", "total_time"], ascending=[False, True, True])
    df.reset_index()
    df.to_csv(path / "results.csv", index=False)
    logging.info("Saving details of results for test cases to file: results.csv")

    return results
