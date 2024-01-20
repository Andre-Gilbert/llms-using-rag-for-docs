"""ReAct agent implementation."""
import ast
import json
import logging
import traceback

import requests

from llms.agents.utils import extract, num_tokens_from_messages
from llms.clients.gpt import GPTClient
from llms.rag.coala import CoALA
from llms.rag.faiss import FAISS
from llms.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%d/%m/%y %H:%M:%S")

_RAG_TOOL_INSTRUCTION = """
Available tools:
RAG: use this tool to access additional information from the pandas documentation.

In order to use the tools, just name them in the Action like so 'RAG'.
"""

_CoALA_TOOL_INSTRUCTION = """
Available tools:
CoALA: use this tool to access additional information from the pandas documentation and to access question & code answers.

In order to use the tools, just name them in the Action like so 'CoALA'.
"""

_SYSTEM_PROMPT = """
You are an AI assistant who can write code using pandas.
All necessary code that is part of the answer must be in a single python function called response_function.
If you have one argument given to you in the user prompt, write your response function so that it takes one argument.
If you have two arguments given in the user prompt, write your response function so that it takes two arguments.
Do not write any text or code outside this function when constructing an answer or action.
In the first line of code inside the function please always import pandas as pd or pyarrow as pa, depending on what you need.
Pandas, numpy and pyarrow are the only non-standard packages you are allowed to use.

Always use the following JSON response format:
{
    "Thought": you should always think about what to do
    "Action": "name of the tool OR for code use the following: def response_function(arguments as required by the user prompt):\ncode goes here\"
}
The users system that is interacting with you will then add an observation to the conversation by executing the action.
{"Observation": the result of the action}
... (this Thought/Action/Observation can repeat N times)
When you are confident that you found the final answer and the observation contains positive feedback, answer:
{
    "Thought": I now know the final answer
    "Answer": the final Python code to the original input question
}

Do not write any text or code outside the given JSON framework. Also do not write any observations by yourself.
Always escape special characters to enable parsing with json.loads().
"""


class ReActAgent:
    """Class that implements an LLM-powered AI agent using ReAct (Reasoning + Acting).

    AI agent built on top of the ReAct methodology as proposed in https://arxiv.org/abs/2210.03629.
    This handles the conversation between the user and the LLM by providing and interface
    for the user and parsing the response from the model.

    Attributes:
        llm_client: The client that handles requests.
        tools: RAG/CoALA as tools the AI agent can use.
        rag: RAG/CoALA implementation the AI agent has access to.
        conversation: The conversation history.
        token_limit: The context length of the LLM.
        reasoning: The AI agent Chain-of-Thought.
    """

    def __init__(
        self,
        llm_client: GPTClient,
        tools: dict[str, FAISS | CoALA] | None = None,
        system_prompt: str = _SYSTEM_PROMPT,
        rag: FAISS | CoALA = None,
    ):
        self.llm_client = llm_client
        self.tools = tools
        self.rag = rag
        self.conversation = [
            {
                "role": "system",
                "content": system_prompt
                if tools is None
                else system_prompt + _RAG_TOOL_INSTRUCTION
                if tools is not None and "RAG" in tools
                else system_prompt + _CoALA_TOOL_INSTRUCTION,
            }
        ]
        self.token_limit = 32768 if llm_client.deployment_id == "gpt-4-32k" else 16385  # gpt-35-turbo-16k
        self.reasoning = []

    def _parse_response(self, response: requests.Response.json) -> tuple:
        """Parses the LLM response."""
        try:
            response = json.loads(response, strict=False)
            thought = extract(response, "Thought")
            action = extract(response, "Action")
            answer = extract(response, "Answer")
            parsed = True
            observation = "Your response format was correct."
        except json.decoder.JSONDecodeError as e:
            logging.error(traceback.format_exc())
            parsed = False
            thought = None
            action = None
            answer = None
            observation = f"Your response format was incorrect. Please correct as specified in the first message. The error was: {e}"
        return thought, action, answer, parsed, observation

    def run(self, user_prompt: str) -> str:
        """Runs the AI agent given a user input.

        Args:
            user_prompt: The question/input of the user.

        Returns:
            The AI agent' proposed answer to the question.
        """
        logging.info("User prompt: %s", user_prompt)
        self.reasoning = [{"User prompt": user_prompt}]
        if self.rag is None:
            self.conversation.append({"role": "user", "content": user_prompt})
        else:
            context = self.rag.similarity_search(text=user_prompt)
            if isinstance(self.rag, FAISS):
                context = "\n\n".join([doc for doc, _ in context])
                context = "\n\npandas documentation, sorted by relevancy:\n" + context
            logging.info("Additional information from docs vector store: %s", context)
            self.conversation.append(
                {
                    "role": "user",
                    "content": f"{user_prompt}\n\nUse the following information to solve the user's question. {context}",
                }
            )

        iterations = 0
        while iterations <= settings.AGENT_MAX_ITERATIONS:
            iterations += 1
            # Makes sure the conversation does not exceed the context length limit.
            self._trim_conversation()

            # Now get the response from the LLM
            response = self.llm_client.get_completion(self.conversation)
            response_content = response["choices"][0]["message"]["content"]
            logging.info("API response content: \n%s", response_content)
            thought, action, answer, parsed, observation = self._parse_response(response_content)
            self.conversation.append({"role": "assistant", "content": response_content})

            # Handle AI agent thought
            if thought is not None:
                logging.info("AI agent thought: %s", thought)
                self.reasoning.append({"Thought": thought})
                observation = (
                    "I only expressed a thought. Next up, I will make use of one of my tools or write an answer."
                )

            # Handle AI agent action
            if parsed and action is not None:
                observation = None
                self.reasoning.append({"Tool": action})
                if "RAG" in action:
                    logging.info("AI agent tool: %s", action)
                    docs_result = self.tools["RAG"].similarity_search(text=user_prompt)
                    context = "\n\n".join([doc for doc, _ in docs_result])
                    observation = context
                    self.reasoning.append({"Tool response": context})
                    logging.info("Additional information from vector store: %s", context)
                elif "CoALA" in action:
                    logging.info("AI agent tool: %s", action)
                    context = self.tools["CoALA"].similarity_search(text=user_prompt)
                    observation = context
                    self.reasoning.append({"Tool response": context})
                    logging.info("Additional information from vector store: %s", context)
                else:
                    # Check if the code is valid
                    # Parse code and look for syntax or schema errors.
                    logging.info("Checking code for syntax errors.")
                    code_is_valid, code_error = self._code_is_valid(action)
                    if code_is_valid:
                        logging.info("Code is valid.")
                        observation = "Your response format was correct and the code does not have any syntax errors."
                    else:
                        logging.info("Code is not valid. Error: %s", code_error)
                        observation = (
                            f"Your response format was correct but there seems to be a syntax error: {code_error}"
                        )

            # Handle AI agent answer
            if action is None and answer is not None:
                if isinstance(self.rag, CoALA):
                    # Push result into the long-term memory (i.e. the FAISS code storage)
                    logging.info("Pushing question & answer pair to episodic memory.")
                    self.rag.add_answer_to_code_storage(f"Question: {user_prompt} Final Answer: {answer}")

                logging.info("AI agent final answer: \n%s", answer)
                self.reasoning.append({"Answer": answer})
                return answer

            # Handle environment observation
            logging.info("Appending observation to the conversation history. Observation: %s", observation)
            self.conversation.append({"role": "assistant", "content": f"Observation: {observation}"})

    def _code_is_valid(self, code: str) -> tuple:
        """Checks the code for syntax errors.

        Args:
            code: The Python code to be checked.

        Returns:
            A tuple containing a boolean indicating code validity and an error message (empty string if no error).
        """
        try:
            _ = ast.parse(code, mode="exec")
            is_valid = True
            error_message = ""
        except SyntaxError as e:
            is_valid = False
            error_message = str(e)
        return is_valid, error_message

    def reset_conversation(self) -> None:
        """Resets the conversation."""
        self.conversation = [self.conversation[0]]
        self.llm_client.chat_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.llm_client.embeddings_usage = {"prompt_tokens": 0, "total_tokens": 0}

    def _trim_conversation(self) -> None:
        """Trims the conversation length to prevent LLM context length overflow."""
        conversation_history_tokens = num_tokens_from_messages(self.conversation)
        while conversation_history_tokens + self.llm_client.max_response_tokens >= self.token_limit:
            del self.conversation[1]
            conversation_history_tokens = num_tokens_from_messages(self.conversation)
