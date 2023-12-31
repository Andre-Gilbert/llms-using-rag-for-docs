"""ReAct agent implementation."""
import ast
import json
import logging

import requests

from llms.agents.utils import extract, num_tokens_from_messages
from llms.clients.gpt import GPTClient
from llms.rag.coala import CoALA
from llms.rag.faiss import FAISS
from llms.settings import settings

logging.basicConfig(level=logging.DEBUG, format="%(process)d - %(levelname)s - %(message)s")


class ReActAgent:
    """
    AI agent built on top of the ReAct methodology as proposed in https://arxiv.org/abs/2210.03629.
    This handles the conversation between the user and the LLM by providing and interface
    for the user and parsing the response from the model. It uses the Chat class to
    keep track of the context/chat history.
    """

    def __init__(
        self,
        llm_client: GPTClient,
        tools: dict[FAISS, CoALA] or None = None,
        system_prompt: str = settings.STANDARD_SYSTEM_INSTRUCTION,
        rag: FAISS or CoALA = None,
        rag_num_search_results: int or None = 3,
    ):
        self.llm_client = llm_client
        self.tools = tools
        self.rag = rag
        self.rag_num_search_results = rag_num_search_results
        self.conversation = [{"role": "system", "content": system_prompt if self.tools is None else system_prompt + settings.TOOL_INSTRUCTION}]

    def _parse_response(self, response: requests.Response) -> tuple:
        try:
            response = json.loads(response, strict=False)
            thought = extract(response, "Thought")
            action = extract(response, "Action")
            answer = extract(response, "Answer")

            parsed = True
            observation = {"Observation": "Your response format was correct."}

        except Exception as e:
            import traceback

            logging.error(traceback.format_exc())
            parsed = False
            thought = None
            action = None
            answer = None
            observation = {
                "Observation": f"Your response format was incorrect. Please correct as specified in the first message. The error was: {e}"
            }

        return thought, action, answer, parsed, observation

    def run(self, user_prompt: str) -> str:
        "Runs the AI agent given a user input."
        if self.rag is None:
            self.conversation.append({"role": "user", "content": user_prompt})
        else:
            context = self.rag.similarity_search(text=user_prompt, num_search_results=self.rag_num_search_results)
            self.conversation.append({"role": "user", "content": f"{user_prompt} \nContext: \n{context}"})
        print(self.conversation)
        # Make sure the conversation does not exceed the token limit as we iterate to get a final answer.
        iterations = 0
        while iterations <= settings.AGENT_MAX_ITERATIONS:
            iterations += 1
            self._trim_conversation()

            # Now get the response from the LLM
            response = self.llm_client.get_completion(self.conversation)
            response_content = response["choices"][0]["message"]["content"]
            # logging.debug(response_content)
            thought, action, answer, parsed, observation = self._parse_response(response_content)
            self.conversation.append({"role": "assistant", "content": response_content})
            logging.debug(f"Final answer log: \n{answer}")

            # Now the model created a step in the chain of thought and will evaluate and potentially automatically refine it.
            if thought is not None:
                observation = {
                    "Observation": "I only expressed a thought. Next up, I will make use of one of my tools or write an answer."
                }
            if parsed and action is not None:
                observation = {
                    "Observation": ""
                }
                if "RAG" in action:
                    observation["Observation"] += str(self.tools["RAG"].similarity_search(text=user_prompt, num_search_results=self.rag_num_search_results))
                elif "CoALA" in action:
                    logging.debug(f"Action contains CoALA: {'CoALA' in action}")
                    observation["Observation"] += self.tools["CoALA"].similarity_search(text=user_prompt, num_search_results=self.rag_num_search_results)
                else:
                    # Check if the code is valid
                    # Parse code and look for syntax or schema errors.
                    logging.debug("Now checking the code for syntax errors.")
                    code_validity, code_error = self._code_is_valid(action)
                    if code_validity:
                        observation = {
                            "Observation": "Your response format was correct and the code does not have any syntax errors."
                        }
                    else:
                        observation = {
                            "Observation": f"Your response format was correct but there seems to be a syntax error: {code_error}"
                        }
            if action is None and answer is not None:
                # TODO: Remove this and embed it in the test framework in case the generated function's output is correct.
                if isinstance(self.rag, CoALA):
                    # Push result into the long-term memory (i.e. the FAISS code storage)
                    logging.info("Assuming that the answer was correct, I'll add that to the long-term storage.")
                    self.rag.add_answer_to_code_storage(f"Question: \n{user_prompt}\nFinal Answer: \n{answer}")
                return answer
            # logging.debug("Final observation: ", observation)
            print(f"\n\nThe entire conversation: \n{self.conversation}")
            self.conversation.append({"role": "assistant", "content": str(observation)})

    def _code_is_valid(self, code: str) -> tuple:
        """
        Checks the code for syntax errors and returns a tuple with a bool result
        and the error message in case there is one.

        Parameters:
        - code (str): The Python code to be checked.

        Returns:
        - tuple: A tuple containing a boolean indicating code validity and an error message (empty string if no error).
        """
        try:
            parsed_code = ast.parse(code, mode="exec")
            is_valid = True
            error_message = ""
        except SyntaxError as e:
            is_valid = False
            error_message = str(e)
        return is_valid, error_message

    def reset_conversation(self) -> None:
        "Resets the conversation."
        self.conversation = [self.conversation[0]]

    def _trim_conversation(self) -> None:
        "Trims the conversation length to prevent LLM context length overflow."
        conversation_history_tokens = num_tokens_from_messages(self.conversation)
        while conversation_history_tokens + settings.AGENT_MAX_RESPONSE_TOKENS >= settings.AGENT_TOKEN_LIMIT:
            del self.conversation[1]
            conversation_history_tokens = num_tokens_from_messages(self.conversation)
