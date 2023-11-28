"""AI agent implementation."""
import requests

from clients import OpenAIClient
from rag import RetrievalAugmentedGeneration
from settings import settings
from utils import num_tokens_from_messages
from helpers import extract
import re
import time
import logging
import json


logging.basicConfig(
    level=logging.INFO, format="%(process)d - %(levelname)s - %(message)s"
)


class AIAgent:
    """
    AI agent built on top of the ReAct methodology as proposed in https://arxiv.org/abs/2210.03629.
    This handles the conversation between the user and the LLM by providing and interface 
    for the user and parsing the response from the model. It uses the Chat class to 
    keep track of the context/chat history.
    """

    def __init__(
        self,
        llm_client: OpenAIClient,
        tools: dict = dict(),
        system_prompt: str = settings.STANDARD_SYSTEM_INSTRUCTION,
        rag: RetrievalAugmentedGeneration = None,
    ):
        self.llm_client = llm_client
        self.tools = tools
        self.rag = rag
        self.conversation = [{"role": "system", "content": system_prompt}]


    def _parse_response(self, response: requests.Response) -> tuple:
        try:
            response = json.loads(response, strict=False)
            thought = extract(response, "Thought")
            action = extract(response, "Action")
            answer = extract(response, "Answer")
            # print(answer)

            # Now extract the code
            # pattern = r'```python(.*?)```'
            # answer = re.search(pattern, answer, re.DOTALL)
            
            parsed = True
            observation = {"Observation": "Your response format was correct."}

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            parsed = False
            thought = None
            action = None
            answer = None
            observation = {"Observation": f"Your response format was incorrect. Please correct as specified in the first message. The error was: {e}"}

        return thought, action, answer, parsed, observation
    

    def run(self, user_prompt: str) -> str:
        "Runs the AI agent given a user input."

        if self.rag:
            # retrieve relevant content
            #
            # context = self.rag.search(user_prompt)
            # self.conversation.append({"role": "user", "content": f"{user_prompt} Context: {context}"})
            #
            pass
        else:
            self.conversation.append({"role": "user", "content": user_prompt})
        
        # Make sure the conversation does not exceed the token limit as we iterate to get a final answer.
        iterations = 0
        while iterations <= settings.AGENT_MAX_ITERATIONS:
            iterations += 1
            self._trim_conversation()

            # Now get the response from the LLM
            raw_response = self.llm_client.get_completion(self.conversation)

            # Retry a few times if the API cannot be reached.
            api_retries = 0
            while not raw_response.ok:
                if api_retries <= settings.REQUEST_TIMEOUT:
                    raise Exception("The LLM API could not be reached!")
                else:
                    api_retries += 1
                    logging.debug("The response code from the LLM API was not less than 400. Retry in 2 sec...")
                    time.sleep(2)
                    raw_response = self.llm_client.get_completion(self.conversation)
            else:
                response = raw_response.json()
                response_content = response["choices"][0]["message"]["content"]
                print(response_content)
                thought, action, answer, parsed, observation = self._parse_response(response_content)
                self.conversation.append({"role": "assistant", "content": response_content})
                print(observation)
                print("Final answer: \n", answer)

                if parsed:
                    # TODO: Handle missing thoughts or actions if necessary.
                    pass

                if action == None and answer != None:
                    return answer
                self.conversation.append({"role": "assistant", "content": str(observation)})


    def reset_conversation(self) -> None:
        "Resets the conversation."

        self.conversation = [self.conversation[0]]


    def _trim_conversation(self) -> None:
        "Trims the conversation length to prevent LLM context length overflow."

        conversation_history_tokens = num_tokens_from_messages(self.conversation)
        while conversation_history_tokens + settings.AGENT_MAX_RESPONSE_TOKENS >= settings.AGENT_TOKEN_LIMIT:
            del self.conversation[1]
            conversation_history_tokens = num_tokens_from_messages(self.conversation)


class Chat:
    """
    Takes care of the chat history to provide the LLM with context on previous requests.
    """

    def __init__(self) -> None:
        pass
