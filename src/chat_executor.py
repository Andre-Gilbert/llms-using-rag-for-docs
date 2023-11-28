from agent import AIAgent
from clients import OpenAIClient
import dotenv
import os
from settings import settings


dotenv.load_dotenv()
service_key = eval(os.getenv('SERVICE_KEY'))

client = OpenAIClient(service_key, settings.LLM_CONFIG)
agent = AIAgent(client)

# Get the user's order
user_prompt = input("What do you want me to do? Type here: ")
agent.run(user_prompt)

