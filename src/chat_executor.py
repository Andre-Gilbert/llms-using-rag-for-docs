from agent import AIAgent
from clients import GPTClient
from settings import settings


client = GPTClient(
    client_id=settings.CLIENT_ID,
    client_secret=settings.CLIENT_SECRET,
    auth_url=settings.AUTH_URL,
    api_base=settings.API_BASE,
    llm_deployment_id="gpt-4-32k",
    llm_max_response_tokens=1000,
    llm_temperature=0.0,
)
agent = AIAgent(client)

# Get the user's order
user_prompt = input("What do you want me to do? Type here: ")
agent.run(user_prompt)
