from llms.agent import AIAgent
from llms.clients import GPTClient
from llms.rag import FAISS, CoALA
from llms.settings import settings

client = GPTClient(
    client_id=settings.CLIENT_ID,
    client_secret=settings.CLIENT_SECRET,
    auth_url=settings.AUTH_URL,
    api_base=settings.API_BASE,
    llm_deployment_id="gpt-4-32k",
    llm_max_response_tokens=1000,
    llm_temperature=0.0,
)

rag = FAISS.create_index_from_texts(
    texts=["print('Hello Wourld')", "None", "Irrelevant content"],
    llm_client=client,
)
code_storage = FAISS.create_index_from_texts(
    texts=["Question: Print Hello World\nFinal Answer:def response_function():\n    print('Hello World')"],
    llm_client=client,
)
coala_rag = CoALA(docs_storage=rag, code_storage=code_storage)

agent = AIAgent(client, rag=coala_rag)

# Get the user's order
while True:
    user_prompt = input("What do you want me to do? Type here: ")
    agent.run(user_prompt)
