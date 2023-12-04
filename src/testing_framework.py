import pandas as pd
from test_cases import TEST_CASES
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

for test_case in TEST_CASES:
    # get response function from agent
    final_answer = agent.run(test_case['user_prompt'])
    namespace_agent = {}
    exec(final_answer, namespace_agent)
    response_function = namespace_agent['response_function']

    # get desired result and save it in a variable called data
    data_string = test_case['data']
    local_vars = {}
    exec(data_string, globals(), local_vars)
    data = local_vars.get('data', None)

    # retrieve the correct function
    correct_function_string = test_case['correct_function']
    namespace_correct = {}
    exec(correct_function_string, namespace_correct)
    correct_function = namespace_correct['correct_function']

    # execute the correct function with the data as parameter and save it as desired result
    desired_result = correct_function(data)

    # execute the agent function with the data as parameter and save it as agent_result
    agent_result = response_function(data)

    # this has to be extended, each time we expect another data type as the desired output
    if isinstance(agent_result, pd.DataFrame):
        if desired_result.equals(agent_result):
            print() # for better reading in the console
            print(f"Agent output was correct for test case {test_case['id']}.")
        else:
            print()
            print(f"Agent output was not correct for test case {test_case['id']}.")

    else:
        if agent_result == desired_result:
            print()
            print(f"Agent output was correct for test case {test_case['id']}.")
        else:
            print()
            print(f"Agent output was not correct for test case {test_case['id']}.")