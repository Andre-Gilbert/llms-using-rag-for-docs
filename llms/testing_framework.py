import pandas as pd
from agent import AIAgent
from clients import GPTClient
from settings import settings
from test_cases import TEST_CASES

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

for test_case in TEST_CASES[0]:
    # get response function from agent
    final_answer = agent.run(test_case["user_prompt"])
    namespace_agent = {}
    exec(final_answer, namespace_agent)
    response_function = namespace_agent["response_function"]

    # get desired result and save it in a variable called data
    data_string = test_case["data"]
    local_vars = {}
    exec(data_string, globals(), local_vars)
    if len(local_vars) == 1:
        data = local_vars.get("data", None)
    else:  # the maximum input of variables we have in the test cases is 2
        data_1 = local_vars.get("data_1", None)
        data_2 = local_vars.get("data_2", None)

    # retrieve the correct function
    correct_function_string = test_case["correct_function"]
    namespace_correct = {}
    exec(correct_function_string, namespace_correct)
    correct_function = namespace_correct["correct_function"]

    # execute the correct function with the data as parameter and save it as desired result
    if len(local_vars) == 1:
        desired_result = correct_function(data)
    else:
        desired_result = correct_function(data_1, data_2)

    # execute the agent function with the data as parameter and save it as agent_result
    if len(local_vars) == 1:
        agent_result = response_function(data)
    else:
        agent_result = response_function(data_1, data_2)

    # this has to be extended, each time we expect another data type as the desired output
    if isinstance(agent_result, pd.DataFrame):
        if desired_result.equals(agent_result):
            print(f"Agent output was correct for test case {test_case['id']}.")
        else:
            print(f"Agent output was not correct for test case {test_case['id']}.")

    else:
        if agent_result == desired_result:
            print(f"Agent output was correct for test case {test_case['id']}.")
        else:
            print(f"Agent output was not correct for test case {test_case['id']}.")
