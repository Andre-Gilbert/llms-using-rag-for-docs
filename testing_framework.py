import pandas as pd
from tests.pandas import TEST_CASES
from llms.agents.react import ReActAgent
from llms.clients.gpt import GPTClient
from llms.settings import settings
import json

client = GPTClient(
    client_id=settings.CLIENT_ID,
    client_secret=settings.CLIENT_SECRET,
    auth_url=settings.AUTH_URL,
    api_base=settings.API_BASE,
    deployment_id="gpt-4-32k",
    max_response_tokens=1000,
    temperature=0.0,
)
agent = ReActAgent(client)

test_results = []

for test_case in TEST_CASES:
    agent_error = None # variable to store errors that the agent code produces

    # get response function from agent
    final_answer = agent.run(test_case['user_prompt'])
    namespace_agent = {}
    exec(final_answer, namespace_agent)
    response_function = namespace_agent['response_function']

    # get desired result and save it in a variable called data
    data_string = test_case['data']
    local_vars = {}
    exec(data_string, globals(), local_vars)
    if 'data_1' in local_vars and 'data_2' in local_vars and 'data_3' in local_vars:
        data_1 = local_vars.get('data_1', None)
        data_2 = local_vars.get('data_2', None)
        data_3 = local_vars.get('data_3', None)
    elif 'data_1' in local_vars and 'data_2' in local_vars:
        data_1 = local_vars.get('data_1', None)
        data_2 = local_vars.get('data_2', None)
    else: # the maximum input of variables we have in the test cases is 3
        data = local_vars.get('data', None)

    # retrieve the correct function
    correct_function_string = test_case['correct_function']
    namespace_correct = {}
    exec(correct_function_string, namespace_correct)
    correct_function = namespace_correct['correct_function']

    # execute the correct function with the data as parameter and save it as desired result
    if 'data_1' in local_vars and 'data_2' in local_vars and 'data_3' in local_vars:
        desired_result = correct_function(data_1, data_2, data_3)
    elif 'data_1' in local_vars and 'data_2' in local_vars:
        desired_result = correct_function(data_1, data_2)
    else:
        desired_result = correct_function(data)

    # execute the agent function with the data as parameter and save it as agent_result, store error, if agent code produces an error
    if 'data_1' in local_vars and 'data_2' in local_vars and 'data_3' in local_vars:
        try:
            agent_result = response_function(data_1, data_2, data_3)
        except Exception as e:
            agent_result = None
            agent_error = e

    elif 'data_1' in local_vars and 'data_2' in local_vars:
        try:
            agent_result = response_function(data_1, data_2)
        except Exception as e:
            agent_result = None
            agent_error = e
    else:
        try:
            agent_result = response_function(data)
        except Exception as e:
            agent_result = None
            agent_error = e

    # this has to be extended, each time we expect another data type as the desired output
    if isinstance(desired_result, pd.DataFrame) or isinstance(desired_result, pd.Series) or isinstance(desired_result, pd.Index):
        if desired_result.equals(agent_result):
            print(f"Agent output was correct for test case {test_case['id']}.")
            test_results.append({'result': f"correct for test case {test_case['id']}", 'agent_result': agent_result, 'desired_result': desired_result, 'agent_error': agent_error})
        else:
            print(f"Agent output was not correct for test case {test_case['id']}.")
            test_results.append({'result': f"false for test case {test_case['id']}", 'agent_result': agent_result, 'desired_result': desired_result, 'agent_error': agent_error})

    else:
        if agent_result == desired_result:
            print(f"Agent output was correct for test case {test_case['id']}.")
            test_results.append({'result': f"correct for test case {test_case['id']}", 'agent_result': agent_result, 'desired_result': desired_result, 'agent_error': agent_error})
        else:
            print(f"Agent output was not correct for test case {test_case['id']}.")
            test_results.append({'result': f"false for test case {test_case['id']}", 'agent_result': agent_result, 'desired_result': desired_result, 'agent_error': agent_error})

for result in test_results:
    if 'agent_result' in result and isinstance(result['agent_result'], pd.DataFrame):
        result['agent_result'] = result['agent_result'].to_dict(orient='records')
    if 'desired_result' in result and isinstance(result['desired_result'], pd.DataFrame):
        result['desired_result'] = result['desired_result'].to_dict(orient='records')

def convert_datetime(obj):
    if isinstance(obj, (pd.Timestamp)):
        return obj.isoformat()

destination_path = './results/results.json'

with open(destination_path, 'w') as json_file:
    json.dump(test_results, json_file, default=convert_datetime, indent=2)

print(f'Test is finished. Results have been written to {destination_path}.')