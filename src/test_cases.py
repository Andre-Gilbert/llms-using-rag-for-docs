"""Test cases."""

TEST_CASES = [
    {
        "id": 0,
        "user_prompt": """How can I convert this dataframe: df = pd.DataFrame({"col1_a": [1, 0, 1], "col1_b": [0, 1, 0], "col2_a": [0, 1, 0], "col2_b": [1, 0, 0], "col2_c": [0, 0, 1]}) into a categorical dataframe? """, # prompt that we send the agent
        "data": """data = pd.DataFrame({"col1_a": [1, 0, 1], "col1_b": [0, 1, 0], "col2_a": [0, 1, 0], "col2_b": [1, 0, 0], "col2_c": [0, 0, 1]})""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data=pd.from_dummies(data, sep="_")\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    }
]