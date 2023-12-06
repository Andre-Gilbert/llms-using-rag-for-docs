"""Test cases."""

TEST_CASES = [
     {
        "id": 0,
        "user_prompt": """How can I convert this dataframe: df = pd.DataFrame({"col1_a": [1, 0, 1], "col1_b": [0, 1, 0], "col2_a": [0, 1, 0], "col2_b": [1, 0, 0], "col2_c": [0, 0, 1]}) into a categorical dataframe? """, # prompt that we send the agent
        "data": """data = pd.DataFrame({"col1_a": [1, 0, 1], "col1_b": [0, 1, 0], "col2_a": [0, 1, 0], "col2_b": [1, 0, 0], "col2_c": [0, 0, 1]})""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data=pd.from_dummies(data, sep="_")\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 1,
        "user_prompt": """This is my Dataframe:({'Name': ['Alice', 'Bob', 'Aritra'], 'Age': [25, 30, 35], 'Location': ['Seattle', 'New York', 'Kona']},index=([10, 20, 30])) Please show the df while making sure to change the index to 100,200and 300 """, # prompt that we send the agent
        "data": """data = pd.DataFrame=({'Name': ['Alice', 'Bob', 'Aritra'], 'Age': [25, 30, 35], 'Location': ['Seattle', 'New York', 'Kona']},index=([10, 20, 30]))""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data.index = [100, 200, 300]\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 2,
        "user_prompt": """({'animal': ['alligator', 'bee', 'falcon', 'lion','monkey', 'parrot', 'shark', 'whale', 'zebra']}) This is my dataframe. Please show me the head of all columns but the last 4 """, # prompt that we send the agent
        "data": """data = pd.DataFrame=({'animal': ['alligator', 'bee', 'falcon', 'lion','monkey', 'parrot', 'shark', 'whale', 'zebra']})""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data = data.iloc[:, :-4].head()\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 3,
        "user_prompt": """ts = pd.Timestamp('2017-01-01 09:10:11') This is your argument. Please add 2 Months to that timestamp """, # prompt that we send the agent
        "data": """data = pd.Timestamp('2017-01-01 09:10:11')""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data = data + pd.DateOffset(months=2)\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 4,
        "user_prompt": """ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd']). Please calculate the expending sum of that series. Make sure to show each Row """, # prompt that we send the agent
        "data": """data = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n   data = data.expanding().sum()\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 5,
        "user_prompt": """data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]] df = pd.DataFrame(data, columns=["a", "b", "c"] , index=["tiger", "leopard", "cheetah", "lion"]) Given Data is my Data and df is my Dataframe. Both are part of your argument. Please groupby that dataframe by a  and give the product as well """, # prompt that we send the agent
        #more data args are needed
        #talk w MG
        error in test_cases.py id5
        "data": """data = pd.DataFrame=""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data = df.groupby('a').prod()\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 6,
        "user_prompt": """a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd']) b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])  Please take a and b as your arguments and divide a by b. Please also use the fill value 0 """, # prompt that we send the agent
        #more data args are needed
        #talk w MG
        error in test_cases.py id6
        "data": """data = pd.DataFrame=""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data = a.div(b, fill_value=0)\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 7,
        "user_prompt": """data = {('level_1', 'c', 'a'): [3, 7, 11],('level_1', 'd', 'b'): [4, 8, 12],('level_2', 'e', 'a'): [5, 9, None],('level_2', 'f', 'b'): [6, 10, None],}Please drop a""", # prompt that we send the agent
        "data": """data = pd.DataFrame={('level_1', 'c', 'a'): [3, 7, 11],('level_1', 'd', 'b'): [4, 8, 12],('level_2', 'e', 'a'): [5, 9, None],('level_2', 'f', 'b'): [6, 10, None],}""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    df = pd.DataFrame(data)\n    data = df.droplevel(2, axis=1)\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 8,
        "user_prompt": """Please take follwing Series and order it ascending while making sure NAN values are at the beginning s = pd.Series([np.nan, 1, 3, 10, 5, np.nan]) """, # prompt that we send the agent
        "data": """data = pd.Series([np.nan, 1, 3, 10, 5, np.nan])""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data = s.sort_values(na_position='first')\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 9,
        "user_prompt": """data1 = {'Name': ['Alice', 'Bob', 'Charlie'],'Age': [25, 30, 22],'City': ['New York', 'San Francisco', 'Los Angeles']}data2= {'Name': ['Alice', 'John', 'Charlie'],'Age': [25, 31, 22],'City': ['New York', 'San Francisco', 'Los Angeles']}Please calculate the average age of people who are in both dataframes """, # prompt that we send the agent
         #more data args are needed
        #talk w MG
        error in test_cases.py id9
        "data": """data = pd.DataFrame=""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data1 = pd.DataFrame(data1)\n    data2 = pd.DataFrame(data2)\n    merged_df = pd.merge(data1, data2, on='Name')\n    data = merged_df['Age_x'].mean()\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 10,
        "user_prompt": """data = { 'Timestamp': [ '2023-01-01 12:01:00', '2023-01-01 12:10:00', '2023-01-01 12:25:00', '2023-01-01 13:05:00', '2023-01-01 13:25:00', '2023-01-01 14:00:00', '2023-01-02 08:30:00', '2023-01-02 09:00:00', '2023-01-02 09:35:00' ], 'User': [1, 1, 1, 2, 2, 2, 3, 3, 3], 'Page': ['Home', 'Product', 'Checkout', 'Home', 'Product', 'Home', 'Home', 'Product', 'Checkout'] } Using the pandas DataFrame df provided, implement the following operation: Create a new column called 'Session_ID' that labels each row with a unique session identifier. Define a session as a series of consecutive interactions by the same user with no gap greater than 30 minutes between interactions. Ensure that each session has a unique identifier. Make sure to show the hole code """, # prompt that we send the agent
        "data": """data = pd.DataFrame={ 'Timestamp': [ '2023-01-01 12:01:00', '2023-01-01 12:10:00', '2023-01-01 12:25:00', '2023-01-01 13:05:00', '2023-01-01 13:25:00', '2023-01-01 14:00:00', '2023-01-02 08:30:00', '2023-01-02 09:00:00', '2023-01-02 09:35:00' ], 'User': [1, 1, 1, 2, 2, 2, 3, 3, 3], 'Page': ['Home', 'Product', 'Checkout', 'Home', 'Product', 'Home', 'Home', 'Product', 'Checkout'] }""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data['Timestamp'] = pd.to_datetime(data['Timestamp'])\n    data = data.sort_values(by=['User', 'Timestamp'])\n    data['TimeDiff'] = data.groupby('User')['Timestamp'].diff()\n    data['Session_ID'] = (data['TimeDiff'] > pd.Timedelta(minutes=30)).cumsum()\n    data = data.drop('TimeDiff', axis=1)\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 11,
        "user_prompt": """Please return the rolling rank(3) of this Series [1, 4, 2, 3, 5, 3]. Make sure to code your solution using the pandas lib """, # prompt that we send the agent
        "data": """data = pd.Series([1, 4, 2, 3, 5, 3])""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data.rolling(3).rank()\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 12,
        "user_prompt": """ Please create a dic using follwing Dataframe. This dataframe is your argument. Make sure to order it tight. pd.DataFrame([[1, 3], [2, 4]],index=pd.MultiIndex.from_tuples([("a", "b"), ("a", "c")],names=["n1", "n2"]),columns=pd.MultiIndex.from_tuples([("x", 1), ("y", 2)], names=["z1", "z2"]),)""", # prompt that we send the agent
        "data": """data = pd.DataFrame.from_records([[1, 3], [2, 4]],index=pd.MultiIndex.from_tuples([("a", "b"), ("a", "c")],names=["n1", "n2"]),columns=pd.MultiIndex.from_tuples([("x", 1), ("y", 2)], names=["z1", "z2"]),)""", # the data needed should always be named 'data'"correct_function": """\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
        "correct_function":"""import pandas as pd\ndef correct_function(data):\n    data.to_dict(orient='tight')\n    return data"""
    },
    {
        "id": 13,
        "user_prompt": """Please take following dataframe (your argument) and group it for column A. Make sure to exclude the last value of each group.["g", "g0"], ["g", "g1"], ["g", "g2"], ["g", "g3"],["h", "h0"], ["h", "h1"]], columns=["A", "B"] """, # prompt that we send the agent
        "data": """data = pd.DataFrame=["g", "g0"], ["g", "g1"], ["g", "g2"], ["g", "g3"],["h", "h0"], ["h", "h1"]], columns=["A", "B"]""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data.groupby("A").head(-1)\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 14,
        "user_prompt": """TPlease remove follwinf suffix “_str” from following Series (["foo_str","_strhead" , "text_str_text" , "bar_str", "no_suffix"]) """, # prompt that we send the agent
        "data": """data = pd.Series(["foo_str","_strhead" , "text_str_text" , "bar_str", "no_suffix"])""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data.str.removesuffix("_str")\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 15,
        "user_prompt": """I have 2 Dataframes. which are you arguments The first one: pd.DataFrame({'key': ['K0', 'K1', 'K1', 'K3', 'K0', 'K1'],  'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})And the second one: pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': ['B0', 'B1', 'B2']})How do i join the second one on the first one using the key. And making sure it is a m:1 relation """, # prompt that we send the agent
         #more data args are needed
        #talk w MG
        error in test_cases.py id15
        "data": """data1 = data2 = pd.DataFrame=""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data1.join(data2.set_index('key'), on='key', validate='m:1')\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 16,
         #more data args are needed
        #talk w MG
        error in test_cases.py id16
        "user_prompt": """Please read in a csv called Test and make sure to use a data type called ArrowDtype """, # prompt that we send the agent
        "data": """data = pd.DataFrame=""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data = pd.read_csv(data, dtype_backend="pyarrow", engine="pyarrow")\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 17,
        "user_prompt": """What are the value counts of this function pd.Series(['quetzal', 'quetzal', 'elk'], name='animal') """, # prompt that we send the agent
        "data": """data = pd.Series(['quetzal', 'quetzal', 'elk'], name='animal')""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data=pd.Series(['quetzal', 'quetzal', 'elk'], name='animal').value_counts()\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 18,
        "user_prompt": """Please compute the difference between these concecutive values as a index object: pd.Index([10, 20, 30, 40, 50]) """, # prompt that we send the agent
        "data": """data = pd.Index([10, 20, 30, 40, 50])""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data = pd.Index([10, 20, 30, 40, 50])\n    data.diff()\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    },
    {
        "id": 19,
        "user_prompt": """df = pd.DataFrame({"a": [1, 1, 2, 1], "b": [np.nan, 2.0, 3.0, 4.0]}, dtype="Int64") This is my Dataframe. Please convert the Int64 to Int64[pyarrow] and use df.sum() at the end""", # prompt that we send the agent
        "data": """data = pd.DataFrame({"a": [1, 1, 2, 1], "b": [np.nan, 2.0, 3.0, 4.0]}, dtype="Int64")""", # the data needed should always be named 'data'
        "correct_function": """import pandas as pd\ndef correct_function(data):\n    data= data.astype"int64[pyarrow]")\n    data.sum()\n    return data""", # this is a response function that takes the parameter 'data' and does the correct thing with it
    }
]