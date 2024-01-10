"""Test cases."""
from llms.evaluation.code import CodeTestCase

TEST_CASES = [
    CodeTestCase(
        id=0,
        prompt="""How can I convert this dataframe: df = pd.DataFrame({"col1_a": [1, 0, 1], "col1_b": [0, 1, 0], "col2_a": [0, 1, 0], "col2_b": [1, 0, 0], "col2_c": [0, 0, 1]}) into a categorical dataframe?""",
        data="""data = pd.DataFrame({"col1_a": [1, 0, 1], "col1_b": [0, 1, 0], "col2_a": [0, 1, 0], "col2_b": [1, 0, 0], "col2_c": [0, 0, 1]})""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = pd.from_dummies(data, sep="_")\n    return result""",
    ),
    CodeTestCase(
        id=1,
        prompt="""This is my Dataframe:({'Name': ['Alice', 'Bob', 'Aritra'], 'Age': [25, 30, 35], 'Location': ['Seattle', 'New York', 'Kona']},index=([10, 20, 30])) Please display the dataframe while making sure to change the index to 100, 200 and 300.""",
        data="""data = pd.DataFrame({'Name': ['Alice', 'Bob', 'Aritra'], 'Age': [25, 30, 35], 'Location': ['Seattle', 'New York', 'Kona']},index=([10, 20, 30]))""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    data.index = [100, 200, 300]\n    return data""",
    ),
    CodeTestCase(
        id=2,
        prompt="""({'animal': ['alligator', 'bee', 'falcon', 'lion','monkey', 'parrot', 'shark', 'whale', 'zebra']}) This is my dataframe. Please display all but the last 3 rows of the dataframe.""",
        data="""data = pd.DataFrame({'animal': ['alligator', 'bee', 'falcon', 'lion','monkey', 'parrot', 'shark', 'whale', 'zebra']})""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.iloc[:-3, :]\n    return result""",
    ),
    CodeTestCase(
        id=3,
        prompt="""ts = pd.Timestamp('2017-01-01 09:10:11') This is your argument. Please add 2 Months to that timestamp.""",
        data="""data = pd.Timestamp('2017-01-01 09:10:11')""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data + pd.DateOffset(months=2)\n    return result""",
    ),
    CodeTestCase(
        id=4,
        prompt="""ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd']). Please calculate the expending sum of that series. Make sure to display each row.""",
        data="""data = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.expanding().sum()\n    return result""",
    ),
    CodeTestCase(
        id=5,
        prompt="""data1 = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]], data2 = pd.DataFrame(data, columns=["a", "b", "c"] , index=["tiger", "leopard", "cheetah", "lion"]) Given Data is my Data and df is my Dataframe. Both are part of your argument. Please group that dataframe by "a" and compute the product aswell.""",
        data="""data_1 = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]\ndata_2 = pd.DataFrame(data_1, columns=["a", "b", "c"] , index=["tiger", "leopard", "cheetah", "lion"])""",
        correct_function="""import pandas as pd\ndef correct_function(data_1, data_2):\n    result = data_2.groupby('a').prod()\n    return result""",
    ),
    CodeTestCase(
        id=6,
        prompt="""a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd']) b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])  Please take a and b as your arguments and divide a by b. Please also use the fill value 0.""",
        data="""import numpy as np\ndata_1 = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])\ndata_2 = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])""",
        correct_function="""import pandas as pd\nimport numpy as np\ndef correct_function(*args):\n    data_1, data_2 = args[1:]\n    result = data_1.div(data_2, fill_value=0)\n    return result""",
    ),
    CodeTestCase(
        id=7,
        prompt="""data = {('level_1', 'c', 'a'): [3, 7, 11],('level_1', 'd', 'b'): [4, 8, 12],('level_2', 'e', 'a'): [5, 9, None],('level_2', 'f', 'b'): [6, 10, None],}Please drop column a.""",
        data="""data = pd.DataFrame({('level_1', 'c', 'a'): [3, 7, 11],('level_1', 'd', 'b'): [4, 8, 12],('level_2', 'e', 'a'): [5, 9, None],('level_2', 'f', 'b'): [6, 10, None],})""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.droplevel(2, axis=1)\n    return result""",
    ),
    CodeTestCase(
        id=8,
        prompt="""Please take following Series and order it ascending while making sure NAN values are at the beginning s = pd.Series([np.nan, 1, 3, 10, 5, np.nan]) """,
        data="""import numpy as np\ndata = pd.Series([np.nan, 1, 3, 10, 5, np.nan])""",
        correct_function="""import pandas as pd\ndef correct_function(*args):\n    temp = args[1:]\n    data = pd.Series(temp)\n    result = data.sort_values(na_position='first')\n    return result""",
    ),
    CodeTestCase(
        id=9,
        prompt="""data1 = {'Name': ['Alice', 'Bob', 'Charlie'],'Age': [25, 30, 22],'City': ['New York', 'San Francisco', 'Los Angeles']} data2= {'Name': ['Alice', 'John', 'Charlie'],'Age': [25, 31, 22],'City': ['New York', 'San Francisco', 'Los Angeles']}Please calculate the average age of the people who appear in both dataframes.""",
        data="""data_1 = {'Name': ['Alice', 'Bob', 'Charlie'],'Age': [25, 30, 22],'City': ['New York', 'San Francisco', 'Los Angeles']}\ndata_2 = {'Name': ['Alice', 'John', 'Charlie'],'Age': [25, 31, 22],'City': ['New York', 'San Francisco', 'Los Angeles']}""",
        correct_function="""import pandas as pd\ndef correct_function(data_1, data_2):\n    df_1 = pd.DataFrame(data_1)\n    df_2 = pd.DataFrame(data_2)\n    merged_df = pd.merge(df_1, df_2, on='Name')\n    result = merged_df['Age_x'].mean()\n    return result""",
    ),
    CodeTestCase(
        id=10,
        prompt="""data = { 'Timestamp': [ '2023-01-01 12:01:00', '2023-01-01 12:10:00', '2023-01-01 12:25:00', '2023-01-01 13:05:00', '2023-01-01 13:25:00', '2023-01-01 14:00:00', '2023-01-02 08:30:00', '2023-01-02 09:00:00', '2023-01-02 09:35:00' ], 'User': [1, 1, 1, 2, 2, 2, 3, 3, 3], 'Page': ['Home', 'Product', 'Checkout', 'Home', 'Product', 'Home', 'Home', 'Product', 'Checkout'] } Using the pandas DataFrame df provided, implement the following operation: Create a new column called 'Session_ID' that labels each row with a unique session identifier. Define a session as a series of consecutive interactions by the same user with no gap greater than 30 minutes between interactions. Ensure that each session has a unique identifier. Make sure to give me the full code.""",
        data="""data = pd.DataFrame({'Timestamp': ['2023-01-01 12:01:00', '2023-01-01 12:10:00', '2023-01-01 12:25:00', '2023-01-01 13:05:00','2023-01-01 13:25:00', '2023-01-01 14:00:00', '2023-01-02 08:30:00', '2023-01-02 09:00:00','2023-01-02 09:35:00'],'User': [1, 1, 1, 2, 2, 2, 3, 3, 3],'Page': ['Home', 'Product', 'Checkout', 'Home', 'Product', 'Home', 'Home', 'Product', 'Checkout']})""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    data['Timestamp'] = pd.to_datetime(data['Timestamp'])\n    data = data.sort_values(by=['User', 'Timestamp'])\n    data['TimeDiff'] = data.groupby('User')['Timestamp'].diff()\n    data['Session_ID'] = (data['TimeDiff'] > pd.Timedelta(minutes=30)).cumsum()\n    data = data.drop('TimeDiff', axis=1)\n    return data""",
    ),
    CodeTestCase(
        id=11,
        prompt="""Please return the rolling rank(3) of this Series [1, 4, 2, 3, 5, 3]. Make sure to code your solution using the pandas lib.""",
        data="""data = pd.Series([1, 4, 2, 3, 5, 3])""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.rolling(3).rank()\n    return result""",
    ),
    CodeTestCase(
        id=12,
        prompt=""" Please create a dictionary using the following Dataframe. This dataframe is your argument. Make sure to order it tight. pd.DataFrame([[1, 3], [2, 4]],index=pd.MultiIndex.from_tuples([("a", "b"), ("a", "c")],names=["n1", "n2"]),columns=pd.MultiIndex.from_tuples([("x", 1), ("y", 2)], names=["z1", "z2"]),)""",
        data="""data = pd.DataFrame.from_records([[1, 3], [2, 4]],index=pd.MultiIndex.from_tuples([("a", "b"), ("a", "c")],names=["n1", "n2"]),columns=pd.MultiIndex.from_tuples([("x", 1), ("y", 2)], names=["z1", "z2"]),)""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.to_dict(orient='tight')\n    return result""",
    ),
    CodeTestCase(
        id=13,
        prompt="""Please take following dataframe (your argument) and group it for column A. Make sure to exclude the last value of each group. This is your argument data = pd.DataFrame(["g", "g0"], ["g", "g1"], ["g", "g2"], ["g", "g3"],["h", "h0"], ["h", "h1"]], columns=["A", "B"]).""",
        data="""data = pd.DataFrame([["g", "g0"], ["g", "g1"], ["g", "g2"], ["g", "g3"], ["h", "h0"], ["h", "h1"]],columns=["A", "B"])""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.groupby("A").head(-1)\n    return result""",
    ),
    CodeTestCase(
        id=14,
        prompt="""Please remove the following suffix “_str” from following Series (["foo_str","_strhead" , "text_str_text" , "bar_str", "no_suffix"]) """,
        data="""data = pd.Series(["foo_str","_strhead" , "text_str_text" , "bar_str", "no_suffix"])""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.str.removesuffix("_str")\n    return result""",
    ),
    CodeTestCase(
        id=15,
        prompt="""I have 2 Dataframes which are you arguments. The first one: pd.DataFrame({'key': ['K0', 'K1', 'K1', 'K3', 'K0', 'K1'],  'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']}) And the second one: pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': ['B0', 'B1', 'B2']})How do I join the second one on the first one using the key and making sure it is a m:1 relation?""",
        data="""data_1 = pd.DataFrame({'key': ['K0', 'K1', 'K1', 'K3', 'K0', 'K1'],  'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})\ndata_2 = pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': ['B0', 'B1', 'B2']})""",
        correct_function="""import pandas as pd\ndef correct_function(data_1, data_2):\n    result = data_1.join(data_2.set_index('key'), on='key', validate='m:1')\n    return result""",
    ),
    CodeTestCase(
        id=16,
        prompt="""This is your Index:pd.MultiIndex.from_tuples([('bird', 'falcon'),('bird', 'parrot'),('mammal', 'lion'),('mammal', 'monkey')],names=['class', 'name']) These are your columns: pd.MultiIndex.from_tuples([('speed', 'max'),('species', 'type')]) And this is your input: pd.DataFrame([(389.0, 'fly'),(24.0, 'fly'),(80.5, 'run'),(np.nan, 'jump')],index=index,columns=columns).Index, Columns and Input are your arguments. Please create a dataframe and rename the index to classes and names""",
        data="""import numpy as np\ndata_1 = pd.MultiIndex.from_tuples([('bird', 'falcon'),('bird', 'parrot'),('mammal', 'lion'),('mammal', 'monkey')],names=['class', 'name'])\ndata_2 = pd.MultiIndex.from_tuples([('speed', 'max'),('species', 'type')])\ndata_3 = pd.DataFrame([(389.0, 'fly'),(24.0, 'fly'),(80.5, 'run'),(np.nan, 'jump')],index=data_1,columns=data_2)""",
        correct_function="""import pandas as pd\ndef correct_function(*args):\n    data_1, data_2, data_3 = args[1:]\n    result = data_3.reset_index(names=['classes', 'names'])\n    return result""",
    ),
    CodeTestCase(
        id=17,
        prompt="""What are the value counts of this function pd.Series(['quetzal', 'quetzal', 'elk'], name='animal')?""",
        data="""data = pd.Series(['quetzal', 'quetzal', 'elk'], name='animal')""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.value_counts()\n    return data""",
    ),
    CodeTestCase(
        id=18,
        prompt="""Please compute the difference between these consecutive values as an index object: pd.Index([10, 20, 30, 40, 50]).""",
        data="""data = pd.Index([10, 20, 30, 40, 50])""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    sum = data.diff()\n    return sum""",
    ),
    CodeTestCase(
        id=19,
        prompt="""df = pd.DataFrame({"a": [1, 1, 2, 1], "b": [None, 2.0, 3.0, 4.0]}, dtype="Int64") This is my Dataframe. Please convert the Int64 to Int64[pyarrow] and use df.sum() at the end.""",
        data="""data = pd.DataFrame({"a": [1, 1, 2, 1], "b": [None, 2.0, 3.0, 4.0]}, dtype="Int64")""",
        correct_function="""\nimport pandas as pd\nimport pyarrow as pa\ndef correct_function(data):\n    data = data.astype("int64[pyarrow]")\n    data.sum()\n    return data""",
    ),
]
