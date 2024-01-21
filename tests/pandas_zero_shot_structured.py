"""Test cases."""
from llms.evaluation.code import CodeTestCase

# pylint: disable=all
TEST_CASES = [
    CodeTestCase(
        prompt="""
        I have a one-hot encoded DataFrame with '_' as the separator.
        How can I revert this one-hot encoded DataFrame back into a categorical DataFrame using pandas?

        The following DataFrame will be the only function argument:
        df = pd.DataFrame({
            'col1_a': [1, 0, 1],
            'col1_b': [0, 1, 0],
            'col2_a': [0, 1, 0],
            'col2_b': [1, 0, 0],
            'col2_c': [0, 0, 1],
        })
        """,
        data="""data = pd.DataFrame({"col1_a": [1, 0, 1], "col1_b": [0, 1, 0], "col2_a": [0, 1, 0], "col2_b": [1, 0, 0], "col2_c": [0, 0, 1]})""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = pd.from_dummies(data, sep="_")\n    return result""",
    ),
    CodeTestCase(
        prompt="""
        I want to change the indices of the DataFrame to 100, 200 and 300.

        The following DataFrame will be the only function argument:
        df = pd.DataFrame({
                'Name': ['Alice', 'Bob', 'Aritra'],
                'Age': [25, 30, 35],
                'Location': ['Seattle', 'New York', 'Kona'],
            },
            index=([10, 20, 30]),
        )
        """,
        data="""data = pd.DataFrame({'Name': ['Alice', 'Bob', 'Aritra'], 'Age': [25, 30, 35], 'Location': ['Seattle', 'New York', 'Kona']}, index=([10, 20, 30]))""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    data.index = [100, 200, 300]\n    return data""",
    ),
    CodeTestCase(
        prompt="""
        Return all rows of the DataFrame except for the last 3 rows.

        The following DataFrame will be the only function argument:
        df = pd.DataFrame({'animal': ['alligator', 'bee', 'falcon', 'lion','monkey', 'parrot', 'shark', 'whale', 'zebra']})
        """,
        data="""data = pd.DataFrame({'animal': ['alligator', 'bee', 'falcon', 'lion','monkey', 'parrot', 'shark', 'whale', 'zebra']})""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.iloc[:-3, :]\n    return result""",
    ),
    CodeTestCase(
        prompt="""
        Please add 2 months to the timestamp.

        The following DataFrame will be the only function argument:
        ts = pd.Timestamp('2017-01-01 09:10:11')
        """,
        data="""data = pd.Timestamp('2017-01-01 09:10:11')""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data + pd.DateOffset(months=2)\n    return result""",
    ),
    CodeTestCase(
        prompt="""
        Calculate the sum using the expanding window of the Series.

        The following Series will be the only function argument:
        data = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        """,
        data="""data = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.expanding().sum()\n    return result""",
    ),
    CodeTestCase(
        prompt="""
        First group the DataFrame by 'a'. Then compute the product of the grouped DataFrame.

        The following DataFrame will be the only function argument:
        df = pd.DataFrame([[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]], columns=['a', 'b', 'c'] , index=['tiger', 'leopard', 'cheetah', 'lion'])
        """,
        data="""data = pd.DataFrame([[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]], columns=["a", "b", "c"] , index=["tiger", "leopard", "cheetah", "lion"])""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.groupby('a').prod()\n    return result""",
    ),
    CodeTestCase(
        prompt="""
        Please give me the floating division of DataFrame and other.
        Divide the Series a by the Series b and use 0 as the fill value.

        The following 2 Series will be the only function arguments:
        data_1 = pd.Series([1, 1, 1, None], index=['a', 'b', 'c', 'd'])
        data_2 = pd.Series([1, None, 1, None], index=['a', 'b', 'd', 'e'])
        """,
        data="""data_1 = pd.Series([1, 1, 1, None], index=['a', 'b', 'c', 'd'])\ndata_2 = pd.Series([1, None, 1, None], index=['a', 'b', 'd', 'e'])""",
        correct_function="""import pandas as pd\nimport numpy as np\ndef correct_function(*args):\n    data_1, data_2 = args\n    result = data_1.div(data_2, fill_value=0)\n    return result""",
    ),
    CodeTestCase(
        prompt="""
        Please drop column 'a' of the DataFrame.

        The following DataFrame will be the only function argument:
        df = pd.DataFrame({
            ('level_1', 'c', 'a'): [3, 7, 11],
            ('level_1', 'd', 'b'): [4, 8, 12],
            ('level_2', 'e', 'a'): [5, 9, None],
            ('level_2', 'f', 'b'): [6, 10, None],
        })
        """,
        data="""data = pd.DataFrame({('level_1', 'c', 'a'): [3, 7, 11],('level_1', 'd', 'b'): [4, 8, 12],('level_2', 'e', 'a'): [5, 9, None],('level_2', 'f', 'b'): [6, 10, None],})""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.droplevel(2, axis=1)\n    return result""",
    ),
    CodeTestCase(
        prompt="""
        Sort the pandas Series in ascending order and put NaNs at the beginning.

        The following Series will be the only function argument:
        data = pd.Series([None, 1, 3, 10, 5, None])
        """,
        data="""data = pd.Series([None, 1, 3, 10, 5, None])""",
        correct_function="""import pandas as pd\ndef correct_function(*args):\n    data = pd.Series(args)\n    result = data.sort_values(na_position='first')\n    return result""",
    ),
    CodeTestCase(
        prompt="""
        Convert the following dictionaries into a pandas DataFrame and calculate the average age of the people who appear in both DataFrames.

        The following 2 dictionaries will be the only function arguments:
        data1 = {
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 22],
            'City': ['New York', 'San Francisco', 'Los Angeles']
        }
        data2 = {
            'Name': ['Alice', 'John', 'Charlie'],
            'Age': [25, 31, 22],
            'City': ['New York', 'San Francisco', 'Los Angeles']
        }
        """,
        data="""data_1 = {'Name': ['Alice', 'Bob', 'Charlie'],'Age': [25, 30, 22],'City': ['New York', 'San Francisco', 'Los Angeles']}\ndata_2 = {'Name': ['Alice', 'John', 'Charlie'],'Age': [25, 31, 22],'City': ['New York', 'San Francisco', 'Los Angeles']}""",
        correct_function="""import pandas as pd\ndef correct_function(data_1, data_2):\n    df_1 = pd.DataFrame(data_1)\n    df_2 = pd.DataFrame(data_2)\n    merged_df = pd.merge(df_1, df_2, on='Name')\n    result = merged_df['Age_x'].mean()\n    return result""",
    ),
    CodeTestCase(
        prompt="""
        First, convert the Timestamp of the DataFrame to datetime.
        Then, sort the values of the DataFrame by User, Timestamp.
        Then, group the DataFrame by User.
        Then, create a new column TimeDiff using the Timestamp column.
        Then, create a new column called Session_ID using the cumulative sum where the TimeDiff is greater than 30 minutes.
        Lastly, drop the TimeDiff column.

        The following DataFrame will be the only function argument:
        df = pd.DataFrame({
            'Timestamp': ['2023-01-01 12:01:00', '2023-01-01 12:10:00', '2023-01-01 12:25:00', '2023-01-01 13:05:00', '2023-01-01 13:25:00', '2023-01-01 14:00:00', '2023-01-02 08:30:00', '2023-01-02 09:00:00', '2023-01-02 09:35:00'],
            'User': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'Page': ['Home', 'Product', 'Checkout', 'Home', 'Product', 'Home', 'Home', 'Product', 'Checkout']
        })
        """,
        data="""data = pd.DataFrame({'Timestamp': ['2023-01-01 12:01:00', '2023-01-01 12:10:00', '2023-01-01 12:25:00', '2023-01-01 13:05:00','2023-01-01 13:25:00', '2023-01-01 14:00:00', '2023-01-02 08:30:00', '2023-01-02 09:00:00','2023-01-02 09:35:00'],'User': [1, 1, 1, 2, 2, 2, 3, 3, 3],'Page': ['Home', 'Product', 'Checkout', 'Home', 'Product', 'Home', 'Home', 'Product', 'Checkout']})""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    data['Timestamp'] = pd.to_datetime(data['Timestamp'])\n    data = data.sort_values(by=['User', 'Timestamp'])\n    data['TimeDiff'] = data.groupby('User')['Timestamp'].diff()\n    data['Session_ID'] = (data['TimeDiff'] > pd.Timedelta(minutes=30)).cumsum()\n    data = data.drop('TimeDiff', axis=1)\n    return data""",
    ),
    CodeTestCase(
        prompt="""
        Calculate the rolling rank of the Series. Use a window size of 3.

        The following Series will be the only function argument:
        data = pd.Series([1, 4, 2, 3, 5, 3])
        """,
        data="""data = pd.Series([1, 4, 2, 3, 5, 3])""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.rolling(3).rank()\n    return result""",
    ),
    CodeTestCase(
        prompt="""
        Please create a dictionary from the following DataFrame. Use 'tight' as the orientation.

        The following DataFrame will be the only function argument:
        df = pd.DataFrame(
            [[1, 3], [2, 4]],
            index=pd.MultiIndex.from_tuples([("a", "b"), ("a", "c")],
            names=["n1", "n2"]),
            columns=pd.MultiIndex.from_tuples([("x", 1), ("y", 2)],
            names=["z1", "z2"]),
        )
        """,
        data="""data = pd.DataFrame.from_records([[1, 3], [2, 4]],index=pd.MultiIndex.from_tuples([("a", "b"), ("a", "c")],names=["n1", "n2"]),columns=pd.MultiIndex.from_tuples([("x", 1), ("y", 2)], names=["z1", "z2"]),)""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.to_dict(orient='tight')\n    return result""",
    ),
    CodeTestCase(
        prompt="""
        Please group the DataFrame by column 'A' and return all rows of the DataFrame except for the last row.

        The following DataFrame will be the only function argument:
        df = pd.DataFrame([
                ["g", "g0"],
                ["g", "g1"],
                ["g", "g2"],
                ["g", "g3"],
                ["h", "h0"],
                ["h", "h1"],
            ],
            columns=["A", "B"],
        )
        """,
        data="""data = pd.DataFrame([["g", "g0"], ["g", "g1"], ["g", "g2"], ["g", "g3"], ["h", "h0"], ["h", "h1"]],columns=["A", "B"])""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.groupby("A")\n    result = data.iloc[:-1, :]\n    return result""",
    ),
    CodeTestCase(
        prompt="""
        Remove the following suffix '_str' from the Series.

        The following Series will be the only function argument:
        data = Series(['foo_str', '_strhead', 'text_str_text', 'bar_str', 'no_suffix'])
        """,
        data="""data = pd.Series(["foo_str","_strhead" , "text_str_text" , "bar_str", "no_suffix"])""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.str.removesuffix("_str")\n    return result""",
    ),
    CodeTestCase(
        prompt="""
        I have 2 DataFrames. How do I join the second one on the first one using the key and making sure it is a m:1 relation?

        The following 2 DataFrames will be the only function arguments:
        df1 = pd.DataFrame({
            'key': ['K0', 'K1', 'K1', 'K3', 'K0', 'K1'],
            'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5'],
        })
        df2 = pd.DataFrame({
            'key': ['K0', 'K1', 'K2'],
            'B': ['B0', 'B1', 'B2'],
        })
        """,
        data="""data_1 = pd.DataFrame({'key': ['K0', 'K1', 'K1', 'K3', 'K0', 'K1'],  'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})\ndata_2 = pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': ['B0', 'B1', 'B2']})""",
        correct_function="""import pandas as pd\ndef correct_function(data_1, data_2):\n    result = data_1.join(data_2.set_index('key'), on='key', validate='m:1')\n    return result""",
    ),
    CodeTestCase(
        prompt="""
        Please create a DataFrame using the provided data, index and, columns.
        Then, reset the index and rename the index to classes, names.

        The following variables will be the only function arguments:
        data = [
            (389.0, 'fly'),
            (24.0, 'fly'),
            (80.5, 'run'),
            (None, 'jump')
        ]
        index = pd.MultiIndex.from_tuples([
                ('bird', 'falcon'),
                ('bird', 'parrot'),
                ('mammal', 'lion'),
                ('mammal', 'monkey')
            ],
            names=['class', 'name']
        )
        columns = pd.MultiIndex.from_tuples([
            ('speed', 'max'),
            ('species', 'type')
        ])
        """,
        data="""index = pd.MultiIndex.from_tuples([('bird', 'falcon'),('bird', 'parrot'),('mammal', 'lion'),('mammal', 'monkey')],names=['class', 'name'])\ncolumns = pd.MultiIndex.from_tuples([('speed', 'max'),('species', 'type')])\ndata = [(389.0, 'fly'),(24.0, 'fly'),(80.5, 'run'),(None, 'jump')]""",
        correct_function="""import pandas as pd\ndef correct_function(*args):\n    index, columns, data = args\n    df = pd.DataFrame(data, index=index, columns=columns)\n    result = df.reset_index(names=['classes', 'names'])\n    return result""",
    ),
    CodeTestCase(
        prompt="""
        Please return the count of unique values in the pandas Series.

        The following Series will be the only function argument:
        data = pd.Series(['quetzal', 'quetzal', 'elk'], name='animal')
        """,
        data="""data = pd.Series(['quetzal', 'quetzal', 'elk'], name='animal')""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    result = data.value_counts()\n    return data""",
    ),
    CodeTestCase(
        prompt="""
        Please compute the difference between these consecutive values as an index object.

        The following Index will be the only function argument:
        data = pd.Index([10, 20, 30, 40, 50])
        """,
        data="""data = pd.Index([10, 20, 30, 40, 50])""",
        correct_function="""import pandas as pd\ndef correct_function(data):\n    return data.diff()""",
    ),
    CodeTestCase(
        prompt="""
        Please change the data type of all columns of the DataFrame to 'int64[pyarrow]'.
        Lastly, return the sum of the DataFrame.

        The following DataFrame will be the only function argument:
        df = pd.DataFrame({"a": [1, 1, 2, 1], "b": [None, 2.0, 3.0, 4.0]}, dtype="Int64")
        """,
        data="""data = pd.DataFrame({"a": [1, 1, 2, 1], "b": [None, 2.0, 3.0, 4.0]}, dtype="Int64")""",
        correct_function="""\nimport pandas as pd\nimport pyarrow as pa\ndef correct_function(data):\n    data = data.astype("int64[pyarrow]")\n    data.sum()\n    return data""",
    ),
]
