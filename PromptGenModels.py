from Model import Model
from abc import abstractmethod
import pandas as pd
from dotenv import get_key
from databench_eval.utils import load_sample, load_table
import numpy as np

test_path = get_key('.env', 'TEST_PATH')

if test_path == None:
    raise(
        Exception(
            '''No test path has been defined. Download the test data at 
            https://drive.google.com/file/d/1IpSi0gNPYj9a9lNbWPsL3TxIBILoLsfE/
            view and add the path to it to your .env file.
            '''
            )
        )

def our_load_sample(name: str) -> pd.DataFrame:
    try:
        toReturn = load_sample(name)
    except:
        toReturn = pd.read_parquet(f"{test_path}{name}/sample.parquet")
    return toReturn
    
def our_load_table(name: str) -> pd.DataFrame:
    try:
        toReturn = load_table(name)
    except:
        toReturn = pd.read_parquet(f"{test_path}{name}/all.parquet")
    return toReturn
    
class PromptGenModel(Model):

    def __init__(self):
        pass

    @abstractmethod
    def generate_prompt(self, row: dict) -> str:
        '''Given a question and a table outputs a prompt
        @param question: The question about table
        @param table: The table in question
        @return: a string prompt
        '''
        pass

    @abstractmethod
    def postprocess(self, response: str, dataset: str, load_func):
        pass

class CompetitionBaseline(PromptGenModel):

    def generate_prompt(self, row: dict) -> str:
            dataset = row["dataset"]
            question = row["question"]
            df = our_load_sample(dataset)
            return f"""
        You are a pandas code generator. Your goal is to complete the function provided.
        * You must not write any more code apart from that.
        * You only have access to pandas and numpy.
        * Pay attention to the type formatting .
        * You cannot read files from disk.
        * The answer should be short and concise, in the format I specify.
        * DO NOT do anything else other than filling the function provided.
        * Answer in one of the following formats, depending on the question
            1. True/False (do not answer with np.True_ or np.False_, but rather True or False)
            2. with a value from the dataframe, (category/number)
            3. with a list of values (either categories or numbers)

        import pandas as pd
        import numpy as np

        # This is an example
        def example(df: pd.DataFrame):
            '''Returns the answer to the question: How many rows are there in the dataframe? '''
            df.columns = {list(df.columns)}
            return df.shape[0]

        # This is the question you have to answer
        def answer(df: pd.DataFrame):
            '''Returns the answer to the question: {question} '''
            df.columns = {list(df.columns)}
            return"""

    def postprocess(self, response, dataset, load_func):
        return super().postprocess(response, dataset, load_func)

class ZiCL(PromptGenModel):

    def generate_prompt(self, row: dict) -> str:
        dataset = row["dataset"]
        question = row["question"]
        df = our_load_sample(dataset)
        return f"""
        You are a pandas code generator. Your goal is to complete the function provided.
        * You must not write any more code apart from that.
        * You only have access to pandas and numpy.
        * Pay attention to the type formatting.
        * You cannot read files from disk.
        * The answer should be short and concise, in the format I specify.
        * DO NOT do anything else other than filling the function provided.
        * Answer in one of the following formats, depending on the question
            1. True/False (do not answer with np.True_ or np.False_, but rather True or False)
            2. with a value from the dataframe, (category/number)
            3. with a list of values (either categories or numbers)

        import pandas as pd
        import numpy as np

        # This is an example
        def example(df: pd.DataFrame):
            '''Returns the answer to the question: How many rows are there in the dataframe? '''
            df.columns = {list(df.columns)}
            return df.shape[0]

        # This is the question you have to answer
        def answer(df: pd.DataFrame):
            '''Returns the answer to the question: {question} '''
            df.columns = {list(df.columns)}
            return"""
    
    def postprocess(self, row, dataset, load_func):
        return super().postprocess(row, dataset, load_func)
    
class CodeBased(PromptGenModel):

    def generate_prompt(self, row: dict) -> str:
        dataset = row["dataset"]
        question = row["question"]
        print("------------------------")
        print(dataset)
        df = our_load_sample(dataset)
        print(question)
        datatypes = ""
        for column, dtype in df.dtypes.items():
            datatypes.join(f"'{column}' dtype('{dtype}')\n")

        toReturn = f'''
        You are a pandas code generator. Your goal is to complete the function provided.
        * Pay attention to the type formatting.
        * The answer should be short and concise, in the format I specify.
        * DO NOT do anything else other than filling the function provided.
        * DO NOT tag your response with markdown

        import pandas as pd
        import numpy as np

        # This is an example
        def example(df: pd.DataFrame):
            """Returns the answer to the question: How many rows are there in the dataframe? """
            df.columns = {list(df.columns)}
            return df.shape[0]

        import pandas as pd
        import numpy as np
        def answer (df: pd.DataFrame):
            """
            Returns: {question}
            The columns are: {list(df.columns)}
            The df dtypes are: {list(df.dtypes)}"""

            return 
        '''

        return toReturn
    
    def postprocess(self, response: str, dataset: str, load_func):
        try:
            df = load_func(dataset)
            global ans
            lead = """
    def answer(df):
        return """
            exec_string = (
                lead +
                response
                + "\nans = answer(df)"
            )
            local_vars = {"df": df, "pd": pd, "np": np}
            print(exec_string)
            exec(exec_string, local_vars)
            ans = local_vars['ans']
            if isinstance(ans, pd.Series):
                ans = ans.tolist()
            elif isinstance(ans, pd.DataFrame):
                ans = ans.iloc[:, 0].tolist()
            return ans.split('\n')[0] if '\n' in str(ans) else ans
        except Exception as e:
            print(e)
            return f"__CODE_ERROR__: {e}"
