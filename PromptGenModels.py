from abc import ABC, abstractmethod
import pandas as pd
from dotenv import get_key
from completion import call_gpt3_5
from databench_eval.utils import load_sample, load_table
import numpy as np
import json

from sammo.instructions import Section, Paragraph, InputData
from sammo.search_op import one_of

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
    
class PromptGenModel(ABC):
    def __init__(self, load_func=our_load_sample):
        self.load_func = load_func

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
        return response

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
        df = self.load_func(dataset).to_csv()
        toReturn = f"""
        You are an assistant tasked with answering the questions asked of a given CSV in JSON format.
        You must answer in a single JSON with three fields:
        * "answer": answer using information from the provided CSV only.
        * "columns_used": list of columns from the CSV used to get the answer.
        * "explanation": A short explanation on why you gave that answer.
        
        Requirements:
        * Only respond with the JSON

        In the following CSV: {df}

        USER: {question}
        ASSISTANT: {{
        """
        return toReturn
    
    def postprocess(self, response, dataset, load_func):
        if response[:9] == "__Error__":
            return {"answer": response}
        response_dict = json.loads("{" + response)
        return response_dict["answer"]
    
class CodeBased(PromptGenModel):

    def __init__(self, num_retries=0):
        super().__init__()
        self._num_retries = num_retries

    def generate_prompt(self, row: dict) -> str:
        dataset = row["dataset"]
        question = row["question"]
        df = self.load_func(dataset)

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
        
    def fixerror(self, code: str, error: str, df: pd.DataFrame):
        datatypes = ""
        for column, dtype in df.dtypes.items():
            datatypes += (f"'{column}' dtype('{dtype}')\n")
        prompt =f"""
        A code and error for the code are the following. Please fix the error and give only the fixed code as output.
        * You only have access to pandas and numpy.
        * You only generate one function
        * Pay attention to the type formatting.
        * You cannot read files from disk.
        * You only generate the code. No need sample data or anything else.

        #Code
        {code}

        #Error
        {error}

        #Additional Info
        The given parameter for the function, df, is consisted of the following datatypes.
        {datatypes}

        #Output (complete this and return only this)
        def answer(df):
        """

        return call_gpt3_5([prompt])[0]
    
    def postprocess(self, response: str, dataset: str, load_func):
        try:
            df = load_func(dataset)
            global ans
            lead = "def answer():\n\t"
            
            local_vars = {"df": df, "pd": pd, "np": np}
            for attempt in range(self._num_retries+1):

                response_split = response.split("return")
                if len(response_split) > 1:
                    response = response_split[1].strip()
                exec_string = (
                    lead
                    + f"return {response}\n"
                    + "ans = answer()"
                )
                try:
                    exec(exec_string, local_vars)
                    break
                except Exception as e:
                    if attempt <= self._num_retries - 1:
                        response = self.fixerror(exec_string, e, df)
                    else:
                        return f"__CODE_ERROR__: {e}"
            ans = local_vars['ans']
            if isinstance(ans, pd.Series):
                ans = ans.tolist()
            elif isinstance(ans, pd.DataFrame):
                ans = ans.iloc[:, 0].tolist()
            return ans.split('\n')[0] if '\n' in str(ans) else ans
        except Exception as e:
            return f"error: {e}"
    
# This is the Few Shot In Context Learning Model.
class FiCL(PromptGenModel):

    def generate_prompt(self, row: dict) -> str:
        dataset = row["dataset"]
        question = row["question"]
        df = self.load_func(dataset)
        return f"""
        You are an assistant tasked with answering the questions asked of a given CSV in JSON format.
        You must answer in a single JSON with three fields:
        * "answer": answer using information from the provided CSV only.
        * "columns_used": list of columns from the CSV used to get the answer.
        * "explanation": A short explanation on why you gave that answer.
        
        Requirements:
        * Only respond with the JSON

        In the following CSV
        “csv
        passenger, wealth($)
        value1,value2.
        “

        USER: What is the name of the richest passenger?
        ASSISTANT: {{answer

        Consider this example to help you out:

        “csv
        student, GPA
        Alice, 4.0
        Bob, 3.5
        Charlie, 2.7
        “

        USER: What is the average GPA of the students?
        ASSISTANT: The average GPA is 3.4.
        """
    
    def postprocess(self, response, dataset, load_func):
        return super().postprocess(response, dataset, load_func)

    # This is the Chain of Thought model.
    class CoT(PromptGenModel):

        def generate_prompt(self, row: dict) -> str:
            dataset = row["dataset"]
            question = row["question"]
            df = self.load_func(dataset)
            return f"""
            You are an assistant tasked with answering the questions asked of a given CSV in JSON format.
            You must answer in a single JSON with three fields:
            * "answer": answer using information from the provided CSV only.
            * "columns_used": list of columns from the CSV used to get the answer.
            * "explanation": A short explanation on why you gave that answer.
            
            Requirements:
            * Only respond with the JSON

            In the following CSV
            “csv
            passenger, wealth($)
            value1,value2.
            “

            USER: What is the name of the richest passenger?
            ASSISTANT: {{answer

            Consider this example to help you out:

            “csv
            student, GPA
            Alice, 4.0
            Bob, 3.5
            Charlie, 2.7
            “

            USER: What is the average GPA of the students?
            ASSISTANT: The average GPA is 3.4. Here is how I calculated it:
            First you calculate the sum of all GPAs's which is 10.2
            Then you get the total number of students which is 3
            Then you divide sum of all GPA's by number of students which is 10.2/3=3.4
            """
    
    def postprocess(self, response, dataset, load_func):
        return super().postprocess(response, dataset, load_func)


# from databench_eval import Runner, Evaluator
# from completion import call_llm
# from datasets import Dataset


# qa_df = pd.DataFrame()



# # Convert to Dataset
# qa = Dataset.from_pandas(qa_df.head(100))
# evaluator = Evaluator(qa=qa)


# def load_table(dataset):
#     return pd.read_parquet(f"/Users/takumi/class/CSC-306/coding/Project Fin/CSC306_Final_Project/semeval_test/066_IBM_HR/all.parquet")


# def load_sample(dataset):
#     return pd.read_parquet(f"/Users/takumi/class/CSC-306/coding/Project Fin/CSC306_Final_Project/semeval_test/066_IBM_HR/sample.parquet")

# model=CodeBased()
# runner = Runner(
#     model_call = call_llm,
#     prompt_generator = model.generate_prompt,
#     qa=qa,
#     postprocess=lambda response, dataset: model.postprocess(
#         response, dataset, load_table
#     ),
#     batch_size=10,
# )
# runner_lite = Runner(
#     model_call = call_llm,
#     prompt_generator = model.generate_prompt,
#     qa=qa,
#     postprocess=lambda response, dataset: model.postprocess(
#         response, dataset, load_sample
#     ),
#     batch_size=10,
# )

# responses = runner.run(save="predictions.txt")
# responses_lite = runner_lite.run(save="predictions_lite.txt")
# print(f"DataBench accuracy is {evaluator.eval(responses)}")
# print(f"DataBench_lite accuracy is {evaluator.eval(responses_lite, lite=True)}")
