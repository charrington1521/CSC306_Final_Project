from Model import Model
from abc import abstractmethod
import pandas as pd
import numpy as np
from dotenv import get_key
from databench_eval.utils import load_sample
from completion import call_llm


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

def our_load_sample(name: any) -> pd.DataFrame:
    try:
        load_sample(name)
    except:
        return pd.read_parquet(f"{test_path}{name}/sample.parquet")
    
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
        dataset = row["dataset"]
        question = row["question"]

        pass

    def postprocess(self, response: str, row: dict): 
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

class ZiCL(PromptGenModel):

    def generate_prompt(self, row: dict) -> str:
        dataset = row["dataset"]
        question = row["question"]
        df = our_load_sample(dataset)
        return """
        You are an assistant tasked with answer-
        ing the questions asked of a given CSV in
        JSON format. You must answer in a single
        JSON with three fields:
        * "answer": answer using information from
        the provided CSV only.
        * "columns_used": list of columns from the
        CSV used to get the answer.
        * "explanation": A short explanation on why
        you gave that answer.
        Requirements:
        * Only respond with the JSON.
        In the following CSV
        “csv
        passenger,wealth($)
        value1,value2.
        “
        USER: What is the name of the richest pas-
        senger?
        ASSISTANT: {"answer:̈
        """
    
class CodeBased(PromptGenModel):

    def generate_prompt(self, row: dict) -> str:
        dataset = row["dataset"]
        question = row["question"]
        df = our_load_sample(dataset)

        datatypes = ""
        for column, dtype in df.dtypes.items():
            datatypes += (f"'{column}' dtype('{dtype}')\n")

        toReturn = f'''
        You are a pandas code generator. Your goal is to complete the function provided.
        * You must not write any more code apart from that.
        * You only have access to pandas and numpy.
        * You only generate one function
        * Pay attention to the type formatting.
        * You cannot read files from disk.

        The dtypes of the given input df are:\n
        {datatypes}

        You should complete the following code without changing it so that it gives the right answer for "{question}".
        
        # Code you need to complete
        import pandas as pd
        import numpy as np
        def answer (df):
            return 
        '''
        return toReturn
    

    def postprocess(self, response: str, dataset: str, load_table=our_load_sample, max_retries=3):
        try:
            df = load_table(dataset)
            global ans
            lead = "def"
            
            local_vars = {"df": df, "pd": pd, "np": np}
            for attempt in range(max_retries):
                function_body = response.split("def")[1].split("return")[0]
                return_statement = response.split("def")[1].split("return")[1].strip()
                exec_string = (
                    lead
                    + function_body
                    + f"return {return_statement}\n"
                    + "ans = answer(df)"
                )
                try:
                    exec(exec_string, local_vars)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        # rint(f"Execution failed (attempt {attempt+1}): {str(e)}")
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

        return call_llm([prompt])[0]


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