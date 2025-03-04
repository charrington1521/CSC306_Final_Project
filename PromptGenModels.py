from Model import Model
from abc import abstractmethod
from databench_eval.utils import load_sample

class PromptGenModel(Model):

    @abstractmethod
    def generate_prompt(row: dict) -> str:
        '''Given a question and a table outputs a prompt
        @param question: The question about table
        @param table: The table in question
        @return: a string prompt
        '''
        dataset = row["dataset"]
        question = row["question"]

        pass

class CompetitionBaseline(PromptGenModel):

    def generate_prompt(row: dict) -> str:
            dataset = row["dataset"]
            question = row["question"]
            df = load_sample(dataset)
            return f'''
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
            """Returns the answer to the question: How many rows are there in the dataframe? """
            df.columns = {list(df.columns)}
            return df.shape[0]

        # This is the question you have to answer
        def answer(df: pd.DataFrame):
            """Returns the answer to the question: {question} """
            df.columns = {list(df.columns)}
            return'''

class ZiCL(PromptGenModel):

    def generate_prompt(row: dict) -> str:
        return super().generate_prompt()
    
class CodeBased(PromptGenModel):

    def generate_prompt(row: dict) -> str:
        return super().generate_prompt()
