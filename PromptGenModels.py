from Model import Model
from abc import abstractmethod

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

class ZiCL(PromptGenModel):

    def generate_prompt(row: dict) -> str:
        return super().generate_prompt()
    
class CodeBased(PromptGenModel):

    def generate_prompt(row: dict) -> str:
        return super().generate_prompt()