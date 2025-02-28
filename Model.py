from typing import List
from abc import abstractmethod
import openai

class Model():
    '''Base model for Tabular Question Answering
    >more info here
    '''
    def __init__(self) -> None:
        #link to a Transformer?
        pass

    @abstractmethod
    def generate_prompt(question: str, table: any) -> str:
        '''Given a question and a table outputs a prompt
        @param question: The question about table
        @param table: The table in question
        @return: a string prompt
        '''
        pass
