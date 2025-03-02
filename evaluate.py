from datasets import load_dataset, DatasetDict
from databench_eval import Runner, Evaluator
from databench_eval.utils import load_qa, load_table
from PromptGenModels import PromptGenModel
from completion import complete
from dotenv import get_key
from os import getcwd
import pandas as pd
from typing import List

### Load Environment Variables ################################################
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

save_path = get_key('.env', 'SAVE_PATH')

if save_path == None:
    print("Save path was undefined, defaulting to current working directory.")
    save_path = getcwd()


### Initialize the Test Data ##################################################
qa_df = pd.DataFrame()

with open(test_path+"answers.txt", "r") as f:
    answers = f.read().splitlines()

with open(test_path+"answers_lite.txt", "r") as f:
    sample_answers = f.read().splitlines()

with open(test_path+"semantics.txt", "r") as f:
    semantics = f.read().splitlines()

try:
    qa_df = pd.read_csv(
        test_path + "test_qa.csv"
    )
except:
    raise(Exception(f"Could not find test_qa.csv at the test path {test_path}. Double check your .env file."))

qa_df["answer"] = answers
qa_df["sample_answer"] = sample_answers
qa_df["type"] = semantics

### Evaluation Implementation #################################################
evaluator = Evaluator.eval(qa=qa_df)

#@TODO: allow evaluate model to use datasets other than the test data
#@TODO: custom save_path as an argument instead?
def evaluate_promptGenModel(model: PromptGenModel, save: bool = False) -> List[str]:
    '''Returns a list of responses by a PromptGenerationModel and 
    prints the performance of the responses on the test data.
    @param model: a PromptGenModel
    @param save: False by default. When true the responses will be saved
        to a "{model_name}_predictions.txt" file on the save path defined
        for this module
    '''
    runner = Runner(
        model_call = complete,
        prompt_generator = model.generate_prompt,
        qa=qa_df,
        batch_size=10
    )

    if save: #@TODO: add a runner_lite to follow submission format for task
        model_name = model.__class__.__name__
        path = f"{save_path}{model_name}_predictions.txt"
        responses = runner.run(save=path)
    else:
        responses = runner.run()
    
    print(f"DataBench accuracy is {evaluator.eval(responses)}")
    return responses