from datasets import load_dataset, DatasetDict, Dataset
from databench_eval import Runner, Evaluator
from PromptGenModels import PromptGenModel, our_load_sample, our_load_table
from completion import call_llm_gpt3_5, call_llm_gpt4o_mini
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
    save_path = getcwd()+"/submissions/"


### Initialize the Test Data ##################################################
qa_df = pd.DataFrame()

with open(test_path+"answers/answers.txt", "r") as f:
    answers = f.read().splitlines()

with open(test_path+"answers/answers_lite.txt", "r") as f:
    sample_answers = f.read().splitlines()

with open(test_path+"answers/semantics.txt", "r") as f:
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
qa_test = Dataset.from_pandas(qa_df)

#@TODO: allow evaluate model to use datasets other than the test data
#@TODO: custom save_path as an argument instead?
def evaluate_promptGenModel(model: PromptGenModel, eval_dataset: Dataset, save: bool = False, llm_call=call_llm_gpt3_5) -> List[str]:
    '''Returns a list of responses by a PromptGenerationModel and 
    prints the performance of the responses on the test data.
    @param model: a PromptGenModel
    @param save: False by default. When true the responses will be saved
        to a "{model_name}_predictions.txt" file on the save path defined
        for this module
    '''
    runner = Runner(
        model_call = llm_call,
        prompt_generator = model.generate_prompt,
        postprocess=lambda response, dataset: model.postprocess(
            response, dataset, load_func=our_load_table
        ),
        qa=eval_dataset,
        batch_size=10,
    )

    runner_lite = Runner(
        model_call = llm_call,
        prompt_generator = model.generate_prompt,
        postprocess=lambda response, dataset: model.postprocess(
            response, dataset, load_func=our_load_sample
        ),
        qa=eval_dataset,
        batch_size=10,
    )

    if save: #@TODO: add a runner_lite to follow submission format for task
        model_name = model.__class__.__name__
        path = f"{save_path}{model_name}_predictions.txt"
        with open(path, "tr") as f:
            pass
        responses = runner.run(save=path)
        responses_lite = runner_lite.run(save=path)

    else:
        responses = runner.run()
        responses_lite = runner_lite.run()

    evaluator = Evaluator(qa=eval_dataset)
    print(f"DataBench accuracy is {evaluator.eval(responses)}")
    print(evaluator.evals)
    print(f"DataBench lite accuracy is {evaluator.eval(responses_lite, lite=True)}")
    print(evaluator.evals)

    return responses, responses_lite