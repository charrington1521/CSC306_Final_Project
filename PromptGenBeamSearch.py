import PromptGenModels, evaluate
from datasets import load_dataset, Dataset
from databench_eval import Evaluator

from sammo.components import Output, GenerateText
from sammo.instructions import MetaPrompt, Section, Paragraph, InputData
from sammo.base import Template, EvaluationScore
from sammo.mutators import BagOfMutators, InduceInstructions, Paraphrase
from sammo.search import BeamSearch
from sammo.search_op import one_of
from sammo.dataformatters import PlainFormatter
from sammo.data import DataTable

from PromptGenModels import our_load_sample

from OpenAIRestRunner import OpenAIRestRunner

import os, dotenv

dotenv.load_dotenv()

#@TODO: We can make a fewshot dataset by creating correct code statements for some entries

#sample the data?
# @TODO: Save these changes to a dataset and keep ahold of it somewhere???

def load_data(split="train"):
    d_train = load_dataset("cardiffnlp/databench", "semeval", split=split)
    d_temp  = d_train.to_pandas()
    d_temp_all =  d_temp.rename(columns={"question": "input", "answer": "output"})
    d_temp_lite = d_temp.rename(columns={"question": "input", "sample_answer": "output"})
    columns = dict({})
    for dataset in d_temp_all['dataset']:
        if not dataset in columns:
            columns[dataset] = (list(our_load_sample(dataset).columns), list(our_load_sample(dataset).dtypes))

    d_temp_all["columns"]       = [columns[dataset][0] for dataset in d_temp_all['dataset']]
    d_temp_lite["columns"]      = [columns[dataset][0] for dataset in d_temp_all['dataset']]
    d_temp_all["column_types"]  = [columns[dataset][1] for dataset in d_temp_all['dataset']]
    d_temp_lite["column_types"] = [columns[dataset][1] for dataset in d_temp_all['dataset']]
    
    return DataTable.from_pandas(d_temp_all), DataTable.from_pandas(d_temp_lite)

# d_dev   = load_dataset("cardiffnlp/databench", "semeval", split="dev")

def evaluation(y_true: DataTable, y_pred: DataTable) -> EvaluationScore:
    #From Sammo User Guide
    y_true = y_true.outputs.normalized_values(on_empty="")
    y_pred = y_pred.outputs.normalized_values(on_empty="")
    n_correct = sum([y_p == y_t for y_p, y_t in zip(y_pred, y_true)])
    accuracy = n_correct / len(y_true)
    
    return EvaluationScore(accuracy)

#Apply mutation operators

runner = OpenAIRestRunner(
    api_config={"api_key": os.getenv("OPENAI_API_KEY")}
)

class InitialCandidates():
    def __init__(self, dtrain):
        self.dtrain = dtrain
    
    def __call__(self):
        d_formatter = PlainFormatter(

        )


        instructions = MetaPrompt(
            [
                Paragraph(
                    one_of(
                        [
                            ""
                        ]
                    ),
                    reference_id="instructions"
                ),
                Paragraph(
                    Template(
                        '''
                        {{#with input}}
                        import pandas as pd
                        import numpy as np
                        def answer (df: pd.DataFrame):
                            """
                            The columns are: {{this.columns}}
                            The df dtypes are: {{this.column_types}}
                            Returns: {{this.input}}"""
                            return{{/with}}'''
                    )
                )
            ],
            render_as="raw",
            data_formatter=d_formatter
        )

        return Output(
            instructions.with_extractor("raise"),
            minibatch_size=1,
            on_error="empty_result"
        )

'''
d_train_all, d_train_lite = load_data()

d_train_sample = d_train_lite.sample(10, seed=42)

mutation_operators = BagOfMutators(
    InitialCandidates(d_train_sample),
    InduceInstructions("#instructions", d_train_sample), #What is this doing
    Paraphrase("#instructions"),
    sample_for_init_candidates=False
)

empty_operators = BagOfMutators(
    InitialCandidates(d_train_sample)
)
'''

if __name__ == "__main__":
    pass

    # From Sammo User Guide
    # prompt_optimizer = BeamSearch(
    #     runner,
    #     empty_operators, #Use an actual bag of operators here
    #     evaluation,
    #     maximize=True,
    #     depth=1, #What is a good depth?
    #     mutations_per_beam=2,
    #     n_initial_candidates=1,
    #     beam_width=4, #What is a good width?
    #     add_previous=True
    # )

    # prompt_optimizer.fit(d_train_sample)
    # prompt_optimizer.show_report()

    # print(
    #     Output(
    #         Template(
    #             '''
    #             {{#with input}}
    #             import pandas as pd
    #             import numpy as np
    #             def answer (df: pd.DataFrame):
    #                 """
    #                 The columns are: {{this.columns}}
    #                 The df dtypes are: {{this.column_types}}
    #                 Returns: {{this.input}}
    #                 """
    #                 return{{/with}}'''
    #             )
    #     ).run(runner, d_train_sample)
    # )