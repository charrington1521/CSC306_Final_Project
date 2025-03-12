import PromptGenModels, evaluate
from datasets import load_dataset, Dataset
from databench_eval import Evaluator

from sammo.components import Output, GenerateText
from sammo.instructions import MetaPrompt, Section, Paragraph, InputData
from sammo.base import Template, EvaluationScore
from sammo.mutators import BagOfMutators, InduceInstructions, Paraphrase
from sammo.search import BeamSearch
from sammo.search_op import one_of, many_of
from sammo.dataformatters import PlainFormatter
from sammo.data import DataTable

from PromptGenModels import CodeBased, our_load_sample, our_load_table

from typing import Tuple

from OpenAIRestRunner import OpenAIRestRunner

import os, dotenv

dotenv.load_dotenv()

#@TODO: We can make a fewshot dataset by creating correct code statements for some entries

#sample the data?
# @TODO: Save these changes to a dataset and keep ahold of it somewhere???

def load_data(split="train") -> Tuple[DataTable, DataTable]:
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

const_code_based = CodeBased()

from sammo.base import Result, Component, Runner, VerbatimText
from frozendict import frozendict
from typing import Optional, Union, Any

class Postprocess(Component):
    def __init__(self, child: Union[Any, str], dataset: str, lite, reference_id: Optional[str] = None, reference_classes: Optional[list[str]] = None):
        super().__init__(child, reference_id, reference_classes)
        self.dataset = dataset
        self.load_func = our_load_sample if lite else our_load_table

    async def _call(self, runner: Runner, context: dict, dynamic_context: Optional[frozendict]) -> Result:
        dataset = await self.dataset(runner, context, dynamic_context)
        intermediate_result = await self._child(runner, context, dynamic_context)
        # print(intermediate_result.value[0], dataset.value)
        return Result(const_code_based.postprocess(intermediate_result.value[0], dataset.value, self.load_func), parent=intermediate_result, op=self)

# d_dev   = load_dataset("cardiffnlp/databench", "semeval", split="dev")

def evaluation(y_true: DataTable, y_pred: DataTable) -> EvaluationScore:
    #From Sammo User Guide
    # print(y_pred)
    # print(y_true)
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
    def __init__(self, dtrain, lite=False):
        self.dtrain = dtrain
        self.lite = lite
    
    def __call__(self):
        d_formatter = PlainFormatter(

        )


        instructions = MetaPrompt(
            [
                Paragraph(
                    "You are a pandas code generator. Your goal is to complete the function provided.",
                    reference_id="instructions"
                ),
                Paragraph(
                    many_of(
                        5,
                        [
                            "* Pay attention to the type formatting.",
                            "* The answer should be short and concise, in the format I specify.",
                            "* Do NOT do anything else other than filling in the function provided.",
                            "* Do NOT tag your response with markdown.",
                            "* You must not write any more code apart from that.",
                            "* You cannot read files from disk.",
                            "* You only have access to pandas and numpy.",
                            '''* Answer in one of the following formats, depending on the question
                                1. True/False (do not answer with np.True_ or np.False_, but rather True or False)
                                2. with a value from the dataframe, (category/number)
                                3. with a list of values (either categories or numbers)'''
                        ]
                    ),
                    reference_id="requirements"
                ),
                Paragraph(
                    '''import pandas as pd
                    import numpy as np
                    
                    # This is an example
                    def example(df: pd.DataFrame):
                        """Returns the answer to the question: How many rows are there in the dataframe? """
                        df.columns = {list(df.columns)}
                        return df.shape[0]''',
                    reference_id="example"
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
                    ),
                    reference_id="prompt"
                )
            ],
            render_as="raw",
            data_formatter=d_formatter
        )

        return Output(
            Postprocess(
                instructions.with_extractor("raise"),
                Template("{{input.dataset}}"),
                self.lite
            ),
            minibatch_size=1,
            on_error="empty_result"
        )

d_train_all, d_train_lite = load_data()

d_train_sample = d_train_lite.sample(3, seed=42)

mutation_operators = BagOfMutators(
    InitialCandidates(d_train_sample),
    InduceInstructions('#requirements', d_train_sample),
    Paraphrase("#instructions"),
    Paraphrase("#requirements"),
    sample_for_init_candidates=False
)

empty_operators = BagOfMutators(
    InitialCandidates(d_train_sample)
)

if __name__ == "__main__":
    pass

    # From Sammo User Guide
    prompt_optimizer = BeamSearch(
        runner,
        mutation_operators,
        evaluation,
        maximize=True,
        depth=4,
        mutations_per_beam=2,
        n_initial_candidates=2,
        beam_width=2,
        add_previous=True
    )

    prompt_optimizer.fit(d_train_sample)
    prompt_optimizer.show_report()

    print(prompt_optimizer.best_prompt)

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