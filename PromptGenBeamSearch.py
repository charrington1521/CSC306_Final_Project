import PromptGenModels, evaluate
from datasets import load_dataset, Dataset
from databench_eval import Evaluator

from sammo.base import Template, EvaluationScore
from sammo.mutators import BagOfMutators, InduceInstructions, Paraphrase
from sammo.search import BeamSearch
from sammo.data import DataTable

from OpenAIRestRunner import OpenAIRestRunner

import os, dotenv

dotenv.load_dotenv()

#sample the data?
d_train = load_dataset("cardiffnlp/databench", "semeval", split="train")
d_train = DataTable.from_json(d_train.to_json())

d_dev   = load_dataset("cardiffnlp/databench", "semeval", split="dev")


#How tf to write this?
def evaluation(y_true: DataTable, y_pred: DataTable) -> EvaluationScore:
    Dataset.from_json(y_true.to_json())
    evaluator = Evaluator(y_true)    
    
    return None #@TODO: COMPLETE THIS

#Apply mutation operators

mutation_operators = BagOfMutators(
    #InitialCandidates(d_train),
    InduceInstructions("#instructions"), #What is this doing
    Paraphrase("#instructions"),
    sample_for_init_candidates=False
)

runner = OpenAIRestRunner(
    api_config={"api_key": os.getenv("OPENAI_API_KEY")}
)

prompt_optimizer = BeamSearch(
    runner,
    mutation_operators,
    evaluation,
    maximize=True,
    depth=3, #is this what we want
    mutations_per_beam=2,
    n_initial_candidates=4,
    beam_width=4, #is this what we want
    add_previous=True
)

prompt_optimizer.fit(d_train)
prompt_optimizer.show_report()