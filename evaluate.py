from datasets import load_dataset, DatasetDict
from databench_eval import Runner, Evaluator
from databench_eval.utils import load_qa, load_table
from Model import Model
from dotenv import get_key

test_path = get_key('.env', 'TEST_PATH')

if test_path == None:
    raise(Exception("No test path has been defined. Download the test data at https://drive.google.com/file/d/1IpSi0gNPYj9a9lNbWPsL3TxIBILoLsfE/view and add the path to it to your .env file"))
else:
    pass

semeval_test_qa: DatasetDict = load_dataset(test_path, split="test")
