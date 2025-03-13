# Semeval Task: Question Answering over Tabular Data

## Setup

### 1. Create a virtual environment and set the interpreter

**In Terminal**
- create virtual environment

        python3.9 -m venv \[path_to_venv]

**In VS Code**
- with project folder open:
    ctrl-shift-p > "select python interpreter" > Enter Interpreter Path > \[path_to_venv]
    
**In Terminal**
- navigate to the project folder
    
      cd [path_to_project_folder]

- activate the virtual environment
    
      source [path_to_venv]/bin/activate

- install reqs

      pip install -r requirements.txt

### 2- Download the Test Data

**Assumes a non-Mac OS**

- Visit [this link](https://drive.google.com/file/d/1IpSi0gNPYj9a9lNbWPsL3TxIBILoLsfE/view) to download competition.zip

- Visit [this repo](https://github.com/jorses/databench_eval/blob/main/examples/answers.zip) and select "View raw" to download answers.zip

- Move both of these to an appropriate location.

- Extract both zip files. Rename competition/competition to semeval_test. Move semeval_test outside of the competition folder. Move answers inside this semeval_test folder. 

- You may now delete the .zip files and the competition folder as well as the __MACOSX folder within semeval_test/answers.

When complete it should match the form

<pre>
📦semeval_test
 ┣ 📂066_IBM_HR
 ┃ ┣ 📜.DS_Store
 ┃ ┣ 📜all.parquet
 ┃ ┗ 📜sample.parquet
 ┣ 📂. . .
 ┣ 📂answers
 ┃ ┣ 📜answers.txt
 ┃ ┣ 📜answers_lite.txt
 ┃ ┗ 📜semantics.txt
 ┣ 📜.DS_Store
 ┣ 📜README.md
 ┗ 📜test_qa.csv
 </pre>

### 3- .env Variables

**There are two .env variables to run this project**

- TEST_PATH: The path at which semeval_test was downloaded

- OPENAI_API_KEY: a valid key to use OpenAI

### 4- Create a github branch for your work

**In VS Code**
-   with project folder open:
        
        ctrl-shift-p > Git: Create Branch > \[new_branch_name]

    This automatically transfers you to the new_branch

    Now when changes are committed AND pushed a merge request can be created at the Github repo

## Implementation

### 1- Replication

**PromptGenModel**

The two baselines defined in the paper [TODO ADD PAPER LINK ](https://aclanthology.org/2024.lrec-main.1179/) are both
prompt generation based models. That is, the models are given a question and a table and use the info to generate a prompt for an LLM. In our case this is hooked up to ChatGpt3.5-Turbo.
This means all PromptGenerationModel models we create have a method "generate_prompt" that can be called during evaluation using tools provided by the shared task. This makes the task of implementing baselines straightforward: complete the "generate_prompt" methods.
We will be doing this for the two baselines from the paper: Zero-shot In-Context Learning and Code Based. 
Both of these baselines will be included in our "PromptGenModels.py" module

- Zero-shot In-Context Learning

- Code Based PrompGenModel

**evaluate.py**

This file comes from [this link](https://github.com/jorses/databench_eval/blob/main/examples/stablecode.py), which is in the same repository as the one we got the answers.zip from.

**tracking.py**

This file comes from the last project. It is only added so that we can check our usage.

### 2- Improvements
#TODO Whitelist the gitignore file


#### FiCL (few-shots in Context Learning) model

This is the model with the few-shots prompting and defined as *class FiCL(PromptGenModel)* in "PromptGenModels.py".

#### CoT (Chain of Thought) model

This is the model with the chain of thought strategy, and defined as *class CoT(PromptGenModel)* in "PromptGenModels.py".

#### Code-based with fixerror function model

This is the advanced code-based model that has a function to fix erros which happens on the generated code. 
This is defined as *class CoT(PromptGenModel)* in "PromptGenModels.py".

#### SAMMO model

NOT COMPLETED

##### Sammo BeamSearch Issue

There is a issue with compatibility of our python version and the packages and type hinting used in the search.py file included with sammo. 
When running PromptGenBeamSearch one will need to navigate to 
"path_to_venv/lib/python3.9/site-packages/sammo/seach.py". On line 30 add ``` from typing import Optional, Union ```. 
- Use ctrl-f to locate all uses of the "|" operator. 
- Replace uses of the pipe operator such as 
``` Callable[[], Output] | None``` with ``` Optional[Callable[[], Output]]``` 
and uses such as 
``` str | Path ``` with ``` Union[str, Path] ```.
- Note that on line **163** (or so) the pipe operator **SHOULD NOT** be replaced.

