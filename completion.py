'''UPDATE THIS FILE
'''
from tracking import default_client, read_usage
from typing import List
import re

def complete(client, s: str, model="gpt-3.5-turbo-0125", *args, **kwargs):
    response = client.chat.completions.create(messages=[{"role": "user", "content": s}],
                                              model=model,
                                              temperature=0.6,
                                              *args, **kwargs)
    return [choice.message.content for choice in response.choices][0]

def call_llm_gpt4o_mini(prompts: List[str], *args, **kwargs) -> List[str]:
    results = []
    for prompt in prompts:
        escaped = re.sub('"', '\"', prompt)
        results.append(complete(default_client, escaped, model="gpt-4o-mini"))
    return results

def call_llm_gpt3_5(prompts: List[str], *args, **kwargs) -> List[str]:
    results = []
    for prompt in prompts:
        escaped = re.sub('"', '\"', prompt)
        results.append(complete(default_client, escaped))
    return results
