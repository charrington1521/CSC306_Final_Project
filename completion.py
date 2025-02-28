'''UPDATE THIS FILE
'''
from tracking import new_default_client, read_usage

client = new_default_client()

def complete(client, s: str, model="gpt-3.5-turbo-0125", *args, **kwargs):
    response = client.chat.completions.create(messages=[{"role": "user", "content": s}],
                                              model=model,
                                              *args, **kwargs)
    return [choice.message.content for choice in response.choices]