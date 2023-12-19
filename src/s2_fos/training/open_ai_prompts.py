import sys
from pathlib import Path

# Add the src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json

import numpy as np
import re
import openai
import time
import os
from s2_fos.constants import LABELS, PROJECT_ROOT_PATH

# Define constants
MODEL = "gpt-3.5-turbo"
MAX_RETRIES = 5
RETRY_DELAY = 5  # Time in seconds to wait before retrying

# Check and Set OpenAI API key
if 'OPENAI_API_KEY' in os.environ:
    openai.api_key = os.environ['OPENAI_API_KEY']
else:
    raise Warning("OPENAI_API_KEY not found in environment variables. Please set it before running this script.")

def tokenize(text):
    return re.findall(r'\b\w+\b', text) if text else []

def truncate_entity(entity, max_tokens=800):
    tokens = tokenize(entity)
    return ' '.join(tokens[:max_tokens]) if tokens and len(tokens) > max_tokens else entity

def call_openai_api(model, messages, temperature=0):
    return openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)

def search_for_fos_response(response):
    return [field for field in LABELS if field in response]

def prompt_with_journal(title, abstract, journal_name):
    fos_string = "', '".join(LABELS)
    return [
        {"role": "system", "content": "You are a highly intelligent and accurate information extraction system. "
                                      "You take title, abstract, journal name of a scientific article as input and your"
                                      " task is to classify the scientific field of study of the passage."},
        {"role": "user", "content": f"You need to classify it with key: 'field_of_study' assign as many "
                                    f"'field_of_study' as you find it fit: '{fos_string}'. Only select from the above "
                                    f"list, or 'Other'."},
        {"role": "assistant", "content": f"```python\ntitle = {title}\nabstract = {abstract}\n"
                                         f"journal_name = {journal_name}\nfield_of_study = []"}
    ]

def process_paper(paper_sha, title, abstract, journal_name, model, open_ai_calls):
    key = f'{paper_sha}_{model}'
    if key not in open_ai_calls:
        truncated_title = truncate_entity(title)
        truncated_abstract = truncate_entity(abstract)
        messages = prompt_with_journal(truncated_title, truncated_abstract, journal_name)

        for _ in range(MAX_RETRIES):
            try:
                result = call_openai_api(model, messages, temperature=0)
                if not search_for_fos_response(result['choices'][0]['message']['content']):
                    assistant_reply1 = result['choices'][0]['message']['content'].replace("```", "``")
                    content_string = (f"Can you please map your response {assistant_reply1} to the following fields: "
                                      f"{', '.join(LABELS)}")
                    messages.extend([
                        {"role": "user", "content": content_string},
                        {"role": "assistant", "content": "```python \n{'field_of_study': ["}
                    ])
                    result = call_openai_api(model, messages, temperature=0)
                open_ai_calls[key] = {'prompt': messages, 'result': result}
                break
            except openai.OpenAIError as e:
                print(f"Error: {e}")
                time.sleep(RETRY_DELAY)
        else:
            print("Max retries reached. Skipping this API call.")
    return open_ai_calls

def save_openai_calls(open_ai_calls, filename):
    with open(filename, 'w') as file_handle:
        json.dump(open_ai_calls, file_handle)

def main():
    # Path to your .jsonl file
    jsonl_file_path = os.path.join(PROJECT_ROOT_PATH, '..', 'data',
                                   'paper_title_abstract_example_file.json')

    # Initialize an empty list to store the JSON objects
    json_list = []

    # Open the JSONL file and read each line
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            # Parse the JSON string into a Python dictionary
            json_obj = json.loads(line.strip())
            # Append the dictionary to the list
            json_list.append(json_obj.values())

    # Loading jsonl file
    meta_data_np = np.array(json_list)  # Replace with actual data
    open_ai_calls = {}
    for idx, (paper_sha, title, abstract, journal_name) in enumerate(meta_data_np):
        open_ai_calls = process_paper(paper_sha, title, abstract, journal_name, MODEL, open_ai_calls)
        if (idx + 1) % 100 == 0:
            save_openai_calls(open_ai_calls, os.path.join(PROJECT_ROOT_PATH, '..', 'data',
                                                          f'open_ai_most_popular_{idx + 1}.json'))

    save_openai_calls(open_ai_calls,
                      os.path.join(PROJECT_ROOT_PATH, '..', 'data',
                                   'open_ai_most_popular_final.json'))

if __name__ == "__main__":
    main()