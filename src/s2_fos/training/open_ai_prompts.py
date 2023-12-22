import json

import numpy as np
import re
from openai import OpenAI, OpenAIError

import time
import os
from s2_fos.constants import LABELS, PROJECT_ROOT_PATH

# Define constants
MODEL = "gpt-3.5-turbo"
MAX_RETRIES = 5
RETRY_DELAY = 5  # Time in seconds to wait before retrying

# Check and Set OpenAI API key
if "OPENAI_API_KEY" in os.environ:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
else:
    raise Warning(
        "OPENAI_API_KEY not found in environment variables. Please set it before running this script."
    )


def tokenize(text):
    return re.findall(r"\b\w+\b", text) if text else []


def truncate_entity(entity, max_tokens=800):
    tokens = tokenize(entity)
    return (
        " ".join(tokens[:max_tokens]) if tokens and len(tokens) > max_tokens else entity
    )


def call_openai_api(model, messages, temperature=0):
    return client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )


def search_for_fos_response(response):
    return [field for field in LABELS if field in response]


def prompt_with_journal(title, abstract, journal_name):
    fos_string = "', '".join(LABELS)
    return [
        {
            "role": "system",
            "content": "You are a highly intelligent and accurate information extraction system. "
            "You take title, abstract, journal name of a scientific article as input and your"
            " task is to classify the scientific field of study of the passage.",
        },
        {
            "role": "user",
            "content": f"You need to classify it with key: 'field_of_study' assign as many "
            f"'field_of_study' as you find it fit: '{fos_string}'. Only select from the above "
            f"list, or 'Other'.",
        },
        {
            "role": "assistant",
            "content": f"```python\ntitle = {title}\nabstract = {abstract}\n"
            f"journal_name = {journal_name}\nfield_of_study = []",
        },
    ]


def process_paper(paper_sha, title, abstract, journal_name, model, open_ai_calls):
    key = f"{paper_sha}_{model}"
    if key not in open_ai_calls:
        truncated_title = truncate_entity(title)
        truncated_abstract = truncate_entity(abstract)
        messages = prompt_with_journal(
            truncated_title, truncated_abstract, journal_name
        )

        for _ in range(MAX_RETRIES):
            try:
                chat_completion = call_openai_api(model, messages, temperature=0)
                message_content = chat_completion.choices[0].message.content
                if not search_for_fos_response(message_content):
                    assistant_reply1 = message_content.replace("```", "``")
                    content_string = (
                        f"Can you please map your response {assistant_reply1} to the following fields: "
                        f"{', '.join(LABELS)}"
                    )
                    messages.extend(
                        [
                            {"role": "user", "content": content_string},
                            {
                                "role": "assistant",
                                "content": "```python \n{'field_of_study': [",
                            },
                        ]
                    )
                    chat_completion = call_openai_api(model, messages, temperature=0)
                open_ai_calls[key] = {
                    "prompt": messages,
                    "result": serialize_chat_completion(chat_completion),
                }
                break
            except OpenAIError as e:
                print(f"Error: {e}")
                time.sleep(RETRY_DELAY)
        else:
            print("Max retries reached. Skipping this API call.")
    return open_ai_calls


# Convert the ChatCompletion object to a serializable dictionary
def serialize_chat_completion(completion):
    return {
        "id": completion.id,
        "choices": [
            {
                "finish_reason": choice.finish_reason,
                "index": choice.index,
                "logprobs": choice.logprobs,  # This might be None or another object that needs serialization
                "message": {
                    "content": choice.message.content,
                    "role": choice.message.role,
                    # Include other fields from ChatCompletionMessage if necessary
                },
            }
            for choice in completion.choices
        ],
        "created": completion.created,
        "model": completion.model,
        "object": completion.object,
        "system_fingerprint": completion.system_fingerprint,
        "usage": {
            "completion_tokens": completion.usage.completion_tokens,
            "prompt_tokens": completion.usage.prompt_tokens,
            "total_tokens": completion.usage.total_tokens,
        }
        if completion.usage
        else None,
        # Include other fields from ChatCompletion if necessary
    }


def save_openai_calls(open_ai_calls, filename):
    with open(filename, "w") as file_handle:
        json.dump(open_ai_calls, file_handle)


def main():
    # Path to your .jsonl file
    jsonl_file_path = os.path.join(
        PROJECT_ROOT_PATH, "..", "data", "paper_title_abstract_example_file.json"
    )

    # Initialize an empty list to store the JSON objects
    json_list = []

    # Open the JSONL file and read each line
    with open(jsonl_file_path, "r") as file:
        for line in file:
            # Parse the JSON string into a Python dictionary
            json_obj = json.loads(line.strip())
            # Append the dictionary to the list
            json_list.append(json_obj.values())

    # Loading jsonl file
    meta_data_np = np.array(json_list)  # Replace with actual data
    open_ai_calls = {}
    for idx, (paper_sha, title, abstract, journal_name) in enumerate(meta_data_np):
        open_ai_calls = process_paper(
            paper_sha, title, abstract, journal_name, MODEL, open_ai_calls
        )
        if (idx + 1) % 100 == 0:
            save_openai_calls(
                open_ai_calls,
                os.path.join(
                    PROJECT_ROOT_PATH,
                    "..",
                    "data",
                    f"open_ai_most_popular_{idx + 1}.json",
                ),
            )
    save_openai_calls(
        open_ai_calls,
        os.path.join(
            PROJECT_ROOT_PATH, "..", "data", "open_ai_most_popular_final.json"
        ),
    )


if __name__ == "__main__":
    main()
