import numpy as np

from s2_fos.src import FOS_LIST
FOS_LIST = [
    "Agricultural and Food sciences",
    "Art",
    "Biology",
    "Business",
    "Chemistry",
    "Computer science",
    "Economics",
    "Education",
    "Engineering",
    "Environmental science",
    "Geography",
    "Geology",
    "History",
    "Law",
    "Linguistics",
    "Materials science",
    "Mathematics",
    "Medicine",
    "Philosophy",
    "Physics",
    "Political science",
    "Psychology",
    "Sociology",
    'Other'
]


def search_for_fos_respons(response):
    fields = []
    for field in FOS_LIST:
        if field in response:
            fields.append(field)

    return fields


def structure_open_ai_response(open_ai_calls):
    open_ai_list = []
    for key, value in open_ai_calls.items():
        paper_sha = key[:40]
        model = key[41:]
        # print(paper_sha, model)
        extracted_fields = search_for_fos_respons(value['result']['choices'][0]['message']['content'])
        completion_tokens = value['result']['usage']['completion_tokens']
        prompt_tokens = value['result']['usage']['prompt_tokens']
        open_ai_list.append({'paper_sha': paper_sha, 'model': model,
                             'FoS_openAI': extracted_fields, 'completion_tokens': completion_tokens,
                             'prompt_tokens': prompt_tokens})

    return open_ai_list


from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType


def create_spark_df_open_ai(open_ai_calls):
    open_ai_calls_dict_list = structure_open_ai_response(open_ai_calls)
    schema = StructType([
        StructField("paper_sha", StringType(), True),
        StructField("model", StringType(), True),
        StructField("FoS_openAI", ArrayType(StringType()), True),
        StructField("completion_tokens", IntegerType(), True),
        StructField("prompt_tokens", IntegerType(), True),
    ])
    return spark.createDataFrame(open_ai_calls_dict_list, schema=schema)


import re

def tokenize(text):
    if text:
        return re.findall(r'\b\w+\b', text)
    return text

def truncate_entity(entity, max_tokens=800):
    tokens = tokenize(entity)
    if tokens and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        truncated_entity = ' '.join(tokens)
        return truncated_entity
    else:
        return entity


import openai
import time
import json

model = "gpt-3.5-turbo"
max_retries = 5
retry_delay = 5  # Time in seconds to wait before retrying


def call_openai_api(model, messages, temperature=0):
    return openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)


def process_paper(paper_sha, title, abstract, journal_name, model, open_ai_calls):
    key = f'{paper_sha}_{model}'
    if key not in open_ai_calls.keys():
        if idx % 100 == 0:
            print(idx, paper_sha)

        truncated_title = truncate_entity(title)
        truncated_abstract = truncate_entity(abstract)
        messages = prompt_with_journal(truncated_title, truncated_abstract, journal_name)

        retries = 0
        success = False
        while not success and retries < max_retries:
            try:
                result = call_openai_api(model, messages, temperature=0)
                if len(search_for_fos_respons(result['choices'][0]['message']['content'])) == 0:
                    # Extract the assistant's reply
                    assistant_reply1 = result['choices'][0]['message']['content']

                    # Add the user's follow-up question
                    assistant_reply1 = assistant_reply1.replace("```", "``")
                    # Format the content string
                    content_string = f"Can you please map your response {assistant_reply1} to the following fields: {', '.join(FOS_LIST)}"
                    messages.extend([
                        {"role": "user", "content": content_string},
                        {"role": "assistant",
                         "content": ("```python \n"
                                     "{'field_of_study': [")}, ])

                    result = call_openai_api(model, messages, temperature=0)
                    open_ai_calls[key] = {'prompt': messages, 'result': result}
                else:
                    open_ai_calls[key] = {'prompt': messages, 'result': result}
                success = True
            except openai.OpenAIError as e:
                print(f"Error: {e}")
                if messages:
                    print(f'Length of the message: {len(tokenize(json.dumps(messages)))}')
                if truncated_title and truncated_abstract:
                    print(
                        f'Length of title: {len(tokenize(truncated_title))} Length of Abstract: {len(tokenize(truncated_abstract))}')
                print(
                    f'Paper_sha: {paper_sha} Title: {truncated_title}\n Abstract: {truncated_abstract}\n Journal Name: {journal_name}')
                if retries < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retries += 1
                else:
                    print("Max retries reached. Skipping this API call.")
    return open_ai_calls


def prompt_with_journal(title, abstract, journal_name):
    message = [
        {"role": "system", "content": "You are a highly intelligent and accurate information extraction system. You take title, abstract, journal name of a \
         scientific article as input and your task is to classify the scientific field of study of the passage.",
         "role": "user", "content": "You need to classify it with key: 'field_of_study' assign as many 'field_of_study' as you find it fit: \
    'Agricultural and Food sciences', \
    'Art', \
    'Biology',\
    'Business',\
    'Chemistry',\
    'Computer science',\
    'Economics',\
    'Education',\
    'Engineering',\
    'Environmental science',\
    'Geography',\
    'Geology',\
    'History',\
    'Law',\
    'Linguistics',\
    'Materials science',\
    'Mathematics',\
    'Medicine',\
    'Philosophy',\
    'Physics',\
    'Political science',\
    'Psychology',\
    'Sociology'\
    Only select from the above list, or 'Other'."},

        {"role": "assistant",
         "content": ("```python \n"
                     f"title = {title} \n"
                     f"abstract = {abstract}\n"
                     f"journal_name = {journal_name}\n"
                     "{'field_of_study': ["
                     )}, ]
    return message


idx = 0
model = "gpt-3.5-turbo"
max_retries = 5
retry_delay = 2  # Time in seconds to wait before retrying

meta_data_np = np.array()
for paper_sha, title, abstract, journal_name in meta_data_np:

    open_ai_calls = process_paper(paper_sha, title, abstract, journal_name, model, open_ai_calls)
    idx += 1
    if idx % 100 == 0:
        with open(f'../upsample_under_perform_s2_fos_2/open_ai_most_popular_{idx}.pkl', 'wb') as file_handle:
            pickle.dump(open_ai_calls, file_handle)

with open(f'../upsample_under_perform_s2_fos_2/open_ai_most_popular_final.pkl', 'wb') as file_handle:
    pickle.dump(open_ai_calls, file_handle)