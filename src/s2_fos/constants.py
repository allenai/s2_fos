import os

FASTTEXT_FNAME = 'lid.176.bin'

MODEL_NAME = 'allenai/scibert_scivocab_uncased_fielf_of_study'

LABELS = [
    'Agricultural and Food Sciences',
    'Art',
    'Biology',
    'Business',
    'Chemistry',
    'Computer Science',
    'Economics',
    'Education',
    'Engineering',
    'Environmental Science',
    'Geography',
    'Geology',
    'History',
    'Law',
    'Linguistics',
    'Materials Science',
    'Mathematics',
    'Medicine',
    'Philosophy',
    'Physics',
    'Political Science',
    'Psychology',
    'Sociology']

try:
    PROJECT_ROOT_PATH = os.path.abspath(
        os.path.join(__file__, os.pardir, os.pardir))
except NameError:
    PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.getcwd()))

TOKENIZER_MODEL_NAME = 'allenai/scibert_scivocab_uncased'
