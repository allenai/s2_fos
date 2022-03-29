import re
import pycld2 as cld2
from text_unidecode import unidecode


ACCEPTABLE_CHARS = re.compile(r"[^a-zA-Z\s]+")


def make_inference_text(paper, sep="|", sep_num=5):
    """Makes the combined text to perform inference over, from a dict with
    'title' and 'abstract' fields.
    """
    if "title" in paper and paper["title"] is not None:
        title = normalize_text(paper["title"])
    else:
        title = ""

    if "abstract" in paper and paper["abstract"] is not None:
        abstract = normalize_text(paper["abstract"])
    else:
        abstract = ""

    return f"{title} {sep * sep_num} {abstract}"


def normalize_text(text):
    """
    Normalize text.
    Parameters
    ----------
    text: string
        the text to normalize
    special_case_apostrophie: bool
        whether to replace apostrophes with empty strings rather than spaces
    Returns
    -------
    string: the normalized text
    """
    if text is None or len(text) == 0:
        return ""

    norm_text = unidecode(text).lower()
    norm_text = ACCEPTABLE_CHARS.sub(" ", norm_text)
    norm_text = re.sub(r"\s+", " ", norm_text).strip()

    return norm_text


def detect_language(fasttext, text):
    """
    Detect the language used in the input text with trained language classifer.
    """
    if len(text.split()) <= 1:
        return (False, False, "un")

    # fasttext
    isuppers = [c.isupper() for c in text if c.isalpha()]
    if len(isuppers) == 0:
        return (False, False, "un")
    elif sum(isuppers) / len(isuppers) > 0.9:
        fasttext_pred = fasttext.predict(text.lower().replace("\n", " "))
        predicted_language_ft = fasttext_pred[0][0].split("__")[-1]
    else:
        fasttext_pred = fasttext.predict(text.replace("\n", " "))
        predicted_language_ft = fasttext_pred[0][0].split("__")[-1]

    # cld2
    try:
        cld2_pred = cld2.detect(text)
        predicted_language_2 = cld2_pred[2][0][1]
        if predicted_language_2 == "un":
            predicted_language_2 = "un_2"
    except:  # noqa: E722
        predicted_language_2 = "un_2"

    if predicted_language_ft == "un_ft" and predicted_language_2 == "un_2":
        predicted_language = "un"
        is_reliable = False
    elif predicted_language_ft == "un_ft":
        predicted_language = predicted_language_2
        is_reliable = True
    elif predicted_language_2 == "un_2":
        predicted_language = predicted_language_ft
        is_reliable = True
    elif predicted_language_2 != predicted_language_ft:
        predicted_language = "un"
        is_reliable = False
    else:
        predicted_language = predicted_language_2
        is_reliable = True

    # is_english can now be obtained
    is_english = predicted_language == "en"

    return is_reliable, is_english, predicted_language
