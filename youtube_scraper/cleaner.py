import re
import nltk

def clean_text(text):

    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_sentences(text):

    sentences = nltk.sent_tokenize(text)

    return [s for s in sentences if len(s) > 10]