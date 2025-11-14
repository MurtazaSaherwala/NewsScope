import re
import spacy
import warnings
from config import SPACY_MODEL

# Suppress warnings
warnings.filterwarnings('ignore')

# Load spacy model once when module is imported
try:
    nlp = spacy.load(SPACY_MODEL)
except IOError:
    print(f"Error: spaCy model '{SPACY_MODEL}' not found.")
    print(f"Please run: python -m spacy download {SPACY_MODEL}")
    exit()

stopwords = nlp.Defaults.stop_words

def clean_text(text):
    """Cleans and lemmatizes raw text."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    doc = nlp(text)
    cleaned_tokens = []
    for token in doc:
        if token.text not in stopwords and not token.is_punct and not token.is_space and len(token.lemma_) > 2:
            cleaned_tokens.append(token.lemma_)
    return " ".join(cleaned_tokens)