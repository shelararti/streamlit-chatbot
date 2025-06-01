
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from rapidfuzz import fuzz  # For fuzzy matching

stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Tokenize sentence into words using regex (Streamlit-safe, no nltk downloads needed).
    """
    return re.findall(r'\b\w+\b', sentence.lower())

def stem(word):
    """
    Stemming: reduce word to its root form.
    E.g., "organizing" -> "organ"
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words, threshold=70):
    """
    Return bag-of-words vector using fuzzy matching.
    """
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)

    for i, known_word in enumerate(words):
        for input_word in sentence_words:
            score = fuzz.ratio(known_word, input_word)
            if score >= threshold:
                bag[i] = 1
                break
    return bag
