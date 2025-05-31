import numpy as np
import nltk
# nltk.download('punkt')  # Uncomment if running for the first time

from nltk.stem.porter import PorterStemmer
from rapidfuzz import fuzz  # For fuzzy matching

stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Split sentence into array of words/tokens.
    A token can be a word or punctuation character, or number.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Stemming: reduce word to its root form.
    E.g., "organizing" -> "organ"
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words, threshold=70):
    """
    Return bag-of-words array:
    1 for each known word that matches a token in the sentence,
    using fuzzy matching (based on RapidFuzz similarity ratio).

    Args:
        tokenized_sentence: list of words from user input
        words: list of known (trained) vocabulary words
        threshold: minimum similarity score (0-100) to consider a match
    """
    # Stem each word in the sentence
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    # Initialize bag with 0s
    bag = np.zeros(len(words), dtype=np.float32)

    for i, known_word in enumerate(words):
        for input_word in sentence_words:
            score = fuzz.ratio(known_word, input_word)
            if score >= threshold:
                bag[i] = 1
                # For learning: see which words are matched
                print(f"[MATCH] '{input_word}' matched with '{known_word}' (score={score})")
                break  # Stop checking once one match is found
    return bag
