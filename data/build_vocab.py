from collections import Counter
import re
def build_vocabulary(corpus: list[str], min_df: int = 1):

    word_counts = Counter()
    doc_counts = Counter()

    for doc in corpus:
        tokens = re.findall(r'\b\w+\b', doc.lower())
        word_counts.update(tokens)
        doc_counts.update(set(tokens)) 

    vocabulary = [word for word, count in doc_counts.items() if count >= min_df]
    vocabulary.sort()
    
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    idx_to_word = {i: word for i, word in enumerate(vocabulary)}
    
    return word_to_idx, idx_to_word, vocabulary