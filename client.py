"""TF-IDF Vectorizer & Cosine Similarity Engine
100% Python Standard Library (math, collections, re).
"""

import math
import collections
import re

class TFIDFVectorizerEngine:
    """TF-IDF sparse vectorizer with smooth IDF scaling."""
    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.doc_count = 0

    def fit_transform(self, documents):
        self.doc_count = len(documents)
        df = collections.defaultdict(int)
        doc_tokens = []
        for doc in documents:
            tokens = re.findall(r"\b\w+\b", doc.lower())
            doc_tokens.append(tokens)
            unique_terms = set(tokens)
            for t in unique_terms:
                df[t] += 1

        self.vocab = {term: idx for idx, term in enumerate(sorted(df.keys()))}
        self.idf = {term: math.log((1 + self.doc_count) / (1 + count)) + 1.0 for term, count in df.items()}

        vectors = []
        for tokens in doc_tokens:
            tf = collections.defaultdict(int)
            for t in tokens:
                tf[t] += 1
            vec = [0.0] * len(self.vocab)
            for t, count in tf.items():
                if t in self.vocab:
                    vec[self.vocab[t]] = (1.0 + math.log(count)) * self.idf[t]
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [round(v / norm, 4) for v in vec]
            vectors.append(vec)
        return vectors

    def cosine_similarity(self, vec1, vec2):
        return round(sum(a * b for a, b in zip(vec1, vec2)), 4)
