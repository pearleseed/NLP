
import sys
import unittest

import os

# Adjust path to find src (project root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from preprocessing.tokenizers import RegexTokenizer
from representations.count_vectorizer import CountVectorizer

class TestLab2(unittest.TestCase):
    def test_count_vectorizer(self):
        corpus = [
            "I love NLP.",
            "I love programming.",
            "NLP is a subfield of AI."
        ]
        
        tokenizer = RegexTokenizer()
        vectorizer = CountVectorizer(tokenizer)
        
        # Test fit_transform
        vectors = vectorizer.fit_transform(corpus)
        
        print("\nVocabulary:", vectorizer.vocabulary_)
        print("Vectors:", vectors)
        
        # Basic assertions
        self.assertIn("nlp", vectorizer.vocabulary_)
        self.assertIn("love", vectorizer.vocabulary_)
        
        # Check vector shape
        self.assertEqual(len(vectors), 3)
        self.assertEqual(len(vectors[0]), len(vectorizer.vocabulary_))

if __name__ == '__main__':
    unittest.main()
