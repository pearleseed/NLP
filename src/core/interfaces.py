from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

class Vectorizer(ABC):
    @abstractmethod
    def fit(self, corpus: List[str]):
        """Learns the vocabulary from a list of documents."""
        pass

    @abstractmethod
    def transform(self, documents: List[str]) -> List[List[int]]:
        """Transforms a list of documents into a list of count vectors."""
        pass

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """A convenience method that performs fit and then transform."""
        self.fit(corpus)
        return self.transform(corpus)
