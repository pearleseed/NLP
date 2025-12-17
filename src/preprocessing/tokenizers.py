import re
from typing import List
from core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        # Simple whitespace tokenization
        return text.strip().split()

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern: str = r"\b\w+\b"):
        self.pattern = pattern

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        return re.findall(self.pattern, text)
