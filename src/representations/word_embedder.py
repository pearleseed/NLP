from gensim.models import KeyedVectors
import numpy as np
import os
import zipfile
from typing import List, Tuple

try:
    from src.preprocessing.tokenizers import RegexTokenizer
except ImportError:
    from preprocessing.tokenizers import RegexTokenizer


class WordEmbedder:
    GLOVE_FILES = {
        'glove-wiki-gigaword-50': 'glove.6B.50d.txt',
        'glove-wiki-gigaword-100': 'glove.6B.100d.txt',
        'glove-wiki-gigaword-200': 'glove.6B.200d.txt',
        'glove-wiki-gigaword-300': 'glove.6B.300d.txt',
    }

    def __init__(self, model_name: str = 'glove-wiki-gigaword-50', data_dir: str = None):
        print(f"Đang tải model '{model_name}'... Vui lòng chờ.")
        self.model = None
        self.vector_size = 0
        self.tokenizer = RegexTokenizer()
        
        # Tìm thư mục data: ưu tiên tham số, sau đó thử ../data, ./data, ~/gensim-data
        if data_dir is None:
            for path in ['../data', './data', 'data', os.path.expanduser('~/gensim-data')]:
                if os.path.exists(path):
                    data_dir = path
                    break
            else:
                data_dir = './data'
        
        self.model = self._load_model(model_name, data_dir)
        
        if self.model is not None:
            self.vector_size = self.model.vector_size
            print("Tải model thành công.")
    
    def _load_model(self, model_name: str, cache_dir: str):
        """Load model từ file local"""
        glove_filename = self.GLOVE_FILES.get(model_name)
        if not glove_filename:
            print(f"Không hỗ trợ model: {model_name}")
            return None
        
        glove_file = os.path.join(cache_dir, glove_filename)
        zip_path = os.path.join(cache_dir, 'glove.6B.zip')
        
        # Giải nén nếu chưa có file txt
        if not os.path.exists(glove_file) and os.path.exists(zip_path):
            print(f"Đang giải nén {glove_filename}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extract(glove_filename, cache_dir)
        
        if os.path.exists(glove_file):
            print(f"Đang load {glove_file}...")
            return KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
        
        print(f"Không tìm thấy file: {glove_file}")
        print("Hãy tải glove.6B.zip từ https://nlp.stanford.edu/data/glove.6B.zip")
        print(f"và đặt vào thư mục: {cache_dir}")
        return None

    def get_vector(self, word: str) -> np.ndarray:
        if self.model is not None and word in self.model:
            return self.model[word]
        return np.zeros(self.vector_size)

    def get_similarity(self, word1: str, word2: str) -> float:
        if self.model is not None and word1 in self.model and word2 in self.model:
            return self.model.similarity(word1, word2)
        print(f"Cảnh báo: Một trong hai từ '{word1}' hoặc '{word2}' là OOV.")
        return 0.0

    def get_most_similar(self, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
        if self.model is not None and word in self.model:
            return self.model.most_similar(word, topn=top_n)
        print(f"Cảnh báo: Từ '{word}' là OOV.")
        return []

    def embed_document(self, document: str) -> np.ndarray:
        tokens = self.tokenizer.tokenize(document)
        word_vectors = [self.model[t] for t in tokens if self.model and t in self.model]
        
        if not word_vectors:
            return np.zeros(self.vector_size)
        return np.mean(word_vectors, axis=0)


def load_glove_model(model_name: str = 'glove-wiki-gigaword-50', data_dir: str = None):
    GLOVE_FILES = {
        'glove-wiki-gigaword-50': 'glove.6B.50d.txt',
        'glove-wiki-gigaword-100': 'glove.6B.100d.txt',
        'glove-wiki-gigaword-200': 'glove.6B.200d.txt',
        'glove-wiki-gigaword-300': 'glove.6B.300d.txt',
    }
    
    glove_filename = GLOVE_FILES.get(model_name)
    if not glove_filename:
        raise ValueError(f"Không hỗ trợ model: {model_name}")
    
    # Tìm thư mục data
    if data_dir is None:
        for path in ['../data', './data', 'data', os.path.expanduser('~/gensim-data')]:
            if os.path.exists(path):
                data_dir = path
                break
        else:
            data_dir = './data'
    
    glove_file = os.path.join(data_dir, glove_filename)
    zip_path = os.path.join(data_dir, 'glove.6B.zip')
    
    # Giải nén nếu chưa có file txt
    if not os.path.exists(glove_file) and os.path.exists(zip_path):
        print(f"Đang giải nén {glove_filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract(glove_filename, data_dir)
    
    if os.path.exists(glove_file):
        print(f"Đang load {glove_file}...")
        return KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
    
    raise FileNotFoundError(
        f"Không tìm thấy file: {glove_file}\n"
        f"Hãy tải glove.6B.zip từ https://nlp.stanford.edu/data/glove.6B.zip\n"
        f"và đặt vào thư mục: {data_dir}"
    )
