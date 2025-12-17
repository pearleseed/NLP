# NLP Labs - Natural Language Processing

Tổng hợp các bài thực hành về **Xử lý Ngôn ngữ Tự nhiên (NLP)** sử dụng Python.

## 📚 Nội dung Labs

| Lab | Chủ đề | Mô tả |
|-----|--------|-------|
| Lab 1 | Lexical Analysis | Tokenization, Chunk Extraction, Regex |
| Lab 2 | Count Vectorization | Bag-of-Words, Document-Term Matrix |
| Lab 3 | Word Embeddings | GloVe, Word2Vec, t-SNE visualization |
| Lab 4 | Text Classification | Machine Learning cho phân loại văn bản |
| Lab 5 | Sequence Labeling | POS Tagging, NER |
| Lab 6 | Deep Learning NLP | Neural Networks cho NLP |
| Lab 7 | Transformers | BERT, Hugging Face |

## 🛠️ Cài đặt

```bash
# Clone repository
git clone https://github.com/<username>/nlp-labs.git
cd nlp-labs

# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

## 📁 Cấu trúc Project

```
.
├── notebook/          # Jupyter notebooks cho từng lab
├── src/               # Source code modules
│   ├── core/          # Interfaces, base classes
│   ├── preprocessing/ # Tokenizers, text processing
│   ├── representations/ # Vectorizers, embeddings
│   └── models/        # ML models
├── report/            # Báo cáo markdown cho từng lab
├── data/              # Datasets và pre-trained models
├── lectures/          # Tài liệu bài giảng
└── test/              # Unit tests
```

## 🚀 Sử dụng

### Chạy Jupyter Notebook
```bash
jupyter notebook notebook/
```

### Chạy từng module
```python
from src.preprocessing.tokenizers import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer

tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer=tokenizer)
```

## 📦 Dependencies

- spacy, nltk, stanza - NLP libraries
- scikit-learn - Machine Learning
- gensim - Word Embeddings
- transformers - Hugging Face Transformers
- tensorflow, keras - Deep Learning
- pyspark - Big Data processing
- pandas, numpy - Data manipulation
- matplotlib, seaborn - Visualization
