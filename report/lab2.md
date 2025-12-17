# Báo cáo Lab 2: Count Vectorization

## 1. Mục tiêu

Triển khai kỹ thuật **Count Vectorization** (Bag-of-Words) theo hướng đối tượng để biểu diễn văn bản dưới dạng vector số học.

**Các task thực hiện:**
1. Triển khai Interface cho Tokenizer và Vectorizer
2. Triển khai SimpleTokenizer và RegexTokenizer
3. Triển khai CountVectorizer
4. Test trên toy corpus và dataset UD English EWT
5. So sánh với scikit-learn

---

## 2. Nền tảng Lý thuyết

### 2.1. Bag-of-Words (BoW)
Biểu diễn văn bản bằng vector đếm tần suất từ, bỏ qua thứ tự và ngữ pháp.

### 2.2. Document-Term Matrix
Ma trận với:
- Hàng: Documents
- Cột: Vocabulary (unique tokens)
- Giá trị: Tần suất xuất hiện

---

## 3. Cài đặt

### 3.1. Source Code
```
src/
├── core/interfaces.py           # Abstract base classes
├── preprocessing/tokenizers.py  # SimpleTokenizer, RegexTokenizer
└── representations/count_vectorizer.py
```

### 3.2. Tokenizers
| Tokenizer | Method | Lowercase | Remove Punctuation |
|-----------|--------|-----------|-------------------|
| SimpleTokenizer | `split()` | No | No |
| RegexTokenizer | `\b\w+\b` | Yes | Yes |

### 3.3. CountVectorizer
**fit()**: Học vocabulary từ corpus → `{token: index}`

**transform()**: Chuyển document → count vector

---

## 4. Kết quả

### 4.1. Tokenizer Comparison
| Input | SimpleTokenizer | RegexTokenizer |
|-------|-----------------|----------------|
| `"Hello World!"` | `['Hello', 'World!']` | `['hello', 'world']` |
| `"I love NLP."` | `['I', 'love', 'NLP.']` | `['i', 'love', 'nlp']` |

### 4.2. CountVectorizer (Toy Corpus)
**Corpus**: `["I love NLP.", "I love programming.", "NLP is a subfield of AI."]`

**Vocabulary size**: 9

**Document-Term Matrix**:
```
Doc 0: [0, 0, 1, 0, 1, 1, 0, 0, 0]
Doc 1: [0, 0, 1, 0, 1, 0, 0, 1, 0]
Doc 2: [1, 1, 0, 1, 0, 1, 1, 0, 1]
```

### 4.3. UD English EWT Dataset
| Metric | Value |
|--------|-------|
| Documents | 12,544 |
| Vocabulary size | 15,972 |
| Sparsity | 99.88% |

**Phân tích chi tiết:**
- **Sparsity cao (99.88%)**: Trong 100 documents đầu tiên với 1,597,200 phần tử ma trận, chỉ có 1,880 phần tử khác 0
- **Ý nghĩa**: Mỗi document chỉ chứa một phần rất nhỏ của vocabulary → cần sparse matrix để tiết kiệm bộ nhớ
- **Ví dụ**: Document đầu tiên có 23 tokens nhưng vector có 15,972 chiều, chỉ 19 vị trí khác 0

### 4.4. So sánh với Scikit-Learn
| Metric | My Implementation | Scikit-Learn |
|--------|-------------------|--------------|
| Vocabulary Size | 15,972 | 15,936 |
| Min token length | 1 (`\w+`) | 2 (`\w\w+`) |

**Phân tích sự khác biệt:**
- **Vocabulary size**: Implementation của tôi lớn hơn 36 từ do bao gồm single-character tokens (a, I, ...)
- **Toy corpus test**: 
  - My vocab: `{'a': 0, 'ai': 1, 'i': 2, 'is': 3, 'love': 4, 'nlp': 5, 'of': 6, 'programming': 7, 'subfield': 8}` (9 từ)
  - Sklearn vocab: `{'ai': 0, 'is': 1, 'love': 2, 'nlp': 3, 'of': 4, 'programming': 5, 'subfield': 6}` (7 từ)
  - Sklearn loại bỏ "a" và "i" vì chỉ có 1 ký tự

**Kết luận**: Cả hai implementation cho kết quả tương đương, khác biệt chỉ ở cách xử lý single-character tokens

---

## 5. Nhận xét

**Ưu điểm BoW:**
- Đơn giản, dễ triển khai
- Baseline tốt cho text classification

**Hạn chế:**
- Vector thưa (99%+ zeros), tốn bộ nhớ
- Mất thông tin thứ tự từ
- Không nắm bắt ngữ nghĩa

---

## 6. Trích dẫn
- Scikit-learn CountVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
- UD English EWT: https://universaldependencies.org/treebanks/en_ewt/
