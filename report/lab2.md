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

### 2.1. Vector Space Model (VSM)

#### 2.1.1. Ý tưởng cốt lõi
Vector Space Model là mô hình đại số biểu diễn văn bản dưới dạng vector trong không gian nhiều chiều. Mỗi chiều tương ứng với một term (từ) trong vocabulary.

**Tại sao cần biểu diễn văn bản thành vector?**
- Máy tính chỉ xử lý được số, không hiểu text trực tiếp
- Vector cho phép áp dụng các phép toán đại số (cộng, trừ, nhân)
- Có thể tính độ tương đồng giữa các văn bản (cosine similarity)
- Là input cho các thuật toán Machine Learning

#### 2.1.2. Giả định của VSM
- **Bag-of-Words assumption**: Thứ tự từ không quan trọng
- **Term independence**: Các từ độc lập với nhau
- **Distributional hypothesis**: Văn bản tương tự có phân bố từ tương tự

### 2.2. Bag-of-Words (BoW)

#### 2.2.1. Định nghĩa hình thức
Cho vocabulary `V = {w₁, w₂, ..., wₙ}` với `n` từ unique.

Một document `d` được biểu diễn bởi vector `v(d) ∈ ℝⁿ`:

```
v(d) = [count(w₁, d), count(w₂, d), ..., count(wₙ, d)]
```

Trong đó `count(wᵢ, d)` là số lần từ `wᵢ` xuất hiện trong document `d`.

#### 2.2.2. Ví dụ minh họa chi tiết

**Corpus:**
```
D1: "I love NLP"
D2: "I love programming"  
D3: "NLP is fun"
```

**Bước 1: Xây dựng Vocabulary**
```
V = {i, love, nlp, programming, is, fun}
Index: {i:0, love:1, nlp:2, programming:3, is:4, fun:5}
```

**Bước 2: Tạo Count Vectors**
```
D1: [1, 1, 1, 0, 0, 0]  → "I"=1, "love"=1, "NLP"=1
D2: [1, 1, 0, 1, 0, 0]  → "I"=1, "love"=1, "programming"=1
D3: [0, 0, 1, 0, 1, 1]  → "NLP"=1, "is"=1, "fun"=1
```

**Bước 3: Document-Term Matrix**
```
        i  love  nlp  programming  is  fun
D1  [   1    1    1       0        0    0  ]
D2  [   1    1    0       1        0    0  ]
D3  [   0    0    1       0        1    1  ]
```

### 2.3. TF-IDF - Term Frequency-Inverse Document Frequency

#### 2.3.1. Vấn đề của Raw Count
Raw count có nhược điểm:
- Từ phổ biến ("the", "is", "a") có count cao nhưng ít mang ý nghĩa
- Document dài có count cao hơn document ngắn
- Không phân biệt được từ quan trọng vs từ thông thường

#### 2.3.2. Term Frequency (TF)
Đo tần suất xuất hiện của term trong document, có nhiều biến thể:

| Biến thể | Công thức | Mô tả |
|----------|-----------|-------|
| Raw count | `tf(t,d) = f(t,d)` | Số lần xuất hiện |
| Boolean | `tf(t,d) = 1 if t∈d else 0` | Có/không xuất hiện |
| Log normalization | `tf(t,d) = 1 + log(f(t,d))` | Giảm ảnh hưởng của count cao |
| Augmented | `tf(t,d) = 0.5 + 0.5 × f(t,d)/max{f(t',d)}` | Normalize theo max |

#### 2.3.3. Inverse Document Frequency (IDF)
Đo độ hiếm của term trong toàn bộ corpus:

```
idf(t, D) = log(N / df(t))
```

Trong đó:
- `N`: Tổng số documents trong corpus
- `df(t)`: Số documents chứa term `t` (document frequency)

**Ý nghĩa:**
- Term xuất hiện trong nhiều documents → IDF thấp (ít quan trọng)
- Term xuất hiện trong ít documents → IDF cao (quan trọng, đặc trưng)

**Biến thể smooth IDF (tránh chia cho 0):**
```
idf(t, D) = log((N + 1) / (df(t) + 1)) + 1
```

#### 2.3.4. TF-IDF Score
Kết hợp TF và IDF:

```
tfidf(t, d, D) = tf(t, d) × idf(t, D)
```

**Ví dụ tính toán:**
```
Corpus: D1="cat sat", D2="cat dog", D3="dog bird"
N = 3 documents

Tính TF-IDF cho "cat" trong D1:
- tf("cat", D1) = 1
- df("cat") = 2 (xuất hiện trong D1, D2)
- idf("cat") = log(3/2) = 0.405
- tfidf("cat", D1) = 1 × 0.405 = 0.405

Tính TF-IDF cho "sat" trong D1:
- tf("sat", D1) = 1
- df("sat") = 1 (chỉ xuất hiện trong D1)
- idf("sat") = log(3/1) = 1.099
- tfidf("sat", D1) = 1 × 1.099 = 1.099

→ "sat" có TF-IDF cao hơn vì hiếm hơn trong corpus
```

### 2.4. N-grams

#### 2.4.1. Định nghĩa
N-gram là chuỗi n tokens liên tiếp, giúp capture một phần thông tin về thứ tự từ.

| N | Tên | Ví dụ ("I love NLP") |
|---|-----|----------------------|
| 1 | Unigram | "I", "love", "NLP" |
| 2 | Bigram | "I love", "love NLP" |
| 3 | Trigram | "I love NLP" |

#### 2.4.2. Ưu điểm của N-grams
- Capture được ngữ cảnh cục bộ
- Phân biệt được "not good" vs "good" (với bigrams)
- Cải thiện performance cho nhiều tasks

#### 2.4.3. Nhược điểm
- **Sparsity**: Vocabulary tăng theo cấp số nhân
- **Data sparsity**: Nhiều n-grams hiếm khi xuất hiện

```
Vocabulary size với |V| = 10,000 từ:
- Unigrams: 10,000
- Bigrams: 10,000² = 100,000,000 (lý thuyết)
- Trigrams: 10,000³ = 1,000,000,000,000 (lý thuyết)
```

### 2.5. Sparse Matrix Representation

#### 2.5.1. Vấn đề Sparsity
Document-Term Matrix thường rất thưa (sparse):
- Vocabulary có thể lên đến hàng chục nghìn từ
- Mỗi document chỉ chứa vài chục đến vài trăm từ
- Sparsity thường > 99%

**Ví dụ:**
```
Vocabulary: 10,000 từ
Document trung bình: 100 từ
Sparsity: (10,000 - 100) / 10,000 = 99%
```

#### 2.5.2. Compressed Sparse Row (CSR) Format
Thay vì lưu toàn bộ ma trận, chỉ lưu các phần tử khác 0:

```
Dense matrix:
[1, 0, 0, 2]
[0, 0, 3, 0]
[4, 0, 0, 5]

CSR representation:
data    = [1, 2, 3, 4, 5]      # Giá trị khác 0
indices = [0, 3, 2, 0, 3]      # Chỉ số cột
indptr  = [0, 2, 3, 5]         # Con trỏ hàng
```

**Lợi ích:**
- Tiết kiệm bộ nhớ: O(nnz) thay vì O(m×n)
- Phép nhân ma trận-vector hiệu quả

### 2.6. Cosine Similarity

#### 2.6.1. Định nghĩa
Đo độ tương đồng giữa hai vectors dựa trên góc giữa chúng:

```
cos(θ) = (A · B) / (||A|| × ||B||)

       = Σᵢ(Aᵢ × Bᵢ) / (√Σᵢ(Aᵢ²) × √Σᵢ(Bᵢ²))
```

#### 2.6.2. Tính chất
- Giá trị trong khoảng [-1, 1] (với TF-IDF thường [0, 1])
- `cos = 1`: Hai vector cùng hướng (giống nhau)
- `cos = 0`: Hai vector vuông góc (không liên quan)
- `cos = -1`: Hai vector ngược hướng

#### 2.6.3. Tại sao dùng Cosine thay vì Euclidean?
```
Document A: "cat cat cat" → [3, 0]
Document B: "cat"         → [1, 0]
Document C: "dog"         → [0, 1]

Euclidean distance:
- d(A, B) = |3-1| = 2
- d(A, C) = √(9+1) = 3.16

Cosine similarity:
- cos(A, B) = 3/(3×1) = 1.0  (giống nhau hoàn toàn!)
- cos(A, C) = 0/(3×1) = 0.0  (khác nhau hoàn toàn)
```

→ Cosine không bị ảnh hưởng bởi độ dài document, chỉ quan tâm đến hướng (tỷ lệ từ)

### 2.7. Hạn chế của Count-based Methods

| Hạn chế | Mô tả | Ví dụ |
|---------|-------|-------|
| Mất thứ tự từ | "dog bites man" = "man bites dog" | Ý nghĩa khác nhau hoàn toàn |
| Không capture ngữ nghĩa | "good" và "excellent" là vectors khác nhau | Dù cùng nghĩa |
| High dimensionality | Vocabulary lớn → vector nhiều chiều | Curse of dimensionality |
| Sparsity | Hầu hết giá trị = 0 | Khó học patterns |

→ Các hạn chế này dẫn đến sự phát triển của **Word Embeddings** (Lab 3)

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
