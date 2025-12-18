# Báo cáo Lab 3: Word Embeddings

## 1. Mục tiêu

Tìm hiểu và ứng dụng **Word Embeddings** - kỹ thuật biểu diễn từ dưới dạng dense vectors.

**Các task thực hiện:**
1. Tải và sử dụng model pre-trained (GloVe)
2. Nhúng văn bản (Document Embedding)
3. Huấn luyện Word2Vec trên dữ liệu nhỏ (Gensim)
4. Huấn luyện Word2Vec trên dữ liệu lớn (Spark)
5. Trực quan hóa Embedding với t-SNE/PCA

---

## 2. Nền tảng Lý thuyết

### 2.1. Từ Sparse sang Dense Representations

#### 2.1.1. Vấn đề của One-hot và BoW
Các phương pháp truyền thống (One-hot, BoW, TF-IDF) tạo ra **sparse vectors**:

```
Vocabulary: [cat, dog, king, queen, man, woman] (6 từ)

One-hot encoding:
cat   = [1, 0, 0, 0, 0, 0]
dog   = [0, 1, 0, 0, 0, 0]
king  = [0, 0, 1, 0, 0, 0]
queen = [0, 0, 0, 1, 0, 0]
```

**Nhược điểm:**
- **Không có semantic similarity**: cos(king, queen) = 0, dù có quan hệ ngữ nghĩa
- **Curse of dimensionality**: Vocabulary 100K từ → vector 100K chiều
- **Không generalize**: Mỗi từ là một chiều độc lập

#### 2.1.2. Dense Word Embeddings
Word Embeddings biểu diễn từ bằng **dense vectors** với số chiều nhỏ (50-300):

```
king  = [0.50, 0.68, -0.59, 0.02, 0.60, ...]  (50-300 chiều)
queen = [0.45, 0.72, -0.55, 0.08, 0.58, ...]
```

**Ưu điểm:**
- Capture được semantic similarity
- Số chiều cố định, không phụ thuộc vocabulary size
- Có thể học được các quan hệ ngữ nghĩa (analogy)

### 2.2. Distributional Hypothesis - Nền tảng lý thuyết

#### 2.2.1. Phát biểu
> "You shall know a word by the company it keeps" - J.R. Firth (1957)
> 
> "Những từ xuất hiện trong ngữ cảnh tương tự có xu hướng mang ý nghĩa tương đồng."

#### 2.2.2. Ví dụ minh họa
```
"The ___ sat on the mat."
"The ___ chased the mouse."
"I fed my ___ some milk."

→ Các từ "cat", "dog", "kitten" có thể điền vào chỗ trống
→ Chúng có ngữ cảnh tương tự → có nghĩa liên quan
```

#### 2.2.3. Co-occurrence Matrix
Đếm số lần các từ xuất hiện cùng nhau trong một window:

```
Corpus: "I like deep learning. I like NLP. I enjoy flying."
Window size = 1 (chỉ xét từ liền kề)

         I  like  deep  learning  NLP  enjoy  flying
I        0    3     0      0       0     1      0
like     3    0     1      1       1     0      0
deep     0    1     0      1       0     0      0
learning 0    1     1      0       0     0      0
NLP      0    1     0      0       0     0      0
enjoy    1    0     0      0       0     0      1
flying   0    0     0      0       0     1      0
```

### 2.3. Word2Vec - Chi tiết thuật toán

#### 2.3.1. Kiến trúc tổng quan
Word2Vec (Mikolov et al., 2013) có 2 kiến trúc:

```
CBOW (Continuous Bag of Words):
Context words → [Average] → Hidden Layer → Target word
"The cat ___ on mat" → predict "sat"

Skip-gram:
Target word → Hidden Layer → Context words
"sat" → predict ["The", "cat", "on", "mat"]
```

#### 2.3.2. Skip-gram - Công thức chi tiết

**Mục tiêu:** Maximize xác suất của context words cho target word.

**Objective function:**
```
J(θ) = (1/T) Σₜ Σ_{-c≤j≤c, j≠0} log P(wₜ₊ⱼ | wₜ)
```

Trong đó:
- `T`: Tổng số từ trong corpus
- `c`: Window size (context size)
- `wₜ`: Target word tại vị trí t
- `wₜ₊ⱼ`: Context word

**Softmax probability:**
```
P(wₒ | wᵢ) = exp(vₒ'ᵀ vᵢ) / Σ_{w∈V} exp(vᵥ'ᵀ vᵢ)
```

Trong đó:
- `vᵢ`: Input vector của word i (target)
- `vₒ'`: Output vector của word o (context)
- `V`: Vocabulary

**Vấn đề:** Softmax tính trên toàn bộ vocabulary rất tốn kém (|V| có thể > 100K)

#### 2.3.3. Negative Sampling - Giải pháp tối ưu

Thay vì tính softmax trên toàn bộ V, chỉ sample k negative examples:

**Objective với Negative Sampling:**
```
log σ(vₒ'ᵀ vᵢ) + Σₖ 𝔼_{wₖ~Pₙ(w)} [log σ(-vₖ'ᵀ vᵢ)]
```

Trong đó:
- `σ(x) = 1/(1 + e⁻ˣ)`: Sigmoid function
- `Pₙ(w)`: Noise distribution (thường là unigram^0.75)
- `k`: Số negative samples (thường 5-20)

**Ý tưởng:**
- Positive sample: (target, context) thực sự xuất hiện cùng nhau → maximize
- Negative samples: (target, random_word) không xuất hiện cùng nhau → minimize

#### 2.3.4. CBOW - Continuous Bag of Words

**Mục tiêu:** Dự đoán target word từ context words.

```
Input: Average của context word vectors
       h = (1/2c) Σ_{-c≤j≤c, j≠0} vₜ₊ⱼ

Output: Softmax over vocabulary
       P(wₜ | context) = softmax(Wₒᵀ h)
```

**So sánh CBOW vs Skip-gram:**

| Tiêu chí | CBOW | Skip-gram |
|----------|------|-----------|
| Tốc độ training | Nhanh hơn | Chậm hơn |
| Từ hiếm | Kém hơn | Tốt hơn |
| Dataset nhỏ | Tốt hơn | Kém hơn |
| Syntactic tasks | Tốt hơn | Tương đương |
| Semantic tasks | Tương đương | Tốt hơn |

### 2.4. GloVe - Global Vectors

#### 2.4.1. Ý tưởng chính
GloVe (Pennington et al., 2014) kết hợp:
- **Matrix factorization** (như LSA): Sử dụng thống kê toàn cục
- **Local context window** (như Word2Vec): Học từ ngữ cảnh cục bộ

#### 2.4.2. Co-occurrence Probability Ratio

**Quan sát quan trọng:**
```
Xét các từ: ice, steam, solid, gas, water

P(solid | ice) / P(solid | steam) = large  (solid liên quan ice)
P(gas | ice) / P(gas | steam) = small      (gas liên quan steam)
P(water | ice) / P(water | steam) ≈ 1      (water liên quan cả hai)
```

→ Ratio của co-occurrence probabilities encode thông tin ngữ nghĩa

#### 2.4.3. Objective Function

```
J = Σᵢⱼ f(Xᵢⱼ) (wᵢᵀ w̃ⱼ + bᵢ + b̃ⱼ - log Xᵢⱼ)²
```

Trong đó:
- `Xᵢⱼ`: Co-occurrence count của word i và j
- `wᵢ, w̃ⱼ`: Word vectors
- `bᵢ, b̃ⱼ`: Bias terms
- `f(x)`: Weighting function để giảm ảnh hưởng của từ quá phổ biến

**Weighting function:**
```
f(x) = (x/xₘₐₓ)^α  if x < xₘₐₓ
     = 1           otherwise

(thường α = 0.75, xₘₐₓ = 100)
```

### 2.5. FastText - Subword Embeddings

#### 2.5.1. Vấn đề OOV (Out-of-Vocabulary)
Word2Vec và GloVe không xử lý được từ mới không có trong training data.

#### 2.5.2. Giải pháp của FastText
Biểu diễn từ bằng tổng của character n-grams:

```
word = "where", n = 3

Character n-grams: <wh, whe, her, ere, re>
(< và > là boundary markers)

v("where") = v(<wh) + v(whe) + v(her) + v(ere) + v(re>) + v(<where>)
```

**Ưu điểm:**
- Xử lý được OOV words
- Capture được morphology (tiền tố, hậu tố)
- Tốt cho ngôn ngữ có nhiều biến thể từ (tiếng Đức, tiếng Thổ Nhĩ Kỳ)

### 2.6. Word Analogy - Kiểm chứng Embeddings

#### 2.6.1. Analogy Task
```
"king" - "man" + "woman" ≈ "queen"

v(king) - v(man) + v(woman) ≈ v(queen)
```

#### 2.6.2. Các loại Analogy

| Loại | Ví dụ |
|------|-------|
| Gender | king:queen :: man:woman |
| Plural | cat:cats :: dog:dogs |
| Tense | walk:walked :: run:ran |
| Country-Capital | France:Paris :: Japan:Tokyo |
| Comparative | good:better :: bad:worse |

#### 2.6.3. Giải thích toán học
Analogy hoạt động vì word vectors encode các quan hệ như **linear offsets**:

```
v(king) - v(queen) ≈ v(man) - v(woman) ≈ v(male) - v(female)

→ Có một "gender direction" trong embedding space
```

### 2.7. Cosine Similarity cho Word Embeddings

#### 2.7.1. Công thức
```
similarity(A, B) = cos(θ) = (A · B) / (||A|| × ||B||)
                 = Σᵢ(Aᵢ × Bᵢ) / (√Σᵢ(Aᵢ²) × √Σᵢ(Bᵢ²))
```

#### 2.7.2. Tại sao dùng Cosine?
- Word vectors đã được normalize về cùng scale
- Cosine đo góc giữa vectors, không phụ thuộc magnitude
- Giá trị trong [-1, 1], dễ interpret

### 2.8. Document Embedding từ Word Embeddings

#### 2.8.1. Mean Pooling (Simple Average)
```
doc_vector = (1/n) Σᵢ v(wordᵢ)
```

**Ưu điểm:** Đơn giản, nhanh
**Nhược điểm:** Mất thông tin thứ tự, từ quan trọng bị "pha loãng"

#### 2.8.2. Weighted Average (TF-IDF weighted)
```
doc_vector = Σᵢ tfidf(wordᵢ) × v(wordᵢ) / Σᵢ tfidf(wordᵢ)
```

#### 2.8.3. Các phương pháp nâng cao
- **Doc2Vec (Paragraph Vectors)**: Học document vector cùng với word vectors
- **Sentence-BERT**: Dùng Transformer để tạo sentence embeddings

### 2.9. Hạn chế của Static Word Embeddings

| Hạn chế | Mô tả | Ví dụ |
|---------|-------|-------|
| Polysemy | Mỗi từ chỉ có 1 vector | "bank" (ngân hàng) = "bank" (bờ sông) |
| Context-independent | Không thay đổi theo ngữ cảnh | "I love you" vs "Love is blind" |
| Bias | Học bias từ training data | "doctor" gần "man", "nurse" gần "woman" |

→ Các hạn chế này dẫn đến sự phát triển của **Contextualized Embeddings** (ELMo, BERT - Lab 6)

---

## 3. Cài đặt

### 3.1. Source Code
- `src/representations/word_embedder.py`: Lớp `WordEmbedder`
  - `get_vector(word)`: Lấy vector, trả về vector 0 nếu OOV
  - `get_similarity(w1, w2)`: Cosine similarity
  - `get_most_similar(word, top_n)`: Tìm từ đồng nghĩa
  - `embed_document(doc)`: Mean pooling các word vectors

### 3.2. Model & Dataset
- **Pre-trained**: `glove-wiki-gigaword-50` (50D, ~65MB)
- **Toy corpus**: 6 câu đơn giản để demo huấn luyện

---

## 4. Kết quả

### 4.1. Task 1: Pre-trained Model (GloVe)

**Vector của 'king'** (5 phần tử đầu): `[0.50451, 0.68607, -0.59517, -0.022801, 0.60046]`
- Kích thước vector: 50 chiều

| Cặp từ | Similarity | Giải thích |
|--------|------------|------------|
| king - queen | 0.7839 | Cao vì cùng trường ngữ nghĩa "hoàng gia" |
| king - man | 0.5309 | Thấp hơn, thể hiện mối quan hệ giới tính |

**10 từ tương đồng nhất với 'computer':**
| Từ | Similarity |
|----|------------|
| computers | 0.9165 |
| software | 0.8815 |
| technology | 0.8526 |
| electronic | 0.8126 |
| internet | 0.8060 |
| computing | 0.8026 |
| devices | 0.8016 |
| digital | 0.7992 |
| applications | 0.7913 |
| pc | 0.7883 |

**Nhận xét**: GloVe nắm bắt tốt mối quan hệ ngữ nghĩa - các từ liên quan công nghệ có similarity cao với "computer"

### 4.2. Task 2: Document Embedding
**Câu**: "The queen rules the country."
- **Vector** (5 phần tử đầu): `[0.02444, 0.37802, -0.63817, 0.01280, 0.05243]`
- **Kích thước**: 50 chiều (mean pooling của các word vectors)

### 4.3. Task 3: So sánh Model tự huấn luyện vs Pre-trained

| Metric | Model tự huấn luyện | GloVe Pre-trained |
|--------|---------------------|-------------------|
| Similarity 'king'-'queen' | 0.0560 | 0.7839 |
| Most similar to 'king' | cat, woman, the, prince, is | queen, prince, royal... |

**Phân tích:**
- Model tự huấn luyện cho kết quả **rất kém** (similarity chỉ 0.056)
- Nguyên nhân: Corpus chỉ có 6 câu đơn giản, không đủ dữ liệu để học mối quan hệ ngữ nghĩa
- **Kết luận**: Pre-trained models tiết kiệm tài nguyên và cho kết quả tốt hơn nhiều

### 4.4. Task 4: Spark MLlib
**5 từ đồng nghĩa với 'data':**
| Từ | Similarity |
|----|------------|
| engine | 0.1237 |
| powerful | 0.0811 |
| spark | 0.0651 |
| quickly | 0.0441 |
| unified | 0.0420 |

**Nhận xét**: Kết quả kém do corpus demo quá nhỏ (chỉ 3 câu), nhưng minh họa được quy trình huấn luyện phân tán với Spark

### 4.5. Task 5: Trực quan hóa (t-SNE)
- Các từ cùng trường ngữ nghĩa tạo thành cụm riêng biệt trên biểu đồ 2D
- **Cụm hoàng gia**: king, queen, prince, princess
- **Cụm quốc gia**: country, nation, kingdom
- **Cụm công nghệ**: computer, software, technology
- t-SNE giảm chiều từ 50D xuống 2D để trực quan hóa, giữ được cấu trúc cụm

---

## 5. Nhận xét

**Ưu điểm Pre-trained Models:**
- Tiết kiệm tài nguyên
- Tận dụng tri thức từ corpus khổng lồ

**Hạn chế:**
- OOV: Không xử lý từ mới/hiếm
- Static: Không phân biệt ngữ cảnh (bank = ngân hàng = bờ sông)

---

## 6. Khó khăn & Giải pháp

| Vấn đề | Giải pháp |
|--------|-----------|
| OOV | Trả về vector 0, bỏ qua khi embed |
| RAM | Dùng model nhỏ (50D) hoặc Spark |
| t-SNE chậm | Chỉ visualize 20-30 từ đại diện |

---

## 7. Trích dẫn
- Gensim: https://radimrehurek.com/gensim/
- Scikit-learn: https://scikit-learn.org/
- Apache Spark: https://spark.apache.org/
- GloVe: glove-wiki-gigaword-50 via Gensim API
