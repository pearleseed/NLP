# Báo cáo Lab 4: Text Classification

## 1. Mục tiêu

Xây dựng pipeline phân loại văn bản hoàn chỉnh, từ raw text đến trained model, sử dụng các kỹ thuật tokenization và vectorization đã học.

**Các task thực hiện:**
1. Chuẩn bị dữ liệu và xây dựng TextClassifier
2. Huấn luyện và đánh giá với Logistic Regression (Baseline)
3. Cải tiến với Multinomial Naive Bayes
4. Thử nghiệm với Word2Vec features
5. So sánh và phân tích kết quả

---

## 2. Nền tảng Lý thuyết

### 2.1. Text Classification - Tổng quan

#### 2.1.1. Định nghĩa hình thức
Text Classification là bài toán học có giám sát (supervised learning):
- **Input**: Document `d` (văn bản)
- **Output**: Label `y ∈ {c₁, c₂, ..., cₖ}` (một trong k classes)
- **Mục tiêu**: Học hàm `f: D → C` từ training data

#### 2.1.2. Các loại Text Classification

| Loại | Mô tả | Ví dụ |
|------|-------|-------|
| Binary | 2 classes | Spam vs Not Spam |
| Multi-class | >2 classes, mỗi doc 1 label | Sentiment: Positive/Negative/Neutral |
| Multi-label | Mỗi doc có thể nhiều labels | News: Politics + Economy + International |

#### 2.1.3. Ứng dụng thực tế

| Ứng dụng | Input | Output |
|----------|-------|--------|
| Sentiment Analysis | Review sản phẩm | Positive/Negative/Neutral |
| Spam Detection | Email | Spam/Ham |
| Intent Classification | User query | Intent (book_flight, weather, ...) |
| Topic Labeling | News article | Category (Sports, Politics, ...) |
| Language Detection | Text | Language (en, vi, fr, ...) |

### 2.2. Pipeline Text Classification

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐    ┌────────────┐
│  Raw Text   │ →  │ Preprocessing│ →  │ Vectorization│ →  │ ML Model │ →  │ Prediction │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────┘    └────────────┘
                         │                    │                 │
                         ▼                    ▼                 ▼
                   - Lowercase          - BoW/TF-IDF      - Naive Bayes
                   - Remove noise       - Word2Vec        - Logistic Reg
                   - Tokenization       - BERT            - SVM
                   - Stopwords                            - Neural Networks
```

### 2.3. Logistic Regression - Chi tiết

#### 2.3.1. Binary Logistic Regression

**Mô hình:**
```
P(y=1|x) = σ(wᵀx + b) = 1 / (1 + e^(-(wᵀx + b)))
```

Trong đó:
- `x ∈ ℝⁿ`: Feature vector (TF-IDF vector)
- `w ∈ ℝⁿ`: Weight vector (learned)
- `b ∈ ℝ`: Bias term
- `σ(z)`: Sigmoid function

**Sigmoid Function:**
```
σ(z) = 1 / (1 + e⁻ᶻ)

Tính chất:
- σ(z) ∈ (0, 1) ∀z
- σ(0) = 0.5
- σ(-z) = 1 - σ(z)
```

**Decision boundary:**
```
ŷ = 1  if P(y=1|x) ≥ 0.5  (tức wᵀx + b ≥ 0)
ŷ = 0  otherwise
```

#### 2.3.2. Loss Function - Binary Cross-Entropy

```
L(w, b) = -1/m Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

Trong đó:
- `m`: Số training samples
- `yᵢ`: True label (0 hoặc 1)
- `ŷᵢ`: Predicted probability

**Ý nghĩa:**
- Khi `y=1`: Loss = `-log(ŷ)` → Muốn ŷ → 1
- Khi `y=0`: Loss = `-log(1-ŷ)` → Muốn ŷ → 0

#### 2.3.3. Multinomial Logistic Regression (Softmax)

Mở rộng cho multi-class với K classes:

**Softmax Function:**
```
P(y=k|x) = exp(wₖᵀx + bₖ) / Σⱼ exp(wⱼᵀx + bⱼ)
```

**Tính chất:**
- Σₖ P(y=k|x) = 1 (tổng xác suất = 1)
- P(y=k|x) > 0 ∀k

**Cross-Entropy Loss (Multi-class):**
```
L = -1/m Σᵢ Σₖ yᵢₖ log(ŷᵢₖ)
```

Với one-hot encoding: `yᵢₖ = 1` nếu sample i thuộc class k, ngược lại = 0.

#### 2.3.4. Regularization

Để tránh overfitting, thêm regularization term:

**L2 Regularization (Ridge):**
```
L_reg = L + λ ||w||² = L + λ Σⱼ wⱼ²
```

**L1 Regularization (Lasso):**
```
L_reg = L + λ ||w||₁ = L + λ Σⱼ |wⱼ|
```

| Regularization | Ưu điểm | Nhược điểm |
|----------------|---------|------------|
| L2 | Stable, smooth | Không sparse |
| L1 | Feature selection (sparse) | Không stable |
| Elastic Net | Kết hợp cả hai | Thêm hyperparameter |

### 2.4. Naive Bayes - Chi tiết

#### 2.4.1. Bayes' Theorem

```
P(c|d) = P(d|c) × P(c) / P(d)
```

Trong đó:
- `P(c|d)`: Posterior - xác suất class c cho document d
- `P(d|c)`: Likelihood - xác suất document d trong class c
- `P(c)`: Prior - xác suất class c
- `P(d)`: Evidence - xác suất document d

#### 2.4.2. Naive Assumption

**Giả định độc lập có điều kiện:**
```
P(d|c) = P(w₁, w₂, ..., wₙ|c) = Πᵢ P(wᵢ|c)
```

Các từ trong document độc lập với nhau khi biết class.

**Tại sao "Naive"?**
- Giả định này hiếm khi đúng trong thực tế
- "New York" - "New" và "York" không độc lập
- Nhưng vẫn hoạt động tốt trong practice!

#### 2.4.3. Multinomial Naive Bayes

Phù hợp với text data (word counts):

**Classification rule:**
```
ĉ = argmax_c [log P(c) + Σᵢ count(wᵢ, d) × log P(wᵢ|c)]
```

**Ước lượng parameters:**
```
P(c) = Nᶜ / N                    (Prior)

P(wᵢ|c) = (count(wᵢ, c) + α) / (Σⱼ count(wⱼ, c) + α|V|)  (Likelihood với Laplace smoothing)
```

Trong đó:
- `Nᶜ`: Số documents trong class c
- `N`: Tổng số documents
- `count(wᵢ, c)`: Số lần từ wᵢ xuất hiện trong class c
- `α`: Smoothing parameter (thường = 1)
- `|V|`: Vocabulary size

#### 2.4.4. Laplace Smoothing

**Vấn đề:** Nếu từ wᵢ không xuất hiện trong class c → P(wᵢ|c) = 0 → P(d|c) = 0

**Giải pháp:** Thêm α (pseudo-count) vào mỗi count:
```
P(wᵢ|c) = (count(wᵢ, c) + α) / (Σⱼ count(wⱼ, c) + α|V|)
```

Với α = 1: Laplace smoothing (add-one smoothing)

#### 2.4.5. Ví dụ tính toán Naive Bayes

```
Training data:
- "good movie" → Positive
- "great film" → Positive  
- "bad movie" → Negative
- "terrible film" → Negative

Test: "good film" → ?

Vocabulary: {good, movie, great, film, bad, terrible}

P(Pos) = 2/4 = 0.5
P(Neg) = 2/4 = 0.5

P(good|Pos) = (1+1)/(4+6) = 0.2
P(film|Pos) = (1+1)/(4+6) = 0.2
P(good|Neg) = (0+1)/(4+6) = 0.1
P(film|Neg) = (1+1)/(4+6) = 0.2

P(Pos|"good film") ∝ 0.5 × 0.2 × 0.2 = 0.02
P(Neg|"good film") ∝ 0.5 × 0.1 × 0.2 = 0.01

→ Predict: Positive (0.02 > 0.01)
```

### 2.5. Evaluation Metrics

#### 2.5.1. Confusion Matrix

```
                    Predicted
                 Pos      Neg
Actual  Pos  [   TP   |   FN   ]
        Neg  [   FP   |   TN   ]
```

- **TP (True Positive)**: Dự đoán Positive, thực tế Positive
- **TN (True Negative)**: Dự đoán Negative, thực tế Negative
- **FP (False Positive)**: Dự đoán Positive, thực tế Negative (Type I Error)
- **FN (False Negative)**: Dự đoán Negative, thực tế Positive (Type II Error)

#### 2.5.2. Các Metrics

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Tỷ lệ dự đoán đúng. **Hạn chế:** Không phù hợp với imbalanced data.

**Precision:**
```
Precision = TP / (TP + FP)
```
"Trong các dự đoán Positive, bao nhiêu % đúng?"
Quan trọng khi **FP costly** (spam filter - không muốn đánh nhầm email quan trọng là spam)

**Recall (Sensitivity):**
```
Recall = TP / (TP + FN)
```
"Trong các Positive thực tế, bao nhiêu % được tìm thấy?"
Quan trọng khi **FN costly** (cancer detection - không muốn bỏ sót bệnh nhân)

**F1-Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Harmonic mean của Precision và Recall. Cân bằng giữa hai metrics.

#### 2.5.3. Macro vs Micro vs Weighted Average

Với multi-class classification:

| Averaging | Công thức | Khi nào dùng |
|-----------|-----------|--------------|
| Macro | Trung bình các class metrics | Quan tâm đều các class |
| Micro | Tính trên tổng TP, FP, FN | Class lớn quan trọng hơn |
| Weighted | Weighted by class support | Imbalanced data |

```
Macro-F1 = (F1_class1 + F1_class2 + F1_class3) / 3
Micro-F1 = 2×(Σ TPᵢ) / (2×Σ TPᵢ + Σ FPᵢ + Σ FNᵢ)
```

### 2.6. Xử lý Imbalanced Data

#### 2.6.1. Vấn đề
Khi một class chiếm đa số (ví dụ: 90% Neutral, 5% Positive, 5% Negative):
- Model có xu hướng predict class đa số
- Accuracy cao nhưng không hữu ích

#### 2.6.2. Các giải pháp

| Phương pháp | Mô tả |
|-------------|-------|
| **Oversampling** | Tăng samples của minority class (SMOTE) |
| **Undersampling** | Giảm samples của majority class |
| **Class weights** | Tăng weight cho minority class trong loss |
| **Threshold tuning** | Điều chỉnh decision threshold |

**Class weights trong Logistic Regression:**
```python
model = LogisticRegression(class_weight='balanced')
# Tự động tính: weight_c = n_samples / (n_classes × n_samples_c)
```

### 2.7. So sánh Logistic Regression vs Naive Bayes

| Tiêu chí | Logistic Regression | Naive Bayes |
|----------|---------------------|-------------|
| Loại model | Discriminative | Generative |
| Học gì | P(y\|x) trực tiếp | P(x\|y) và P(y) |
| Giả định | Linear decision boundary | Feature independence |
| Training | Iterative (gradient descent) | Closed-form (counting) |
| Tốc độ training | Chậm hơn | Rất nhanh |
| Data ít | Kém hơn | Tốt hơn |
| Data nhiều | Tốt hơn | Tương đương |
| Correlated features | Xử lý được | Bị ảnh hưởng |

---

## 3. Dataset

### 3.1. Thông tin Dataset
- **Nguồn**: Twitter Financial News Sentiment
- **Số mẫu train**: 9,543
- **Số mẫu validation**: 2,388

### 3.2. Phân bố nhãn
| Label | Tên | Số lượng (train) |
|-------|-----|------------------|
| 0 | Bearish (tiêu cực) | ~1,789 |
| 1 | Bullish (tích cực) | ~2,398 |
| 2 | Neutral (trung tính) | ~7,744 |

**Nhận xét**: Dữ liệu không cân bằng, nhãn Neutral chiếm đa số (~65%).

---

## 4. Cài đặt

### 4.1. Lớp TextClassifier
```python
class TextClassifier:
    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model
    
    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
    
    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)
    
    def evaluate(self, y_true, y_pred):
        # Tính accuracy, precision, recall, f1
        ...
```

### 4.2. Cấu hình Model

**TF-IDF Vectorizer:**
- `max_features=5000`: Giới hạn vocabulary
- `ngram_range=(1, 2)`: Unigrams và bigrams
- `stop_words='english'`: Loại bỏ stop words

**Logistic Regression:**
- `solver='lbfgs'`
- `max_iter=1000`
- `multi_class='multinomial'`

**Naive Bayes:**
- `alpha=0.1`: Laplace smoothing

---

## 5. Kết quả

### 5.1. Bảng So sánh

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression + TF-IDF | 0.8061 | 0.8024 | 0.8061 | 0.7887 |

### 5.2. Phân tích Chi tiết theo từng nhãn

**Classification Report (Logistic Regression + TF-IDF):**

| Nhãn | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| Bearish (0) | 0.78 | 0.40 | 0.53 | 347 |
| Bullish (1) | 0.78 | 0.60 | 0.68 | 475 |
| Neutral (2) | 0.81 | 0.96 | 0.88 | 1566 |

**Phân tích kết quả:**

1. **Nhãn Neutral (2)** - Hiệu suất tốt nhất:
   - Recall rất cao (0.96): Model phát hiện được 96% các mẫu Neutral
   - Nguyên nhân: Chiếm đa số trong dataset (1566/2388 = 65.6%)
   - Model có xu hướng "thiên vị" về nhãn này

2. **Nhãn Bearish (0)** - Hiệu suất kém nhất:
   - Recall thấp (0.40): Chỉ phát hiện được 40% các mẫu Bearish
   - F1-Score thấp (0.53): Mất cân bằng giữa precision và recall
   - Nguyên nhân: Số lượng mẫu ít nhất (347 mẫu)

3. **Nhãn Bullish (1)** - Hiệu suất trung bình:
   - Recall 0.60: Phát hiện được 60% các mẫu Bullish
   - Precision 0.78: Khi dự đoán Bullish, đúng 78%

**Vấn đề chính**: Dữ liệu không cân bằng (imbalanced) khiến model học tốt nhãn đa số (Neutral) nhưng kém với nhãn thiểu số (Bearish, Bullish)

---

## 6. Nhận xét

### 6.1. Ưu điểm của TF-IDF
- Đơn giản, hiệu quả
- Capture được tầm quan trọng của từ
- Phù hợp với văn bản ngắn

### 6.2. Hạn chế
- **Dữ liệu không cân bằng**: Neutral chiếm đa số → Model có xu hướng dự đoán Neutral
- **OOV**: Từ mới không có trong vocabulary
- **Ngữ cảnh**: TF-IDF không capture được ngữ cảnh và thứ tự từ

### 6.3. Đề xuất Cải tiến
1. **Xử lý dữ liệu không cân bằng**: SMOTE, class weights
2. **Tiền xử lý tốt hơn**: Loại bỏ URLs, @mentions, hashtags
3. **Model phức tạp hơn**: BERT, RoBERTa cho domain-specific tasks
4. **Fine-tune embeddings**: Huấn luyện Word2Vec trên corpus tài chính

---

## 7. Khó khăn và Giải pháp

| Vấn đề | Giải pháp |
|--------|-----------|
| Dữ liệu không cân bằng | Sử dụng weighted metrics |
| OOV trong Word2Vec | Trả về vector 0, bỏ qua từ không có |
| Overfitting với TF-IDF | Giới hạn max_features, regularization |
| Tweets có nhiều noise | Tiền xử lý: loại bỏ URLs, mentions |

---

## 8. Hướng dẫn Chạy Code

```bash
# Cài đặt dependencies
pip install scikit-learn pandas numpy matplotlib seaborn gensim

# Chạy notebook
jupyter notebook notebook/lab4.ipynb
```

---

## 9. Trích dẫn
- Scikit-learn: https://scikit-learn.org/
- Gensim (Word2Vec): https://radimrehurek.com/gensim/
- Dataset: Twitter Financial News Sentiment
- TF-IDF: Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval.
