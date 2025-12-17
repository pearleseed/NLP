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

### 2.1. Text Classification
Text Classification là bài toán gán nhãn (categories/labels) cho văn bản. Ứng dụng phổ biến:
- Sentiment Analysis (phân tích cảm xúc)
- Spam Detection (phát hiện spam)
- Topic Labeling (gán chủ đề)

### 2.2. Pipeline
```
Raw Text → Tokenization → Vectorization → ML Model → Prediction
```

### 2.3. Các Model sử dụng
- **Logistic Regression**: Model tuyến tính đơn giản, hiệu quả cho phân loại nhị phân và đa lớp
- **Multinomial Naive Bayes**: Model xác suất dựa trên định lý Bayes, phù hợp với dữ liệu text

### 2.4. Evaluation Metrics
- **Accuracy**: Tỷ lệ dự đoán đúng
- **Precision**: Độ chính xác của các dự đoán positive
- **Recall**: Khả năng tìm được tất cả positive samples
- **F1-Score**: Trung bình điều hòa của Precision và Recall

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
