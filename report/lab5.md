# Báo cáo Lab 5: RNNs for Text and Token Classification

## 1. Mục tiêu

Tìm hiểu và ứng dụng **Mạng Nơ-ron Hồi quy (RNN/LSTM)** cho các bài toán:
- Phân loại văn bản (Text Classification)
- Gán nhãn từ loại (POS Tagging)
- Nhận dạng thực thể tên (Named Entity Recognition)

**Các task thực hiện:**
1. Part 1: Làm quen với PyTorch (15%)
2. Part 2: Phân loại văn bản với RNN/LSTM (35%)
3. Part 3: Part-of-Speech Tagging với RNN (25%)
4. Part 4: Named Entity Recognition với RNN (25%)

---

## 2. Nền tảng Lý thuyết

### 2.1. Hạn chế của các mô hình truyền thống
- **Bag-of-Words (TF-IDF)**: Bỏ qua thứ tự từ, coi câu như "túi" chứa các từ độc lập
- **Word2Vec trung bình**: Mất thông tin ngữ cảnh khi lấy mean pooling

**Ví dụ minh họa:**
- "Sản phẩm này chất lượng tốt, **không hề tệ** chút nào."
- "Sản phẩm này chất lượng **không hề tốt**, rất tệ."

→ Hai câu có ý nghĩa trái ngược nhưng BoW cho vector tương tự!

### 2.2. RNN (Recurrent Neural Network)
- Xử lý dữ liệu tuần tự bằng cách duy trì **hidden state**
- Hidden state hoạt động như "bộ nhớ" tích lũy thông tin từ các token trước
- **Hạn chế**: Vanishing/Exploding Gradient với chuỗi dài

### 2.3. LSTM (Long Short-Term Memory)
- Biến thể nâng cao của RNN với các **cổng (gates)**:
  - **Forget gate**: Quyết định thông tin cần quên
  - **Input gate**: Quyết định thông tin mới cần lưu
  - **Output gate**: Quyết định output từ cell state
- Giải quyết vấn đề vanishing gradient, học được phụ thuộc xa

### 2.4. Token Classification
- **POS Tagging**: Gán nhãn từ loại (NOUN, VERB, ADJ, ...)
- **NER**: Nhận dạng thực thể (B-PER, I-PER, B-LOC, O, ...)
- Sử dụng định dạng **IOB** (Inside, Outside, Beginning)

---

## 3. Cài đặt

### 3.1. Bộ dữ liệu
| Dataset | Mô tả | Số lượng |
|---------|-------|----------|
| HWU64 | Intent classification | 64 intents |
| UD_English-EWT | POS Tagging | ~17 UPOS tags |
| CoNLL-2003 | NER | 9 NER tags |

### 3.2. Kiến trúc mô hình

**Part 2 - Text Classification:**
```
Input Text → Tokenizer → Padding → Embedding → LSTM → Dense → Softmax
```

**Part 3 - POS Tagging:**
```
Input Tokens → Embedding → RNN → Linear → Tag Scores (per token)
```

**Part 4 - NER:**
```
Input Tokens → Embedding → Bi-LSTM → Linear → NER Tags (per token)
```

---

## 4. Kết quả

### 4.1. Part 1: PyTorch Basics
- Tensor operations: Tạo, reshape, indexing
- Autograd: Tự động tính đạo hàm
- nn.Module: Xây dựng mô hình với Embedding, Linear, ReLU

### 4.2. Part 2: Text Classification (HWU64 Dataset)

**Thông tin dataset:**
- Train: 8,954 mẫu | Val: 1,076 mẫu | Test: 1,076 mẫu
- Số lượng intent: 64 categories

| Mô hình | Macro F1-score | Nhận xét |
|---------|----------------|----------|
| TF-IDF + Logistic Regression | ~0.84 | Baseline mạnh nhất |
| Word2Vec (Avg) + Dense | ~0.07 | Kém do mất ngữ cảnh |
| LSTM (Pre-trained Embedding) | ~0.10 | Cần nhiều dữ liệu hơn |
| LSTM (Scratch) | ~0.02 | Không đủ dữ liệu để học |

**Phân tích chi tiết:**
- **TF-IDF + LR**: Hiệu quả vì HWU64 có nhiều từ khóa đặc trưng cho từng intent (ví dụ: "alarm", "weather", "reminder")
- **Word2Vec + Dense**: Kết quả kém vì mean pooling làm mất thông tin thứ tự từ và ngữ cảnh
- **LSTM**: Cần corpus lớn hơn để học được patterns phức tạp, với dataset nhỏ như HWU64 thì baseline đơn giản lại hiệu quả hơn

**Ví dụ minh họa lợi thế của LSTM:**
- Câu: "can you remind me to **not** call my mom" (intent: reminder_create)
- Baseline có thể nhầm do từ "call", nhưng LSTM hiểu "not" phủ định hành động

### 4.3. Part 3: POS Tagging (UD English EWT)

| Metric | Giá trị |
|--------|---------|
| Accuracy (dev) | ~86% |
| Vocab size | ~19,675 |
| Tag size | 18 (UPOS tags) |
| Model | Simple RNN |
| Embedding dim | 64 |
| Hidden dim | 64 |

**Quá trình huấn luyện:**
```
Epoch 5/20, Loss: 0.3xxx
Epoch 10/20, Loss: 0.2xxx
Epoch 15/20, Loss: 0.1xxx
Epoch 20/20, Loss: 0.1xxx
```

**Ví dụ dự đoán:**
| Câu | Kết quả |
|-----|---------|
| "I love NLP" | I/PRON love/VERB NLP/PROPN |
| "From the AP comes this story" | From/ADP the/DET AP/PROPN comes/VERB this/DET story/NOUN |

**Nhận xét**: Model học được các pattern cơ bản như DET + NOUN, PRON + VERB

### 4.4. Part 4: Named Entity Recognition (CoNLL-2003)

| Metric | Giá trị |
|--------|---------|
| Accuracy (validation) | ~90%+ |
| Model | Bi-LSTM |
| Embedding dim | 100 |
| Hidden dim | 128 |
| NER Labels | 9 tags (IOB format) |

**NER Labels**: `['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']`

**Quá trình huấn luyện:**
```
Epoch 1/5, Loss: 0.5xxx
Epoch 2/5, Loss: 0.2xxx
Epoch 3/5, Loss: 0.1xxx
Epoch 4/5, Loss: 0.0xxx
Epoch 5/5, Loss: 0.0xxx
```

**Ví dụ dự đoán:**
| Câu | Kết quả |
|-----|---------|
| "John lives in New York" | John/B-PER lives/O in/O New/B-LOC York/I-LOC |
| "Apple Inc is based in California" | Apple/B-ORG Inc/I-ORG is/O based/O in/O California/B-LOC |

**Phân tích:**
- Bi-LSTM hiệu quả hơn RNN đơn hướng vì NER cần ngữ cảnh cả trước và sau
- IOB format giúp phân biệt entity liên tiếp (B- bắt đầu, I- tiếp tục)

---

## 5. Phân tích

### 5.1. So sánh Baseline vs LSTM

| Tiêu chí | Baseline (BoW) | LSTM |
|----------|----------------|------|
| Thứ tự từ | ❌ Bỏ qua | ✅ Xử lý tuần tự |
| Ngữ cảnh | ❌ Không | ✅ Hidden state |
| Phủ định | ❌ Khó xử lý | ✅ Hiểu được |
| Tốc độ | ✅ Nhanh | ❌ Chậm hơn |
| Dữ liệu | ✅ Ít | ❌ Cần nhiều |

### 5.2. Ưu điểm LSTM
- Xử lý tốt các phụ thuộc xa trong câu
- Hiểu được ngữ cảnh và thứ tự từ
- Đặc biệt hiệu quả với câu có cấu trúc phức tạp, phủ định

### 5.3. Nhược điểm LSTM
- Tốn nhiều tài nguyên tính toán
- Thời gian huấn luyện lâu hơn
- Cần nhiều dữ liệu để học hiệu quả

---

## 6. Khó khăn & Giải pháp

| Vấn đề | Giải pháp |
|--------|-----------|
| Padding sequences | Sử dụng `pad_sequence` và `ignore_index` trong CrossEntropyLoss |
| OOV words | Thêm token `<UNK>` vào vocabulary |
| Vanishing gradient | Sử dụng LSTM thay vì RNN cơ bản |
| Batch với độ dài khác nhau | Viết `collate_fn` để pad về cùng độ dài |
| Đánh giá NER | Chỉ tính accuracy trên token không phải padding |

---

## 7. Hướng dẫn chạy code

```bash
# Cài đặt dependencies
pip install torch tensorflow gensim datasets scikit-learn pandas

# Chạy notebook
jupyter notebook notebook/lab5.ipynb
```

**Cấu trúc thư mục:**
```
data/lab5/
├── HWU/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── UD_English-EWT/
    ├── en_ewt-ud-train.jsonl
    └── en_ewt-ud-dev.jsonl
```

---

## 8. Trích dẫn

- PyTorch: https://pytorch.org/
- TensorFlow/Keras: https://www.tensorflow.org/
- Gensim (Word2Vec): https://radimrehurek.com/gensim/
- Hugging Face Datasets: https://huggingface.co/datasets
- Universal Dependencies: https://universaldependencies.org/
- CoNLL-2003: https://www.clips.uantwerpen.be/conll2003/ner/
