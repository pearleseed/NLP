# Báo cáo Lab 6: Giới thiệu về Transformers

## 1. Mục tiêu

Tìm hiểu và ứng dụng **Transformers** - kiến trúc nền tảng cho các mô hình NLP hiện đại.

**Các task thực hiện:**
1. Khôi phục Masked Token (Masked Language Modeling)
2. Dự đoán từ tiếp theo (Next Token Prediction)
3. Tính toán Vector biểu diễn của câu (Sentence Representation)

---

## 2. Nền tảng Lý thuyết

### 2.1. Kiến trúc Transformer
Được giới thiệu trong bài báo "Attention Is All You Need" (Vaswani et al., 2017), bao gồm:
- **Encoder**: Đọc và hiểu văn bản đầu vào, tạo ra biểu diễn giàu ngữ cảnh
- **Decoder**: Dựa vào biểu diễn của Encoder để sinh văn bản đầu ra
- **Self-Attention**: Cơ chế cốt lõi cho phép mô hình cân nhắc tầm quan trọng của các từ khác nhau

### 2.2. Các loại mô hình Transformer

| Loại | Ví dụ | Đặc điểm | Tác vụ phù hợp |
|------|-------|----------|----------------|
| Encoder-only | BERT, RoBERTa | Bidirectional, hiểu ngữ cảnh hai chiều | Phân loại, NER, MLM |
| Decoder-only | GPT-2, GPT-3 | Unidirectional, auto-regressive | Sinh văn bản |
| Encoder-Decoder | T5, BART | Kết hợp cả hai | Dịch máy, tóm tắt |

### 2.3. Masked Language Modeling (MLM)
- Che một số từ trong câu bằng token `[MASK]`
- Mô hình dự đoán từ gốc dựa vào ngữ cảnh xung quanh
- Phương pháp huấn luyện chính của BERT

### 2.4. Next Token Prediction
- Dự đoán từ tiếp theo dựa vào các từ đã xuất hiện
- Phương pháp huấn luyện chính của GPT
- Cơ chế auto-regressive: sinh từng token một

---

## 3. Cài đặt

### 3.1. Thư viện sử dụng
```bash
pip install transformers torch
```

### 3.2. Pipeline của Hugging Face
- `fill-mask`: Khôi phục masked token
- `text-generation`: Sinh văn bản
- `AutoTokenizer`, `AutoModel`: Tải model và tokenizer

---

## 4. Kết quả

### 4.1. Bài 1: Masked Language Modeling

**Model sử dụng:** `distilbert/distilroberta-base` (tự động tải qua pipeline)

**Input:** `Hanoi is the <mask> of Vietnam.`

| Dự đoán | Độ tin cậy | Câu hoàn chỉnh |
|---------|------------|----------------|
| capital | 0.9341 | Hanoi is the capital of Vietnam. |
| Republic | 0.0300 | Hanoi is the Republic of Vietnam. |
| Capital | 0.0105 | Hanoi is the Capital of Vietnam. |
| birthplace | 0.0054 | Hanoi is the birthplace of Vietnam. |
| heart | 0.0014 | Hanoi is the heart of Vietnam. |

**Phân tích kết quả:**
- Model dự đoán đúng "capital" với độ tin cậy **rất cao (93.41%)**
- Các dự đoán khác cũng hợp lý về mặt ngữ pháp (Republic, birthplace, heart)
- Khoảng cách confidence giữa top-1 và top-2 rất lớn (93% vs 3%) → Model rất tự tin

**Tại sao BERT phù hợp?**
- BERT là **bidirectional**: nhìn được cả "Hanoi" (thủ đô) và "Vietnam" (quốc gia)
- Ngữ cảnh hai chiều giúp hiểu đầy đủ ý nghĩa câu để dự đoán chính xác

### 4.2. Bài 2: Text Generation

**Model sử dụng:** `openai-community/gpt2`

**Input:** `The best thing about learning NLP is`

**Output mẫu:**
```
The best thing about learning NLP is that you learn a lot.

And I hope it helps you in other ways too.

Let's go back to the question: who can make your own version of Google?

I'm not sure if this is the right question to ask. I think that you will 
get a little bit of a feeling...
```

**Phân tích kết quả:**
- Văn bản sinh ra có **ngữ pháp đúng** và **mạch lạc**
- Nội dung phát triển tự nhiên từ câu mồi
- Kết quả có thể khác nhau mỗi lần chạy do tính ngẫu nhiên (sampling)

**Tại sao GPT phù hợp?**
- GPT là **unidirectional** (một chiều): chỉ nhìn các từ đã xuất hiện
- Cơ chế **auto-regressive**: dự đoán từ tiếp theo → thêm vào input → dự đoán tiếp
- Hoàn toàn phù hợp với bản chất sinh văn bản tuần tự từ trái sang phải

### 4.3. Bài 3: Sentence Representation

**Model sử dụng:** `bert-base-uncased`

**Input:** `This is a sample sentence.`

**Tokenization:**
```
Tokens: ['[CLS]', 'this', 'is', 'a', 'sample', 'sentence', '.', '[SEP]']
Input IDs: [101, 2023, 2003, 1037, 7099, 6251, 1012, 102]
```

**Output:**
- `last_hidden_state` shape: `(1, 8, 768)` → 8 tokens, mỗi token có vector 768 chiều
- Sentence embedding (Mean Pooling): Vector 768 chiều
- 10 phần tử đầu: `[-0.0639, -0.4284, -0.0668, -0.3843, -0.0658, -0.2183, 0.4764, 0.4866, 0.0000, -0.0743]`

**Ứng dụng - So sánh độ tương đồng (Cosine Similarity):**

| Index | Câu |
|-------|-----|
| [0] | I love machine learning. |
| [1] | I enjoy studying artificial intelligence. |
| [2] | The weather is nice today. |
| [3] | It is sunny outside. |

**Ma trận Similarity:**
```
     [0]    [1]    [2]    [3]
[0]  1.000  0.863  0.589  0.619
[1]  0.863  1.000  0.607  0.634
[2]  0.589  0.607  1.000  0.839
[3]  0.619  0.634  0.839  1.000
```

**Phân tích:**
- Câu [0] và [1] có similarity cao (0.863) vì cùng chủ đề AI/ML
- Câu [2] và [3] có similarity cao (0.839) vì cùng chủ đề thời tiết
- Các cặp câu khác chủ đề có similarity thấp hơn (~0.6)

---

## 5. Nhận xét

**Ưu điểm của Transformers:**
- Nắm bắt ngữ cảnh tốt nhờ cơ chế Self-Attention
- Pre-trained models tiết kiệm tài nguyên và thời gian
- Hugging Face cung cấp API đơn giản, dễ sử dụng

**So sánh với Word Embeddings (Lab 3):**
| Tiêu chí | Word2Vec/GloVe | Transformers |
|----------|----------------|--------------|
| Ngữ cảnh | Static (1 vector/từ) | Dynamic (phụ thuộc ngữ cảnh) |
| Polysemy | Không xử lý được | Xử lý tốt |
| Tài nguyên | Nhẹ | Nặng hơn |

---

## 6. Khó khăn & Giải pháp

| Vấn đề | Giải pháp |
|--------|-----------|
| Tải model chậm | Model được cache sau lần đầu |
| Bộ nhớ hạn chế | Dùng `torch.no_grad()`, model nhỏ (distilbert) |
| Token mask khác nhau | Kiểm tra tokenizer của model cụ thể |
| Kết quả sinh không ổn định | Điều chỉnh temperature, top_k, top_p |

---

## 7. Trả lời câu hỏi

### Bài 1:
1. **Mô hình đã dự đoán đúng từ `capital` không?**
   - Có, với độ tin cậy 93.41%

2. **Tại sao BERT phù hợp cho MLM?**
   - BERT là bidirectional, nhìn được cả hai chiều để hiểu ngữ cảnh đầy đủ

### Bài 2:
1. **Kết quả sinh ra có hợp lý không?**
   - Có, ngữ pháp đúng và phát triển ý tưởng tự nhiên

2. **Tại sao GPT phù hợp cho text generation?**
   - GPT là unidirectional, auto-regressive - phù hợp với việc sinh văn bản tuần tự

### Bài 3:
1. **Kích thước vector là bao nhiêu?**
   - 768 chiều, tương ứng với `hidden_size` của bert-base-uncased

2. **Tại sao cần `attention_mask` khi Mean Pooling?**
   - Để bỏ qua padding tokens, chỉ tính trung bình trên các token thực sự

---

## 8. Trích dẫn
- Hugging Face Transformers: https://huggingface.co/transformers/
- Vaswani et al., "Attention Is All You Need", 2017
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", 2019
- Radford et al., "Language Models are Unsupervised Multitask Learners", 2019
