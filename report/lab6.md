# Báo cáo Lab 6: Transformers

## 1. Mục tiêu

Tìm hiểu và ứng dụng **Transformers** - kiến trúc nền tảng cho các mô hình NLP hiện đại. Bài lab tập trung vào việc hiểu sâu về cơ chế hoạt động của mô hình và cách sử dụng thư viện `transformers` của Hugging Face để giải quyết các bài toán cơ bản.

**Các task thực hiện:**
1.  **Masked Language Modeling (MLM)**: Hiểu cách mô hình Encoder (BERT) "hiểu" ngữ cảnh hai chiều.
2.  **Next Token Prediction (Text Generation)**: Hiểu cơ chế sinh văn bản tuần tự của mô hình Decoder (GPT).
3.  **Sentence Representation**: Trích xuất đặc trưng câu (Embedding) để so sánh độ tương đồng ngữ nghĩa.

---

## 2. Nền tảng Lý thuyết

### 2.1. Kiến trúc Transformer (Vaswani et al., 2017)
Transformer là kiến trúc mạng nơ-ron dựa hoàn toàn vào cơ chế **Attention** (Chú ý), loại bỏ sự phụ thuộc vào Recurrent (RNN) hay Convolution (CNN). Điều này cho phép mô hình huấn luyện song song hóa cao và nắm bắt sự phụ thuộc xa (long-range dependencies) tốt hơn.

Kiến trúc gốc gồm 2 phần:
-   **Encoder**: Đọc toàn bộ chuỗi đầu vào cùng lúc, hiểu ngữ cảnh và tạo ra các biểu diễn trung gian (hidden states).
-   **Decoder**: Nhận biểu diễn từ Encoder và sinh ra chuỗi đầu ra từng token một (auto-regressive).

### 2.2. Cơ chế Self-Attention
Đây là trái tim của Transformer. Với mỗi từ trong câu, Self-Attention giúp mô hình "nhìn" vào các từ khác để hiểu ngữ cảnh.

Cơ chế này hoạt động dựa trên 3 vector: **Query (Q)**, **Key (K)**, và **Value (V)**.
-   **Query**: "Tôi đang tìm kiếm thông tin gì?"
-   **Key**: "Tôi có thông tin gì để cung cấp?"
-   **Value**: "Nội dung thông tin của tôi là gì?"

**Công thức Scaled Dot-Product Attention:**
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

> **Ví dụ:** Trong câu "The animal didn't cross the street because **it** was too wide", Self-Attention giúp từ "**it**" có trọng số cao với "**street**" (vì đường rộng) thay vì "animal".

### 2.3. Positional Encoding
Vì Transformer xử lý song song và không có tính tuần tự như RNN, nó không biết thứ tự của từ. **Positional Encoding** được cộng vào Input Embeddings để cung cấp thông tin về vị trí tương đối hoặc tuyệt đối của token trong chuỗi.

### 2.4. So sánh Chi tiết: BERT vs. GPT

| Đặc điểm | BERT (Bidirectional Encoder Representations from Transformers) | GPT (Generative Pre-trained Transformer) |
| :--- | :--- | :--- |
| **Kiến trúc** | **Encoder-only** (Chỉ dùng phần Encoder) | **Decoder-only** (Chỉ dùng phần Decoder) |
| **Cơ chế Attention** | **Bidirectional** (Hai chiều): Nhìn thấy cả từ trước và sau. | **Unidirectional** (Một chiều): Chỉ nhìn thấy các từ phía trước. |
| **Objective** | **Masked Language Modeling (MLM)**: Dự đoán từ bị che. | **Causal Language Modeling (CLM)**: Dự đoán từ tiếp theo. |
| **Ưu điểm** | Hiểu sâu ngữ cảnh 2 chiều, tốt cho bài toán hiểu (Classification, QA). | Sinh văn bản tự nhiên, tốt cho bài toán sáng tạo (Generation, Chat). |

---

## 3. Dataset

Bài lab này sử dụng các mô hình đã được huấn luyện trước (Pre-trained Models) trên các tập dữ liệu khổng lồ (Wikipedia, BookCorpus, Common Crawl). Chúng ta không cần train lại mà chỉ sử dụng cho giai đoạn Inference (Dự đoán).

**Dữ liệu đầu vào mẫu:**
-   **Task MLM**: `Hanoi is the <mask> of Vietnam.`
-   **Task Generation**: `The best thing about learning NLP is`
-   **Task Similarity**: Các câu tiếng Anh đơn giản về chủ đề AI và thời tiết.

---

## 4. Cài đặt

### 4.1. Bài 1: Masked Language Modeling (BERT)
**Mã lệnh:** `pipeline("fill-mask", model="distilbert/distilroberta-base")`

1.  **Preprocessing (Tokenization)**: Tokenizer tách câu thành subwords và thêm token đặc biệt (`[CLS]`, `[SEP]`, `<mask>`).
2.  **Forward Pass**: Encoder xử lý context hai chiều để tạo vector đại diện cho vị trí mask.
3.  **Prediction**: Tính softmax để tìm từ có xác suất cao nhất điền vào chỗ trống.

### 4.2. Bài 2: Text Generation (GPT)
**Mã lệnh:** `generator(prompt, max_length=50)`

Cơ chế **Auto-regressive Generation**:
1.  Model dự đoán từ tiếp theo dựa trên chuỗi hiện tại.
2.  Từ mới được thêm vào chuỗi input.
3.  Lặp lại quá trình cho đến khi hoàn thành câu.

### 4.3. Bài 3: Sentence Representation (Mean Pooling)
**Mục tiêu:** Tạo vector cố định (768 chiều) đại diện cho cả câu.

**Thuật toán Mean Pooling:**
1.  **Attention Mask**: Xác định đâu là từ thật (1), đâu là padding (0).
2.  **Lọc**: Loại bỏ các vector của padding tokens (nhân với 0).
3.  **Trung bình**: Tính tổng các vector từ thật chia cho số lượng từ thật.

```python
# Pseudo-code
mask_expanded = attention_mask.unsqueeze(-1)
sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
sentence_embedding = sum_embeddings / sum_mask
```

---

## 5. Kết quả

### 5.1. Masked Language Modeling
-   **Input**: `Hanoi is the <mask> of Vietnam.`
-   **Top 1 Prediction**: `capital` (Confidence: 93.41%)
-   **Nhận xét**: Model rất tự tin nhờ context hai chiều mạnh mẽ.

### 5.2. Text Generation
-   **Input**: `The best thing about learning NLP is`
-   **Output**: "...that you learn a lot. And I hope it helps you in other ways too..."
-   **Nhận xét**: Văn bản sinh ra mạch lạc, đúng ngữ pháp, tuy nhiên có thể lan man nếu không có prompt kỹ.

### 5.3. Sentence Similarity
Ma trận cosine similarity:
-   Nhóm AI ("machine learning", "artificial intelligence"): Tương đồng ~0.86.
-   Nhóm Thời tiết ("weather", "sunny"): Tương đồng ~0.84.
-   Khác nhóm: Tương đồng thấp (~0.60).

---

## 6. Nhận xét

### 6.1. Ưu điểm của Transformers
-   **Hiểu ngữ cảnh sâu sắc**: Nhờ cơ chế Self-Attention, từ đa nghĩa được xử lý tốt hơn hẳn Word2Vec.
-   **Tính đa năng**: Một kiến trúc có thể giải quyết hầu hết các bài toán NLP (Phân loại, Tóm tắt, Dịch, Sinh văn bản).
-   **Hệ sinh thái mạnh**: Hugging Face giúp việc tiếp cận các model SOTA trở nên cực kỳ đơn giản.

### 6.2. Câu hỏi thường gặp
**Q1: Tại sao kích thước vector lại là 768?**
-   Đây là tham số `hidden_size` thiết kế của `bert-base`. Model lớn hơn sẽ có vector lớn hơn (ví dụ 1024).

**Q2: Tại sao không dùng vector [CLS] để đại diện cho câu?**
-   Mặc dù `[CLS]` được thiết kế cho bài toán phân loại, thực nghiệm cho thấy **Mean Pooling** các token thường mang lại biểu diễn ngữ nghĩa tốt hơn cho bài toán so sánh độ tương đồng (Semantic Similarity).

---

## 7. Khó khăn & Giải pháp

| Vấn đề | Giải pháp |
| :--- | :--- |
| **Tải model chậm** | Model Hugging Face thường nặng (vài trăm MB đến vài GB). Giải pháp: Model sẽ được cache lại tại `~/.cache/huggingface`, chỉ cần tải lần đầu. |
| **Bộ nhớ hạn chế** | Khi chạy trên CPU/Laptop, model lớn gây tràn RAM. Giải pháp: Dùng `torch.no_grad()` khi inference, hoặc chọn model nhỏ hơn như `distilbert`. |
| **Token Mask khác nhau** | BERT dùng `[MASK]`, RoBERTa dùng `<mask>`. Giải pháp: Luôn kiểm tra `tokenizer.mask_token` trước khi dùng. |

---

## 8. Hướng dẫn Chạy Code

```bash
# 1. Cài đặt thư viện
pip install transformers torch scikit-learn numpy

# 2. Chạy notebook
jupyter notebook notebook/lab6.ipynb
```

---

## 9. Trích dẫn
-   **Vaswani et al.**, "Attention Is All You Need", 2017.
-   **Devlin et al.**, "BERT: Pre-training of Deep Bidirectional Transformers", 2019.
-   **Hugging Face Documentation**: https://huggingface.co/docs
