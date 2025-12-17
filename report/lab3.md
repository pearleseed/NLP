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

### 2.1. Giả thuyết Phân bố (Distributional Hypothesis)
> "Những từ xuất hiện trong cùng ngữ cảnh có xu hướng mang ý nghĩa tương đồng."

### 2.2. Word2Vec
- **CBOW**: Dự đoán từ đích từ ngữ cảnh. Nhanh, tốt với dữ liệu lớn.
- **Skip-gram**: Dự đoán ngữ cảnh từ từ đích. Tốt với từ hiếm.
- **Hạn chế**: Mô hình tĩnh - mỗi từ chỉ có 1 vector bất kể ngữ cảnh.

### 2.3. GloVe
Kết hợp thống kê toàn cục (ma trận đồng xuất hiện) với phương pháp dự đoán.

### 2.4. FastText
Biểu diễn từ bằng n-gram ký tự → xử lý được từ OOV.

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
