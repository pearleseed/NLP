# Báo cáo Lab 7: Dependency Parsing

## 1. Mục tiêu

Thực hành **Phân tích cú pháp phụ thuộc (Dependency Parsing)** - kỹ thuật hiểu cấu trúc ngữ pháp của câu.

**Các task thực hiện:**
1. Giới thiệu và cài đặt spaCy
2. Phân tích câu và trực quan hóa cây phụ thuộc
3. Truy cập các thành phần trong cây phụ thuộc
4. Duyệt cây phụ thuộc để trích xuất thông tin
5. Bài tập tự luyện

---

## 2. Nền tảng Lý thuyết

### 2.1. Phân tích cú pháp phụ thuộc
Biểu diễn cấu trúc câu dưới dạng các mối quan hệ:
- **Head (điều khiển)**: Từ chính trong quan hệ
- **Dependent (phụ thuộc)**: Từ bổ nghĩa cho head

### 2.2. Các nhãn quan hệ phổ biến
| Nhãn | Ý nghĩa | Ví dụ |
|------|---------|-------|
| nsubj | Chủ ngữ | "The **cat** sleeps" |
| dobj | Tân ngữ trực tiếp | "I eat **apple**" |
| amod | Tính từ bổ nghĩa | "**big** cat" |
| det | Mạo từ | "**the** cat" |
| ROOT | Gốc của câu | Thường là động từ chính |
| prep | Giới từ | "on **the** table" |
| pobj | Tân ngữ của giới từ | "on the **table**" |

### 2.3. spaCy
Thư viện NLP công nghiệp với:
- Pipeline xử lý ngôn ngữ hoàn chỉnh
- Mô hình pre-trained cho nhiều ngôn ngữ
- Công cụ trực quan hóa displaCy

---

## 3. Cài đặt

### 3.1. Cài đặt thư viện
```bash
pip install -U spacy
python -m spacy download en_core_web_md
```

### 3.2. Các thuộc tính Token quan trọng
- `token.text`: Văn bản của token
- `token.dep_`: Nhãn quan hệ phụ thuộc
- `token.head`: Token head (điều khiển)
- `token.children`: Iterator các token con
- `token.pos_`: Part-of-Speech tag

---

## 4. Kết quả

### 4.1. Phân tích câu cơ bản
**Câu**: "The quick brown fox jumps over the lazy dog." (10 tokens)

| TEXT | DEP | HEAD | POS | Giải thích |
|------|-----|------|-----|------------|
| The | det | fox | DET | Mạo từ xác định cho "fox" |
| quick | amod | fox | ADJ | Tính từ bổ nghĩa cho "fox" |
| brown | amod | fox | ADJ | Tính từ bổ nghĩa cho "fox" |
| fox | nsubj | jumps | NOUN | Chủ ngữ của động từ "jumps" |
| jumps | ROOT | jumps | VERB | **Gốc của câu** (động từ chính) |
| over | prep | jumps | ADP | Giới từ bổ nghĩa cho "jumps" |
| the | det | dog | DET | Mạo từ xác định cho "dog" |
| lazy | amod | dog | ADJ | Tính từ bổ nghĩa cho "dog" |
| dog | pobj | over | NOUN | Tân ngữ của giới từ "over" |
| . | punct | jumps | PUNCT | Dấu câu |

**Phân tích cấu trúc:**
- **ROOT**: "jumps" là động từ chính, điều khiển toàn bộ câu
- **Cụm chủ ngữ**: "The quick brown fox" → "fox" là head, các từ khác là dependents
- **Cụm giới từ**: "over the lazy dog" → "over" nối với "jumps", "dog" là tân ngữ của "over"

### 4.2. Trích xuất bộ ba (Subject, Verb, Object)
**Câu**: "The cat chased the mouse and the dog watched them."

**Kết quả:**
| Subject | Verb | Object |
|---------|------|--------|
| cat | chased | mouse |
| dog | watched | them |

**Phân tích**: Câu có 2 mệnh đề nối bằng "and", mỗi mệnh đề có cấu trúc S-V-O riêng

### 4.3. Tìm tính từ bổ nghĩa
**Câu**: "The big, fluffy white cat is sleeping on the warm mat."

| Danh từ | Tính từ bổ nghĩa (amod) |
|---------|-------------------------|
| cat | ['big', 'fluffy', 'white'] |
| mat | ['warm'] |

**Nhận xét**: spaCy nhận diện đúng tất cả tính từ bổ nghĩa, kể cả khi có dấu phẩy ngăn cách

### 4.4. Bài tập tự luyện

**Bài 1 - Tìm động từ chính:**
| Câu | Động từ chính | POS |
|-----|---------------|-----|
| "The cat is sleeping on the mat." | sleeping | VERB |
| "She quickly finished her homework." | finished | VERB |
| "The students are studying for the exam." | studying | VERB |

**Bài 2 - Trích xuất cụm danh từ:**
**Câu**: "The big brown dog chased the small white cat."

| Phương pháp | Kết quả |
|-------------|---------|
| Tự implement | "The big brown dog", "the small white cat" |
| spaCy noun_chunks | "The big brown dog", "the small white cat" |

**Bài 3 - Đường đi đến ROOT:**
**Câu**: "The quick brown fox jumps over the lazy dog."

| Từ | Đường đi đến ROOT |
|----|-------------------|
| quick | quick(amod) → fox(nsubj) → jumps(ROOT) |
| lazy | lazy(amod) → dog(pobj) → over(prep) → jumps(ROOT) |
| dog | dog(pobj) → over(prep) → jumps(ROOT) |

**Nhận xét**: Độ sâu của cây phụ thuộc phản ánh độ phức tạp cú pháp - "lazy" cách ROOT 3 bước trong khi "quick" chỉ cách 2 bước

---

## 5. Nhận xét

**Ưu điểm:**
- spaCy cung cấp API đơn giản, dễ sử dụng
- Mô hình pre-trained cho kết quả tốt
- displaCy giúp trực quan hóa dễ dàng

**Ứng dụng:**
- Trích xuất thông tin (Information Extraction)
- Phân tích cảm xúc (Sentiment Analysis)
- Hỏi đáp tự động (Question Answering)
- Tóm tắt văn bản (Text Summarization)

---

## 6. Trích dẫn
- spaCy Documentation: https://spacy.io/
- displaCy Visualizer: https://explosion.ai/demos/displacy
- Universal Dependencies: https://universaldependencies.org/
