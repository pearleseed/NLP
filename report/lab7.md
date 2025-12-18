# Báo cáo Lab 7: Dependency Parsing

## 1. Mục tiêu

Thực hành **Phân tích cú pháp phụ thuộc (Dependency Parsing)** - một kỹ thuật nền tảng để máy tính hiểu cấu trúc ngữ pháp và mối quan hệ giữa các từ trong câu. Lab này sử dụng thư viện **spaCy**, một công cụ NLP mạnh mẽ trong công nghiệp, để xây dựng cây cú pháp và trích xuất thông tin.

**Các nội dung chính:**
1.  Hiểu bản chất của Dependency Parsing so với Constituency Parsing.
2.  Làm quen với hệ thống nhãn quan hệ (Dependency Labels).
3.  Vận dụng cấu trúc cây (Tree) để trích xuất thông tin (Information Extraction).

---

## 2. Nền tảng Lý thuyết

### 2.1. Dependency Parsing vs. Constituency Parsing
Có 2 cách tiếp cận chính để phân tích cú pháp:

*   **Constituency Parsing**: Chia câu thành các cụm từ (phrase) - NP, VP. Tập trung vào cấu trúc hình thức.
*   **Dependency Parsing**: Tập trung vào mối quan hệ **Head-Dependent** giữa các từ. Không có node ảo, mọi node đều là từ thực. Đây là phương pháp tối ưu cho việc trích xuất thông tin (Information Extraction) vì nó chỉ ra trực tiếp "ai làm gì cho ai".

### 2.2. Hệ thống Nhãn Quan hệ (Universal Dependencies)
Các nhãn phổ biến trong spaCy:

| Nhãn | Giải thích | Ví dụ (Dependent -> Head) |
| :--- | :--- | :--- |
| **ROOT** | Gốc của câu | Main verb |
| **nsubj** | Chủ ngữ | **He** runs |
| **dobj** | Tân ngữ trực tiếp | Eat **cake** |
| **amod** | Tính từ bổ nghĩa | **Red** apple |
| **prep** | Giới từ | Sit **on** chair |
| **pobj** | Tân ngữ của giới từ | on **chair** |

### 2.3. Cơ chế Transition-based Parsing
spaCy sử dụng State Machine để xây dựng cây từ trái sang phải:
-   **Buffer**: Chứa từ chưa xử lý.
-   **Stack**: Chứa từ đang xử lý.
-   **Action**: SHIFT (đẩy từ), LEFT-ARC (tạo cung trái), RIGHT-ARC (tạo cung phải).

---

## 3. Dataset

Lab này sử dụng các mẫu câu tiếng Anh tiêu chuẩn để kiểm thử khả năng phân tích của mô hình:
-   Câu đơn: "The quick brown fox jumps over the lazy dog."
-   Câu ghép: "The cat chased the mouse and the dog watched them."
-   Câu có cấu trúc lồng ghép: "The big, fluffy white cat is sleeping on the warm mat."

Mô hình được sử dụng là **`en_core_web_sm`** (hoặc `md`), được huấn luyện trên tập dữ liệu OntoNotes (bao gồm tin tức, blog, hội thoại...).

---

## 4. Cài đặt

### 4.1. Đối tượng Token trong spaCy
Khi xử lý văn bản bằng `nlp(text)`, mỗi từ trở thành một object `Token`. Các thuộc tính quan trọng:
*   `token.text`: Từ gốc.
*   `token.dep_`: Nhãn quan hệ (VD: "nsubj").
*   `token.head`: Token cha (Head). Dùng để duyệt lên.
*   `token.children`: Danh sách token con. Dùng để duyệt xuống.

### 4.2. Kỹ thuật Duyệt cây (Tree Traversal)
Để trích xuất thông tin, ta thường duyệt từ một node (thường là danh từ hoặc động từ) sang các node lân cận.

**Ví dụ: Tìm tính từ bổ nghĩa cho danh từ**
```python
def get_amods(noun_token):
    return [child.text for child in noun_token.children if child.dep_ == "amod"]
```
Logic: Từ token danh từ -> duyệt tất cả con -> lọc con có nhãn "amod".

---

## 5. Kết quả

### 5.1. Phân tích câu phức
**Input**: "The quick brown fox jumps over the lazy dog."

**Cấu trúc cây thu được:**
*   **ROOT**: `jumps`
*   **Chủ ngữ**: `fox` (được bổ nghĩa bởi `The`, `quick`, `brown`)
*   **Giới từ**: `over`
*   **Tân ngữ giới từ**: `dog` (được bổ nghĩa bởi `the`, `lazy`)

### 5.2. Trích xuất quan hệ S-V-O
Mô hình tách thành công 2 sự kiện từ câu ghép:
1.  **Sự kiện 1**: Cat (S) -> Chased (V) -> Mouse (O)
2.  **Sự kiện 2**: Dog (S) -> Watched (V) -> Them (O)

Mô hình xử lý tốt từ nối "and" và không nhầm lẫn chủ ngữ của 2 động từ.

---

## 6. Nhận xét

### 6.1. Ưu điểm của spaCy
-   **Tốc độ cao**: Được viết bằng Cython, rất nhanh so với NLTK hay Stanford CoreNLP.
-   **Dễ sử dụng**: API `token.head`, `token.children` rất trực quan để duyệt cây.
-   **Độ chính xác tốt**: Với các văn bản tiếng Anh chuẩn `en_core_web_sm` hoạt động rất ổn định.

### 6.2. Câu hỏi thường gặp
**Q: Làm sao để biết một từ là ROOT?**
-   Kiểm tra `token.dep_ == "ROOT"` hoặc `token.head == token`.

**Q: Tại sao độ sâu của cây lại quan trọng?**
-   Độ sâu phản ánh độ phức tạp của câu. Cây càng sâu, quan hệ phụ thuộc càng chồng chéo (lồng ghép giới từ, mệnh đề quan hệ), gây khó khăn cho việc hiểu nghĩa.

---

## 7. Khó khăn & Giải pháp

| Vấn đề | Giải pháp |
| :--- | :--- |
| **Lỗi nhập nhằng (Ambiguity)** | Ví dụ: "I saw the man with a telescope". spaCy có thể gắn "telescope" vào "saw" hoặc "man". Giải pháp: Dùng mô hình lớn hơn (`en_core_web_trf`) để cải thiện ngữ cảnh. |
| **Cài đặt model** | Lỗi `OSError: [E050] Can't find model`. Giải pháp: Chạy lệnh `python -m spacy download en_core_web_sm` trong terminal. |
| **Tùy biến nhãn** | spaCy dùng nhãn chuẩn UD. Nếu muốn nhãn riêng, cần train lại model (phức tạp). |

---

## 8. Hướng dẫn Chạy Code

```bash
# 1. Cài đặt thư viện và tải model
pip install -U spacy
python -m spacy download en_core_web_sm

# 2. Chạy notebook
jupyter notebook notebook/lab7.ipynb
```

---

## 9. Trích dẫn
-   **spaCy Documentation**: https://spacy.io/usage
-   **Universal Dependencies**: https://universaldependencies.org/
-   **Nivre, J.**, "Transition-Based Dependency Parsing", 2008.
