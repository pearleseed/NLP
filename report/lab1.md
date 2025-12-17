# Báo cáo Lab 1: Lexical Analysis

## 1. Mục tiêu

Làm quen với các kỹ thuật cơ bản trong **Lexical Analysis** (Phân tích từ vựng) - bước đầu tiên trong pipeline NLP.

**Các task thực hiện:**
1. Text Segmentation (Tokenization): Tách văn bản thành tokens
2. Chunk Extraction: Trích xuất cụm từ (Named Entities)
3. Token Classification: Trích xuất từ viết hoa và số

---

## 2. Nền tảng Lý thuyết

### 2.1. Tokenization
Quá trình tách văn bản thành các đơn vị nhỏ hơn (tokens): từ, dấu câu, số...

### 2.2. Regular Expressions
Công cụ mạnh mẽ để pattern matching trong text processing:
- `\w+`: Chuỗi ký tự chữ/số
- `[^\w\s]`: Dấu câu
- `[A-Z]\w+`: Từ viết hoa

---

## 3. Cài đặt

### 3.1. Exercise 1: Tokenization
**Pattern**: `r"\w+|[^\w\s]"`
- `\w+`: Bắt từ
- `[^\w\s]`: Bắt dấu câu

### 3.2. Exercise 2: Chunk Extraction
**Pattern**: `r"[A-Z]\w+(?:\s[A-Z]\w+)*"`
- Bắt chuỗi từ viết hoa liên tiếp (Named Entities)

### 3.3. Exercise 3: Information Extraction
**Pattern số**: `r"\d+(?:[.,]\d+)*%?"`
- Xử lý số nguyên, thập phân, phần trăm

**Pattern từ viết hoa**: `r"[A-ZÀ-Ỹ]\w*"`
- Hỗ trợ Unicode tiếng Việt

---

## 4. Kết quả

### 4.1. Exercise 1: Tokenization
**Input**: `"Nepal's Home Minister Bhim Rawal on Sunday held a meeting with CPN-UML leaders..."`

**Output**: 
```
['Nepal's', 'Home', 'Minister', 'Bhim', 'Rawal', 'on', 'Sunday', 'held', 'a', 
'meeting', 'with', 'CPN', '-', 'UML', 'leaders', ',', 'including', 'Madhav', 
'Kumar', 'Nepal', ',', 'Jhalanath', 'Khanal', 'and', 'Bam', 'Dev', 'Gautam', 
',', 'at', 'the', 'party', 'headquarters', 'in', 'Balkhu', '.', ...]
```

**Phân tích kết quả:**
- Pattern `\w+'\w+` xử lý tốt sở hữu cách: "Nepal's", "Ministry's" được giữ nguyên
- Dấu gạch ngang trong "CPN-UML" được tách riêng thành 3 token: `['CPN', '-', 'UML']`
- Dấu câu (`,`, `.`) được tách riêng khỏi từ

### 4.2. Exercise 2: Chunk Extraction
**Output**: 
```
['Nepal's Home Minister Bhim Rawal', 'Sunday', 'CPN', 'UML', 
'Madhav Kumar Nepal', 'Jhalanath Khanal', 'Bam Dev Gautam', 
'Balkhu', 'The', 'Home Ministry's']
```

**Phân tích kết quả:**
- Trích xuất thành công các Named Entities: tên người (Bhim Rawal, Madhav Kumar Nepal), địa danh (Balkhu)
- Các cụm từ viết hoa liên tiếp được nhóm lại: "Nepal's Home Minister Bhim Rawal"
- Một số từ đơn lẻ như "The", "Sunday" cũng được bắt do bắt đầu bằng chữ hoa

### 4.3. Exercise 3: Information Extraction

**Text tiếng Anh**: `"The police are firing 50 rounds in the air. 84.477 people were affected."`

| Loại | Kết quả | Giải thích |
|------|---------|------------|
| Số | `['50', '84.477']` | Bắt được cả số nguyên và số thập phân với dấu `.` |
| Từ viết hoa | `['The']` | Chỉ có "The" bắt đầu bằng chữ hoa |

**Text tiếng Việt**: `"Việt Nam có dân số 99,6 triệu người. Tăng trưởng đạt 2,21%."`

| Loại | Kết quả | Giải thích |
|------|---------|------------|
| Số | `['99,6', '2,21']` | Xử lý đúng số thập phân kiểu Việt Nam (dùng dấu `,`) |
| Từ viết hoa | `['Việt', 'Nam', 'Tăng']` | Pattern Unicode bắt được ký tự có dấu |

**Điểm đáng chú ý:**
- Regex cần điều chỉnh cho từng ngôn ngữ (dấu `.` vs `,` cho số thập phân)
- Tiếng Việt cần pattern Unicode để bắt ký tự có dấu: `[A-ZĐ][a-zà-ỹ]+`

---

## 5. Nhận xét

**Ưu điểm Regex:**
- Nhanh, không cần model ML
- Phù hợp với pattern có cấu trúc rõ ràng

**Hạn chế:**
- Khó bảo trì khi pattern phức tạp
- Không xử lý được ngữ nghĩa mơ hồ

---

## 6. Trích dẫn
- Python `re` module: https://docs.python.org/3/library/re.html
- Regular Expression HOWTO: https://docs.python.org/3/howto/regex.html
