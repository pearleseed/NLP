# Báo cáo Lab 1: Lexical Analysis

## 1. Mục tiêu

Làm quen với các kỹ thuật cơ bản trong **Lexical Analysis** (Phân tích từ vựng) - bước đầu tiên trong pipeline NLP.

**Các task thực hiện:**
1. Text Segmentation (Tokenization): Tách văn bản thành tokens
2. Chunk Extraction: Trích xuất cụm từ (Named Entities)
3. Token Classification: Trích xuất từ viết hoa và số

---

## 2. Nền tảng Lý thuyết

### 2.1. Lexical Analysis trong NLP Pipeline

Lexical Analysis là bước đầu tiên và quan trọng nhất trong pipeline xử lý ngôn ngữ tự nhiên. Mục tiêu là chuyển đổi chuỗi ký tự thô (raw text) thành các đơn vị có ý nghĩa (tokens) để các bước xử lý tiếp theo có thể làm việc.

```
Raw Text → Lexical Analysis → Tokens → Syntactic Analysis → Semantic Analysis → ...
```

**Tại sao Lexical Analysis quan trọng?**
- Máy tính không hiểu "văn bản" như con người, chỉ thấy chuỗi bytes
- Cần xác định ranh giới giữa các đơn vị ngôn ngữ (từ, câu, đoạn)
- Chất lượng tokenization ảnh hưởng trực tiếp đến các bước sau

### 2.2. Tokenization - Các phương pháp chính

#### 2.2.1. Word-level Tokenization
Tách văn bản theo từ, thường dựa vào khoảng trắng và dấu câu.

| Phương pháp | Mô tả | Ưu điểm | Nhược điểm |
|-------------|-------|---------|------------|
| Whitespace | Tách theo khoảng trắng | Đơn giản, nhanh | Không xử lý dấu câu |
| Rule-based | Dùng regex patterns | Linh hoạt, kiểm soát được | Khó bảo trì |
| Statistical | Dùng ML để học ranh giới | Tự động, chính xác | Cần dữ liệu training |

**Ví dụ minh họa:**
```
Input: "I can't believe it's 2024!"

Whitespace:  ["I", "can't", "believe", "it's", "2024!"]
Rule-based:  ["I", "ca", "n't", "believe", "it", "'s", "2024", "!"]
```

#### 2.2.2. Subword Tokenization
Tách từ thành các đơn vị nhỏ hơn, giải quyết vấn đề OOV (Out-of-Vocabulary).

| Thuật toán | Mô tả | Sử dụng bởi |
|------------|-------|-------------|
| BPE (Byte Pair Encoding) | Merge cặp ký tự phổ biến nhất | GPT-2, RoBERTa |
| WordPiece | Tương tự BPE, dùng likelihood | BERT, DistilBERT |
| Unigram | Xác suất unigram, loại bỏ token ít dùng | SentencePiece, T5 |

**Ví dụ BPE:**
```
Vocabulary ban đầu: ['l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'i', 'd']
Sau merge: ['low', 'er', 'new', 'est', 'wid', 'est']

"lowest" → ["low", "est"]
"newest" → ["new", "est"]
```

#### 2.2.3. Character-level Tokenization
Tách thành từng ký tự, vocabulary nhỏ nhưng sequence dài.

```
"Hello" → ['H', 'e', 'l', 'l', 'o']
```

### 2.3. Regular Expressions - Nền tảng toán học

#### 2.3.1. Finite State Automata (FSA)
Regex được triển khai dựa trên lý thuyết **Finite State Automata** - máy trạng thái hữu hạn.

**Định nghĩa hình thức:**
Một FSA là bộ 5 thành phần: `M = (Q, Σ, δ, q₀, F)` trong đó:
- `Q`: Tập hữu hạn các trạng thái
- `Σ`: Bảng chữ cái (alphabet)
- `δ`: Hàm chuyển trạng thái `δ: Q × Σ → Q`
- `q₀`: Trạng thái khởi đầu
- `F`: Tập các trạng thái kết thúc (accepting states)

**Ví dụ FSA cho pattern `\d+` (một hoặc nhiều chữ số):**
```
        ┌─────┐
        │ 0-9 │
        ▼     │
  ──► (q0) ──────► ((q1)) ◄──┐
       start    0-9    accept │
                              │
                         0-9  │
                              │
                        ──────┘
```

#### 2.3.2. Các Regex Patterns quan trọng trong NLP

| Pattern | Mô tả | Ví dụ match |
|---------|-------|-------------|
| `\b\w+\b` | Từ hoàn chỉnh (word boundary) | "hello", "world" |
| `\w+(?:'\w+)?` | Từ có thể có sở hữu cách | "it's", "don't" |
| `[A-Z][a-z]+` | Từ viết hoa đầu | "Hello", "World" |
| `\d{1,3}(?:,\d{3})*` | Số có dấu phẩy ngăn cách | "1,000", "1,000,000" |
| `[A-Z]{2,}` | Viết tắt (acronyms) | "NASA", "FBI" |

#### 2.3.3. Greedy vs Non-greedy Matching
```
Text: "The <b>bold</b> and <i>italic</i> text"

Greedy:     <.*>   → "<b>bold</b> and <i>italic</i>"  (match dài nhất)
Non-greedy: <.*?>  → "<b>", "</b>", "<i>", "</i>"     (match ngắn nhất)
```

### 2.4. Named Entity Recognition (NER) cơ bản

#### 2.4.1. Định nghĩa
NER là task nhận dạng và phân loại các thực thể có tên trong văn bản:
- **PER** (Person): Tên người
- **LOC** (Location): Địa danh
- **ORG** (Organization): Tổ chức
- **DATE/TIME**: Ngày tháng, thời gian
- **MONEY**: Tiền tệ

#### 2.4.2. Phương pháp Rule-based cho NER
Sử dụng các heuristics và patterns:

| Rule | Pattern | Ví dụ |
|------|---------|-------|
| Capitalization | `[A-Z][a-z]+(\s[A-Z][a-z]+)*` | "John Smith", "New York" |
| Title + Name | `(Mr\.|Mrs\.|Dr\.)\s[A-Z]\w+` | "Dr. Smith" |
| Organization suffix | `\w+\s(Inc\.|Corp\.|Ltd\.)` | "Apple Inc." |

**Hạn chế của Rule-based NER:**
- Không xử lý được ambiguity: "Apple" (công ty vs trái cây)
- Cần rules riêng cho từng ngôn ngữ
- Khó mở rộng cho domain mới

### 2.5. Challenges trong Tokenization

#### 2.5.1. Xử lý đa ngôn ngữ
| Ngôn ngữ | Thách thức | Giải pháp |
|----------|------------|-----------|
| Tiếng Việt | Từ ghép không có dấu cách ("học sinh") | Word segmentation tools (VnCoreNLP) |
| Tiếng Trung | Không có khoảng trắng | Character-based hoặc Jieba |
| Tiếng Nhật | 3 hệ chữ viết | MeCab tokenizer |
| Tiếng Đức | Từ ghép dài ("Donaudampfschifffahrt") | Compound splitting |

#### 2.5.2. Xử lý các trường hợp đặc biệt
```
URLs:        https://example.com/path?query=1
Emails:      user@domain.com
Hashtags:    #MachineLearning
Mentions:    @username
Emoticons:   :) :-( ^_^
Contractions: I'm, don't, it's
Hyphenated:  state-of-the-art, well-known
```

### 2.6. Đánh giá Tokenization

Không có "ground truth" tuyệt đối cho tokenization, nhưng có thể đánh giá qua:
- **Downstream task performance**: Tokenization nào giúp model đạt kết quả tốt hơn
- **Vocabulary size**: Cân bằng giữa coverage và efficiency
- **OOV rate**: Tỷ lệ từ không có trong vocabulary

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
