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

### 2.1. Tại sao cần Sequential Models?

#### 2.1.1. Hạn chế của các mô hình truyền thống

**Bag-of-Words (TF-IDF):**
- Bỏ qua hoàn toàn thứ tự từ
- Coi câu như "túi" chứa các từ độc lập
- Không phân biệt được ngữ cảnh

**Word2Vec trung bình:**
- Mất thông tin vị trí khi lấy mean pooling
- Không capture được cấu trúc câu

**Ví dụ minh họa vấn đề:**
```
Câu 1: "Sản phẩm này chất lượng tốt, không hề tệ chút nào."  → Positive
Câu 2: "Sản phẩm này chất lượng không hề tốt, rất tệ."      → Negative

BoW vectors gần như giống nhau! (cùng các từ: tốt, tệ, không, chất lượng...)
```

#### 2.1.2. Dữ liệu tuần tự (Sequential Data)
Nhiều loại dữ liệu có tính tuần tự, thứ tự quan trọng:
- **Text**: Thứ tự từ quyết định nghĩa
- **Time series**: Giá cổ phiếu, nhiệt độ
- **Audio**: Tín hiệu âm thanh
- **Video**: Chuỗi frames

### 2.2. RNN (Recurrent Neural Network) - Chi tiết

#### 2.2.1. Kiến trúc cơ bản

```
        ┌─────────────────────────────────────────────┐
        │                                             │
        ▼                                             │
   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
   │  RNN    │───▶│  RNN    │───▶│  RNN    │───▶│  RNN    │
   │  Cell   │    │  Cell   │    │  Cell   │    │  Cell   │
   └─────────┘    └─────────┘    └─────────┘    └─────────┘
        ▲              ▲              ▲              ▲
        │              │              │              │
       x₁             x₂             x₃             x₄
      "The"         "cat"          "sat"          "down"
```

#### 2.2.2. Công thức Forward Pass

Tại mỗi time step t:

```
hₜ = tanh(Wₓₕ × xₜ + Wₕₕ × hₜ₋₁ + bₕ)
yₜ = Wₕᵧ × hₜ + bᵧ
```

Trong đó:
- `xₜ ∈ ℝᵈ`: Input tại time step t (word embedding)
- `hₜ ∈ ℝʰ`: Hidden state tại time step t
- `hₜ₋₁ ∈ ℝʰ`: Hidden state từ time step trước
- `Wₓₕ ∈ ℝʰˣᵈ`: Weight matrix input → hidden
- `Wₕₕ ∈ ℝʰˣʰ`: Weight matrix hidden → hidden (recurrent)
- `Wₕᵧ ∈ ℝᵒˣʰ`: Weight matrix hidden → output
- `tanh`: Activation function, output ∈ (-1, 1)

#### 2.2.3. Unrolling RNN qua thời gian

```
h₀ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → [RNN] → h₄
       ↑           ↑           ↑           ↑
       x₁          x₂          x₃          x₄
```

**Quan trọng:** Cùng một bộ weights (Wₓₕ, Wₕₕ, Wₕᵧ) được chia sẻ qua tất cả time steps!

#### 2.2.4. Backpropagation Through Time (BPTT)

Gradient được tính ngược qua thời gian:

```
∂L/∂W = Σₜ ∂Lₜ/∂W

∂Lₜ/∂Wₕₕ = Σₖ₌₁ᵗ (∂Lₜ/∂hₜ × ∂hₜ/∂hₖ × ∂hₖ/∂Wₕₕ)
```

**Vấn đề:** `∂hₜ/∂hₖ` là tích của nhiều Jacobians → gradient có thể explode hoặc vanish!

### 2.3. Vanishing/Exploding Gradient Problem

#### 2.3.1. Giải thích toán học

```
∂hₜ/∂hₖ = Πᵢ₌ₖᵗ⁻¹ ∂hᵢ₊₁/∂hᵢ = Πᵢ₌ₖᵗ⁻¹ Wₕₕᵀ × diag(tanh'(hᵢ))
```

Với chuỗi dài (t-k lớn):
- Nếu `||Wₕₕ|| < 1`: Gradient → 0 (vanishing)
- Nếu `||Wₕₕ|| > 1`: Gradient → ∞ (exploding)

#### 2.3.2. Hậu quả

**Vanishing Gradient:**
- Model không học được long-term dependencies
- Chỉ "nhớ" được vài tokens gần nhất
- Ví dụ: "The cat, which was sitting on the mat, **was** sleeping" - khó liên kết "cat" với "was"

**Exploding Gradient:**
- Weights update quá lớn, model không hội tụ
- Loss = NaN

#### 2.3.3. Giải pháp

| Giải pháp | Mô tả |
|-----------|-------|
| Gradient Clipping | Giới hạn norm của gradient |
| LSTM/GRU | Kiến trúc với gating mechanism |
| Skip connections | Residual connections |
| Proper initialization | Xavier/He initialization |

### 2.4. LSTM (Long Short-Term Memory) - Chi tiết

#### 2.4.1. Ý tưởng chính
LSTM (Hochreiter & Schmidhuber, 1997) giải quyết vanishing gradient bằng:
- **Cell state (Cₜ)**: "Highway" cho thông tin chảy qua không bị biến đổi nhiều
- **Gates**: Kiểm soát thông tin nào được thêm/xóa/output

#### 2.4.2. Kiến trúc LSTM Cell

```
                    ┌───────────────────────────────────────┐
                    │              Cell State Cₜ            │
    Cₜ₋₁ ──────────▶│ ──────×────────────+────────────────▶ │ ──────▶ Cₜ
                    │       │            │                  │
                    │    ┌──┴──┐      ┌──┴──┐               │
                    │    │  fₜ │      │  iₜ │               │
                    │    │Forget│      │Input│               │
                    │    │ Gate│      │Gate │               │
                    │    └──┬──┘      └──┬──┘               │
                    │       │            │                  │
    hₜ₋₁ ──────────▶│ ──────┴────────────┴──────────────── │
                    │                    │                  │
         xₜ ───────▶│                 ┌──┴──┐    ┌──┴──┐   │
                    │                 │  C̃ₜ │    │  oₜ │   │
                    │                 │Candi│    │Output│   │
                    │                 │date │    │ Gate│   │
                    │                 └──┬──┘    └──┬──┘   │
                    │                    │          │      │
                    │                    └────×─────┘      │
                    │                         │            │
                    │                        tanh          │
                    │                         │            │
                    └─────────────────────────┼────────────┘
                                              ▼
                                             hₜ
```

#### 2.4.3. Công thức LSTM

**Forget Gate** - Quyết định thông tin cần quên từ cell state:
```
fₜ = σ(Wf × [hₜ₋₁, xₜ] + bf)
```

**Input Gate** - Quyết định thông tin mới cần lưu:
```
iₜ = σ(Wᵢ × [hₜ₋₁, xₜ] + bᵢ)
C̃ₜ = tanh(Wc × [hₜ₋₁, xₜ] + bc)
```

**Cell State Update:**
```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
```

**Output Gate** - Quyết định output từ cell state:
```
oₜ = σ(Wₒ × [hₜ₋₁, xₜ] + bₒ)
hₜ = oₜ ⊙ tanh(Cₜ)
```

Trong đó:
- `σ`: Sigmoid function (output ∈ (0,1) - như "cổng")
- `⊙`: Element-wise multiplication
- `[hₜ₋₁, xₜ]`: Concatenation của hidden state và input

#### 2.4.4. Tại sao LSTM giải quyết Vanishing Gradient?

**Cell state highway:**
```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
```

- Khi `fₜ ≈ 1` và `iₜ ≈ 0`: `Cₜ ≈ Cₜ₋₁` (thông tin được giữ nguyên)
- Gradient có thể chảy qua cell state mà không bị nhân với weight matrix
- Tránh được tích của nhiều số < 1

#### 2.4.5. Ví dụ hoạt động của Gates

```
Câu: "The cat, which was very cute, sat on the mat."

Khi xử lý "sat":
- Forget gate: Quên thông tin về "cute" (không liên quan đến hành động)
- Input gate: Lưu thông tin "sat" là hành động chính
- Output gate: Output hidden state để dự đoán từ tiếp theo

Khi xử lý "mat":
- Cell state vẫn giữ thông tin "cat" là subject (từ đầu câu)
- Có thể liên kết "cat" với "sat" dù cách xa nhau
```

### 2.5. GRU (Gated Recurrent Unit)

#### 2.5.1. Kiến trúc đơn giản hơn LSTM
GRU (Cho et al., 2014) gộp forget và input gate thành **update gate**:

```
zₜ = σ(Wz × [hₜ₋₁, xₜ])           # Update gate
rₜ = σ(Wr × [hₜ₋₁, xₜ])           # Reset gate
h̃ₜ = tanh(W × [rₜ ⊙ hₜ₋₁, xₜ])   # Candidate
hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ   # Final hidden state
```

#### 2.5.2. So sánh LSTM vs GRU

| Tiêu chí | LSTM | GRU |
|----------|------|-----|
| Số gates | 3 (forget, input, output) | 2 (update, reset) |
| Parameters | Nhiều hơn | Ít hơn (~25%) |
| Training | Chậm hơn | Nhanh hơn |
| Long sequences | Tốt hơn | Tương đương |
| Small data | Kém hơn | Tốt hơn |

### 2.6. Bidirectional RNN

#### 2.6.1. Motivation
RNN thông thường chỉ nhìn context từ trái sang phải. Nhưng nhiều tasks cần context cả hai chiều:

```
"I went to the bank to deposit money."  → bank = ngân hàng
"I went to the bank to catch fish."     → bank = bờ sông

Cần nhìn cả "deposit money" và "catch fish" để hiểu "bank"
```

#### 2.6.2. Kiến trúc

```
Forward:   h₁→ → h₂→ → h₃→ → h₄→
              ↘    ↘    ↘    ↘
Output:        y₁    y₂    y₃    y₄
              ↗    ↗    ↗    ↗
Backward:  h₁← ← h₂← ← h₃← ← h₄←
```

**Concatenation:**
```
hₜ = [h→ₜ ; h←ₜ]  (dimension = 2 × hidden_size)
```

#### 2.6.3. Khi nào dùng Bidirectional?

| Task | Bidirectional? | Lý do |
|------|----------------|-------|
| Text Classification | Có thể | Cần hiểu toàn bộ câu |
| POS Tagging | Nên dùng | Cần context cả hai chiều |
| NER | Nên dùng | Entity phụ thuộc context |
| Language Modeling | Không | Chỉ có past context |
| Machine Translation | Encoder: Có, Decoder: Không | |

### 2.7. Token Classification Tasks

#### 2.7.1. POS Tagging (Part-of-Speech)

Gán nhãn từ loại cho mỗi token:

```
Input:  "The   cat   sat   on   the   mat"
Output: "DET   NOUN  VERB  ADP  DET   NOUN"
```

**Universal POS Tags (UPOS):**
| Tag | Meaning | Example |
|-----|---------|---------|
| NOUN | Noun | cat, dog, house |
| VERB | Verb | run, eat, is |
| ADJ | Adjective | big, red, happy |
| ADV | Adverb | quickly, very |
| DET | Determiner | the, a, this |
| ADP | Adposition | in, on, at |
| PRON | Pronoun | I, you, he |
| PROPN | Proper noun | John, Paris |

#### 2.7.2. NER (Named Entity Recognition)

Nhận dạng và phân loại entities:

```
Input:  "John   lives   in   New   York"
Output: "B-PER  O       O    B-LOC I-LOC"
```

**IOB Format (Inside-Outside-Beginning):**
- `B-XXX`: Beginning of entity type XXX
- `I-XXX`: Inside (continuation) of entity type XXX
- `O`: Outside any entity

**Tại sao cần IOB?**
```
"New York City is in New York State"

Không có IOB: "LOC LOC LOC O O LOC LOC LOC" - không biết đâu là entity riêng
Có IOB: "B-LOC I-LOC I-LOC O O B-LOC I-LOC I-LOC" - rõ ràng 2 entities
```

#### 2.7.3. Kiến trúc cho Token Classification

```
Input tokens:  [x₁,    x₂,    x₃,    x₄]
                ↓      ↓      ↓      ↓
Embedding:     [e₁,    e₂,    e₃,    e₄]
                ↓      ↓      ↓      ↓
Bi-LSTM:       [h₁,    h₂,    h₃,    h₄]
                ↓      ↓      ↓      ↓
Linear:        [o₁,    o₂,    o₃,    o₄]
                ↓      ↓      ↓      ↓
Softmax:       [ŷ₁,    ŷ₂,    ŷ₃,    ŷ₄]
```

**Loss function:** Cross-entropy trên mỗi token
```
L = -1/T Σₜ Σₖ yₜₖ log(ŷₜₖ)
```

### 2.8. Padding và Batching cho Sequences

#### 2.8.1. Vấn đề
Các sequences có độ dài khác nhau, nhưng batch cần cùng shape:

```
Batch:
- "I love NLP"        (3 tokens)
- "Deep learning"     (2 tokens)
- "Natural language"  (2 tokens)
```

#### 2.8.2. Padding
Thêm token đặc biệt `<PAD>` để cùng độ dài:

```
- "I love NLP <PAD>"     (4 tokens)
- "Deep learning <PAD> <PAD>"  (4 tokens)
- "Natural language <PAD> <PAD>"  (4 tokens)
```

#### 2.8.3. Attention Mask
Đánh dấu tokens thực vs padding:

```
Attention mask:
- [1, 1, 1, 0]
- [1, 1, 0, 0]
- [1, 1, 0, 0]
```

**Trong loss calculation:**
```python
loss = CrossEntropyLoss(ignore_index=PAD_IDX)
# Không tính loss trên padding tokens
```

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
| Thứ tự từ | Bỏ qua | Xử lý tuần tự |
| Ngữ cảnh | Không | Hidden state |
| Phủ định | Khó xử lý | Hiểu được |
| Tốc độ | Nhanh | Chậm hơn |
| Dữ liệu | Ít | Cần nhiều |

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
