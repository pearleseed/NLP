# Báo cáo Lab X: Nghiên cứu Text-to-Speech (TTS)

## 1. Mục tiêu

Nghiên cứu tổng quan về **Text-to-Speech (TTS)** - công nghệ chuyển đổi văn bản thành giọng nói.

**Các nội dung thực hiện:**
1. Tìm hiểu tổng quan về bài toán TTS và tình hình nghiên cứu
2. Phân tích các phương pháp triển khai (3 Level)
3. Đánh giá ưu/nhược điểm từng hướng tiếp cận
4. Tìm hiểu các pipeline tối ưu hóa

---

## 2. Nền tảng Lý thuyết

### 2.1. Định nghĩa bài toán
Text-to-Speech (TTS) là quá trình sử dụng máy tính để chuyển đổi văn bản đầu vào thành âm thanh tiếng nói con người. Mục tiêu cuối cùng là tạo ra giọng nói tự nhiên, truyền cảm và khó phân biệt với giọng người thật.

### 2.2. Bối cảnh và Xu hướng nghiên cứu
Trong kỷ nguyên AI và các Agent tự động, TTS đóng vai trò là "cổng giao tiếp" đầu ra quan trọng.

**Xu hướng hiện tại:**
- Chuyển từ "đọc rõ chữ" sang "đọc diễn cảm" với cảm xúc
- Voice Cloning với dữ liệu mẫu cực ngắn (vài giây)

**Thách thức chung:**
| Thách thức | Mô tả |
|------------|-------|
| Tốc độ (Latency) | Phản hồi nhanh, Real-time |
| Tài nguyên | Chạy trên Edge devices, giảm chi phí server |
| Tính tự nhiên | Ngữ điệu (prosody), cảm xúc |
| Đa ngôn ngữ | Cross-lingual (nói tiếng nước ngoài bằng giọng bản xứ) |

### 2.3. Ba cấp độ phát triển TTS

| Level | Phương pháp | Đặc điểm chính |
|-------|-------------|----------------|
| 1 | Rule-based / Concatenative | Ghép nối âm vị, nhanh, ít tự nhiên |
| 2 | Deep Learning + Fine-tuning | Tự nhiên hơn, cần dữ liệu fine-tune |
| 3 | Few-shot / Zero-shot (Generative AI) | Clone giọng từ vài giây, tốn tài nguyên |

---

## 3. Phân tích các phương pháp triển khai

### 3.1. Level 1: TTS dựa trên luật (Rule-based / Concatenative)
Phương pháp truyền thống, ghép nối các đơn vị âm thanh nhỏ hoặc sử dụng luật vật lý.

| Ưu điểm | Nhược điểm |
|---------|------------|
| Hiệu suất cực cao, phản hồi tức thì | Thiếu tự nhiên, nghe như robot |
| Tốn ít tài nguyên, chạy offline | Khó tùy biến giọng |
| Dễ mở rộng đa ngôn ngữ | Ngắt nghỉ không đúng ngữ cảnh |
| Kiểm soát tốt, không hallucination | |

**Use Cases:** Thông báo công cộng (sân bay, nhà ga), thiết bị nhúng giá rẻ, Screen readers.

### 3.2. Level 2: Deep Learning + Fine-tuning (Neural TTS)
Mô hình học cách sinh ra phổ âm thanh (mel-spectrogram) từ văn bản (Tacotron, FastSpeech).

| Ưu điểm | Nhược điểm |
|---------|------------|
| Tính tự nhiên cao, mượt mà | Cần dữ liệu ghi âm sạch để fine-tune |
| Cá nhân hóa qua fine-tuning | Khó training đa ngôn ngữ |
| Tài nguyên vừa phải, phù hợp deploy | |

**Use Cases:** Trợ lý ảo (Siri, Google Assistant), Audiobooks, Call Center.

### 3.3. Level 3: Few-shot / Zero-shot (Generative AI)
Coi TTS như bài toán Language Modeling cho âm thanh (VALL-E, XTTS).

| Ưu điểm | Nhược điểm |
|---------|------------|
| Chỉ cần vài giây ghi âm | Tốn GPU mạnh, latency cao |
| Đa dạng cảm xúc & ngữ cảnh | Kém ổn định (lặp từ, bỏ từ) |
| Cross-lingual | Rủi ro đạo đức (Deepfake) |

**Use Cases:** Lồng tiếng phim, Content Creator, Podcast AI, Avatar ảo.

---

## 4. Pipeline tối ưu hóa

### 4.1. Tối ưu hiệu năng (Level 2 & 3 Hybrid)
- **Distillation:** Dùng model lớn (Teacher) dạy model nhỏ (Student)
- **Streaming Inference:** Sinh và phát âm thanh theo chunk để giảm latency
- **Vocoder nhẹ:** HiFi-GAN chuyển phổ âm thành sóng âm nhanh

### 4.2. Đảm bảo đa dạng và cá nhân hóa (Level 2 cải tiến)
- **Speaker Adaptation:** Fine-tune adapter layers thay vì toàn bộ model
- **Phoneme Mapping:** Dùng IPA làm trung gian cho đa ngôn ngữ

### 4.3. Giải pháp đạo đức (Ethical Pipeline)
- **Audio Watermarking:** Nhúng tín hiệu ẩn vào audio đầu ra
  - Giúp nhận diện giọng AI vs giọng thật
  - Ngăn chặn tin giả và voice phishing

---

## 5. Nhận xét

**Kết luận theo từng Level:**

| Yêu cầu | Level phù hợp |
|---------|---------------|
| Tốc độ và ổn định tuyệt đối | Level 1 |
| Cân bằng chất lượng và chi phí | Level 2 |
| Trải nghiệm đột phá, sáng tạo | Level 3 |

**Xu hướng tương lai:**
- Level 3 là hướng đi tiên tiến nhất
- Cần đi kèm biện pháp kiểm soát đạo đức (Watermarking)
- Tối ưu hóa chi phí vận hành qua Distillation và Streaming

---

## 6. Trích dẫn
- Harito ID - more_research: https://harito.id.vn/
- Tacotron: https://arxiv.org/abs/1703.10135
- FastSpeech: https://arxiv.org/abs/1905.09263
- VALL-E: https://arxiv.org/abs/2301.02111
- HiFi-GAN: https://arxiv.org/abs/2010.05646
