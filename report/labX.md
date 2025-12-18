# Báo cáo Lab X: Nghiên cứu Text-to-Speech (TTS)

## 1. Mục tiêu

Nghiên cứu toàn diện về công nghệ **Text-to-Speech (TTS)**, từ các khái niệm cơ bản về xử lý tín hiệu âm thanh đến các kiến trúc Generative AI tiên tiến nhất hiện nay. Báo cáo cung cấp cái nhìn sâu sắc về quy trình chuyển đổi văn bản thành giọng nói, so sánh các phương pháp tiếp cận và phân tích xu hướng tương lai.

**Nội dung chính:**
1.  Tổng quan về xử lý tín hiệu âm thanh trong TTS.
2.  Phân tích kiến trúc Neural TTS hiện đại (Encoder - Decoder - Vocoder).
3.  So sánh các chiến lược triển khai: Autoregressive vs. Non-autoregressive.

---

## 2. Nền tảng Lý thuyết

### 2.1. Biểu diễn Tín hiệu Âm thanh
Máy tính xử lý âm thanh thông qua các dạng biểu diễn số:
*   **Waveform (Sóng âm)**: Dữ liệu thô (raw audio) biên độ theo thời gian. Chứa rất nhiều chiều (nghìn điểm dữ liệu/giây), khó mô hình hóa trực tiếp.
*   **Mel-spectrogram**: Biểu diễn tần số theo thời gian, được nén theo thang đo Mel (mô phỏng thính giác người). Đây là biểu diễn trung gian tiêu chuẩn cho hầu hết các mô hình TTS hiện đại.

### 2.2. Ba Cấp độ Phát triển của TTS
1.  **Level 1 - Concatenative**: Cắt ghép giọng thu âm sẵn. Nhanh nhưng thiếu tự nhiên.
2.  **Level 2 - Neural Vectors**: Học mối quan hệ Text-to-Spectrogram (Tacotron, FastSpeech). Tự nhiên, mượt mà nhưng cần fine-tuning cho giọng mới.
3.  **Level 3 - Generative / Zero-shot**: Coi TTS là bài toán Language Modeling với Audio Tokens (VALL-E). Có thể clone giọng bất kỳ chỉ từ vài giây mẫu.

---

## 3. Dataset (Dữ liệu Huấn luyện)

Các mô hình TTS hiện đại đòi hỏi lượng dữ liệu khổng lồ để đạt độ tự nhiên cao:

*   **Dữ liệu Single-speaker (Cho Level 2)**: LJSpeech (24 giờ, 1 người đọc), đảm bảo sự nhất quán tối đa.
*   **Dữ liệu Multi-speaker (Cho Level 3)**: LibriTTS (585 giờ, 2400 người đọc), VCTK.
*   **Dữ liệu thô (In-the-wild)**: Common Voice, VoxCeleb. Dùng để huấn luyện khả năng zero-shot learning trên đa dạng điều kiện thu âm.

---

## 4. Pipeline Triển khai (Cài đặt)

Một hệ thống Neural TTS điển hình (Tacotron 2 / FastSpeech 2) gồm 3 module chính kết nối tuần tự:

### 4.1. Text Frontend (Text Analysis)
*   **Input**: Văn bản thô ("Hello world, it's 2024").
*   **Xử lý**: Chuẩn hóa số, ngày tháng, viết tắt -> Chuyển sang chuỗi âm vị (Phonemes) dùng bảng IPA.
*   **Output**: Chuỗi vector âm vị.

### 4.2. Acoustic Model (Mel Generator)
*   **Nhiệm vụ**: Chuyển chuỗi âm vị thành Mel-spectrogram.
*   **Chiến lược**:
    *   *Autoregressive (Tacotron)*: Sinh tuần tự từng khung thời gian. Chất lượng cao nhưng chậm.
    *   *Non-autoregressive (FastSpeech)*: Sinh song song toàn bộ phổ âm. Cực nhanh, phù hợp realtime.

### 4.3. Vocoder (Neural Vocoder)
*   **Nhiệm vụ**: "Vẽ" lại sóng âm (Waveform) từ bản thiết kế (Mel-spectrogram).
*   **Công nghệ**: HiFi-GAN (Generative Adversarial Networks) hiện là chuẩn mực vì tốc độ nhanh và chất lượng âm thanh trung thực, loại bỏ tiếng rè kim khí.

---

## 5. Kết quả & So sánh

### 5.1. So sánh Chất lượng (MOS - Mean Opinion Score)
*   **Concatenative**: MOS ~3.0 - 3.5 (Nghe rõ tiếng máy).
*   **Neural TTS (Tacotron 2)**: MOS ~4.0 - 4.2 (Gần giống người).
*   **Generative TTS (VALL-E)**: MOS ~4.5 (Khó phân biệt với bản thu gốc).

### 5.2. So sánh Tốc độ (Real-time Factor - RTF)
*   **Tacotron 2**: RTF ~0.1 (Nhanh hơn thời gian thực 10 lần).
*   **FastSpeech 2**: RTF ~0.01 (Cực nhanh, xử lý câu dài trong tíc tắc).
*   **VALL-E**: RTF cao hơn do mô hình ngôn ngữ lớn, cần GPU mạnh.

---

## 6. Nhận xét

### 6.1. Xu hướng Tương lai
Ngành TTS đang chuyển dịch mạnh mẽ sang hướng **Zero-shot Learning**: Khả năng nói bất kỳ giọng nào, bất kỳ ngôn ngữ nào chỉ với một mẫu gợi ý (prompt) cực ngắn mà không cần huấn luyện lại.

### 6.2. Đạo đức AI
Sự phát triển của Voice Cloning đặt ra thách thức lớn về **Deepfake**. Các mô hình thế hệ mới cần đi kèm với công nghệ Watermarking (đánh dấu thủy vân) để phân biệt giọng máy và giọng thật.

---

## 7. Trích dẫn
1.  **Tacotron 2** (Google, 2018) - Paper nền tảng của Neural TTS.
2.  **FastSpeech 2** (Microsoft, 2020) - Chuẩn mực về tốc độ.
3.  **VALL-E** (Microsoft, 2023) - Kỷ nguyên Audio LM.
4.  **Hugging Face Audio Course**: https://huggingface.co/learn/audio-course
