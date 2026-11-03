# 36. LLM Sampling: Temperature, Top-k, Top-p

**Độ khó:** Medium · **Liên quan:** Phần 5 AI Engineering (LLM) · **Chấm bài:** `python run.py 36`

## Đề bài

Khi LLM sinh văn bản, mỗi bước mô hình cho ra vector `logits` (một điểm cho mỗi token trong từ điển). Bài này viết các cách chọn token tiếp theo.

1. `top_k_filter(logits, k)`: giữ nguyên k logits lớn nhất, các vị trí còn lại đặt bằng `-np.inf`. Trả về mảng mới.
2. `top_p_filter(logits, p)` (nucleus sampling):
   - Tính xác suất bằng softmax, sắp xếp giảm dần.
   - Giữ tập token **nhỏ nhất** có tổng xác suất **≥ p** (luôn giữ ít nhất 1 token).
   - Các token còn lại đặt logits bằng `-np.inf`. Trả về mảng mới.
3. `sample(logits, temperature=1.0, rng=None)`: trả về chỉ số token (int).
   - `temperature == 0`: chọn argmax (greedy decoding).
   - Ngược lại: lấy mẫu theo phân phối `softmax(logits / temperature)` bằng `rng.choice` (nếu `rng` là `None` thì dùng `np.random.default_rng()`). Logits bằng `-inf` có xác suất 0.

## Ví dụ

```
logits = log([0.5, 0.3, 0.15, 0.05])
top_k_filter(logits, 2)    -> giữ token 0, 1
top_p_filter(logits, 0.75) -> giữ token 0, 1    (0.5 + 0.3 = 0.8 ≥ 0.75)
top_p_filter(logits, 0.4)  -> giữ token 0       (0.5 ≥ 0.4)
```

## Gợi ý

<details><summary>Gợi ý 1: top-p</summary>

`order = np.argsort(-probs)`, `cum = np.cumsum(probs[order])`. Số token cần giữ là `np.searchsorted(cum, p) + 1` (vị trí đầu tiên mà tổng tích luỹ ≥ p, cộng 1).
</details>

<details><summary>Gợi ý 2: -inf</summary>

`np.exp(-np.inf) = 0`, nên softmax ổn định (trừ max) vẫn chạy đúng khi có -inf, miễn là còn ít nhất một logit hữu hạn.
</details>

## Câu hỏi phỏng vấn thêm

- Temperature cao và thấp ảnh hưởng thế nào đến văn bản sinh ra?
- Vì sao top-p thường tốt hơn top-k? (Số token giữ lại tự co giãn theo độ tự tin của mô hình.)
- Greedy decoding có nhược điểm gì? Beam search khác gì?
- Muốn câu trả lời của API có thể lặp lại được (reproducible) thì đặt tham số nào?
