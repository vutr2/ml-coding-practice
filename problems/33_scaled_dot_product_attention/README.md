# 33. Scaled Dot-Product Attention

**Độ khó:** Medium · **Liên quan:** Bài 27 Transformer · **Chấm bài:** `python run.py 33`

## Đề bài

**1. `scaled_dot_product_attention(Q, K, V, mask=None)`**
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$
- `Q` (..., T_q, d_k), `K` (..., T_k, d_k), `V` (..., T_k, d_v). Dấu `...` là các chiều batch/head tuỳ ý.
- `mask` (tuỳ chọn): mảng bool broadcast được với `(..., T_q, T_k)`. **`True` = được nhìn**, `False` = bị chặn (trọng số attention bằng 0). Giống quy ước của `torch.nn.functional.scaled_dot_product_attention`.
- Trả về `(output, weights)`: output shape (..., T_q, d_v), weights shape (..., T_q, T_k).

**2. `causal_mask(T)`**: trả về mảng bool (T, T), `True` ở đường chéo và phía dưới (token chỉ nhìn được chính nó và các token trước).

## Ví dụ

```
causal_mask(3) -> [[ True, False, False],
                   [ True,  True, False],
                   [ True,  True,  True]]
```

## Gợi ý

<details><summary>Gợi ý 1</summary>

Chuyển vị hai chiều cuối của K: `np.swapaxes(K, -1, -2)`. Toán tử `@` tự xử lý các chiều batch.
</details>

<details><summary>Gợi ý 2: mask</summary>

`scores = np.where(mask, scores, -1e9)` trước khi softmax (dùng softmax ổn định của bài 05).
</details>

## Câu hỏi phỏng vấn thêm

- Vì sao chia cho $\sqrt{d_k}$? (Tích vô hướng có phương sai $d_k$, không chia thì softmax bị bão hoà, gradient nhỏ.)
- Độ phức tạp theo độ dài chuỗi T là bao nhiêu? (O(T²·d).) FlashAttention tối ưu điều gì?
- KV cache là gì, vì sao làm suy luận LLM nhanh hơn?
