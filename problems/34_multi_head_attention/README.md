# 34. Multi-Head Self-Attention

**Độ khó:** Hard · **Liên quan:** Bài 27 Transformer · **Chấm bài:** `python run.py 34`

## Đề bài

Viết hàm `multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, causal=False)`:
- `X`: shape (T, d_model). Các ma trận `Wq, Wk, Wv, Wo`: shape (d_model, d_model). Không có bias.
- Nếu `d_model` không chia hết cho `num_heads` thì `raise ValueError`.

Các bước:
1. $Q = XW_q$, $K = XW_k$, $V = XW_v$, mỗi cái shape (T, d_model).
2. Tách thành `num_heads` head: reshape (T, h, d_k) rồi đổi trục thành **(h, T, d_k)**, với $d_k = d_{model}/h$. Head thứ j dùng các cột `j*d_k` đến `(j+1)*d_k - 1`.
3. Tính scaled dot-product attention cho từng head (dùng lại bài 33). Nếu `causal=True` thì áp causal mask.
4. Ghép các head lại thành (T, d_model) (đảo lại bước 2), rồi nhân $W_o$.

Trả về mảng (T, d_model).

## Gợi ý

<details><summary>Gợi ý 1: tách head</summary>

`Q.reshape(T, h, d_k).transpose(1, 0, 2)` cho shape (h, T, d_k). Ghép lại: `out.transpose(1, 0, 2).reshape(T, d_model)`.
</details>

<details><summary>Gợi ý 2</summary>

Nhờ `@` tự xử lý chiều batch, bạn có thể tính attention cho mọi head cùng lúc, không cần vòng lặp.
</details>

## Câu hỏi phỏng vấn thêm

- Vì sao dùng nhiều head thay vì một head lớn? (Mỗi head học một kiểu quan hệ khác nhau.)
- Số tham số của một lớp multi-head attention là bao nhiêu? Có phụ thuộc số head không? (4·d², không phụ thuộc.)
- Multi-Query Attention và Grouped-Query Attention là gì? Vì sao giúp KV cache nhỏ hơn?
