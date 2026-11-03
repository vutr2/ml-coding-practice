# 32. LSTM Cell & Sequence

**Độ khó:** Medium · **Liên quan:** Bài 26 RNN/LSTM · **Chấm bài:** `python run.py 32`

## Đề bài

**1. `lstm_cell(x, h_prev, c_prev, W, U, b)`**: một bước LSTM cho batch.
- `x` (N, D), `h_prev` (N, H), `c_prev` (N, H), `W` (D, 4H), `U` (H, 4H), `b` (4H,).
- Tính `gates = x @ W + h_prev @ U + b` rồi chia thành 4 khối theo thứ tự **i, f, g, o** (giống PyTorch):
$$i = \sigma(\cdot), \quad f = \sigma(\cdot), \quad g = \tanh(\cdot), \quad o = \sigma(\cdot)$$
$$c = f \odot c_{prev} + i \odot g, \qquad h = o \odot \tanh(c)$$
- Trả về `(h, c)`.

**2. `lstm_forward(xs, h0, c0, W, U, b)`**: chạy qua cả chuỗi `xs` shape (T, N, D). Trả về `(hs, c_T)` với `hs` shape (T, N, H) chứa h ở mọi bước, và `c_T` là cell state cuối.

## Ý nghĩa các cổng

- **f (forget)**: giữ lại bao nhiêu phần trí nhớ cũ.
- **i (input)** và **g**: ghi bao nhiêu thông tin mới vào trí nhớ.
- **o (output)**: đưa bao nhiêu phần trí nhớ ra ngoài thành h.

## Gợi ý

<details><summary>Gợi ý</summary>

`i, f, g, o = np.split(gates, 4, axis=1)`.
</details>

## Câu hỏi phỏng vấn thêm

- Vì sao RNN thường bị vanishing gradient còn LSTM thì đỡ hơn? (Đường đi của c chỉ qua phép nhân với f và phép cộng.)
- GRU khác LSTM thế nào?
- Vì sao Transformer thay thế RNN trong NLP? (Huấn luyện song song được, nhìn xa tốt hơn.)
