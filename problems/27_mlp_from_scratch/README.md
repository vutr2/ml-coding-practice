# 27. 2-Layer MLP from Scratch

**Độ khó:** Hard · **Liên quan:** Bài 21–22 MLP & Backprop · **Chấm bài:** `python run.py 27`

## Đề bài

Xây mạng MLP 2 lớp cho phân loại C lớp, chỉ dùng NumPy:

$$h = \text{ReLU}(XW_1 + b_1), \qquad \text{logits} = hW_2 + b_2, \qquad L = \text{CrossEntropy}(\text{softmax}(\text{logits}), y)$$

`params` là dict gồm `"W1"` (D, H), `"b1"` (H,), `"W2"` (H, C), `"b2"` (C,).

1. `mlp_loss_and_grads(params, X, y)`: trả về `(loss, grads)`, `grads` là dict có cùng các khoá và shape với `params`.
2. `train_mlp(X, y, hidden=32, lr=0.5, epochs=2000, seed=0)`:
   - Số lớp `C = max(y) + 1`.
   - Khởi tạo **He**: `W = rng.normal(size=...) * sqrt(2 / fan_in)`, bias bằng 0, với `rng = np.random.default_rng(seed)`.
   - Gradient descent toàn batch. Trả về `params`.
3. `mlp_predict(params, X)`: trả về nhãn dự đoán (argmax của logits).

## Gợi ý

<details><summary>Gợi ý 1: thứ tự backward</summary>

Đi ngược từ loss:
1. `dlogits = (p - onehot) / N` (bài 26)
2. `dW2 = h.T @ dlogits`, `db2 = dlogits.sum(0)`, `dh = dlogits @ W2.T` (bài 25)
3. `dz1 = dh * (z1 > 0)` (đạo hàm ReLU, bài 24)
4. `dW1 = X.T @ dz1`, `db1 = dz1.sum(0)`
</details>

<details><summary>Gợi ý 2</summary>

Lưu `z1` (trước ReLU) trong forward để dùng khi backward.
</details>

## Câu hỏi phỏng vấn thêm

- Nếu khởi tạo tất cả trọng số bằng 0 thì chuyện gì xảy ra? (Đối xứng: mọi nơ-ron ẩn học giống nhau.)
- Vì sao khởi tạo He dùng $\sqrt{2/\text{fan\_in}}$ cho ReLU, còn Xavier dùng cho tanh?
- Mạng 1 lớp (logistic regression) có giải được dữ liệu hình tròn trong test không? Vì sao?
