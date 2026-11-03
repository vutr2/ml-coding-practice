# 14. Logistic Regression from Scratch

**Độ khó:** Medium · **Liên quan:** Bài 14 Logistic regression · **Chấm bài:** `python run.py 14`

## Đề bài

Phân loại nhị phân với nhãn `y ∈ {0, 1}`.

1. `sigmoid(z)`: $\sigma(z) = 1 / (1 + e^{-z})$, **ổn định số học** (không ra `nan` hay cảnh báo tràn số với `z = ±1000`).
2. `bce_loss_and_grad(X, y, w, b)`: với $p = \sigma(Xw + b)$, trả về `(loss, dw, db)`:
$$L = -\frac{1}{N}\sum_i \big[y_i \log p_i + (1-y_i)\log(1-p_i)\big], \quad \frac{\partial L}{\partial w} = \frac{1}{N}X^\top(p - y), \quad \frac{\partial L}{\partial b} = \frac{1}{N}\sum_i (p_i - y_i)$$
Kẹp `p` vào khoảng `[1e-12, 1 - 1e-12]` khi tính log.
3. `train_logistic_regression(X, y, lr=0.1, epochs=1000)`: khởi tạo `w = 0`, `b = 0`, gradient descent toàn batch, trả về `(w, b)`.
4. `predict(X, w, b, threshold=0.5)`: trả về mảng nhãn 0/1 (kiểu int).

## Ví dụ

```
sigmoid(0) = 0.5
X = [[1], [-1]], y = [1, 0], w = [0], b = 0  ->  loss = log 2 = 0.6931,  dw = [-0.5],  db = 0
```

## Gợi ý

<details><summary>Gợi ý 1: sigmoid ổn định</summary>

Một cách gọn: $\sigma(z) = \tfrac{1}{2}\big(1 + \tanh(z/2)\big)$. Cách khác: với z ≥ 0 dùng `1 / (1 + exp(-z))`, với z < 0 dùng `exp(z) / (1 + exp(z))`.
</details>

<details><summary>Gợi ý 2: gradient</summary>

Gradient của BCE qua sigmoid gọn đến bất ngờ: `p - y`. Đây là lý do logistic regression đi với cross-entropy.
</details>

## Câu hỏi phỏng vấn thêm

- Tự chứng minh $\partial L / \partial z = p - y$.
- Vì sao không dùng MSE cho logistic regression? (Loss không lồi, gradient nhỏ khi dự đoán sai nhiều.)
- Ngưỡng 0.5 có phải lúc nào cũng tốt? Khi nào nên đổi?
