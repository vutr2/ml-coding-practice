# 13. Linear Regression (Gradient Descent)

**Độ khó:** Medium · **Liên quan:** Bài 13 Linear regression · **Chấm bài:** `python run.py 13`

## Đề bài

1. `mse_loss_and_grad(X, y, w, b)`: với dự đoán $\hat{y} = Xw + b$, trả về `(loss, dw, db)`:
$$L = \frac{1}{N}\sum_i (\hat{y}_i - y_i)^2, \quad \frac{\partial L}{\partial w} = \frac{2}{N} X^\top(\hat{y} - y), \quad \frac{\partial L}{\partial b} = \frac{2}{N}\sum_i(\hat{y}_i - y_i)$$
2. `train_linear_regression_gd(X, y, lr=0.1, epochs=1000)`: khởi tạo `w = 0`, `b = 0`, chạy gradient descent toàn batch `epochs` lần, trả về `(w, b)`.

## Ví dụ

```
X = [[1], [2]], y = [2, 4], w = [0], b = 0
loss = (4 + 16) / 2 = 10,  dw = [-10],  db = -6
```

## Gợi ý

<details><summary>Gợi ý</summary>

Tính `err = X @ w + b - y` một lần rồi dùng lại cho cả loss và gradient. Viết vector hoá, không dùng vòng lặp theo mẫu.
</details>

## Câu hỏi phỏng vấn thêm

- Tự đạo hàm công thức dw ở trên.
- Vì sao nên chuẩn hoá đặc trưng trước khi chạy gradient descent?
- Nếu loss tăng dần qua các epoch thì nghi ngờ điều gì đầu tiên?
