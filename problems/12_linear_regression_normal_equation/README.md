# 12. Linear Regression (Normal Equation)

**Độ khó:** Easy · **Liên quan:** Bài 13 Linear regression · **Chấm bài:** `python run.py 12`

## Đề bài

Viết hàm `fit_linear_regression(X, y)` tìm `w` (shape `(d,)`) và `b` (số thực) tối thiểu hoá
$$\text{MSE} = \frac{1}{N}\sum_i (x_i \cdot w + b - y_i)^2$$
bằng **công thức nghiệm đóng** (normal equation), không dùng gradient descent:

$$\theta = (\tilde{X}^\top \tilde{X})^{-1} \tilde{X}^\top y$$

với $\tilde{X}$ là `X` thêm một cột toàn số 1 (cột này ứng với bias). Trả về `(w, b)`.

## Ví dụ

```
X = [[1], [2], [3]], y = [3, 5, 7]   ->  w = [2], b = 1     (y = 2x + 1)
```

## Ràng buộc

- Không dùng scikit-learn. Được dùng `np.linalg.solve`, `np.linalg.lstsq` hoặc `np.linalg.pinv`.

## Gợi ý

<details><summary>Gợi ý</summary>

`Xb = np.hstack([X, np.ones((N, 1))])`. Dùng `np.linalg.lstsq(Xb, y, rcond=None)[0]` ổn định hơn tự nghịch đảo ma trận. Phần tử cuối của θ là b.
</details>

## Câu hỏi phỏng vấn thêm

- Công thức trên từ đâu ra? (Đạo hàm MSE theo θ rồi cho bằng 0.)
- Khi nào $X^\top X$ không khả nghịch? Xử lý thế nào? (Đặc trưng phụ thuộc tuyến tính; dùng pseudo-inverse hoặc ridge.)
- Với 10 triệu mẫu và 100.000 đặc trưng, vì sao người ta dùng gradient descent thay vì công thức này?
