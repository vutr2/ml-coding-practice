# 20. Gradient Boosting with Stumps

**Độ khó:** Medium · **Liên quan:** Bài 17 Ensemble · **Chấm bài:** `python run.py 20`

## Đề bài

Gradient boosting cho hồi quy với **một đặc trưng** `x` (mảng 1 chiều), mỗi "cây" là một **stump** (cây cao 1 tầng: một ngưỡng, hai giá trị lá).

1. `fit_stump(x, r)`: tìm ngưỡng `t` (trung điểm giữa các giá trị khác nhau liên tiếp của `x`) tối thiểu tổng bình phương sai số khi dự đoán `r`. Lá trái (`x <= t`) và lá phải lần lượt là **trung bình** của `r` ở mỗi bên. Trả về `(t, left_value, right_value)`. Hoà thì chọn `t` nhỏ hơn (sai khác < 1e-12). Nếu `x` là hằng số, trả về `(None, mean(r), mean(r))`.
2. `predict_stump(stump, x)`: trả về mảng dự đoán.
3. `gradient_boosting_fit_predict(x_train, y_train, x_test, n_estimators, lr)`:
   - $F_0 = \text{mean}(y)$ (hằng số).
   - Lặp `n_estimators` lần: tính residual $r = y - F$ trên tập train, fit stump vào r, cập nhật $F \leftarrow F + \text{lr} \cdot \text{stump}(x)$.
   - Trả về dự đoán trên `x_test`.

## Ví dụ

```
fit_stump([1, 2, 3, 4], [1, 1, 5, 5]) -> (2.5, 1.0, 5.0)

y = [1, 1, 5, 5], n_estimators = 1, lr = 0.5
F0 = 3, residual = [-2, -2, 2, 2], stump = (2.5, -2, 2)
dự đoán = 3 + 0.5 * [-2, -2, 2, 2] = [2, 2, 4, 4]
```

## Gợi ý

<details><summary>Gợi ý</summary>

Với MSE, residual $y - F$ chính là **gradient âm** của loss theo F. Mỗi cây mới học cách sửa phần sai của các cây trước.
</details>

## Câu hỏi phỏng vấn thêm

- Bagging (Random Forest) khác boosting thế nào? Cái nào giảm variance, cái nào giảm bias?
- Learning rate nhỏ và nhiều cây thường tốt hơn learning rate lớn và ít cây. Vì sao?
- XGBoost và LightGBM cải tiến gì so với gradient boosting cơ bản?
