# 19. Decision Tree: Gini & Best Split

**Độ khó:** Medium · **Liên quan:** Bài 16 Cây quyết định · **Chấm bài:** `python run.py 19`

## Đề bài

1. `gini(y)`: độ vẩn đục Gini của mảng nhãn: $G = 1 - \sum_k p_k^2$, với $p_k$ là tỉ lệ nhãn k. Mảng rỗng có Gini bằng 0.
2. `best_split(X, y)`: tìm cách chia tốt nhất cho **một nút** của cây. Trả về `(feature, threshold, weighted_gini)`:
   - Thử mọi cột `feature` và mọi ngưỡng là **trung điểm giữa hai giá trị khác nhau liên tiếp** (sau khi sắp xếp) của cột đó.
   - Nhánh trái: `X[:, feature] <= threshold`, nhánh phải: phần còn lại.
   - $\text{weighted\_gini} = \frac{n_L}{n} G_L + \frac{n_R}{n} G_R$, chọn cách chia có giá trị nhỏ nhất.
   - Nếu hoà (sai khác < 1e-12): chọn `feature` nhỏ hơn, rồi `threshold` nhỏ hơn.
   - Nếu không chia được (mọi cột đều là hằng số): trả về `(None, None, gini(y))`.

## Ví dụ

```
X = [[2], [3], [10], [19]], y = [0, 0, 1, 1]
best_split(X, y) -> (0, 6.5, 0.0)
```

## Gợi ý

<details><summary>Gợi ý 1</summary>

`np.unique(col)` trả về các giá trị khác nhau đã sắp xếp. Trung điểm: `(u[:-1] + u[1:]) / 2`.
</details>

<details><summary>Gợi ý 2</summary>

Cập nhật kết quả tốt nhất chỉ khi `g < best - 1e-12`. Duyệt feature và threshold theo thứ tự tăng dần thì luật hoà tự động đúng.
</details>

## Câu hỏi phỏng vấn thêm

- Gini khác entropy (information gain) thế nào? Kết quả có khác nhiều không?
- Cây quá sâu thì sao? Có những cách nào để chống overfit cho cây?
- Vì sao cây quyết định không cần chuẩn hoá đặc trưng?
