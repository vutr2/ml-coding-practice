# 17. Ridge Regression

**Độ khó:** Easy · **Liên quan:** Bài 15 Overfitting & regularization · **Chấm bài:** `python run.py 17`

## Đề bài

Viết hàm `fit_ridge(X, y, alpha)` trả về `(w, b)` tối thiểu hoá
$$\sum_i (x_i \cdot w + b - y_i)^2 + \alpha \|w\|^2$$

**Không phạt bias b.** Cách làm chuẩn:
1. Trừ trung bình: $X_c = X - \bar{X}$, $y_c = y - \bar{y}$.
2. $w = (X_c^\top X_c + \alpha I)^{-1} X_c^\top y_c$
3. $b = \bar{y} - \bar{X} \cdot w$

## Ví dụ

```
alpha = 0      ->  giống hệt linear regression thường
alpha rất lớn  ->  w gần 0, b gần mean(y)
```

## Gợi ý

<details><summary>Gợi ý</summary>

`np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ yc)`.
</details>

## Câu hỏi phỏng vấn thêm

- Vì sao thêm $\alpha I$ giúp ma trận luôn khả nghịch?
- L1 (Lasso) khác L2 (Ridge) thế nào? Vì sao L1 cho ra w thưa (nhiều số 0)?
- Vì sao không phạt bias?
- Ridge tương đương MLE với prior nào trên w? (Gaussian prior, tức MAP.)
