# 21. Linear SVM (Hinge Loss)

**Độ khó:** Medium · **Liên quan:** Bài 18 SVM · **Chấm bài:** `python run.py 21`

## Đề bài

SVM tuyến tính với nhãn `y ∈ {-1, +1}`. Đặt margin $m_i = y_i (x_i \cdot w + b)$.

1. `svm_loss_and_grad(X, y, w, b, C)`: trả về `(loss, dw, db)` với
$$L = \tfrac{1}{2}\|w\|^2 + \frac{C}{N}\sum_i \max(0,\, 1 - m_i)$$
Subgradient: chỉ các mẫu **vi phạm margin** ($m_i < 1$) đóng góp:
$$\frac{\partial L}{\partial w} = w - \frac{C}{N}\sum_{m_i < 1} y_i x_i, \qquad \frac{\partial L}{\partial b} = -\frac{C}{N}\sum_{m_i < 1} y_i$$
2. `train_svm(X, y, C=1.0, lr=0.01, epochs=1000)`: khởi tạo `w = 0`, `b = 0`, gradient descent toàn batch, trả về `(w, b)`.
3. `svm_predict(X, w, b)`: trả về `sign(Xw + b)` dạng -1/+1 (giá trị 0 tính là +1).

## Ví dụ

```
X = [[2, 0], [-2, 0]], y = [1, -1], w = [0, 0], b = 0, C = 1
margin = [0, 0]  ->  loss = 1,  dw = [-2, 0],  db = 0
```

## Gợi ý

<details><summary>Gợi ý</summary>

`viol = margins < 1` rồi `dw = w - C / N * (X[viol].T @ y[viol])`.
</details>

## Câu hỏi phỏng vấn thêm

- Support vector là gì? Vì sao chỉ chúng ảnh hưởng đến lời giải?
- C lớn và C nhỏ khác nhau thế nào (hard margin hay soft margin)?
- Kernel trick là gì? Vì sao SVM với kernel RBF tách được dữ liệu không tuyến tính?
