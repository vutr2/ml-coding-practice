# 25. Linear Layer Forward & Backward

**Độ khó:** Medium · **Liên quan:** Bài 22 Backprop bằng NumPy · **Chấm bài:** `python run.py 25`

## Đề bài

Lớp Linear (fully connected) với batch `X` shape `(N, D)`, trọng số `W` shape `(D, M)`, bias `b` shape `(M,)`:

$$Y = XW + b \quad \text{(shape } (N, M))$$

1. `linear_forward(X, W, b)`: trả về `Y`.
2. `linear_backward(X, W, dY)`: cho gradient của loss theo đầu ra $dY = \partial L / \partial Y$ (shape `(N, M)`), trả về `(dX, dW, db)`:
$$dX = dY\, W^\top, \qquad dW = X^\top dY, \qquad db = \sum_{n} dY_n$$

Đây là "viên gạch" của backprop: mọi mạng MLP chỉ là các lớp Linear xếp chồng với hàm kích hoạt xen giữa.

## Mẹo nhớ: khớp shape

Không cần thuộc công thức. Gradient luôn có cùng shape với biến: `dW` phải là `(D, M)`, mà chỉ có `X.T @ dY` = `(D, N) @ (N, M)` cho ra đúng shape đó.

## Gợi ý

<details><summary>Gợi ý</summary>

`db = dY.sum(axis=0)`: bias được cộng vào mọi mẫu trong batch, nên gradient của nó là tổng theo batch.
</details>

## Câu hỏi phỏng vấn thêm

- Tự chứng minh $dW = X^\top dY$ bằng cách viết $Y_{nm} = \sum_d X_{nd}W_{dm} + b_m$.
- Vì sao cần lưu X trong forward để dùng cho backward? Điều này ảnh hưởng thế nào đến bộ nhớ GPU khi huấn luyện? (Activation memory, gradient checkpointing.)
