# 04. Single Neuron Backprop (Chain Rule)

**Độ khó:** Medium · **Liên quan:** Bài 4 Chain rule · **Chấm bài:** `python run.py 4`

## Đề bài

Một nơ-ron với đầu vào `x` (vector), trọng số `w`, bias `b` và nhãn thật `y`:

$$z = w \cdot x + b, \qquad a = \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad L = (a - y)^2$$

Viết hàm `neuron_forward_backward(x, w, b, y)` trả về bộ ba `(L, dw, db)`:
- `L`: giá trị loss (float)
- `dw`: $\partial L / \partial w$, mảng cùng shape với `w`
- `db`: $\partial L / \partial b$ (float)

Tính đạo hàm bằng **chain rule**, không dùng gradient số.

## Ví dụ

```
x = [1, 2], w = [0.5, -0.25], b = 0.1, y = 1
z = 0.1, a = 0.52498, L = 0.22564
```

## Gợi ý

<details><summary>Gợi ý 1</summary>

Chain rule: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$
</details>

<details><summary>Gợi ý 2</summary>

- $\partial L / \partial a = 2(a - y)$
- $\partial a / \partial z = a(1 - a)$ (đạo hàm sigmoid)
- $\partial z / \partial w = x$, $\partial z / \partial b = 1$
</details>

## Câu hỏi phỏng vấn thêm

- Khi `z` rất lớn hoặc rất nhỏ, `a(1-a)` gần 0. Điều này gây ra vấn đề gì khi huấn luyện? (Vanishing gradient.)
- Vì sao logistic regression dùng cross-entropy thay vì MSE như bài này?
