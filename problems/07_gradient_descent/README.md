# 07. Gradient Descent

**Độ khó:** Easy · **Liên quan:** Bài 7 Gradient descent · **Chấm bài:** `python run.py 7`

## Đề bài

Viết hàm `gradient_descent(grad_fn, x0, lr, n_steps)`:
- `grad_fn(x)` trả về gradient của hàm cần tối thiểu tại `x`.
- Bắt đầu từ `x0`, lặp `n_steps` lần: $x \leftarrow x - \text{lr} \cdot \nabla f(x)$
- Trả về **lịch sử** các điểm: mảng shape `(n_steps + 1, d)`, hàng đầu là `x0`, hàng cuối là điểm sau bước cuối.
- Không làm thay đổi `x0`.

## Ví dụ

```
f(x) = (x - 3)^2,  grad = 2(x - 3),  x0 = [0], lr = 0.1
bước 1: x = 0 - 0.1 * (-6) = 0.6
bước 2: x = 0.6 - 0.1 * (-4.8) = 1.08
...  hội tụ về 3
```

## Gợi ý

<details><summary>Gợi ý</summary>

Dùng `x = np.array(x0, dtype=float)` để tạo bản sao, lưu từng bước vào list rồi `np.array(list)` ở cuối.
</details>

## Câu hỏi phỏng vấn thêm

- Learning rate quá lớn thì sao? Quá nhỏ thì sao?
- Với f(x) = x², lr lớn hơn bao nhiêu thì gradient descent phân kỳ? (lr > 1.)
- Khác nhau giữa batch GD, SGD và mini-batch GD?
