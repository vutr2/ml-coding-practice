# 24. Activation Functions & Derivatives

**Độ khó:** Easy · **Liên quan:** Bài 20 Neuron & hàm kích hoạt · **Chấm bài:** `python run.py 24`

## Đề bài

Viết các hàm kích hoạt và đạo hàm của chúng. Mọi hàm nhận mảng NumPy và trả về mảng cùng shape.

| Hàm | Công thức | Đạo hàm |
|---|---|---|
| `relu(x)` | $\max(0, x)$ | `relu_grad(x)`: 1 nếu x > 0, ngược lại 0 |
| `leaky_relu(x, alpha=0.01)` | x nếu x > 0, ngược lại αx | `leaky_relu_grad(x, alpha=0.01)`: 1 hoặc α |
| `sigmoid(x)` | $1/(1+e^{-x})$ | `sigmoid_grad(x)`: $\sigma(x)(1-\sigma(x))$ |
| `tanh(x)` | $\tanh(x)$ | `tanh_grad(x)`: $1 - \tanh^2(x)$ |

Quy ước: tại x = 0, đạo hàm ReLU là 0, đạo hàm Leaky ReLU là α.

## Ví dụ

```
relu([-2, 0, 3])        -> [0, 0, 3]
relu_grad([-2, 0, 3])   -> [0, 0, 1]
sigmoid_grad([0])       -> [0.25]
```

## Gợi ý

<details><summary>Gợi ý</summary>

`np.where(x > 0, a, b)` cho các hàm có hai nhánh. `sigmoid_grad` và `tanh_grad` nhận x (không phải đầu ra), nên tính hàm trước rồi dùng công thức.
</details>

## Câu hỏi phỏng vấn thêm

- Vì sao mạng sâu dùng ReLU thay vì sigmoid? (Gradient của sigmoid tối đa 0.25, nhân qua nhiều lớp sẽ biến mất.)
- "Dying ReLU" là gì? Leaky ReLU giải quyết thế nào?
- Vì sao cần hàm kích hoạt phi tuyến? Nếu bỏ hết thì mạng 10 lớp tương đương gì?
- GELU là gì, dùng ở đâu? (Transformer, BERT, GPT.)
