# 28. SGD with Momentum & Adam

**Độ khó:** Medium · **Liên quan:** Bài 23 Optimizers & Adam · **Chấm bài:** `python run.py 28`

## Đề bài

Viết hai optimizer theo đúng công thức của PyTorch. Mỗi class có hàm `step(params, grads)`: `params` và `grads` là dict `tên -> mảng NumPy`, hàm **cập nhật trực tiếp** `params` (không trả về gì). Trạng thái (velocity, m, v, số bước t) lưu riêng cho từng tham số theo tên.

**1. `SGDMomentum(lr, momentum=0.9)`**
$$v \leftarrow \mu v + g, \qquad \theta \leftarrow \theta - \text{lr}\cdot v \quad (v \text{ khởi tạo bằng } 0)$$

**2. `Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)`**, với t là số bước (bắt đầu từ 1):
$$m \leftarrow \beta_1 m + (1-\beta_1) g, \qquad v \leftarrow \beta_2 v + (1-\beta_2) g^2$$
$$\hat{m} = \frac{m}{1-\beta_1^t}, \qquad \hat{v} = \frac{v}{1-\beta_2^t}, \qquad \theta \leftarrow \theta - \text{lr}\cdot\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

## Ví dụ

```
SGDMomentum(lr=0.1, momentum=0.9), θ = 1, gradient luôn bằng 1
bước 1: v = 1,    θ = 0.9
bước 2: v = 1.9,  θ = 0.71
```

Điều thú vị của Adam: ở bước đầu tiên $\hat{m}/\sqrt{\hat{v}} = g/|g|$, nên **mọi tham số di chuyển đúng một khoảng lr**, bất kể gradient lớn hay nhỏ.

## Gợi ý

<details><summary>Gợi ý</summary>

Trong `__init__` tạo `self.state = {}`. Trong `step`, với mỗi tên k: nếu chưa có trong state thì khởi tạo `np.zeros_like(params[k])`. Cập nhật bằng `params[k] -= ...` để sửa trực tiếp mảng.
</details>

## Câu hỏi phỏng vấn thêm

- Momentum giúp gì so với SGD thường? (Đi nhanh hơn theo hướng ổn định, giảm dao động.)
- Vì sao Adam cần bias correction $1 - \beta^t$? Bỏ đi thì những bước đầu sao?
- AdamW khác Adam ở điểm nào? Vì sao LLM thường dùng AdamW?
- Adam tốn bao nhiêu bộ nhớ so với số tham số? (Thêm 2 bản sao m và v.)
