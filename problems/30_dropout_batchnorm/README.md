# 30. Dropout & BatchNorm

**Độ khó:** Medium · **Liên quan:** Bài 15 Regularization · Bài 25 CNN · **Chấm bài:** `python run.py 30`

## Đề bài

**1. `dropout(x, p, training=True, rng=None)`** (inverted dropout):
- Khi `training=True`: mỗi phần tử bị đặt về 0 với xác suất `p`, các phần tử còn lại được **chia cho (1 − p)**. Dùng `rng` (một `np.random.Generator`) để sinh ngẫu nhiên; nếu `rng` là `None` thì dùng `np.random.default_rng()`.
- Khi `training=False`: trả về `x` không đổi.

**2. `batchnorm_forward(x, gamma, beta, eps=1e-5)`** (chế độ huấn luyện), `x` shape `(N, D)`:
$$\mu = \frac{1}{N}\sum_n x_n, \quad \sigma^2 = \frac{1}{N}\sum_n (x_n - \mu)^2, \quad \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$
Thống kê tính theo **từng cột** (từng đặc trưng) trên cả batch.

## Ví dụ

```
dropout([1, 1, 1, 1], p=0.5)  ->  ví dụ [2, 0, 2, 0]   (phần còn lại nhân 2)
```

## Gợi ý

<details><summary>Gợi ý</summary>

- Dropout: `mask = rng.random(x.shape) >= p`, rồi `x * mask / (1 - p)`.
- BatchNorm: `x.mean(axis=0)`, `x.var(axis=0)`.
</details>

## Câu hỏi phỏng vấn thêm

- Vì sao gọi là "inverted" dropout? Nếu không chia (1 − p) khi train thì lúc test phải làm gì?
- Khi suy luận (inference), BatchNorm dùng thống kê nào? (running mean/var lưu trong lúc huấn luyện.)
- Vì sao Transformer dùng LayerNorm thay vì BatchNorm?
- BatchNorm với batch size = 1 có vấn đề gì?
