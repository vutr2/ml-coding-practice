# 06. Gaussian MLE

**Độ khó:** Easy · **Liên quan:** Bài 6 MLE · **Chấm bài:** `python run.py 6`

## Đề bài

Cho dữ liệu `x` (mảng 1 chiều) giả sử lấy từ phân phối chuẩn $\mathcal{N}(\mu, \sigma^2)$.

1. `gaussian_log_likelihood(x, mu, var)`: trả về tổng log-likelihood
$$\log L = \sum_i \left[ -\tfrac{1}{2}\log(2\pi\sigma^2) - \frac{(x_i - \mu)^2}{2\sigma^2} \right]$$
2. `gaussian_mle(x)`: trả về `(mu, var)` là ước lượng hợp lý cực đại (MLE).

## Ví dụ

```
gaussian_mle([2, 4, 4, 4, 5, 5, 7, 9]) -> (5.0, 4.0)
```

## Ràng buộc

- MLE của phương sai chia cho **N** (không phải N − 1).

## Gợi ý

<details><summary>Gợi ý</summary>

Đạo hàm log-likelihood theo μ và σ² rồi cho bằng 0, bạn được: μ là trung bình mẫu, σ² là trung bình của $(x_i - \mu)^2$.
</details>

## Câu hỏi phỏng vấn thêm

- Vì sao tối đa hoá log-likelihood thay vì likelihood?
- Chứng minh: tối thiểu hoá MSE trong linear regression tương đương MLE khi nhiễu có phân phối chuẩn.
- Vì sao ước lượng phương sai chia N bị lệch (biased)?
