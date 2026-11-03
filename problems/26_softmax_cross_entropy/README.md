# 26. Softmax Cross-Entropy (Forward & Backward)

**Độ khó:** Medium · **Liên quan:** Bài 22 Backprop bằng NumPy · **Chấm bài:** `python run.py 26`

## Đề bài

Viết hàm `softmax_cross_entropy(logits, labels)`:
- `logits`: shape `(N, C)`, `labels`: mảng số nguyên shape `(N,)`.
- Trả về `(loss, dlogits)`:
$$L = -\frac{1}{N}\sum_n \log p_{n, y_n}, \qquad p = \text{softmax}(\text{logits})$$
$$\frac{\partial L}{\partial \text{logits}} = \frac{1}{N}\big(p - \text{onehot}(y)\big)$$

Phải **ổn định số học** với logits rất lớn (ví dụ 1000).

## Ví dụ

```
logits = [[2.0, 1.0, 0.1]], labels = [0]
p = [0.659, 0.242, 0.099]
loss = -log(0.659) = 0.4170
dlogits = [[-0.341, 0.242, 0.099]]
```

## Gợi ý

<details><summary>Gợi ý 1: ổn định số học</summary>

Tính log-softmax trực tiếp: $\log p_{nc} = z_{nc} - \max_c z - \log\sum_c e^{z_{nc} - \max_c z}$. Không tính softmax rồi mới lấy log (log(0) = -inf).
</details>

<details><summary>Gợi ý 2: gradient</summary>

`dlogits = p.copy(); dlogits[np.arange(N), labels] -= 1; dlogits /= N`.
</details>

## Câu hỏi phỏng vấn thêm

- Tự chứng minh gradient bằng $p - y$.
- Vì sao PyTorch có `nn.CrossEntropyLoss` nhận logits thay vì xác suất?
- Label smoothing là gì? Thay đổi gradient thế nào?
