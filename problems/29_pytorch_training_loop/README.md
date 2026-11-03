# 29. PyTorch Training Loop

**Độ khó:** Medium · **Liên quan:** Bài 24 PyTorch · **Chấm bài:** `python run.py 29`

## Đề bài

Viết vòng lặp huấn luyện PyTorch chuẩn. Bài này **cần cài PyTorch** (`pip install torch`).

1. `build_model(in_dim, hidden, out_dim)`: trả về `nn.Sequential(Linear(in_dim, hidden), ReLU(), Linear(hidden, out_dim))`.
2. `train(model, X, y, epochs=30, lr=0.01, batch_size=32)`:
   - `X`: tensor float `(N, in_dim)`, `y`: tensor long `(N,)`.
   - Dùng `TensorDataset` + `DataLoader(shuffle=True)`, loss `nn.CrossEntropyLoss()`, optimizer `torch.optim.Adam`.
   - Mỗi epoch: `model.train()`, với mỗi batch: `zero_grad` → forward → loss → `backward` → `step`.
   - Trả về list độ dài `epochs`, phần tử thứ i là **loss trung bình** của epoch i (trung bình theo số mẫu).
3. `evaluate(model, X, y)`: chuyển `model.eval()`, chạy trong `torch.no_grad()`, trả về accuracy (float từ 0 đến 1).

## Gợi ý

<details><summary>Gợi ý: khung vòng lặp</summary>

```python
for epoch in range(epochs):
    model.train()
    total = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    history.append(total / len(X))
```
</details>

## Câu hỏi phỏng vấn thêm

- Quên `optimizer.zero_grad()` thì chuyện gì xảy ra? (Gradient cộng dồn qua các batch.)
- `model.eval()` khác `torch.no_grad()` thế nào? (eval đổi hành vi Dropout/BatchNorm; no_grad tắt việc lưu đồ thị tính toán để tiết kiệm bộ nhớ.)
- Vì sao `nn.CrossEntropyLoss` nhận logits, không cần softmax ở cuối mạng?
- Muốn chạy trên GPU cần đổi những gì?
