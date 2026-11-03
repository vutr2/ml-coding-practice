# 05. Stable Softmax

**Độ khó:** Easy · **Liên quan:** Bài 5 Xác suất · **Chấm bài:** `python run.py 5`

## Đề bài

Viết hàm `softmax(z)` biến vector điểm số (logits) thành phân phối xác suất:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

- `z` có thể là mảng 1 chiều (một mẫu) hoặc 2 chiều (mỗi **hàng** là một mẫu, tính softmax theo hàng).
- Phải **ổn định số học**: với `z = [1000, 1001, 1002]` không được trả về `nan`.

## Ví dụ

```
softmax([1, 2, 3])          -> [0.0900, 0.2447, 0.6652]
softmax([1000, 1001, 1002]) -> [0.0900, 0.2447, 0.6652]
```

## Gợi ý

<details><summary>Gợi ý 1</summary>

`np.exp(1000)` bị tràn số thành `inf`, và `inf / inf = nan`.
</details>

<details><summary>Gợi ý 2</summary>

Softmax không đổi khi cộng cùng một số vào mọi phần tử: $\text{softmax}(z) = \text{softmax}(z - c)$. Chọn $c = \max(z)$ thì số mũ lớn nhất là $e^0 = 1$. Với mảng 2 chiều dùng `z.max(axis=-1, keepdims=True)`.
</details>

## Câu hỏi phỏng vấn thêm

- Chứng minh softmax không đổi khi trừ đi một hằng số.
- "Temperature" trong LLM tác động lên softmax thế nào? (Chia logits cho T trước khi softmax.)
