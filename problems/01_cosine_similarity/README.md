# 01. Cosine Similarity

**Độ khó:** Easy · **Liên quan:** Bài 1 Vector · **Chấm bài:** `python run.py 1`

## Đề bài

Viết hàm `cosine_similarity(a, b)` trả về độ tương đồng cosine giữa hai vector:

$$\cos(a, b) = \frac{a \cdot b}{\|a\| \, \|b\|}$$

Nếu một trong hai vector là vector 0 thì trả về `0.0`.

Đây là phép đo dùng trong tìm kiếm bằng embedding và RAG: hai câu có nghĩa gần nhau thì embedding của chúng có cosine gần 1.

## Ví dụ

```
cosine_similarity([1, 2, 3], [4, 5, 6])   ->  0.9746
cosine_similarity([1, 0], [0, 1])         ->  0.0     (vuông góc)
cosine_similarity([1, 2], [-2, -4])       -> -1.0     (ngược chiều)
```

## Ràng buộc

- Không dùng `np.dot`, `@`, `np.linalg.norm`. Tự tính tích vô hướng bằng phép nhân từng phần tử và `np.sum` (hoặc vòng lặp).
- Đầu vào có thể là list hoặc mảng NumPy 1 chiều, cùng độ dài.

## Gợi ý

<details><summary>Gợi ý 1</summary>

Tích vô hướng là tổng của tích từng cặp phần tử: `sum(a[i] * b[i])`.
</details>

<details><summary>Gợi ý 2</summary>

Độ dài vector: `||a|| = sqrt(a · a)`. Kiểm tra mẫu số bằng 0 trước khi chia.
</details>

## Câu hỏi phỏng vấn thêm

- Vì sao tìm kiếm embedding hay dùng cosine thay vì khoảng cách Euclid?
- Nếu mọi vector đã được chuẩn hoá về độ dài 1 thì cosine bằng gì? (Đây là lý do nhiều vector database chỉ cần tính tích vô hướng.)
