# 10. One-Hot Encoding

**Độ khó:** Easy · **Liên quan:** Bài 11 Feature engineering · **Chấm bài:** `python run.py 10`

## Đề bài

Viết hàm `one_hot(labels, num_classes=None)`:
- `labels`: mảng số nguyên từ 0 đến K − 1, độ dài N.
- `num_classes`: số lớp K. Nếu `None` thì K = `max(labels) + 1`.
- Trả về ma trận `float` shape `(N, K)`, hàng i có số 1 ở cột `labels[i]`, còn lại là 0.
- Nếu có nhãn âm hoặc nhãn `>= num_classes` thì `raise ValueError`.

## Ví dụ

```
one_hot([0, 2, 1, 2]) -> [[1, 0, 0],
                          [0, 0, 1],
                          [0, 1, 0],
                          [0, 0, 1]]
```

## Gợi ý

<details><summary>Gợi ý</summary>

Tạo ma trận 0 rồi dùng fancy indexing: `out[np.arange(N), labels] = 1`.
</details>

## Câu hỏi phỏng vấn thêm

- Vì sao không đưa thẳng nhãn 0, 1, 2 vào mô hình tuyến tính cho đặc trưng dạng loại (ví dụ thành phố)?
- Khi một cột có 100.000 giá trị khác nhau thì one-hot gặp vấn đề gì? Có cách nào khác? (Embedding, target encoding, hashing.)
