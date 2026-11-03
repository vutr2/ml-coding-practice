# 02. Matrix Multiplication

**Độ khó:** Easy · **Liên quan:** Bài 2 Ma trận · **Chấm bài:** `python run.py 2`

## Đề bài

Viết hàm `matmul(A, B)` nhân ma trận `A` (m × n) với `B` (n × p), trả về ma trận m × p:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

Nếu số cột của `A` khác số hàng của `B` thì `raise ValueError`.

Mọi lớp `Linear` trong mạng nơ-ron đều là một phép nhân ma trận. Hiểu rõ shape `(m, n) @ (n, p) -> (m, p)` giúp bạn đọc được code deep learning.

## Ví dụ

```
A = [[1, 2, 3],        B = [[7,  8],
     [4, 5, 6]]             [9, 10],
                            [11, 12]]
matmul(A, B) -> [[ 58,  64],
                 [139, 154]]
```

## Ràng buộc

- Không dùng `@`, `np.dot`, `np.matmul`, `np.einsum`, `np.tensordot`.
- Được dùng vòng lặp, phép nhân từng phần tử và `np.sum`.

## Gợi ý

<details><summary>Gợi ý 1</summary>

Cách cơ bản: 3 vòng lặp lồng nhau theo i, j, k.
</details>

<details><summary>Gợi ý 2</summary>

Cách nhanh hơn: phần tử `C[i, j]` là tích vô hướng của hàng i của A và cột j của B: `np.sum(A[i, :] * B[:, j])`. Bạn còn có thể bỏ hết vòng lặp bằng broadcasting: `A[:, :, None] * B[None, :, :]` có shape (m, n, p).
</details>

## Câu hỏi phỏng vấn thêm

- Độ phức tạp của phép nhân ma trận n × n là bao nhiêu?
- Vì sao GPU nhân ma trận nhanh hơn CPU rất nhiều?
