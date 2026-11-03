# 23. K-Means Clustering

**Độ khó:** Medium · **Liên quan:** Phần 3 ML cổ điển (học không giám sát) · **Chấm bài:** `python run.py 23`

## Đề bài

Viết hàm `kmeans(X, init_centroids, n_iters=100)` trả về `(centroids, labels)`.

Lặp tối đa `n_iters` lần:
1. **Gán nhãn**: mỗi điểm gán vào tâm gần nhất (khoảng cách Euclid). Hoà thì chọn tâm có chỉ số nhỏ hơn.
2. **Cập nhật tâm**: mỗi tâm là trung bình các điểm được gán cho nó. Cụm **rỗng** thì giữ nguyên tâm cũ.
3. Nếu các tâm không đổi thì dừng sớm.

`labels` trả về phải là phép gán ứng với **các tâm cuối cùng**. Không làm thay đổi `init_centroids`.

## Ví dụ

```
X = [[0, 0], [0, 1], [10, 10], [10, 11]]
init = [[0, 0], [10, 10]]
-> centroids = [[0, 0.5], [10, 10.5]],  labels = [0, 0, 1, 1]
```

## Gợi ý

<details><summary>Gợi ý</summary>

Ma trận khoảng cách bình phương shape (N, K): `((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)`, rồi `argmin(axis=1)`.
</details>

## Câu hỏi phỏng vấn thêm

- K-means tối thiểu hoá hàm mục tiêu nào? Vì sao thuật toán chắc chắn hội tụ?
- Kết quả phụ thuộc vào khởi tạo. K-means++ khởi tạo thế nào?
- Chọn K thế nào? (Elbow, silhouette.)
- Khi nào K-means thất bại? (Cụm không có dạng cầu, kích thước rất khác nhau.)
