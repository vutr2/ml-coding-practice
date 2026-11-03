# 18. K-Nearest Neighbors

**Độ khó:** Easy · **Liên quan:** Phần 3 ML cổ điển · **Chấm bài:** `python run.py 18`

## Đề bài

Viết hàm `knn_predict(X_train, y_train, X_test, k)`:
- Với mỗi điểm trong `X_test`, tìm `k` điểm gần nhất trong `X_train` theo **khoảng cách Euclid**.
- Dự đoán nhãn xuất hiện nhiều nhất trong k láng giềng (nhãn là số nguyên không âm).
- Nếu nhiều nhãn hoà số phiếu, chọn **nhãn nhỏ nhất**.
- Nếu khoảng cách bằng nhau, ưu tiên điểm có **chỉ số nhỏ hơn** trong `X_train` (dùng sort ổn định).

Trả về mảng nhãn số nguyên, độ dài bằng số điểm test.

## Ví dụ

```
X_train = [[0, 0], [1, 0], [5, 5], [6, 5]],  y_train = [0, 0, 1, 1]
knn_predict(X_train, y_train, [[0.5, 0], [5.5, 5]], k=2)  ->  [0, 1]
```

## Ràng buộc

- Tính ma trận khoảng cách bằng broadcasting, không lặp hai vòng theo cặp điểm.

## Gợi ý

<details><summary>Gợi ý 1</summary>

`D = np.sqrt(((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(-1))` có shape (n_test, n_train).
</details>

<details><summary>Gợi ý 2</summary>

`np.argsort(D, axis=1, kind="stable")[:, :k]` lấy chỉ số k láng giềng. `np.bincount(labels).argmax()` trả về nhãn nhiều phiếu nhất, hoà thì lấy nhãn nhỏ nhất.
</details>

## Câu hỏi phỏng vấn thêm

- k nhỏ thì overfit hay underfit? k lớn thì sao?
- Độ phức tạp khi dự đoán là bao nhiêu? Làm sao tăng tốc? (KD-tree, approximate nearest neighbor như FAISS, HNSW.)
- "Curse of dimensionality" ảnh hưởng đến KNN thế nào?
