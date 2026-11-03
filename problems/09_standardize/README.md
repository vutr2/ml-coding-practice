# 09. Standard Scaler (no leakage)

**Độ khó:** Easy · **Liên quan:** Bài 11–12 Feature engineering & leakage · **Chấm bài:** `python run.py 9`

## Đề bài

Viết hai hàm giống `StandardScaler` của scikit-learn:

1. `fit_standardizer(X_train)`: trả về `(mean, std)` của từng cột (std chia cho N, tức `ddof=0`).
2. `transform(X, mean, std)`: trả về $(X - \text{mean}) / \text{std}$. Cột có `std == 0` (cột hằng số) thì coi std là 1 để tránh chia cho 0.

Điểm mấu chốt: **chỉ fit trên tập train**, rồi dùng chính mean/std đó để transform tập test. Nếu fit trên cả tập test là bị **data leakage**.

## Ví dụ

```
X_train = [[1, 10],        mean = [2, 10]
           [2, 10],        std  = [0.8165, 0]   (cột 2 là hằng số)
           [3, 10]]
transform(X_train) = [[-1.2247, 0], [0, 0], [1.2247, 0]]
```

## Gợi ý

<details><summary>Gợi ý</summary>

`X.mean(axis=0)`, `X.std(axis=0)`, rồi `np.where(std == 0, 1.0, std)`.
</details>

## Câu hỏi phỏng vấn thêm

- Data leakage là gì? Cho một ví dụ ngoài việc chuẩn hoá.
- Mô hình nào cần chuẩn hoá đặc trưng (KNN, SVM, gradient descent) và mô hình nào không cần (cây quyết định)? Vì sao?
