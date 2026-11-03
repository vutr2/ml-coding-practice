# 11. K-Fold Cross-Validation Split

**Độ khó:** Medium · **Liên quan:** Bài 12 Train/test & leakage · **Chấm bài:** `python run.py 11`

## Đề bài

Viết hàm `kfold_indices(n, k, shuffle=False, seed=0)` chia chỉ số `0..n-1` thành k fold, giống `KFold` của scikit-learn. Trả về **list** gồm k cặp `(train_idx, val_idx)` (mảng NumPy số nguyên).

- Kích thước fold: `n % k` fold đầu có `n // k + 1` phần tử, các fold sau có `n // k` phần tử.
- Fold thứ i là tập validation, các fold còn lại ghép thành tập train.
- `shuffle=False`: các fold là các đoạn liên tiếp theo thứ tự `0, 1, ..., n-1`.
- `shuffle=True`: trước khi chia, xáo trộn bằng `np.random.default_rng(seed).permutation(n)`.

## Ví dụ

```
kfold_indices(10, 3)
fold 0: val = [0, 1, 2, 3]   train = [4, 5, 6, 7, 8, 9]
fold 1: val = [4, 5, 6]      train = [0, 1, 2, 3, 7, 8, 9]
fold 2: val = [7, 8, 9]      train = [0, 1, 2, 3, 4, 5, 6]
```

## Gợi ý

<details><summary>Gợi ý</summary>

Tạo mảng kích thước fold `sizes`, rồi dùng biến `start` chạy dọc mảng chỉ số. `np.concatenate` để ghép phần trước và sau fold validation.
</details>

## Câu hỏi phỏng vấn thêm

- Vì sao dùng cross-validation thay vì một lần chia train/val?
- Với dữ liệu chuỗi thời gian, vì sao không được shuffle? Nên chia thế nào?
- Stratified K-Fold khác gì K-Fold thường? Khi nào cần?
