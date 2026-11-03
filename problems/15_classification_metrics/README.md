# 15. Confusion Matrix, Precision, Recall, F1

**Độ khó:** Easy · **Liên quan:** Bài 14 Logistic regression (metrics) · **Chấm bài:** `python run.py 15`

## Đề bài

Phân loại nhị phân, nhãn 0/1.

1. `confusion_matrix(y_true, y_pred)`: trả về mảng 2 × 2 theo bố cục của scikit-learn:
```
[[TN, FP],
 [FN, TP]]
```
2. `precision_recall_f1(y_true, y_pred)`: trả về bộ ba `(precision, recall, f1)`:
$$P = \frac{TP}{TP + FP}, \quad R = \frac{TP}{TP + FN}, \quad F_1 = \frac{2PR}{P + R}$$
Nếu mẫu số bằng 0 thì giá trị đó là `0.0`.

## Ví dụ

```
y_true = [1, 0, 1, 1, 0, 0, 1]
y_pred = [1, 0, 0, 1, 1, 0, 1]
TP = 3, FP = 1, FN = 1, TN = 2
precision = 0.75, recall = 0.75, f1 = 0.75
```

## Gợi ý

<details><summary>Gợi ý</summary>

`TP = np.sum((y_true == 1) & (y_pred == 1))`, tương tự cho FP, FN, TN.
</details>

## Câu hỏi phỏng vấn thêm

- Phát hiện ung thư thì ưu tiên precision hay recall? Lọc thư rác thì sao?
- Vì sao accuracy dễ gây hiểu nhầm khi dữ liệu mất cân bằng (ví dụ 99% mẫu âm)?
- Vì sao F1 dùng trung bình điều hoà thay vì trung bình cộng?
