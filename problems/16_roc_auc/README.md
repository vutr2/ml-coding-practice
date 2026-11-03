# 16. ROC AUC

**Độ khó:** Medium · **Liên quan:** Bài 14 Logistic regression (ROC) · **Chấm bài:** `python run.py 16`

## Đề bài

Viết hàm `roc_auc(y_true, scores)` tính diện tích dưới đường ROC.

Cách hiểu dễ nhất: **AUC là xác suất một mẫu dương chọn ngẫu nhiên được chấm điểm cao hơn một mẫu âm chọn ngẫu nhiên.**

$$\text{AUC} = \frac{1}{P \cdot N}\sum_{i \in \text{dương}} \sum_{j \in \text{âm}} \Big( [s_i > s_j] + \tfrac{1}{2}[s_i = s_j] \Big)$$

- Hoà điểm tính 0.5.
- Nếu `y_true` chỉ có một lớp thì `raise ValueError`.

## Ví dụ

```
y_true = [0, 0, 1, 1], scores = [0.1, 0.4, 0.35, 0.8]  ->  0.75
```
Có 4 cặp (dương, âm): (0.35, 0.1) đúng, (0.35, 0.4) sai, (0.8, 0.1) đúng, (0.8, 0.4) đúng → 3/4.

## Gợi ý

<details><summary>Gợi ý 1</summary>

Tách `pos = scores[y == 1]`, `neg = scores[y == 0]`. Dùng broadcasting `pos[:, None] > neg[None, :]` để so mọi cặp một lúc.
</details>

<details><summary>Gợi ý 2: cách O(n log n)</summary>

Xếp hạng tất cả điểm (hoà thì lấy hạng trung bình). Khi đó $\text{AUC} = \frac{\sum \text{hạng dương} - P(P+1)/2}{P \cdot N}$ (Mann–Whitney U).
</details>

## Câu hỏi phỏng vấn thêm

- AUC = 0.5 nghĩa là gì? AUC = 0.3 thì sao?
- Vì sao AUC không phụ thuộc vào ngưỡng phân loại?
- Khi dữ liệu rất mất cân bằng, vì sao người ta hay xem PR-AUC thay vì ROC-AUC?
