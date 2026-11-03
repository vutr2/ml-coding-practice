# ML Coding Practice

![tests](https://github.com/vutr2/ml-coding-practice/actions/workflows/tests.yml/badge.svg)

36 bài tự cài đặt thuật toán Machine Learning và Deep Learning từ đầu bằng NumPy (và PyTorch), luyện theo kiểu NeetCode: mỗi bài có đề, code khung để điền và bộ test tự chấm. Chạy một lệnh là biết **Accepted** hay **Wrong Answer**.

Các bài đi theo thứ tự: Toán → Dữ liệu → ML cổ điển → Deep Learning → LLM. Đây là những dạng hay gặp ở vòng phỏng vấn ML coding cho vị trí AI / ML Engineer: softmax ổn định số học, backprop, K-means, attention, ...

## Cài đặt

```bash
git clone https://github.com/vutr2/ml-coding-practice.git
cd ml-coding-practice
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch chỉ bắt buộc cho bài 29. Nếu chưa cài, các bài khác vẫn chạy bình thường.

## Cách làm một bài

1. Đọc đề: `problems/05_softmax/README.md`
2. Viết code vào `problems/05_softmax/solution.py` (xoá dòng `raise NotImplementedError`)
3. Chấm bài:

```bash
python run.py 5            # hoặc: python run.py softmax
```

```
━━ 05. Stable Softmax  [Easy] ━━
  ✓ softmax([1, 2, 3])                              0.2 ms
  ✗ Logits rất lớn [1000, 1001, 1002] không ra nan
      Kết quả: sai giá trị
        nhận:   [nan nan nan]
        đáp án: [0.09   0.2447 0.6652]
Kết quả: ❌ Wrong Answer  (2/4 test đúng)
```

## Các lệnh

| Lệnh | Tác dụng |
|---|---|
| `python run.py` | Bảng tiến độ tất cả các bài và gợi ý bài tiếp theo |
| `python run.py 5` | Chấm bài số 5 |
| `python run.py 5 --solution` | Xem lời giải mẫu (chỉ khi đã bí thật sự) |
| `python run.py --readme` | Ghi bảng tiến độ vào README này |
| `pytest` | Chạy toàn bộ test bằng pytest |

## Quy tắc luyện tập

- **Không đọc `test_solution.py` trước khi làm.** Trên NeetCode test bị ẩn, ở đây bạn tự giữ kỷ luật.
- Tự làm 25–45 phút. Bí thì mở phần **Gợi ý** trong đề, vẫn bí mới xem lời giải mẫu.
- Làm xong, trả lời to các câu trong mục **Câu hỏi phỏng vấn thêm** như đang phỏng vấn thật.
- Lời giải mẫu nằm trong thư mục `_solutions/`, thư mục này không được đẩy lên GitHub (xem `.gitignore`), nên repo chỉ chứa code của bạn.

## Lưu tiến độ lên GitHub

Mỗi khi một bài được Accepted:

```bash
python run.py --readme
git add .
git commit -m "Solve 05 softmax"
git push
```

GitHub Actions tự chấm lại mỗi lần push. Huy hiệu ở đầu trang màu xanh khi mọi bài bạn đã làm đều Accepted, bài chưa làm không bị tính là lỗi. Bảng tiến độ của mỗi lần chấm nằm trong tab **Actions**.

## Tiến độ

<!-- PROGRESS:START -->
**Đã giải: 2/36**

| # | Bài | Độ khó | Bài học | Trạng thái |
|---|-----|--------|---------|------------|
| 01 | [Cosine Similarity](problems/01_cosine_similarity/README.md) | Easy | Bài 1 Vector | ✅ Accepted |
| 02 | [Matrix Multiplication](problems/02_matmul/README.md) | Easy | Bài 2 Ma trận | ✅ Accepted |
| 03 | [Numerical Gradient](problems/03_numerical_gradient/README.md) | Easy | Bài 3 Đạo hàm & gradient | ⬜ Chưa làm |
| 04 | [Single Neuron Backprop (Chain Rule)](problems/04_neuron_chain_rule/README.md) | Medium | Bài 4 Chain rule | ⬜ Chưa làm |
| 05 | [Stable Softmax](problems/05_softmax/README.md) | Easy | Bài 5 Xác suất | ⬜ Chưa làm |
| 06 | [Gaussian MLE](problems/06_gaussian_mle/README.md) | Easy | Bài 6 MLE | ⬜ Chưa làm |
| 07 | [Gradient Descent](problems/07_gradient_descent/README.md) | Easy | Bài 7 Gradient descent | ⬜ Chưa làm |
| 08 | [Clean a DataFrame (Pandas)](problems/08_clean_dataframe/README.md) | Easy | Bài 9–10 Pandas & làm sạch dữ liệu | ⬜ Chưa làm |
| 09 | [Standard Scaler (no leakage)](problems/09_standardize/README.md) | Easy | Bài 11–12 Feature engineering & leakage | ⬜ Chưa làm |
| 10 | [One-Hot Encoding](problems/10_one_hot/README.md) | Easy | Bài 11 Feature engineering | ⬜ Chưa làm |
| 11 | [K-Fold Cross-Validation Split](problems/11_kfold_split/README.md) | Medium | Bài 12 Train/test & leakage | ⬜ Chưa làm |
| 12 | [Linear Regression (Normal Equation)](problems/12_linear_regression_normal_equation/README.md) | Easy | Bài 13 Linear regression | ⬜ Chưa làm |
| 13 | [Linear Regression (Gradient Descent)](problems/13_linear_regression_gd/README.md) | Medium | Bài 13 Linear regression | ⬜ Chưa làm |
| 14 | [Logistic Regression from Scratch](problems/14_logistic_regression/README.md) | Medium | Bài 14 Logistic regression | ⬜ Chưa làm |
| 15 | [Confusion Matrix, Precision, Recall, F1](problems/15_classification_metrics/README.md) | Easy | Bài 14 Logistic regression (metrics) | ⬜ Chưa làm |
| 16 | [ROC AUC](problems/16_roc_auc/README.md) | Medium | Bài 14 Logistic regression (ROC) | ⬜ Chưa làm |
| 17 | [Ridge Regression](problems/17_ridge_regression/README.md) | Easy | Bài 15 Overfitting & regularization | ⬜ Chưa làm |
| 18 | [K-Nearest Neighbors](problems/18_knn/README.md) | Easy | Phần 3 ML cổ điển | ⬜ Chưa làm |
| 19 | [Decision Tree: Gini & Best Split](problems/19_decision_tree_split/README.md) | Medium | Bài 16 Cây quyết định | ⬜ Chưa làm |
| 20 | [Gradient Boosting with Stumps](problems/20_gradient_boosting_stumps/README.md) | Medium | Bài 17 Ensemble | ⬜ Chưa làm |
| 21 | [Linear SVM (Hinge Loss)](problems/21_svm_hinge_loss/README.md) | Medium | Bài 18 SVM | ⬜ Chưa làm |
| 22 | [PCA from Scratch](problems/22_pca/README.md) | Medium | Bài 19 PCA | ⬜ Chưa làm |
| 23 | [K-Means Clustering](problems/23_kmeans/README.md) | Medium | Phần 3 ML cổ điển (học không giám sát) | ⬜ Chưa làm |
| 24 | [Activation Functions & Derivatives](problems/24_activations/README.md) | Easy | Bài 20 Neuron & hàm kích hoạt | ⬜ Chưa làm |
| 25 | [Linear Layer Forward & Backward](problems/25_linear_layer_backward/README.md) | Medium | Bài 22 Backprop bằng NumPy | ⬜ Chưa làm |
| 26 | [Softmax Cross-Entropy (Forward & Backward)](problems/26_softmax_cross_entropy/README.md) | Medium | Bài 22 Backprop bằng NumPy | ⬜ Chưa làm |
| 27 | [2-Layer MLP from Scratch](problems/27_mlp_from_scratch/README.md) | Hard | Bài 21–22 MLP & Backprop | ⬜ Chưa làm |
| 28 | [SGD with Momentum & Adam](problems/28_optimizers/README.md) | Medium | Bài 23 Optimizers & Adam | ⬜ Chưa làm |
| 29 | [PyTorch Training Loop](problems/29_pytorch_training_loop/README.md) | Medium | Bài 24 PyTorch | ⬜ Chưa làm |
| 30 | [Dropout & BatchNorm](problems/30_dropout_batchnorm/README.md) | Medium | Bài 15 Regularization · Bài 25 CNN | ⬜ Chưa làm |
| 31 | [Conv2D & MaxPool from Scratch](problems/31_conv2d_maxpool/README.md) | Medium | Bài 25 CNN | ⬜ Chưa làm |
| 32 | [LSTM Cell & Sequence](problems/32_lstm_cell/README.md) | Medium | Bài 26 RNN/LSTM | ⬜ Chưa làm |
| 33 | [Scaled Dot-Product Attention](problems/33_scaled_dot_product_attention/README.md) | Medium | Bài 27 Transformer | ⬜ Chưa làm |
| 34 | [Multi-Head Self-Attention](problems/34_multi_head_attention/README.md) | Hard | Bài 27 Transformer | ⬜ Chưa làm |
| 35 | [Sinusoidal Positional Encoding](problems/35_positional_encoding/README.md) | Easy | Bài 27 Transformer | ⬜ Chưa làm |
| 36 | [LLM Sampling: Temperature, Top-k, Top-p](problems/36_llm_sampling/README.md) | Medium | Phần 5 AI Engineering (LLM) | ⬜ Chưa làm |
<!-- PROGRESS:END -->
