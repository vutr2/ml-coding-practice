# 22. PCA from Scratch

**Độ khó:** Medium · **Liên quan:** Bài 19 PCA · **Chấm bài:** `python run.py 22`

## Đề bài

Viết hàm `pca(X, k)` trả về bộ ba `(components, explained_variance, X_proj)`:

1. Trừ trung bình mỗi cột: $X_c = X - \bar{X}$.
2. Ma trận hiệp phương sai: $\Sigma = \frac{1}{N-1} X_c^\top X_c$.
3. Lấy k vector riêng ứng với k trị riêng **lớn nhất**, sắp xếp giảm dần.
4. **Quy ước dấu**: với mỗi vector riêng, đổi dấu sao cho phần tử có trị tuyệt đối lớn nhất là số dương (vì v và −v đều là vector riêng).
5. Trả về:
   - `components`: shape `(k, d)`, mỗi hàng là một thành phần chính.
   - `explained_variance`: shape `(k,)`, các trị riêng tương ứng.
   - `X_proj`: shape `(N, k)`, bằng $X_c \cdot \text{components}^\top$.

## Ràng buộc

- Không dùng scikit-learn. Được dùng `np.linalg.eigh` hoặc `np.linalg.svd`.

## Gợi ý

<details><summary>Gợi ý 1</summary>

`np.linalg.eigh` trả về trị riêng **tăng dần** và vector riêng ở **cột**. Đảo ngược thứ tự rồi chuyển vị.
</details>

<details><summary>Gợi ý 2: quy ước dấu</summary>

`idx = np.argmax(np.abs(comps), axis=1)`, `signs = np.sign(comps[np.arange(k), idx])`, `comps *= signs[:, None]`.
</details>

## Câu hỏi phỏng vấn thêm

- Vì sao phải trừ trung bình trước khi làm PCA?
- PCA và SVD liên hệ thế nào?
- Chọn k thế nào? (Tỉ lệ phương sai giải thích được, ví dụ giữ 95%.)
- PCA khác t-SNE/UMAP ở điểm nào?
