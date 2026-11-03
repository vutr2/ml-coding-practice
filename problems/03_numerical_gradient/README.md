# 03. Numerical Gradient

**Độ khó:** Easy · **Liên quan:** Bài 3 Đạo hàm & gradient · **Chấm bài:** `python run.py 3`

## Đề bài

Viết hàm `numerical_gradient(f, x, h=1e-5)` tính gradient của hàm `f: R^n -> R` tại điểm `x` bằng **sai phân trung tâm**:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h}$$

Trong đó $e_i$ là vector có số 1 ở vị trí i, còn lại là 0.

- `x` có thể là mảng 1 chiều hoặc nhiều chiều (ví dụ ma trận trọng số). Kết quả có cùng shape với `x`.
- Không được làm thay đổi `x` của người gọi.

Đây là công cụ **gradient checking**: kỹ sư dùng nó để kiểm tra code backprop tự viết có đúng không (bạn sẽ dùng lại ở bài 25–27).

## Ví dụ

```
f(x) = x0^2 + x1^2,  x = [3, 4]   ->  [6, 8]
```

## Ràng buộc

- Dùng sai phân trung tâm, không dùng sai phân một phía `(f(x+h) - f(x)) / h`.

## Gợi ý

<details><summary>Gợi ý 1</summary>

Tạo bản sao `x = np.array(x, dtype=float)` rồi duyệt từng phần tử. Với mảng nhiều chiều, dùng `np.ndindex(x.shape)` để lấy từng chỉ số.
</details>

<details><summary>Gợi ý 2</summary>

Với mỗi chỉ số i: lưu giá trị cũ, cộng h rồi tính f, trừ h rồi tính f, cuối cùng **trả lại giá trị cũ**.
</details>

## Câu hỏi phỏng vấn thêm

- Vì sao sai phân trung tâm chính xác hơn sai phân một phía? (Sai số O(h²) so với O(h).)
- Vì sao không dùng gradient số để huấn luyện mạng thật? (Mỗi tham số cần 2 lần gọi f: mạng 1 tỷ tham số cần 2 tỷ lần forward.)
