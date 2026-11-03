# 31. Conv2D & MaxPool from Scratch

**Độ khó:** Medium · **Liên quan:** Bài 25 CNN · **Chấm bài:** `python run.py 31`

## Đề bài

**1. `conv2d(x, w, b, stride=1, padding=0)`**
- `x`: ảnh shape `(C_in, H, W)`; `w`: bộ lọc shape `(C_out, C_in, kH, kW)`; `b`: shape `(C_out,)`.
- Đệm `padding` số 0 quanh ảnh, rồi trượt bộ lọc với bước `stride`.
- Giống PyTorch, đây là **cross-correlation** (không lật kernel):
$$\text{out}[o, i, j] = b_o + \sum_{c, u, v} x_{\text{pad}}[c,\; i\cdot s + u,\; j \cdot s + v] \cdot w[o, c, u, v]$$
- Kích thước đầu ra: $H_{out} = \lfloor (H + 2P - kH)/S \rfloor + 1$ (tương tự cho W).

**2. `maxpool2d(x, size=2, stride=2)`**: `x` shape `(C, H, W)`, lấy max trong mỗi cửa sổ `size × size`.

## Ví dụ

```
x = [[1, 2, 3],      w = [[1,  0],      conv2d -> [[-4, -4],
     [4, 5, 6],           [0, -1]]                  [-4, -4]]
     [7, 8, 9]]   (1 kênh vào, 1 kênh ra, b = 0)

maxpool2d([[1, 2, 5, 6],
           [3, 4, 7, 8],      ->  [[ 4,  8],
           [9, 10, 13, 14],        [12, 16]]
           [11, 12, 15, 16]])
```

## Gợi ý

<details><summary>Gợi ý 1</summary>

`np.pad(x, ((0, 0), (padding, padding), (padding, padding)))` đệm số 0, không đệm theo trục kênh.
</details>

<details><summary>Gợi ý 2</summary>

Hai vòng lặp theo (i, j) là đủ: cắt `patch = xp[:, i*s:i*s+kH, j*s:j*s+kW]` (shape `(C_in, kH, kW)`), rồi `out[:, i, j] = (w * patch).sum(axis=(1, 2, 3)) + b`.
</details>

## Câu hỏi phỏng vấn thêm

- Số tham số của một lớp conv `(C_out, C_in, k, k)` là bao nhiêu? So với lớp Linear nối toàn bộ ảnh?
- Receptive field là gì? Vì sao xếp nhiều conv 3×3 tốt hơn một conv 7×7?
- Thư viện thật tính conv nhanh thế nào? (im2col biến conv thành nhân ma trận, hoặc FFT, Winograd.)
