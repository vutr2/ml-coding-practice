# 35. Sinusoidal Positional Encoding

**Độ khó:** Easy · **Liên quan:** Bài 27 Transformer · **Chấm bài:** `python run.py 35`

## Đề bài

Viết hàm `positional_encoding(T, d_model)` trả về ma trận (T, d_model) theo bài báo "Attention Is All You Need" (giả sử `d_model` chẵn):

$$PE[pos, 2i] = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right), \qquad PE[pos, 2i+1] = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Cột chẵn dùng sin, cột lẻ dùng cos, mỗi cặp cột (2i, 2i+1) dùng chung một tần số.

## Ví dụ

```
positional_encoding(2, 4) ->
[[0,        1,        0,       1      ],
 [sin(1),   cos(1),   sin(0.01), cos(0.01)]]
```

## Gợi ý

<details><summary>Gợi ý</summary>

`pos = np.arange(T)[:, None]`, `i = np.arange(0, d_model, 2)`, `angle = pos / 10000 ** (i / d_model)` có shape (T, d_model/2). Gán `pe[:, 0::2] = np.sin(angle)`, `pe[:, 1::2] = np.cos(angle)`.
</details>

## Câu hỏi phỏng vấn thêm

- Vì sao Transformer cần positional encoding? (Attention không biết thứ tự, hoán vị đầu vào thì đầu ra hoán vị theo.)
- Tính chất hay: $PE[p] \cdot PE[p+k]$ chỉ phụ thuộc k, không phụ thuộc p. Vì sao? (Test cuối kiểm tra điều này.)
- RoPE (rotary position embedding) mà LLaMA dùng khác gì?
