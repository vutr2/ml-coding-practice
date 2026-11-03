# 08. Clean a DataFrame (Pandas)

**Độ khó:** Easy · **Liên quan:** Bài 9–10 Pandas & làm sạch dữ liệu · **Chấm bài:** `python run.py 8`

## Đề bài

Viết hàm `clean(df)` nhận một `pandas.DataFrame` và trả về DataFrame **mới** đã làm sạch:

1. Xoá các hàng trùng lặp hoàn toàn (giữ lần xuất hiện đầu tiên).
2. Cột số (`int`, `float`): điền giá trị thiếu (NaN) bằng **median** của cột đó (median tính sau bước 1).
3. Cột chữ (`object`): điền giá trị thiếu bằng chuỗi `"unknown"`.
4. Đánh lại index từ 0 (`reset_index(drop=True)`).

Không được thay đổi `df` đầu vào.

## Ví dụ

```
     age  city        ->      age   city
0   25.0  Hanoi            0  25.0  Hanoi
1    NaN  HCM              1  30.0  HCM
2   35.0  NaN              2  35.0  unknown
3   25.0  Hanoi   (trùng hàng 0, bị xoá)
```

## Gợi ý

<details><summary>Gợi ý</summary>

- `df.drop_duplicates()`, `pd.api.types.is_numeric_dtype(df[col])`, `df[col].median()`, `df[col].fillna(...)`.
- Các hàm như `drop_duplicates()` trả về bản sao, nên dữ liệu gốc không bị sửa.
</details>

## Câu hỏi phỏng vấn thêm

- Khi nào điền bằng median tốt hơn mean? (Khi có outlier.)
- Vì sao phải tính median **chỉ trên tập train** rồi dùng lại cho tập test?
- Khi nào nên xoá hàng thiếu dữ liệu thay vì điền?
