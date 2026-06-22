# Hướng dẫn biên dịch

## Yêu cầu

- **TeX distribution**: TeX Live 2020+ (có `xelatex`, `bibtex`)
- **Font**: Times New Roman (hoặc `fontspec` sẽ fallback)

## Cách biên dịch

### Cách 1 — Dùng script (khuyên dùng)

```bash
./compile.sh           # biên dịch cả 3 bản (main, baocao_tung, baocao_phuc)
./compile.sh main      # chỉ biên dịch bản chính (main.tex)
./compile.sh tung      # chỉ bản của Tùng
./compile.sh phuc      # chỉ bản của Phúc
```

### Cách 2 — Thủ công (nếu script lỗi)

```bash
xelatex main
bibtex  main
xelatex main
xelatex main
```

### Cách 3 — Overleaf

Upload toàn bộ thư mục `baocao_thuctap/` lên Overleaf. Sau đó:

1. Overleaf → **Menu** → **Main document** → chọn `main.tex`
2. Overleaf → **Menu** → **Compiler** → đổi thành **XeLaTeX**
3. Nhấn **Recompile**

Dòng `% !TEX program = xelatex` đầu `main.tex` cũng tự báo Overleaf dùng XeLaTeX, nhưng đổi tay trong Menu là chắc chắn nhất.

## Cấu trúc file

```
main.tex                     ← entry point chính (dùng \documentclass{baocaothuctap})
baocaothuctap.cls            ← class file (preamble, packages, định dạng)
baocao_body.tex              ← input các file con bên dưới

baocao_phuc.tex              ← bản của Phúc (có trang bìa PDF)
baocao_tung.tex              ← bản của Tùng (có trang bìa PDF)

frontmatter/
├── loicamon.tex
├── mucluc.tex
├── danhsach_hinh.tex
├── danhsach_bang.tex
└── danhmuc_tuviettat.tex

chapters/
├── chuong1.tex              ← Tổng quan
├── chuong2.tex              ← Cơ sở lý thuyết
├── chuong3.tex              ← Phương pháp thực hiện
├── chuong4.tex              ← Thực nghiệm & đánh giá
└── chuong5.tex              ← Kết luận

backmatter/
├── phuluc_a.tex
└── tailieu_thamkhao.tex

bib.bib                      ← BibTeX database
```

## Debug

- Mở file `.tex` tương ứng trong `frontmatter/`, `chapters/`, `backmatter/` để sửa.
- File `baocao_body.tex` chỉ chứa `\input{}` — khỏi đụng vào.
- File `baocaothuctap.cls` chứa toàn bộ định dạng, font, packages.
- Chạy `xelatex` 2 lần (3 nếu có mục lục/bib) để cập nhật cross-reference.
