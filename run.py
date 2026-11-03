#!/usr/bin/env python3
"""Trình chấm bài ML coding, dùng giống nút "Submit" của NeetCode.

    python run.py              bảng tiến độ tất cả các bài
    python run.py 5            chấm bài số 5
    python run.py softmax      chấm bài theo tên (một phần tên là đủ)
    python run.py 5 --solution xem lời giải mẫu bài 5 (chỉ khi đã bí thật sự)
    python run.py --readme     ghi bảng tiến độ vào README.md
    python run.py --ref        chấm bằng lời giải mẫu (kiểm tra bộ test)
    python run.py --ci         chế độ cho GitHub Actions
"""
import argparse
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback
import warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
PROBLEMS_DIR = os.path.join(ROOT, "problems")
README = os.path.join(ROOT, "README.md")
START, END = "<!-- PROGRESS:START -->", "<!-- PROGRESS:END -->"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ACCEPTED, WRONG, TODO, RUNTIME, COMPILE, SKIPPED = (
    "accepted", "wrong", "todo", "runtime", "compile", "skipped")
LABEL = {
    ACCEPTED: "✅ Accepted",
    WRONG: "❌ Wrong Answer",
    TODO: "⬜ Chưa làm",
    RUNTIME: "💥 Runtime Error",
    COMPILE: "🛑 Compile Error",
    SKIPPED: "⚠️ Cần cài torch",
}


def load_index():
    with open(os.path.join(PROBLEMS_DIR, "index.json"), encoding="utf-8") as f:
        return json.load(f)


def find_problem(index, query):
    if query.isdigit():
        for p in index:
            if p["id"] == int(query):
                return p
    q = query.lower().replace("-", "_")
    matches = [p for p in index if q in p["slug"] or q in p["title"].lower()]
    if len(matches) == 1:
        return matches[0]
    if matches:
        names = ", ".join(f'{p["id"]:02d} {p["slug"]}' for p in matches)
        sys.exit(f"Có nhiều bài khớp '{query}': {names}")
    sys.exit(f"Không tìm thấy bài '{query}'. Gõ `python run.py` để xem danh sách.")


def short_error(exc, user_only=True):
    """Lấy vài dòng traceback, ưu tiên dòng nằm trong solution.py."""
    tb = traceback.extract_tb(exc.__traceback__)
    frames = [fr for fr in tb if fr.filename.endswith(("solution.py",))] or tb[-1:]
    fr = frames[-1]
    where = f"{os.path.relpath(fr.filename, ROOT)}:{fr.lineno}"
    return f"{type(exc).__name__}: {exc}  (tại {where})"


def trim(msg, n=12):
    lines = [l for l in str(msg).strip().splitlines() if l.strip()]
    if len(lines) > n:
        lines = lines[:n] + ["..."]
    return "\n".join(lines)


def judge(problem, verbose=False):
    """Chạy toàn bộ test của một bài. Trả về (status, số test đúng, tổng, chi tiết)."""
    warnings.simplefilter("ignore", RuntimeWarning)  # chỉ báo kết quả, không in cảnh báo tràn số
    folder = os.path.join(PROBLEMS_DIR, problem["folder"])
    test_path = os.path.join(folder, "test_solution.py")
    spec = importlib.util.spec_from_file_location(f"test_{problem['folder']}", test_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except NotImplementedError:
        return TODO, 0, 0, []
    except ModuleNotFoundError as e:
        if e.name == "torch":
            return SKIPPED, 0, 0, ["Bài này cần PyTorch: pip install torch"]
        return COMPILE, 0, 0, [short_error(e)]
    except SyntaxError as e:
        return COMPILE, 0, 0, [f"SyntaxError: {e.msg} (dòng {e.lineno})"]
    except Exception as e:
        return COMPILE, 0, 0, [short_error(e)]

    tests = [(name, fn) for name, fn in vars(module).items()
             if name.startswith("test_") and inspect.isfunction(fn)]
    passed, details, status = 0, [], ACCEPTED
    for name, fn in tests:
        label = (fn.__doc__ or name).strip().splitlines()[0]
        t0 = time.perf_counter()
        try:
            fn()
            ms = (time.perf_counter() - t0) * 1000
            passed += 1
            details.append(f"  ✓ {label:<46} {ms:7.1f} ms")
        except NotImplementedError:
            return TODO, 0, len(tests), []
        except AssertionError as e:
            if status == ACCEPTED:
                status = WRONG
            details.append(f"  ✗ {label}")
            msg = trim(e) or "Kết quả không khớp đáp án."
            details.append("      " + msg.replace("\n", "\n      "))
        except Exception as e:
            status = RUNTIME
            details.append(f"  ✗ {label}")
            details.append("      " + short_error(e))
    return status, passed, len(tests), details


def cmd_one(problem, show_solution=False):
    title = f"{problem['id']:02d}. {problem['title']}  [{problem['difficulty']}]"
    print(f"\n━━ {title} ━━")
    print(f"   Đề bài: problems/{problem['folder']}/README.md")
    if show_solution:
        path = os.path.join(ROOT, "_solutions", problem["folder"] + ".py")
        if not os.path.exists(path):
            sys.exit("Không có lời giải mẫu trong thư mục _solutions/.")
        print("\n--- Lời giải mẫu ---\n")
        with open(path, encoding="utf-8") as f:
            print(f.read())
        return
    status, passed, total, details = judge(problem)
    print()
    for line in details:
        print(line)
    print()
    if status == TODO:
        print("Kết quả: ⬜ Chưa làm. Mở solution.py, xoá dòng `raise NotImplementedError` và viết code.")
    elif status in (COMPILE, SKIPPED):
        print(f"Kết quả: {LABEL[status]}")
    else:
        print(f"Kết quả: {LABEL[status]}  ({passed}/{total} test đúng)")
    if status == ACCEPTED:
        print("Làm tốt lắm! Chạy `python run.py --readme` để cập nhật tiến độ rồi commit.")


def run_all(index):
    results = []
    for p in index:
        status, passed, total, _ = judge(p)
        results.append((p, status, passed, total))
    return results


def cmd_table(index):
    results = run_all(index)
    solved = sum(1 for r in results if r[1] == ACCEPTED)
    width = 30
    bar = "█" * round(width * solved / len(index)) + "░" * (width - round(width * solved / len(index)))
    part = None
    print()
    for p, status, passed, total in results:
        if p["part"] != part:
            part = p["part"]
            print(f"\n  {part}")
        print(f"   {p['id']:02d}  {p['title']:<44} {p['difficulty']:<7} {LABEL[status]}")
    print(f"\n  Tiến độ: {bar} {solved}/{len(index)}")
    nxt = next((r[0] for r in results if r[1] != ACCEPTED), None)
    if nxt:
        print(f"  Bài tiếp theo: python run.py {nxt['id']}   ({nxt['title']})\n")
    else:
        print("  Bạn đã giải hết! 🎉\n")


def progress_markdown(results):
    solved = sum(1 for r in results if r[1] == ACCEPTED)
    lines = [f"**Đã giải: {solved}/{len(results)}**", "",
             "| # | Bài | Độ khó | Bài học | Trạng thái |",
             "|---|-----|--------|---------|------------|"]
    for p, status, _, _ in results:
        link = f"[{p['title']}](problems/{p['folder']}/README.md)"
        lines.append(f"| {p['id']:02d} | {link} | {p['difficulty']} | {p['lesson']} | {LABEL[status]} |")
    return "\n".join(lines)


def cmd_readme(index):
    results = run_all(index)
    with open(README, encoding="utf-8") as f:
        text = f.read()
    a, b = text.index(START) + len(START), text.index(END)
    text = text[:a] + "\n" + progress_markdown(results) + "\n" + text[b:]
    with open(README, "w", encoding="utf-8") as f:
        f.write(text)
    solved = sum(1 for r in results if r[1] == ACCEPTED)
    print(f"Đã cập nhật README.md: {solved}/{len(index)} bài Accepted.")


def cmd_ci(index):
    results = run_all(index)
    md = "## Tiến độ ML coding\n\n" + progress_markdown(results)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    print(md)
    bad = [r for r in results if r[1] in (WRONG, RUNTIME, COMPILE)]
    if bad:
        print("\nCác bài đã làm nhưng chưa Accepted:")
        for p, status, _, _ in bad:
            print(f"  {p['id']:02d} {p['title']}: {LABEL[status]}")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="Trình chấm bài ML coding")
    ap.add_argument("problem", nargs="?", help="số thứ tự hoặc tên bài")
    ap.add_argument("--solution", action="store_true", help="xem lời giải mẫu")
    ap.add_argument("--readme", action="store_true", help="ghi tiến độ vào README.md")
    ap.add_argument("--ref", action="store_true", help="chấm bằng lời giải mẫu")
    ap.add_argument("--ci", action="store_true", help="chế độ GitHub Actions")
    args = ap.parse_args()

    if args.ref:
        if not os.path.isdir(os.path.join(ROOT, "_solutions")):
            sys.exit("Không có thư mục _solutions/.")
        os.environ["ML_REF"] = "1"
    index = load_index()
    if args.ci:
        cmd_ci(index)
    elif args.readme:
        cmd_readme(index)
    elif args.problem:
        cmd_one(find_problem(index, args.problem), args.solution)
    else:
        cmd_table(index)


if __name__ == "__main__":
    main()
