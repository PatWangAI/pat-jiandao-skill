"""术语表改字（讲稿可选，第二期）：没有讲稿时，专有名词只能靠术语表改对。

术语表是 Markdown，每个词一行，以“- ”开头：
  - Claude Code：cloud code、克劳德扣的      正确写法：听错的写法（几种用顿号或逗号隔开）
  - FunASR                                  只写正确写法：大小写、空格不对时改成这样（比如 funasr → FunASR）
其余行（标题、说明）不管，所以说明文字别用“- ”开头。读哪几份：设置里 glossary 列的文件（相对仓库根目录，默认 术语表/通用.md），
再加上视频文件夹里的 术语表.md（这一条视频的，由用户给）。

只改字，不改时间、不改哪几个字进哪一行：找到的是转写稿里连续的几个字（字序号范围），整段换成正确写法。
不看上下文就改，所以每处都列进修字表，改错了改术语表重跑。做法参考 chengfeng-videocut-skills 的“词典修字、出修字表”（Apache-2.0），代码是自己写的。

用法：uv run python scripts/terms.py <视频文件夹> [runs/<时间> 文件夹，默认最新一次]
  读 <运行>/subtitles_auto.json，写 <运行>/subtitles_terms.json（每行的字按术语表改过，格式同 subtitles.json）
  和 <运行>/修字表.md（原词 → 改成 → 上下文；跨了两行的不改，单列出来）。
"""

import re
import sys
from pathlib import Path

from common import CONFIG, ROOT, join_text, latest_run, normalize, read_json, write_json


def glossary_files(folder: Path | None) -> list[Path]:
    files = []
    for f in CONFIG.get("glossary") or []:
        p = Path(str(f)).expanduser()
        files.append(p if p.is_absolute() else ROOT / p)
    if folder is not None:
        files.append(folder / "术语表.md")
    return [p for p in files if p.is_file()]


def load_glossary(files: list[Path]) -> list[tuple[str, list[str]]]:
    """[(正确写法, [听错的写法, …])]。只写了正确写法的，听错的写法就是它自己（只差大小写、空格）。"""
    out = []
    for p in files:
        for ln in p.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*[-*]\s+(.+?)\s*$", ln)
            if not m:
                continue
            parts = re.split(r"[：:]", m.group(1), maxsplit=1)
            right = parts[0].strip()
            forms = [w.strip() for w in re.split(r"[、，,]", parts[1] if len(parts) > 1 else "") if w.strip()]
            if right:
                out.append((right, forms + [right]))
    return out


def find_fixes(tok: list, glossary: list, only: list[int] | None = None) -> list[dict]:
    """在转写稿里找听错的写法。only 给了就只在这些字里找（比如只看进字幕的字），按它们的顺序拼起来找。
    返回 [{"from_token", "to_token", "wrong", "right"}]，按位置排好，重叠的只留先找到的长的那个。"""
    idx = list(range(len(tok))) if only is None else list(only)
    s, owner = "", []
    for i in idx:
        n = normalize(tok[i]["text"])
        s += n
        owner += [i] * len(n)
    hits = []
    for right, forms in glossary:
        for w in sorted({normalize(f) for f in forms if normalize(f)}, key=len, reverse=True):
            for m in re.finditer(re.escape(w), s):
                a, b = owner[m.start()], owner[m.end() - 1]
                # 只认整字：开头和结尾都要落在字的边界上（英文词不能只匹配半个）
                if (m.start() > 0 and owner[m.start() - 1] == a) or (m.end() < len(s) and owner[m.end()] == b):
                    continue
                wrong = join_text([tok[i]["text"] for i in idx[idx.index(a):idx.index(b) + 1]])
                if wrong != right:
                    hits.append({"from_token": a, "to_token": b, "wrong": wrong, "right": right})
    hits.sort(key=lambda h: (h["from_token"], -(h["to_token"] - h["from_token"])))
    out, last = [], -1
    for h in hits:
        if h["from_token"] > last:
            out.append(h)
            last = h["to_token"]
    return out


def fixed_text(idx: list[int], tok: list, fixes: list[dict], tail: str = "") -> str:
    """这几个字拼成的文字，改到的地方换成正确写法。"""
    words, k = [], 0
    starts = {f["from_token"]: f for f in fixes}
    while k < len(idx):
        f = starts.get(idx[k])
        if f:
            words.append(f["right"])
            while k < len(idx) and idx[k] <= f["to_token"]:
                k += 1
        else:
            words.append(tok[idx[k]]["text"])
            k += 1
    return join_text(words) + tail


def move_across(rows: list[dict], tok: list, fixes: list[dict]) -> int:
    """词被断在两行之间（比如“pixa｜被”）：把它在下一行的那几个字挪到上一行末尾，放不下（超过 16 个字）就把上一行的
    那几个字挪到下一行开头。每个字仍然恰好在一行里、顺序不变。返回挪了几处。"""
    from common import MAX_CHARS, text_width
    n = 0
    for f in fixes:
        k = next((k for k, r in enumerate(rows[:-1]) if r["idx"] and f["from_token"] in r["idx"]
                  and f["to_token"] > r["idx"][-1]), None)
        if k is None or not rows[k + 1]["idx"]:
            continue
        a, b = rows[k], rows[k + 1]
        head = [i for i in a["idx"] if i >= f["from_token"]]
        tail = [i for i in b["idx"] if i <= f["to_token"]]
        if len(head) + len(tail) != f["to_token"] - f["from_token"] + 1 or a["tail"]:
            continue  # 中间夹着删掉的字，或者上一行是问句：不动
        width = lambda idx: text_width(fixed_text(idx, tok, [x for x in fixes if x["from_token"] >= idx[0] and x["to_token"] <= idx[-1]]))
        if width(a["idx"] + tail) <= MAX_CHARS:
            a["idx"], b["idx"] = a["idx"] + tail, b["idx"][len(tail):]
        elif width(head + b["idx"]) <= MAX_CHARS:
            a["idx"], b["idx"] = a["idx"][:-len(head)], head + b["idx"]
        else:
            continue
        n += 1
    return n


def context(tok: list, f: dict, k: int = 6) -> str:
    left = join_text([x["text"] for x in tok[max(0, f["from_token"] - k):f["from_token"]]])
    right = join_text([x["text"] for x in tok[f["to_token"] + 1:f["to_token"] + 1 + k]])
    return f"……{left}【{f['wrong']}】{right}……"


def main() -> None:
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    folder = Path(sys.argv[1]).expanduser().resolve()
    run = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else latest_run(folder)
    files = glossary_files(folder)
    glossary = load_glossary(files)
    edl = read_json(run / "edl_final.json")
    tok = read_json(folder / edl["transcript"])["tokens"]
    from make_subtitles import line_tokens, subtitle_tokens
    subs = subtitle_tokens(edl, len(tok))
    lines = read_json(run / "subtitles_auto.json")["lines"]
    fixes = find_fixes(tok, glossary, subs)
    rows = [{"idx": line_tokens(ln, subs), "tail": "？" if ln["text"].endswith("？") else ""} for ln in lines]
    moved = move_across(rows, tok, fixes)
    out, used, across = [], [], []
    for r in rows:
        idx = r["idx"]
        if not idx:
            continue
        inside = [f for f in fixes if idx[0] <= f["from_token"] and f["to_token"] <= idx[-1]]
        across += [f for f in fixes if f["from_token"] <= idx[-1] < f["to_token"]]
        out.append({"text": fixed_text(idx, tok, inside, r["tail"]), "from_token": idx[0], "to_token": idx[-1]})
        used += inside
    write_json(run / "subtitles_terms.json", {"lines": out, "glossary": [str(p) for p in files]})
    L = ["# 修字表", "", f"术语表：{'、'.join(str(p) for p in files) or '没有'}。字幕里改了 {len(used)} 处"
         f"（其中 {moved} 处原来跨了两行，挪进了同一行），跨行没改 {len(across)} 处。"
         "术语表不看上下文，改错了就改术语表重跑。", "", "| 原词 | 改成 | 上下文 | 字序号 |", "| --- | --- | --- | --- |"]
    L += [f"| {f['wrong']} | {f['right']} | {context(tok, f)} | {f['from_token']}-{f['to_token']} |" for f in used] or ["| — | | 没有 | |"]
    if across:
        L += ["", "## 跨了两行、没改的（调整断行后手动改）", ""]
        L += [f"- {f['wrong']} → {f['right']}：{context(tok, f)}（字 {f['from_token']}-{f['to_token']}）" for f in across]
    (run / "修字表.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"按术语表改了 {len(used)} 处（跨行没改 {len(across)} 处）→ {run / 'subtitles_terms.json'}、{run / '修字表.md'}")


if __name__ == "__main__":
    main()
