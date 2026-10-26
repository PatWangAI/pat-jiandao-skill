"""第 8 步之二 字幕初稿（SPEC 第 6 节）。

用法：uv run python scripts/make_subtitles.py <视频文件夹> [runs/<时间> 文件夹，默认最新一次]

读 edl_final.json 和转写稿，取保留下来、要进字幕的字（删掉的不进，连读待定的语气词也不进），
重新加标点，按逗号、句号、问号断开，超过 16 个字的在字和词的边界上切开，写成 subtitles_auto.json。
每行有 text、text_asr（转写原文）、from_token、to_token（这一行管的第一个和最后一个字）。

接下来由 Claude 对照讲稿改认错的词和数字写法、调整断行，写成同目录的 subtitles.json
（只写 text、from_token、to_token，不写时间）。时间由 render_rough.py 渲染时按字算。
"""

import sys
import time
from pathlib import Path

from common import MAX_CHARS, add_timing, join_text, latest_run, read_json, text_width, write_json

PUNC = {1: "", 2: "，", 3: "。", 4: "？", 5: "、"}


def subtitle_tokens(edl: dict, n: int) -> list[int]:
    """该进字幕的字：没删掉的，也不是连读待定（用户留下的语气词不进字幕）。
    没有讲稿时（剪辑表记了 no_script，第二期）只有语气词的连读待定不进字幕：说断、重说进了连读待定、声音里留着的，
    字幕也照有（10-24 盲测：第 03 期“得打开剪映，手动找出”留在声音里，字幕却没有这一段）。"""
    skip = set()
    for d in edl["deletions"]:
        skip.update(range(d["from_token"], d["to_token"] + 1))
    for d in edl["pending"]:
        if not edl.get("no_script") or d.get("orig_type") == "语气词":
            skip.update(range(d["from_token"], d["to_token"] + 1))
    return [i for i in range(n) if i not in skip]


def line_tokens(line: dict, subs: list[int]) -> list[int]:
    return [i for i in subs if line["from_token"] <= i <= line["to_token"]]


def check_lines(lines: list, edl: dict, n: int) -> list[str]:
    """校验字幕：该进字幕的字每个恰好属于一行，顺序不变；每行不超过 16 个字。"""
    errs = []
    subs = subtitle_tokens(edl, n)
    got = []
    prev_to = -1
    for k, ln in enumerate(lines, 1):
        f, t = ln.get("from_token"), ln.get("to_token")
        if not isinstance(f, int) or not isinstance(t, int) or f > t:
            errs.append(f"第 {k} 行的 from_token、to_token 不对：{f!r}、{t!r}")
            continue
        if f <= prev_to:
            errs.append(f"第 {k} 行（字 {f}—{t}）和上一行重叠或顺序颠倒")
        prev_to = t
        cov = line_tokens(ln, subs)
        if not cov:
            errs.append(f"第 {k} 行（字 {f}—{t}）里没有要进字幕的字")
        got += cov
        if not str(ln.get("text", "")).strip():
            errs.append(f"第 {k} 行没有文字")
        w = text_width(str(ln.get("text", "")))
        if w > MAX_CHARS:
            errs.append(f"第 {k} 行“{ln.get('text')}”有 {w:g} 个字，超过 {MAX_CHARS} 个")
    if got != subs:
        miss = sorted(set(subs) - set(got))
        dup = sorted({i for i in got if got.count(i) > 1})
        if miss:
            errs.append(f"这些字没有进任何一行：{miss[:20]}")
        if dup:
            errs.append(f"这些字进了不止一行：{dup[:20]}")
    return errs


def split_long(idx: list[int], tok: list) -> list[list[int]]:
    """超过 16 个字的一段，平均切成几行，只在字和英文词的边界上切。"""
    def width(ix):  # 按拼好的文字算，英文词之间的空格也算宽度
        return text_width(join_text([tok[i]["text"] for i in ix]))

    w = width(idx)
    k = int(-(-w // MAX_CHARS))
    if k <= 1:
        return [idx]
    out, cur, target = [], [], w / k
    for i in idx:
        if cur and (width(cur + [i]) > MAX_CHARS or (width(cur) >= target and len(out) < k - 1)):
            out.append(cur)
            cur = []
        cur.append(i)
    out.append(cur)
    return out


def main() -> None:
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    t0 = time.time()
    folder = Path(sys.argv[1]).expanduser().resolve()
    run = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else latest_run(folder)
    edl = read_json(run / "edl_final.json")
    tok = read_json(folder / edl["transcript"])["tokens"]
    subs = subtitle_tokens(edl, len(tok))

    from funasr import AutoModel
    punc = AutoModel(model="ct-punc", disable_update=True, device="cpu")
    res = punc.generate(input=" ".join(tok[i]["text"] for i in subs))[0]
    marks = [int(x) for x in res["punc_array"]]
    if len(marks) != len(subs):
        sys.exit(f"标点模型给的标号 {len(marks)} 个，字有 {len(subs)} 个，对不上")

    lines, cur = [], []
    for i, m in zip(subs, marks):
        cur.append((i, m))
        if m in (2, 3, 4):
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        tail = "？" if ln[-1][1] == 4 else ""  # 句末标点去掉，问号保留
        pieces = split_long([i for i, _ in ln], tok)
        for p, idx in enumerate(pieces):
            asr = join_text([tok[i]["text"] for i in idx])
            text = asr + (tail if p == len(pieces) - 1 else "")
            out.append({"text": text, "text_asr": asr, "from_token": idx[0], "to_token": idx[-1]})
    errs = check_lines(out, edl, len(tok))
    if errs:
        sys.exit("字幕初稿校验不通过（是脚本的问题）：\n  - " + "\n  - ".join(errs))
    write_json(run / "subtitles_auto.json", {"lines": out})
    add_timing(run, "make_subtitles", time.time() - t0)
    print(f"字幕初稿 {len(out)} 行 → {run / 'subtitles_auto.json'}")
    for k, ln in enumerate(out, 1):
        print(f"  {k:>2}. [{ln['from_token']}-{ln['to_token']}] {ln['text']}")


if __name__ == "__main__":
    main()
