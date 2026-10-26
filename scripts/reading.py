"""阅读稿（讲稿可选，第二期）：没有讲稿时，把转写稿按停顿分成一段一行，给 Claude 判语气词、说断和重说。

用法：
  uv run python scripts/reading.py <视频文件夹>
      写 work/阅读稿.md
  uv run python scripts/reading.py <视频文件夹> --tokens 起-止
      逐字列出这几个字的序号、文字、时间，和下一个字之间空了多久、这个交界能不能下刀
      （和 plan_cuts.py 找刀点同一个办法：交界附近有没有够长的静音）。写 deletions.json、进了连读待定要换删法时用

分段：两个字之间有 0.5 秒以上的停顿就另起一段（静音和剪切用同一套 analyze()，中间只隔一口气的几段静音连起来算；
FunASR 的字时间常常偏早，所以按和剪切一样的窗口找：交界点往前 0.15 秒、往后 0.6 秒）。每段一行：
  段号 [起—止] #首字序号—末字序号 文字（句末标点借转写稿的分句）  ⏸ 后面停了几秒（静音和字时间的空当取长的）
同一句说了几遍，多半是相邻的几段开头一样，这样排比逐字表好认。做法借 video-use 的
“按 0.5 秒以上停顿分段、带时间的阅读稿”（MIT），代码是自己写的。
有术语表时（见 terms.py），段落下面另起一行写这一段里术语表改得到的字，只显示，不改转写稿。
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf

from common import AFTER, BEFORE, SR, analyze, fmt, join_text, read_json
from terms import find_fixes, glossary_files, load_glossary

PAUSE = 0.5   # 停这么久以上另起一段
BREATH = 0.3  # 两段静音之间的声音短于这个（一口气、杂音），连成一段停顿


def pauses(loud) -> list[tuple[float, float]]:
    """长停顿：静音段之间只隔一小口气（不到 0.3 秒的声音）的连成一段，连起来 0.5 秒以上的才算。
    10-24 盲测：一段 3 秒的停顿中间有气声，只按最长的一小截报 0.7 秒，看着不像停过。"""
    spans = []
    for a, b in loud.quiet:
        if spans and a - spans[-1][1] < BREATH:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return [(a, b) for a, b in spans if b - a >= PAUSE]


def phrases(tok: list, loud) -> list[dict]:
    """按长停顿分段。每段停顿只分一次：分在落进这段静音里的交界，没有就分在离它最近的交界。
    FunASR 常把停顿后面那个字的时间标早，落进静音里（10-24 第 03 期“定下来的。每 ⏸ 天”，“每”大半在静音里），
    这时往前挪一个字，最多挪两个。不看标点：标点模型也会标错位置（第 05 期“谷贡市，律里”）。"""
    junctions = [(tok[k]["end"] + tok[k + 1]["start"]) / 2 for k in range(len(tok) - 1)]
    breaks, used = {}, []

    def inside(k: int, a: float, b: float) -> bool:  # 第 k 个字的时间一大半落在静音里
        t0, t1 = tok[k]["start"], tok[k]["end"]
        return t1 > t0 and (min(t1, b) - max(t0, a)) > 0.5 * (t1 - t0)

    for a, b in pauses(loud):
        # 交界点往后 0.6 秒、往前 0.15 秒内有这段静音，才算这个交界的停顿（和剪切找静音同一个窗口）
        near = [k for k, j in enumerate(junctions) if a - AFTER <= j <= b + BEFORE]
        if not near:
            continue
        dist = lambda k: 0 if a <= junctions[k] <= b else min(abs(junctions[k] - a), abs(junctions[k] - b))
        k = min(near, key=lambda k: (dist(k), k))
        for _ in range(2):
            if k > 0 and inside(k, a, b) and k - 1 not in breaks:
                k -= 1
        gap = tok[k + 1]["start"] - tok[k]["end"]
        breaks[k] = max(breaks.get(k, 0), b - a, gap)
        used.append((a, b))
    for k in range(len(tok) - 1):  # 字时间本身就隔了 0.5 秒以上、又不是上面那几段静音的，也分开
        gap = (tok[k]["end"], tok[k + 1]["start"])
        if gap[1] - gap[0] >= PAUSE and k not in breaks and not any(a < gap[1] and b > gap[0] for a, b in used):
            breaks[k] = gap[1] - gap[0]
    out, start = [], 0
    for k in sorted(breaks) + [len(tok) - 1]:
        if k < start:
            continue
        out.append({"from_token": start, "to_token": k, "pause_after": round(breaks.get(k, 0), 2)})
        start = k + 1
    return out


def punctuated(tok: list, sentences: list, a: int, b: int) -> str:
    tails = {s["last_token"]: s["text"][-1] for s in sentences if s["text"] and s["text"][-1] in "，。？！、"}
    words = []
    for i in range(a, b + 1):
        words.append(tok[i]["text"] + tails.get(i, ""))
    return join_text(words)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--tokens")
    a = ap.parse_args()
    folder = Path(a.folder).expanduser().resolve()
    tr = read_json(folder / "work" / "transcript.json")
    tok = tr["tokens"]
    audio, sr = sf.read(folder / "work" / "audio.wav", dtype="float32")
    if sr != SR:
        sys.exit(f"work/audio.wav 应是 {SR} Hz，先跑 preprocess.py")
    loud = analyze(audio)
    if a.tokens:
        from plan_cuts import silence_cut
        f, _, t = a.tokens.partition("-")
        f, t = int(f), int(t or f)
        # plan_cuts.py 在交界往后 0.6 秒内找静音，连着几个交界会找到同一段；只算在离它最近的那个交界上
        rng = [i for i in range(max(0, f), min(len(tok) - 1, t + 1))]
        cuts = {i: silence_cut(loud, tok[i], tok[i + 1]) for i in rng}
        j = lambda i: (tok[i]["end"] + tok[i + 1]["start"]) / 2
        owner = {}
        for i, c in cuts.items():
            if c:
                q = tuple(c["quiet"])
                d = 0 if q[0] <= j(i) <= q[1] else min(abs(j(i) - q[0]), abs(j(i) - q[1]))
                if q not in owner or d < owner[q][1]:
                    owner[q] = (i, d)
        print("序号\t字\t时间\t到下一个字\t这个交界能不能下刀")
        for i in range(max(0, f), min(len(tok), t + 1)):
            if i in cuts:
                c = cuts[i]
                gap = f"{tok[i + 1]['start'] - tok[i]['end']:.2f} 秒"
                if not c:
                    can = "不能，连着说"
                elif owner[tuple(c["quiet"])][0] == i:
                    d = owner[tuple(c["quiet"])][1]
                    can = f"能，静音 {c['quiet'][0]:.2f}—{c['quiet'][1]:.2f}" + (f"（在交界后 {d:.2f} 秒，字时间常偏早，可能是下一个字后面的停顿）" if d > 0.2 else "")
                else:
                    can = f"不能，附近那段静音在第 {owner[tuple(c['quiet'])][0]} 个字后面"
            else:
                gap, can = "—", "片尾" if i == len(tok) - 1 else "—"
            print(f"{i}\t{tok[i]['text']}\t{tok[i]['start']:.2f}-{tok[i]['end']:.2f}\t{gap}\t{can}")
        return
    files = glossary_files(folder)
    fixes = find_fixes(tok, load_glossary(files))
    ps = phrases(tok, loud)
    L = [f"# 阅读稿：{tr['source_name']}", "",
         f"原片 {fmt(tr['duration'])}，{len(tok)} 个字，按 {PAUSE} 秒以上的停顿分成 {len(ps)} 段。"
         f"每段：段号 [起—止] #首字—末字 文字，⏸ 是这段后面停了几秒。时间是转写给的，常常偏早零点几秒。",
         f"术语表：{'、'.join(str(p) for p in files) or '没有'}；能改到的字写在段落下面（→ 开头），只是提示，转写稿没改。", ""]
    for n, p in enumerate(ps, 1):
        f, t = p["from_token"], p["to_token"]
        pause = f"  ⏸ {p['pause_after']:.1f}" if p["pause_after"] else ""
        L.append(f"{n:03d} [{fmt(tok[f]['start'])}—{fmt(tok[t]['end'])}] #{f}—{t} {punctuated(tok, tr['sentences'], f, t)}{pause}")
        here = [x for x in fixes if f <= x["from_token"] <= t]
        if here:
            L.append("    → " + "；".join(f"#{x['from_token']}—{x['to_token']}“{x['wrong']}”改成“{x['right']}”" for x in here))
    out = folder / "work" / "阅读稿.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"阅读稿 {len(ps)} 段，术语表改得到的 {len(fixes)} 处 → {out}")


if __name__ == "__main__":
    main()
