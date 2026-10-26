"""第 8 步之三 渲染粗剪：按 edl_final.json 剪画面、烧字幕、合上剪好的声音。

用法：uv run python scripts/render_rough.py <视频文件夹> [runs/<时间> 文件夹，默认最新一次]

先校验 subtitles.json（Claude 改过字的字幕），再按字算每行的时间写成 subtitles.ass：
一行从第一个字的开始到最后一个字的结束，换到粗剪的时间上；
开始时间落在静音里（FunASR 的时间偏早）挪到这段静音结束，落在删掉的范围里推到下一个保留段开头；
结束时间落在删掉的范围里收到上一个保留段结尾，落在声音里延到这段声音结束（不越过下一行开始）。

画面按保留段一段一条命令精确切（从帧号起，-frames:v 指定帧数，重新编码），concat 拼起来，
最后一遍烧字幕并合上粗剪.wav（声音已在 cut_audio.py 里按采样拼好，这里只编码一次 AAC）。
输出 subtitles.ass、粗剪.srt（时间和 .ass 一样，开源首发加）、粗剪.mp4，中间文件在 segs/。同一个运行目录重渲时，上一版（连同它的自检结果）
先挪进 旧版_<时间>/，不覆盖。
"""

import shutil
import sys
import time
from pathlib import Path

import soundfile as sf

from common import (CONFIG, FPS, ROOT, SR, add_timing, analyze, check_font, ffprobe, keep_segments,
                    latest_run, read_json, run_ffmpeg, to_rough)
from make_subtitles import check_lines, line_tokens, subtitle_tokens

# 思源黑体粗体的 ASS 字号和单个汉字宽度之比。2026-10-07 实测：字号 100 时 16 个字宽 1102 像素，
# 比值约 1.452；取 1.40 给黑边留余量，720 宽的画面字号 56，16 个字连黑边约 625 像素，不超过 90%（648）
FONT_EM_RATIO = 1.40


def centis(t: float) -> int:
    """取到百分之一秒。.ass、.srt、剪映草稿的字幕时间都从这里算，几份一样。"""
    return int(round(max(0.0, t) * 100))


def ass_time(t: float) -> str:
    cs = centis(t)
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def srt_time(t: float) -> str:
    cs = centis(t)
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:02d}:{m:02d}:{s:02d},{cs * 10:03d}"


def write_srt(path: Path, lines: list, en: bool = False) -> None:
    """SRT 字幕（开源首发加，给剪映、Premiere、播放器和平台单独上传字幕用）：每行起止和 .ass 一样，
    en 为真时英文放在中文下一行。只多写这一个文件，画面和 .ass 不变。"""
    rows = [f"{k}\n{srt_time(x['start'])} --> {srt_time(x['end'])}\n{x['text']}" + (f"\n{x['en'].strip()}" if en else "") + "\n"
            for k, x in enumerate(lines, 1)]
    path.write_text("\n".join(rows), encoding="utf-8")


def ass_escape(s: str) -> str:
    return s.replace("\\", "＼").replace("{", "｛").replace("}", "｝")


def time_lines(lines: list, edl: dict, tok: list, loud) -> list[dict]:
    segs = keep_segments(edl)
    subs = subtitle_tokens(edl, len(tok))
    total = sum(b - a for a, b in segs) / FPS
    out = []
    for ln in lines:
        idx = line_tokens(ln, subs)
        s, e = tok[idx[0]]["start"], tok[idx[-1]]["end"]
        for a, b in loud.quiet:
            if a <= s <= b:
                s = b
                break
        if not loud.in_quiet(e):  # 落在声音里：延到这段声音结束，最多 0.5 秒
            nxt = [a for a, b in loud.quiet if a > e]
            if nxt:
                e = min(nxt[0], e + 0.5)
        rs = to_rough(s, segs, "next")
        re_ = to_rough(e, segs, "prev")
        rs = total if rs is None else rs
        re_ = rs if re_ is None else re_
        out.append({**ln, "start": rs, "end": max(re_, rs + 0.3)})
    for a, b in zip(out, out[1:]):  # 一次只显示一行：不越过下一行开始
        if a["end"] > b["start"]:
            a["end"] = max(a["start"] + 0.1, b["start"])
        if b["start"] < a["end"]:
            b["start"] = a["end"]
    for x in out:
        x["start"], x["end"] = round(min(x["start"], total), 2), round(min(x["end"], total), 2)
    return out


def write_ass(path: Path, lines: list, w: int, h: int) -> int:
    size = int(0.9 * w / 16 * FONT_EM_RATIO)
    outline = max(2, round(size * 0.07))
    margin_v = round(h * 0.08)
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{CONFIG['font_name']},{size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{outline},0,2,{round(w * 0.05)},{round(w * 0.05)},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    rows = [f"Dialogue: 0,{ass_time(x['start'])},{ass_time(x['end'])},Sub,,0,0,0,,{ass_escape(x['text'])}"
            for x in lines]
    path.write_text(head + "\n".join(rows) + "\n", encoding="utf-8")
    return size


OUTPUTS = ["粗剪.mp4", "subtitles.ass", "粗剪.srt", "video_concat.mp4", "segs", "check_report.md", "check.json",
           "接缝试听.mp4", "seams.ass"]


def archive_previous(run: Path) -> Path | None:
    """这个运行目录里已经渲染过（比如只改了字幕重跑），先把上一版挪进 旧版_<时间>/，翻车画面不丢。"""
    if not (run / "粗剪.mp4").exists():
        return None
    old = run / time.strftime("旧版_%H%M%S")
    old.mkdir()
    for name in OUTPUTS:
        if (run / name).exists():
            (run / name).rename(old / name)
    if (run / "subtitles.json").exists():
        shutil.copy2(run / "subtitles.json", old / "subtitles.json")
    return old


def main() -> None:
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    t0 = time.time()
    folder = Path(sys.argv[1]).expanduser().resolve()
    run = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else latest_run(folder)
    edl = read_json(run / "edl_final.json")
    tok = read_json(folder / edl["transcript"])["tokens"]
    if not (run / "subtitles.json").exists():
        sys.exit(f"{run / 'subtitles.json'} 还没有：先由 Claude 对照讲稿改 subtitles_auto.json 再写成它")
    lines = read_json(run / "subtitles.json")["lines"]
    errs = check_lines(lines, edl, len(tok))
    if errs:
        sys.exit("subtitles.json 校验不通过：\n  - " + "\n  - ".join(errs))

    a16, sr = sf.read(folder / "work" / "audio.wav", dtype="float32")
    assert sr == SR
    loud = analyze(a16)
    proxy = folder / "work" / "proxy.mp4"
    v = ffprobe(proxy)["streams"][0]
    w, h = v["width"], v["height"]
    check_font([ln["text"] for ln in lines], w, h, run)
    if old := archive_previous(run):
        print(f"上一版挪进了 {old.name}/")
    timed = time_lines(lines, edl, tok, loud)
    size = write_ass(run / "subtitles.ass", timed, w, h)
    write_srt(run / "粗剪.srt", timed)  # 粗剪只有中文

    segs = keep_segments(edl)
    seg_dir = run / "segs"
    seg_dir.mkdir(exist_ok=True)
    names = []
    for k, (a, b) in enumerate(segs):
        name = f"{k:03d}.mp4"
        # 从半帧之前起读，第一帧正好是第 a 帧
        run_ffmpeg(["-ss", f"{max(0.0, (a - 0.5) / FPS):.6f}", "-i", str(proxy), "-frames:v", str(b - a),
                    "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
                    "-r", str(FPS), str(seg_dir / name)])
        names.append(name)
    (seg_dir / "list.txt").write_text("".join(f"file '{n}'\n" for n in names), encoding="utf-8")
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "../video_concat.mp4"], cwd=seg_dir)
    fonts = (ROOT / CONFIG["font_file"]).parent
    run_ffmpeg(["-i", "video_concat.mp4", "-i", "粗剪.wav", "-map", "0:v", "-map", "1:a",
                "-vf", f"subtitles=subtitles.ass:fontsdir={fonts}",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "粗剪.mp4"], cwd=run)
    add_timing(run, "render_rough", time.time() - t0)
    print(f"粗剪渲染完成：{len(segs)} 段，字幕 {len(timed)} 行（字号 {size}），用时 {time.time() - t0:.1f} 秒 → {run / '粗剪.mp4'}")


if __name__ == "__main__":
    main()
