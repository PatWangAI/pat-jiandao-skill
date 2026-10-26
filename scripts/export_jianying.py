"""把粗剪导成剪映草稿（开源首发，立项卡 v1.2 范围第 5 项）：剪好的画面段（从原片截、带原声）加中文字幕；
subtitles.json 每行都写了英文（出过成片的），再加一条英文字幕。配的素材、转场、背景音乐不进草稿。

用法：uv run python scripts/export_jianying.py <视频文件夹> [runs/<时间> 文件夹，默认最新一次] [--drafts <剪映草稿文件夹>] [--name <草稿名>]

- 画面段和剪辑表 edl_final.json 的保留段一样（common.keep_segments），帧号按 30 帧/秒换成原片里的时间；
  字幕时间和 粗剪.srt 一样（render_rough.time_lines，同样取到百分之一秒）。写完把草稿读回来逐段比一遍。
- 原片复制进草稿自己的文件夹：剪映只读得到它自己的文件夹，原片放在别处，打开时要手动“链接媒体”（10-24 测过）。
- 剪映草稿文件夹：--drafts，或设置里的 jianying_drafts，默认是 Mac 版剪映专业版的
  ~/Movies/JianyingPro/User Data/Projects/com.lveditor.draft。
- 只新建草稿：同名草稿已经有了就停下；不改剪映的草稿索引，不碰别的草稿，不解密剪映自己存的文件。
  草稿写成明文 draft_content.json（pyJianYingDraft 写的名字），再复制一份叫 draft_info.json（Mac 版剪映读这个名字）。
- 不跑这个脚本，剪片的结果一点不变。
- 非官方功能：靠 pyJianYingDraft（Apache-2.0）写草稿，剪映升级后可能打不开。测过：剪映专业版 9.6.0（Mac）、竖屏。
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import soundfile as sf

from common import CONFIG, FPS, analyze, has_picture, keep_segments, latest_run, read_json
from render_rough import centis, time_lines

DEFAULT_DRAFTS = "~/Movies/JianyingPro/User Data/Projects/com.lveditor.draft"
US = 1_000_000  # 剪映草稿里的时间单位是微秒

# 字幕样式（10-24 可行性测试第 2 版在剪映里看过）：中文字号 10、加粗、黑边，英文字号 6 放在中文下面。
# transform_y 是从画面中心往下的距离占半个画面高度的比例（-1 是底边）。竖屏放在底部账号文案上面，测过；横屏没在剪映里看过。
STYLE = {"竖": {"zh_y": -0.35, "en_y": -0.43}, "横": {"zh_y": -0.72, "en_y": -0.84}}


def frame_us(k: int) -> int:
    return round(k * US / FPS)


def plan(segs: list, lines: list, en: bool) -> dict:
    """要写进草稿的东西，全部换成微秒：画面段（原片里的起止、成片里的起止）、中文和英文字幕各行的起止。"""
    video, pos = [], 0
    for a, b in segs:
        n = b - a
        video.append({"src": (frame_us(a), frame_us(b) - frame_us(a)), "dst": (frame_us(pos), frame_us(pos + n) - frame_us(pos))})
        pos += n
    subs = [{"text": x["text"], "en": str(x.get("en") or "").strip(),
             "start": centis(x["start"]) * 10_000, "end": centis(x["end"]) * 10_000} for x in lines]
    return {"video": video, "subs": subs, "en": en, "total_us": frame_us(pos)}


def write_draft(drafts: Path, name: str, src: Path, picture: bool, size: tuple, p: dict) -> Path:
    import pyJianYingDraft as draft
    from pyJianYingDraft import trange

    w, h = size
    script = draft.DraftFolder(str(drafts)).create_draft(name, w, h, FPS)
    d = drafts / name
    media = d / "素材" / src.name
    media.parent.mkdir()
    shutil.copy2(src, media)
    if picture:
        script.append_track(draft.TrackSpec(draft.TrackType.video))
        for v in p["video"]:
            script.add_segment(draft.VideoSegment(str(media), trange(*v["dst"]), source_timerange=trange(*v["src"])))
    else:  # 只有声音的原片（无人脸模式）：放音轨，画面是黑的
        script.append_track(draft.TrackSpec(draft.TrackType.audio))
        for v in p["video"]:
            script.add_segment(draft.AudioSegment(str(media), trange(*v["dst"]), source_timerange=trange(*v["src"])))
    st = STYLE["竖" if h > w else "横"]
    tracks = [("中文字幕", "text", 10, True, st["zh_y"])] + ([("英文字幕", "en", 6, False, st["en_y"])] if p["en"] else [])
    for track, key, fsize, bold, y in tracks:
        script.append_track(draft.TrackSpec(draft.TrackType.text, track))
        for s in p["subs"]:
            script.add_segment(draft.TextSegment(
                s[key], trange(s["start"], s["end"] - s["start"]),
                style=draft.TextStyle(size=fsize, bold=bold, align=1), border=draft.TextBorder(width=40),
                clip_settings=draft.ClipSettings(transform_y=y)), track)
    script.save()
    shutil.copy2(d / "draft_content.json", d / "draft_info.json")
    return d


def verify(d: Path, p: dict) -> list[str]:
    """把写好的草稿读回来，和剪辑表、字幕逐段比：段数、每段在原片和成片里的起止、每行字幕的字和起止。"""
    errs = []
    for fn in ("draft_content.json", "draft_info.json"):
        if not (d / fn).exists():
            errs.append(f"缺 {fn}")
    j = json.loads((d / "draft_content.json").read_text(encoding="utf-8"))
    texts = {m["id"]: json.loads(m["content"])["text"] for m in j["materials"]["texts"]}
    by_type = {}
    for t in j["tracks"]:
        by_type.setdefault(t["type"], []).append(t)
    media = (by_type.get("video") or by_type.get("audio") or [{}])[0].get("segments", [])
    if len(media) != len(p["video"]):
        errs.append(f"画面段 {len(media)} 段，剪辑表 {len(p['video'])} 段")
    for k, (seg, v) in enumerate(zip(media, p["video"]), 1):
        got_src = (seg["source_timerange"]["start"], seg["source_timerange"]["duration"])
        got_dst = (seg["target_timerange"]["start"], seg["target_timerange"]["duration"])
        if got_src != v["src"] or got_dst != v["dst"]:
            errs.append(f"第 {k} 段：草稿 原片{got_src} 成片{got_dst}，剪辑表 原片{v['src']} 成片{v['dst']}")
    keys = [("中文字幕", "text")] + ([("英文字幕", "en")] if p["en"] else [])
    text_tracks = {t.get("name"): t["segments"] for t in by_type.get("text", [])}
    for track, key in keys:
        segs = text_tracks.get(track, [])
        if len(segs) != len(p["subs"]):
            errs.append(f"{track} {len(segs)} 行，字幕 {len(p['subs'])} 行")
        for k, (seg, s) in enumerate(zip(segs, p["subs"]), 1):
            tr = seg["target_timerange"]
            if (tr["start"], tr["start"] + tr["duration"]) != (s["start"], s["end"]) or texts.get(seg["material_id"]) != s[key]:
                errs.append(f"{track}第 {k} 行对不上：草稿 {tr} {texts.get(seg['material_id'])!r}，字幕 {s['start']}—{s['end']} {s[key]!r}")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("run", nargs="?")
    ap.add_argument("--drafts", help="剪映的草稿文件夹")
    ap.add_argument("--name", help="草稿名，默认 <视频文件夹名>_粗剪")
    a = ap.parse_args()
    folder = Path(a.folder).expanduser().resolve()
    run = Path(a.run).resolve() if a.run else latest_run(folder)
    if not (run / "edl_final.json").exists() or not (run / "subtitles.json").exists():
        sys.exit(f"{run} 里还没有 edl_final.json 和 subtitles.json：先剪出粗剪（第 7 步）再导剪映")
    drafts = Path(a.drafts or CONFIG.get("jianying_drafts") or DEFAULT_DRAFTS).expanduser()
    if not drafts.is_dir():
        sys.exit(f"找不到剪映的草稿文件夹 {drafts}。装了剪映专业版、打开过一次才有；"
                 "不在默认位置的，加 --drafts <文件夹>，或在 config.local.yaml 写 jianying_drafts")
    name = a.name or f"{folder.name}_粗剪"
    if (drafts / name).exists():
        sys.exit(f"剪映里已经有叫“{name}”的草稿，不覆盖：换个名字加 --name，或者先在剪映里删掉它")

    inputs = read_json(folder / "inputs.json")
    src = (folder / inputs["source"]).resolve()
    if not src.exists():
        sys.exit(f"找不到原片 {src}")
    proxy = read_json(folder / "work" / "preprocess.json")["proxy"]
    vertical = proxy["height"] > proxy["width"]
    size = (1080, 1920) if vertical else (1920, 1080)
    edl = read_json(run / "edl_final.json")
    tok = read_json(folder / edl["transcript"])["tokens"]
    a16, _ = sf.read(folder / "work" / "audio.wav", dtype="float32")
    lines = time_lines(read_json(run / "subtitles.json")["lines"], edl, tok, analyze(a16))
    en = all(str(x.get("en") or "").strip() for x in lines) and bool(lines)
    p = plan(keep_segments(edl), lines, en)

    d = write_draft(drafts, name, src, has_picture(src), size, p)
    errs = verify(d, p)
    mb = src.stat().st_size / 1e6
    print(f"剪映草稿写好了：{d}\n  {'竖' if vertical else '横'}屏 {size[0]}×{size[1]}，{len(p['video'])} 段，共 {p['total_us'] / US:.2f} 秒；"
          f"字幕 {len(p['subs'])} 行{'，中英两条' if en else '，只有中文（subtitles.json 没写英文）'}；原片复制了一份（{mb:.1f} MB）")
    if errs:
        sys.exit("读回来和剪辑表、字幕对不上：\n  - " + "\n  - ".join(errs))
    print("  读回来逐段比过：段数、每段起止、每行字幕的字和起止都和剪辑表、字幕一样。打开剪映专业版就能看到这个草稿。")


if __name__ == "__main__":
    main()
