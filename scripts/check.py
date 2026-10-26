"""第 9 步 自检（SPEC 第 5 节）：只检查和报告，不自动改。

用法：uv run python scripts/check.py <视频文件夹> [runs/<时间> 文件夹，默认最新一次]
      uv run python scripts/check.py --calibrate <阶段 0 运行目录>   （量阶段 0 接缝的爆音分数，标定门限用）

写出 check_report.md 和 接缝试听.mp4：
1. 文件：H.264 加 AAC，帧率 30/1，帧数和时长对得上，音视频时长差不超过 0.1 秒，
   粗剪.wav 的采样数等于保留帧数 × 1600，方向和原片一致；
2. 复核每一轮发现了什么、挪了哪几刀，最后还剩什么；
3. 剪切点清单：删除处和压停顿的每一刀（片头片尾不算），运行记录的“剪切点数”就数这份；
4. 爆音：每个接缝前后 20 毫秒的峰值，比两侧各 100 毫秒里较安静一侧的平均音量高出多少；
5. 字幕：每行不超过 16 个字，时间在片长以内、互不重叠，列出每行从转写原文改成了什么；
6. 要重点听的接缝。
接缝试听.mp4 是粗剪画面加声音，到删除处的接缝，屏幕上方提示第几处、删了什么；压停顿的接缝在右上角标小号编号。
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from common import (CONFIG, FPS, ROOT, SR, SR_OUT, add_timing, analyze, check_font, ffprobe, fmt, join_text,
                    keep_segments, latest_run, read_json, run_ffmpeg, seam_times, text_width)
from make_subtitles import line_tokens, subtitle_tokens
from render_rough import ass_escape, ass_time, time_lines


def pop_score(x: np.ndarray, sr: int, t: float) -> dict:
    """接缝前后 20 毫秒内的峰值（dBFS），减去两侧各 100 毫秒里较安静一侧的平均音量（dBFS）。"""
    if x.ndim > 1:
        x = x.mean(axis=1)
    i, w20, w100 = int(round(t * sr)), int(0.02 * sr), int(0.1 * sr)

    def db(v):
        return 20 * math.log10(v + 1e-10)

    peak = db(float(np.max(np.abs(x[max(0, i - w20):i + w20])))) if len(x) else -200
    left, right = x[max(0, i - w20 - w100):max(0, i - w20)], x[i + w20:i + w20 + w100]
    sides = [db(float(np.sqrt(np.mean(s.astype(np.float64) ** 2)))) for s in (left, right) if len(s)]
    quiet = min(sides) if sides else -200
    return {"peak_db": round(peak, 1), "quiet_db": round(quiet, 1), "diff_db": round(peak - quiet, 1)}


def calibrate(run0: Path) -> None:
    """阶段 0 的粗剪（人试听过、没听出爆音）里每个接缝的爆音分数。门限 = 最大值 + 6 dB。"""
    cuts = read_json(run0 / "cuts.json")
    x, sr = sf.read(run0 / "rough_cut.wav", dtype="float32")
    # 按阶段 0 脚本的拼法重算每个接缝在粗剪里的时间
    pos, clock, seams = 0.0, 0.0, []
    total = len(x) / sr
    for j in cuts["joins"]:
        a, b = pos, j["cut_out"]["time"]
        if b - a > 0.02:
            clock += round((b - a) * sr) / sr
        seams.append(clock)
        pos = j["cut_in"]["time"]
    scores = [pop_score(x, sr, t) for t in seams if 0.15 < t < total - 0.15]
    for t, s in zip(seams, scores):
        print(f"  接缝 {t:6.2f} 秒：{s}")
    mx = max(s["diff_db"] for s in scores)
    print(f"最大差值 {mx} dB，门限建议 {mx + 6:.1f} dB")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--calibrate":
        calibrate(Path(sys.argv[2]).resolve())
        return
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    t0 = time.time()
    folder = Path(sys.argv[1]).expanduser().resolve()
    run = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else latest_run(folder)
    inputs = read_json(folder / "inputs.json")
    pre = read_json(folder / "work" / "preprocess.json")
    edl = read_json(run / "edl_final.json")
    tok = read_json(folder / edl["transcript"])["tokens"]
    recheck = read_json(run / "recheck.json")
    segs = keep_segments(edl)
    keep_frames = sum(b - a for a, b in segs)
    total = keep_frames / FPS
    a16, _ = sf.read(folder / "work" / "audio.wav", dtype="float32")
    loud = analyze(a16)
    rough, sr48 = sf.read(run / "粗剪.wav", dtype="float32", always_2d=True)
    problems = []

    # 1. 文件
    info = ffprobe(run / "粗剪.mp4")
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), {})
    vdur, adur = float(v.get("duration", 0)), float(a.get("duration", 0))
    nb = int(v.get("nb_frames", 0))
    src_orient = pre["source"]["orientation"]
    out_orient = "竖屏" if v["height"] > v["width"] else "横屏"
    file_checks = [
        ("视频 H.264、音频 AAC", v.get("codec_name") == "h264" and a.get("codec_name") == "aac",
         f"{v.get('codec_name')} + {a.get('codec_name')}"),
        ("帧率标记 30/1", v.get("r_frame_rate") == "30/1", v.get("r_frame_rate")),
        ("帧数等于保留帧数（差不超过 1 帧）", abs(nb - keep_frames) <= 1, f"{nb} 帧，应为 {keep_frames}"),
        ("帧数和时长对得上", abs(vdur - nb / FPS) <= 1 / FPS, f"{vdur:.3f} 秒"),
        ("音视频时长差不超过 0.1 秒", abs(vdur - adur) <= 0.1, f"差 {abs(vdur - adur):.3f} 秒"),
        ("粗剪.wav 采样数 = 保留帧数 × 1600", sr48 == SR_OUT and len(rough) == keep_frames * 1600,
         f"{len(rough)}，应为 {keep_frames * 1600}"),
        ("显示方向和原片一致", out_orient == src_orient, f"成片{out_orient}，原片{src_orient}"),
    ]
    problems += [f"文件：{n}（{d}）" for n, ok, d in file_checks if not ok]

    # 3. 剪切点清单
    seam_of = {}
    for st in seam_times(edl):
        for owner in st["owners"]:
            seam_of[owner] = st["rough_s"]
    cuts = []
    for d in edl["deletions"]:
        if d["out_how"] == "片头" or d["in_how"] == "片尾":  # 这两刀就是粗剪的起点或终点，不算剪切点
            continue
        for side, label in (("out", "前一刀"), ("in", "后一刀")):
            how = d[f"{side}_how"]
            f = d[f"{side}_frame"]
            t = f / FPS
            db = loud.db_at(t)
            ok = loud.in_quiet(t) or db < loud.thr
            cuts.append({"id": len(cuts) + 1, "kind": "删除", "owner": d["id"], "side": label, "frame": f,
                         "src_s": t, "rough_s": seam_of.get(("删除", d["id"])), "how": how,
                         "moved": bool([m for m in d["moves"] if m["cut"] == label]),
                         "db": round(db, 1), "ok": ok})
    for p in edl["pauses"]:
        for side, label in (("out", "前一刀"), ("in", "后一刀")):
            f = p[f"{side}_frame"]
            t = f / FPS
            db = loud.db_at(t)
            ok = loud.in_quiet(t) or db < loud.thr
            cuts.append({"id": len(cuts) + 1, "kind": "长停顿", "owner": p["id"], "side": label, "frame": f,
                         "src_s": t, "rough_s": seam_of.get(("长停顿", p["id"])), "how": "压停顿",
                         "moved": False, "db": round(db, 1), "ok": ok})
    problems += [f"剪切点 {c['id']}：程序判断有声，可能切到字" for c in cuts if not c["ok"]]

    # 4. 爆音
    thr_pop = CONFIG.get("pop_threshold_db")
    pops = []
    for st in seam_times(edl):
        s = pop_score(rough, sr48, st["rough_s"])
        s["rough_s"] = st["rough_s"]
        s["owners"] = st["owners"]
        s["flag"] = thr_pop is not None and s["diff_db"] > thr_pop
        pops.append(s)
    problems += [f"爆音：粗剪 {fmt(s['rough_s'])} 的接缝高出 {s['diff_db']} dB" for s in pops if s["flag"]]

    # 5. 字幕
    sub_path = run / "subtitles.json"
    sub_rows, sub_errs = [], []
    if sub_path.exists():
        lines = time_lines(read_json(sub_path)["lines"], edl, tok, loud)
        subs = subtitle_tokens(edl, len(tok))
        for k, ln in enumerate(lines, 1):
            asr = join_text([tok[i]["text"] for i in line_tokens(ln, subs)])
            w = text_width(ln["text"])
            if w > 16:
                sub_errs.append(f"第 {k} 行有 {w:g} 个字")
            if not (0 <= ln["start"] < ln["end"] <= total + 0.01):
                sub_errs.append(f"第 {k} 行时间 {ln['start']}—{ln['end']} 不在片长 {total:.2f} 秒以内")
            if k > 1 and ln["start"] < lines[k - 2]["end"]:
                sub_errs.append(f"第 {k} 行和上一行时间重叠")
            sub_rows.append((k, ln, asr))
    else:
        sub_errs.append("没有 subtitles.json")
    problems += ["字幕：" + e for e in sub_errs]

    # 6. 要重点听的接缝
    final = next(r for r in recheck["rounds"] if r["round"] == recheck["final_round"])
    focus = []
    for c in cuts:
        if c["how"] == "交界处最小音量":
            focus.append((c["rough_s"], f"剪切点 {c['id']}（第 {c['owner']} 处{c['side']}）在交界处音量最小的地方切的"))
        if c["moved"]:
            focus.append((c["rough_s"], f"剪切点 {c['id']}（第 {c['owner']} 处{c['side']}）复核时挪过"))
        if not c["ok"]:
            focus.append((c["rough_s"], f"剪切点 {c['id']} 程序判断有声"))
    # 最终那一轮的发现全列；前几轮报过的替换和没挪成的也列上（那一处后来没挪刀，声音没变，问题可能还在）
    seen = set()
    moved_ids = {d["id"] for d in edl["deletions"] if d["moves"]}
    earlier = [f for r in recheck["rounds"] if r["round"] < final["round"] for f in r["findings"]
               if (f["kind"] == "替换" or f.get("unfixed")) and f["deletion"] not in moved_ids]
    for f in final["findings"] + earlier:
        key = (f["deletion"], f["note"])
        if key in seen:
            continue
        seen.add(key)
        extra = f"，没能自动挪：{f['unfixed']}" if f.get("unfixed") else ""
        t = next((st["rough_s"] for st in seam_times(edl) if ("删除", f["deletion"]) in st["owners"]), f["seam_rough_s"])
        focus.append((t, f"第 {f['deletion']} 处接缝复核时{f['kind']}：{f['note']}{extra}"))
    for s in pops:
        if s["flag"]:
            focus.append((s["rough_s"], f"可能有爆音（高出 {s['diff_db']} dB）"))
    focus.sort(key=lambda x: x[0] or 0)
    merged = []  # 同一个接缝的几条合成一行
    for t, why in focus:
        if merged and merged[-1][0] is not None and t is not None and abs(merged[-1][0] - t) < 0.01:
            merged[-1] = (merged[-1][0], merged[-1][1] + "；" + why)
        else:
            merged.append((t, why))
    focus = merged

    # 报告
    L = [f"# 自检报告：{inputs['name']}", "",
         f"运行目录 runs/{run.name}。原片 {fmt(pre['proxy']['duration'])}，粗剪 {fmt(total)}；"
         f"删除 {len(edl['deletions'])} 处、压停顿 {len(edl['pauses'])} 处、连读待定 {len(edl['pending'])} 处没删；"
         f"剪切点 {len(cuts)} 个；复核 {len(recheck['rounds'])} 轮，用第 {recheck['final_round']} 轮的结果。", "",
         f"程序查出的问题 {len(problems)} 条，要重点听的接缝 {len(focus)} 处。本报告只列问题，不自动改。", ""]
    if problems:
        L += ["程序查出的问题：", ""] + [f"- {p}" for p in problems] + [""]
    L += ["## 1. 文件", "", "| 检查 | 结果 | 实际 |", "| --- | --- | --- |"]
    L += [f"| {n} | {'通过' if ok else '不通过'} | {d} |" for n, ok, d in file_checks]
    L += ["", "## 2. 复核", ""]
    for r in recheck["rounds"]:
        fs = "；".join(f"第 {f['deletion']} 处{f['kind']}：{f['note']}" + (f"（没能挪：{f['unfixed']}）" if f.get("unfixed") else "")
                      for f in r["findings"]) or "没有发现"
        mv = "；".join(r["moves"]) or "没挪刀"
        L.append(f"- 第 {r['round']} 轮（转写 {r['transcribe_seconds']} 秒）：{fs}。{mv}。")
    L += ["", f"用第 {recheck['final_round']} 轮的结果，最后还剩："
          + ("；".join(f"第 {f['deletion']} 处{f['kind']}" for f in final["findings"]) or "没有") + "。", ""]
    L += [f"## 3. 剪切点清单（{len(cuts)} 个，片头片尾不算）", "",
          f"静音门限 {loud.thr:.1f} dB。", "",
          "| 刀 | 属于 | 哪一刀 | 原片时间 | 粗剪时间 | 找法 | 刀点音量（dB） | 程序判断 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in cuts:
        L.append(f"| {c['id']} | {c['kind']}第 {c['owner']} 处 | {c['side']} | {c['src_s']:.3f} | "
                 f"{fmt(c['rough_s'] or 0)} | {c['how']} | {c['db']} | {'在静音里' if c['ok'] else '有声，可能切到字'} |")
    L += ["", "## 4. 爆音", "",
          (f"门限 {thr_pop} dB（阶段 0 标定，见 config.yaml）。" if thr_pop is not None else
           "门限还没标定（config.yaml 的 pop_threshold_db），只列数值。"), "",
          "| 粗剪时间 | 接缝属于 | 峰值（dBFS） | 较安静一侧（dBFS） | 高出（dB） | 判断 |", "| --- | --- | --- | --- | --- | --- |"]
    for s in pops:
        who = "、".join(f"{k}第 {n} 处" for k, n in s["owners"]) or "—"
        L.append(f"| {fmt(s['rough_s'])} | {who} | {s['peak_db']} | {s['quiet_db']} | {s['diff_db']} | "
                 f"{'可能有爆音' if s['flag'] else '正常'} |")
    L += ["", "## 5. 字幕", ""]
    L += ([f"- {e}" for e in sub_errs] or ["每行不超过 16 个字，时间都在片长以内、互不重叠。"]) + [""]
    L += ["| 行 | 粗剪时间 | 转写原文 | 最终文字 |", "| --- | --- | --- | --- |"]
    for k, ln, asr in sub_rows:
        same = asr.replace(" ", "") == ln["text"].replace(" ", "")
        L.append(f"| {k} | {fmt(ln['start'])}—{fmt(ln['end'])} | {asr} | {'（同左）' if same else ln['text']} |")
    L += ["", f"## 6. 要重点听的接缝（{len(focus)} 处）", ""]
    L += [f"- 粗剪 {fmt(t or 0)}：{why}" for t, why in focus] or ["没有。"]
    if edl["pending"]:
        L += ["", "连读待定（没删，在粗剪里还听得到）："]
        L += [f"- 第 {'、'.join(map(str, p['items']))} 条“{p['text']}”：{p['reason']}" for p in edl["pending"]]
    (run / "check_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # 接缝试听.mp4
    w, h = v["width"], v["height"]
    big, small = round(w / 15), round(w / 26)
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Head,{CONFIG['font_name']},{small},&H00C0C0C0,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,3,2,0,8,20,20,{round(h * 0.03)},1
Style: Label,{CONFIG['font_name']},{big},&H0000D7FF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,8,20,20,{round(h * 0.10)},1
Style: Gone,{CONFIG['font_name']},{small},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,8,20,20,{round(h * 0.10) + big + 16},1
Style: Pause,{CONFIG['font_name']},{small},&H0080FF80,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,9,20,20,{round(h * 0.03) + small + 20},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    rows = [f"Dialogue: 0,{ass_time(0)},{ass_time(total)},Head,,0,0,0,,"
            f"接缝试听：删 {len(edl['deletions'])} 处，压停顿 {len(edl['pauses'])} 处。听提示前后的字是否完整"]
    dels = {d["id"]: d for d in edl["deletions"]}
    for st in seam_times(edl):
        t = st["rough_s"]
        for kind, n in st["owners"]:
            ids = "、".join(str(c["id"]) for c in cuts if c["kind"] == kind and c["owner"] == n)
            if kind == "删除":
                gone = dels[n]["text"]
                gone = gone if len(gone) <= 14 else gone[:7] + "……" + gone[-5:]
                rows.append(f"Dialogue: 0,{ass_time(t - 0.7)},{ass_time(t + 0.8)},Label,,0,0,0,,"
                            f"{ass_escape(f'第 {n} 处接缝（刀 {ids}）')}")
                rows.append(f"Dialogue: 0,{ass_time(t - 0.7)},{ass_time(t + 0.8)},Gone,,0,0,0,,"
                            f"{ass_escape('删掉：' + gone)}")
            else:
                rows.append(f"Dialogue: 0,{ass_time(t - 0.4)},{ass_time(t + 0.4)},Pause,,0,0,0,,"
                            f"{ass_escape(f'停顿 {n}（刀 {ids}）')}")
    check_font([r.split(",,", 2)[2] for r in rows], w, h, run)
    (run / "seams.ass").write_text(head + "\n".join(rows) + "\n", encoding="utf-8")
    fonts = (ROOT / CONFIG["font_file"]).parent
    run_ffmpeg(["-i", "粗剪.mp4", "-vf", f"subtitles=seams.ass:fontsdir={fonts}", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "23", "-c:a", "copy", "接缝试听.mp4"], cwd=run)
    write_json_path = run / "check.json"
    write_json_path.write_text(json.dumps({"cuts": len(cuts), "problems": problems, "focus": len(focus),
                                           "recheck_rounds": len(recheck["rounds"])}, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    add_timing(run, "check", time.time() - t0)
    print(f"自检完成：剪切点 {len(cuts)} 个，程序查出的问题 {len(problems)} 条，要重点听 {len(focus)} 处 → "
          f"{run / 'check_report.md'}")


if __name__ == "__main__":
    main()
