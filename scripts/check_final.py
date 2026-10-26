"""第 9 步 成片自检（阶段 3—5 的验收里程序能查的部分）：只检查和报告，不自动改。

用法：uv run python scripts/check_final.py <视频文件夹> [runs/<时间> 文件夹，默认最新一次]

查的是：
1. 文件：1920×1080、H.264 加 AAC、帧率 30/1、帧数对得上、音画时长差不超过 0.1 秒（阶段 4）；
   响度在目标 ±1 LU 以内，峰值不超过 -1 dB。
2. 每句都有画面：每句取开头、中间、结尾三帧，平均亮度不能接近全黑（阶段 3“无黑屏”）；
   四条边不能有整条的黑边（阶段 5“无黑边”）。
3. 预览图和成片一致：预览里每句那一帧，到成片里取同一帧比（字幕那一条不比），差得多就报（阶段 3）。
4. 小窗不挡字幕：最宽那行字幕的左边在小窗右边 20 像素以外（阶段 4）。无人脸模式没有小窗，只报字幕居中的位置。
   竖屏（成片表的 vertical，第二期）：1080×1920，没有小窗，查字幕在不在平台界面挡住的地方以外（右侧按钮那一列、顶部）。
5. 动效：哪几句用了哪种动效、截图聚焦的高亮框画在哪句话上；有动效和无动效的渲染时间比（阶段 5，要求不超过 2 倍）。
6. 配错句数要人看了才知道：列出每句的讲稿原文、素材和 Claude 写的描述，方便对照。
7. 转场（阶段 6）：每处转场取正中那一帧，不能接近全黑、不能有黑边；加转场和不加的渲染时间比（要求不超过 1.5 倍）。
8. 背景音乐（阶段 6）：只有音乐的那一轨（bgm_track.wav）每 10 秒取中间 1 秒，不能是静音；来源清单里有音乐那一条。
另写 来源清单.md：成片里用到的第三方素材和背景音乐的来源（画面上不标来源，发布时贴进简介）。
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from common import CONFIG, FFMPEG, FPS, ffprobe, latest_run, read_json, write_json
from composite import video_frames
from effects import H, W

BLACK = 16          # 平均亮度低于 16（满分 255）算黑屏
EDGE = 8            # 边上 8 像素宽的一条
SAME_DIFF = 12      # 预览和成片同一帧，平均每像素差不超过 12（压缩带来的差别）算一致
BGM_QUIET_REL = -30  # 音乐轨一秒的平均音量比整首（去掉淡入淡出）低 30 dB 以上算没声音


def frame_at(path: Path, k: int):
    return next(video_frames(path, k, 1))


def luma(img) -> np.ndarray:
    a = np.asarray(img.convert("L"), dtype=np.float32)
    return a


def dark_or_bars(L: np.ndarray) -> tuple[bool, list]:
    """这一帧是不是接近全黑、哪几条边是整条黑边。"""
    strips = {"上": L[:EDGE], "下": L[-EDGE:], "左": L[:, :EDGE], "右": L[:, -EDGE:]}
    return L.mean() < BLACK, [side for side, st in strips.items() if st.max() < 20]


def source_lines(comp: dict, bgm: dict | None) -> tuple[list, list]:
    """来源清单（画面上不标，发布时贴进简介）：按每句实际用的那个文件记（compose.py 写进成片表的 source）。
    返回 (清单的行, 缺链接这类问题)。"""
    lines, problems = [], []
    for sc in comp["scenes"]:
        so = sc.get("source")
        if not so or sc["kind"] == "missing":
            continue
        who = f"{so['media']}，作者 {so.get('author')}" if so["media"] == "Pixabay" else f"{so['media']}，{so.get('date')}"
        lines.append(f"- 第 {sc['sentence']} 句：{who}，{so.get('url') or '（缺链接）'}")
        if not so.get("url"):
            problems.append(f"来源：第 {sc['sentence']} 句用了 {so['media']} 的素材，但没记链接，补进素材清单的 url")
    if bgm:
        # bgm.json 的五项 render_final.py 渲染前已经查过，这里照抄；几首轮流放就一首一行，片子短、没轮到的不列
        for t in [t for t in bgm.get("tracks") or [bgm] if t.get("plays", 1)]:
            lines.append(f"- 背景音乐：《{t['title']}》，{t['author']}，{t['license']}，{t['url']}（{t['date']} 获取）")
    return lines, problems


def main() -> None:
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    folder = Path(sys.argv[1]).expanduser().resolve()
    run = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else latest_run(folder)
    comp = read_json(run / "composition.json")
    rep = read_json(run / "render_report.json")
    if "effects" not in rep:
        sys.exit("还没渲染过成片，先跑 render_final.py")
    from common import archive, sha256
    if rep["effects"].get("composition_sha256") != sha256(run / "composition.json"):
        sys.exit("成片表在渲染之后改过（重跑过 compose.py），成片不是按现在的成片表做的：先重跑 render_final.py")
    name = rep["effects"]["file"]
    out = run / name
    total = comp["total_frames"]
    problems = []

    # 1. 文件
    info = ffprobe(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), {})
    nb, vdur, adur = int(v.get("nb_frames", 0)), float(v.get("duration", 0)), float(a.get("duration", 0))
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", str(out), "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    tail = r.stderr[r.stderr.rfind("Summary:"):]
    lufs = float(re.search(r"I:\s+(-?[\d.]+) LUFS", tail).group(1))
    peak = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", tail).group(1))
    target = comp["loudness_lufs"]
    # 画面时间戳要一帧一帧连续：中间跳了，后面的画面就和声音错开（10-08 的成片在第 8 句开头跳了 3 帧）
    raw = subprocess.run([CONFIG["ffprobe"], "-v", "error", "-select_streams", "v", "-show_entries", "frame=pts_time",
                          "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout.split()
    raw = [t.strip(",") for t in raw if t.strip(",")]
    pts = [float(t) for t in raw if re.fullmatch(r"-?[\d.]+", t)]
    jumps = [round(pts[i - 1] * FPS) + 1 for i in range(1, len(pts)) if abs((pts[i] - pts[i - 1]) * FPS - 1) > 0.05]
    pts_ok = len(pts) == nb and nb > 0 and not jumps
    pts_note = ("连续" if pts_ok else f"第 {'、'.join(map(str, jumps[:5]))} 帧处跳了" if jumps
                else f"读到 {len(pts)} 个时间戳（有 {len(raw) - len(pts)} 个读不出），应为 {nb} 个")
    files = [
        (f"画面 {comp['size'][0]}×{comp['size'][1]}", [v["width"], v["height"]] == list(comp["size"]), f"{v['width']}×{v['height']}"),
        ("视频 H.264、音频 AAC", v.get("codec_name") == "h264" and a.get("codec_name") == "aac", f"{v.get('codec_name')} + {a.get('codec_name')}"),
        ("帧率 30/1", v.get("r_frame_rate") == "30/1", v.get("r_frame_rate")),
        ("帧数等于粗剪帧数（差不超过 1 帧）", abs(nb - total) <= 1, f"{nb}，应为 {total}"),
        ("音画时长差不超过 0.1 秒", abs(vdur - adur) <= 0.1, f"差 {abs(vdur - adur):.3f} 秒"),
        ("画面时间戳逐帧连续（不跳帧）", pts_ok, pts_note),
        (f"响度在 {target}±1 LUFS", abs(lufs - target) <= 1, f"{lufs} LUFS"),
        ("峰值不超过 -1 dBFS", peak <= -1.0, f"{peak} dBFS"),
    ]
    problems += [f"文件：{n}（{d}）" for n, ok, d in files if not ok]

    # 2. 每句有画面、没黑边。无人脸模式草稿里缺素材的句子本来就是黑底“待补素材”，不查
    no_face = bool(comp.get("no_face"))
    blank = {sc["sentence"] for sc in comp["scenes"] if no_face and sc["kind"] == "missing"}
    rows = []
    tf = rep.get("transition_frames") or 0
    for sc in comp["scenes"]:
        if sc["sentence"] in blank:
            rows.append((sc, True, []))
            continue
        a_, b = sc["start_frame"], sc["end_frame"]
        # 有转场时，前 tf 帧是上一句和这一句的混合画面，“开头”那一帧取在转场结束以后（新画面完全出现的第一帧）
        a_in = a_ + (tf if rep.get("transition_count") and sc.get("transition") else 0)
        picks = sorted({min(a_in, b - 1), (a_ + b) // 2, max(a_, b - 4)})
        dark, bars = [], []
        for k in picks:
            is_dark, b_ = dark_or_bars(luma(frame_at(out, k)))
            if is_dark:
                dark.append(k)
            bars += b_
        ok = not dark and not bars
        if dark:
            problems.append(f"第 {sc['sentence']} 句：第 {'、'.join(map(str, dark))} 帧接近全黑")
        if bars:
            problems.append(f"第 {sc['sentence']} 句：画面{'、'.join(sorted(set(bars)))}边有黑边")
        rows.append((sc, ok, sorted(set(bars))))

    # 3. 预览和成片一致：拿 预览.jpg 里每句那一格，和成片同一帧缩成同样大小比（底下字幕那一条不比）
    from PIL import Image
    sheet = Image.open(run / "预览.jpg").convert("L")
    tw, th, cols, pad, lab = 640, 360, 3, 10, 34  # 和 compose.py 拼预览图的排法一致
    if comp.get("preview_layout"):  # 竖屏的排法 compose.py 记在成片表里
        pl = comp["preview_layout"]
        (tw, th), cols, pad, lab = pl["tile"], pl["cols"], pl["pad"], pl["label"]
    same = []
    for i, p in enumerate(comp.get("preview", [])):
        x0, y0 = pad + (i % cols) * (tw + pad), pad + (i // cols) * (th + lab + pad)
        A = np.asarray(sheet.crop((x0, y0, x0 + tw, y0 + th)), dtype=np.float32)[: int(th * 0.8)]
        B = np.asarray(frame_at(out, p["frame"]).convert("L").resize((tw, th), Image.BICUBIC), dtype=np.float32)[: int(th * 0.8)]
        d = float(np.abs(A - B).mean())
        same.append((p["sentence"], p["frame"], round(d, 1), d <= SAME_DIFF))
        if d > SAME_DIFF:
            problems.append(f"第 {p['sentence']} 句：预览和成片同一帧差得多（平均每像素差 {d:.1f}）")

    # 4. 小窗不挡字幕（无人脸模式没有小窗）；竖屏查平台界面
    sub = rep["subtitles"]
    vertical = bool(comp.get("vertical"))
    if vertical and not sub.get("clear_of_ui"):
        problems.append(f"字幕：竖屏字幕进了平台界面挡住的地方（右边到 {sub['right_px']} 像素，限 {sub['right_limit_px']}；"
                        f"最上面在 {sub['top_px']} 像素，限 {comp['vertical']['safe']['top']}）")
    if not no_face and not vertical and not sub["clear_of_pip"]:
        problems.append(f"字幕：最宽那行左边在 {sub['left_px']} 像素"
                        + (f"、最宽那行英文左边在 {sub['en_left_px']} 像素" if sub.get("english") else "")
                        + f"，离小窗右边（{sub['pip_right_px']}）不到 20 像素")

    # 5. 动效和渲染时间：只拿同一份成片表、同一首音乐出的两版比
    def comparable(key: str) -> bool:
        o = rep.get(key)
        return bool(o) and all(o.get(f) == rep["effects"].get(f) for f in ("composition_sha256", "bgm_sha256"))

    ratio = None
    if comparable("no_effects"):
        ratio = rep["effects"]["seconds"] / max(0.1, rep["no_effects"]["seconds"])
        if ratio > 2:
            problems.append(f"动效：有动效的渲染时间是无动效的 {ratio:.2f} 倍，超过 2 倍")

    # 7. 转场：每处取正中一帧和最后一帧
    tr_rows = []
    for sc in comp["scenes"] if rep.get("transition_count") else []:
        if not sc.get("transition") or sc["sentence"] in blank:
            continue
        ok = True
        for k in sorted({sc["start_frame"] + tf // 2, sc["start_frame"] + tf - 1}):
            is_dark, b_ = dark_or_bars(luma(frame_at(out, k)))
            ok = ok and not is_dark and not b_
            if is_dark:
                problems.append(f"转场：进入第 {sc['sentence']} 句的{sc['transition_label']}，第 {k} 帧接近全黑")
            if b_:
                problems.append(f"转场：进入第 {sc['sentence']} 句的{sc['transition_label']}，第 {k} 帧{'、'.join(b_)}边有黑边")
        tr_rows.append((sc, ok))
    tr_ratio = None
    if comparable("no_transitions"):
        tr_ratio = rep["effects"]["seconds"] / max(0.1, rep["no_transitions"]["seconds"])
        if tr_ratio > 1.5:
            problems.append(f"转场：加转场的渲染时间是不加的 {tr_ratio:.2f} 倍，超过 1.5 倍")

    # 8. 背景音乐：音乐那一轨每 10 秒取中间 1 秒
    bgm = rep.get("bgm")
    bgm_rows, quiet_db = [], None
    if bgm:
        track = run / bgm["track"]
        if not track.exists() or sha256(track) != bgm.get("track_sha256"):
            problems.append(f"背景音乐：{bgm['track']} 和这次成片对不上（不见了或被改过），重跑 render_final.py")
        else:
            x, sr = sf.read(track, dtype="float32", always_2d=True)
            dur = len(x) / sr
            body = x[int(bgm["fade_in"] * sr):max(int(bgm["fade_in"] * sr) + 1, len(x) - int(bgm["fade_out"] * sr))]
            # 门限按这首音乐自己的平均音量算：比它低 30 dB 才算没声音（安静的段落不误报），最低不低于 -70 dBFS
            quiet_db = max(-70.0, float(20 * np.log10(np.sqrt(np.mean(body ** 2)) + 1e-12)) + BGM_QUIET_REL)
            for t0 in np.arange(0, dur, 10.0):
                a_, b = t0 + 4.5, min(t0 + 5.5, dur - bgm["fade_out"])  # 结尾淡出那 2 秒不算
                if b - a_ < 0.5:
                    continue
                seg = x[int(a_ * sr):int(b * sr)]
                db = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)
                bgm_rows.append((a_, round(float(db), 1), db > quiet_db))
                if db <= quiet_db:
                    problems.append(f"背景音乐：{a_:.1f}—{b:.1f} 秒音乐轨没声音（{db:.1f} dBFS）")

    src_lines, src_problems = source_lines(comp, bgm)
    problems += src_problems
    archive(run, ["成片自检.md", "来源清单.md", "check_final.json"])
    (run / "来源清单.md").write_text("# 素材来源（发布时贴进视频简介）\n\n" + ("\n".join(src_lines) or "没有第三方素材。") + "\n",
                                   encoding="utf-8")

    L = [f"# 成片自检：{run.name}/{name}", "",
         f"{'草稿（有句子缺素材）' if comp['draft'] else '正式'}；{len(comp['scenes'])} 句，{total / FPS:.2f} 秒。"
         f"程序查出的问题 {len(problems)} 条。本报告只列问题，不自动改。", ""]
    todo = "黑底" if no_face else "全屏人脸"
    if comp["draft"]:
        L += [f"缺素材、放了{todo}的：第 {'、'.join(map(str, comp['missing_sentences']))} 句。", ""]
    if problems:
        L += ["程序查出的问题：", ""] + [f"- {p}" for p in problems] + [""]
    L += ["## 1. 文件", "", "| 检查 | 结果 | 实际 |", "| --- | --- | --- |"]
    L += [f"| {n} | {'通过' if ok else '不通过'} | {d} |" for n, ok, d in files]
    L += ["", "## 2. 每句的画面（判断配错没有，就看这张表）", "",
          "| 句 | 时间 | 讲稿原文 | 素材 | 动效 | 画面描述 | 黑屏黑边 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for sc, ok, bars in rows:
        eff = sc["effect"] + ("（高亮：" + (sc.get("params") or {}).get("target", "highlight") + "）" if sc["effect"] == "截图聚焦" else "")
        L.append(f"| {sc['sentence']} | {sc['start_frame'] / FPS:.1f}—{sc['end_frame'] / FPS:.1f} 秒 | {sc['script_text']} | "
                 f"{sc['asset']}{f'（缺，{todo}）' if sc['kind'] == 'missing' else ''} | {eff} | {sc['desc']} | {'待补' if sc['sentence'] in blank else '正常' if ok else '有问题'} |")
    L += ["", "## 3. 预览图和成片", "", "| 句 | 帧 | 平均每像素差 | 一致 |", "| --- | --- | --- | --- |"]
    L += [f"| {s} | {k} | {d} | {'是' if ok else '否'} |" for s, k, d, ok in same]
    en_note = (f"；英文字幕在中文下面，字号 {sub['en_font_size']}，最宽一行约 {sub['en_widest_px']} 像素（“{sub['en_widest_text']}”），左边在 {sub['en_left_px']} 像素"
               if sub.get("english") else "")
    if vertical:
        safe = comp["vertical"]["safe"]
        L += ["", "## 4. 小窗和字幕", "",
              f"竖屏 {comp['size'][0]}×{comp['size'][1]}，不叠小窗。字幕字号 {sub['font_size']}，最宽一行 {sub['widest_chars']:g} 个字约 {sub['widest_px']} 像素，"
              f"右边到 {sub['right_px']} 像素（右侧按钮那一列从 {sub['right_limit_px']} 起）；字幕从 {sub['top_px']} 到 {sub['bottom_px']} 像素"
              f"（底部界面从 {comp['size'][1] - safe['bottom']} 起）{en_note}：{'不进平台界面' if sub.get('clear_of_ui') else '进了平台界面'}。", ""]
    elif no_face:
        L += ["", "## 4. 小窗和字幕", "",
              f"无人脸模式：不叠小窗。字幕水平居中，字号 {sub['font_size']}，最宽一行 {sub['widest_chars']:g} 个字约 {sub['widest_px']} 像素，"
              f"左右各留约 {sub['left_px']} 像素{en_note}。", ""]
    else:
        L += ["", "## 4. 小窗和字幕", "",
              f"小窗直径 {comp['pip']['diameter']} 像素，在左下角（右边到 {sub['pip_right_px']} 像素）；字幕字号 {sub['font_size']}，"
              f"最宽一行 {sub['widest_chars']:g} 个字约 {sub['widest_px']} 像素，左边在 {sub['left_px']} 像素{en_note}"
              f"：{'不挡' if sub['clear_of_pip'] else '太近'}。", ""]
    L += ["## 5. 动效和渲染时间", ""]
    fx = {}
    for sc in comp["scenes"]:
        fx.setdefault(sc["effect"], []).append(sc["sentence"])
    L += [f"- {k}：第 {'、'.join(map(str, v))} 句" for k, v in fx.items()]
    L.append(f"- 渲染用时：有动效 {rep['effects']['seconds']} 秒" +
             (f"，无动效 {rep['no_effects']['seconds']} 秒，是 {ratio:.2f} 倍（要求不超过 2 倍）" if ratio else
              "；无动效的还没跑，或者是按别的成片表、别的音乐出的（重跑 render_final.py --no-effects）"))
    L += ["", "## 6. 转场", ""]
    if "transition_frames" in comp:
        by = {}
        for sc in comp["scenes"][1:]:
            by.setdefault(sc.get("transition_label") or "无", []).append(sc["sentence"])
        L += [f"- {k}：进入第 {'、'.join(map(str, v))} 句" for k, v in by.items()]
        hard = [f"第 {sc['sentence']} 句（{sc['transition_note']}）" for sc in comp["scenes"] if "硬切" in sc.get("transition_note", "")]
        if hard:
            L.append(f"- 太短改硬切的：{'、'.join(hard)}")
        L.append(f"- 每处 {tf} 帧（{tf / FPS:.2f} 秒），共 {len(tr_rows)} 处；正中和最后一帧都不黑、没黑边：{'是' if all(ok for _, ok in tr_rows) else '否'}")
    else:
        L.append("- 这份成片没有转场（成片表是阶段 6 之前出的）。")
    L.append(f"- 渲染用时：加转场 {rep['effects']['seconds']} 秒" +
             (f"，不加 {rep['no_transitions']['seconds']} 秒，是 {tr_ratio:.2f} 倍（要求不超过 1.5 倍）" if tr_ratio
              else "；不加转场的还没跑，或者是按别的成片表、别的音乐出的（重跑 render_final.py --no-transitions）"))
    L += ["", "## 7. 背景音乐", ""]
    if bgm:
        tracks = bgm.get("tracks") or [bgm]
        sw = bgm.get("switch_seconds")
        unit = "段" if sw else "遍"
        L += [f"《{t['title']}》，{t['author']}，{t['license']}。去掉首尾静音 {t['music_seconds']} 秒，"
              f"{'放了 ' + str(t['plays']) + ' ' + unit if t['plays'] else '没轮到'}。" for t in tracks]
        swt = ("依次每段 " + "、".join(f"{v:g}" for v in sw) + " 秒，之后一直每首 " + f"{sw[-1]:g}" + " 秒") if isinstance(sw, list) else (f"每首 {sw:g} 秒" if sw else "")
        how = (f"几首按时间轮流放：{swt}换下一首，再轮到时接着上次停的地方放，换歌处交叉淡化 {bgm.get('crossfade', 2):g} 秒"
               if sw else "几首按顺序轮流放、循环，换歌处交叉淡化 " + format(bgm.get('crossfade', 0.5), 'g') + " 秒" if len(tracks) > 1 else "一首循环")
        L += ["", f"{how}；"
              f"开头 {bgm['fade_in']:g} 秒淡入、结尾 {bgm['fade_out']:g} 秒淡出。", "",
              f"成片里量到：人声约 {bgm['voice_lufs_final']} LUFS，音乐 {bgm['music_lufs_final']} LUFS，音乐比人声低 {bgm['below_voice_db']:g} dB"
              f"（要求 {bgm['target_below_voice_db']:g} dB，config.yaml 的 bgm_lufs 定）。", ""]
        if bgm_rows:
            L.append(f"音乐轨每 10 秒取中间 1 秒：{len(bgm_rows)} 段，都有声音：{'是' if all(ok for *_, ok in bgm_rows) else '否'}"
                     f"（最小 {min(d for _, d, _ in bgm_rows)} dBFS，低于 {quiet_db:.1f} 算没声音）。人声清不清楚要你自己听。")
    else:
        L.append("没加背景音乐（素材/ 里没有 bgm.*）。")
    L += ["", "画面会不会抖、高亮框有没有落在要的句子上，程序按坐标算，最后要你看成片确认。", "",
          f"来源清单：{run.name}/来源清单.md（画面上不标来源，发布时贴进简介）。", ""]
    (run / "成片自检.md").write_text("\n".join(L), encoding="utf-8")
    write_json(run / "check_final.json", {"problems": problems, "lufs": lufs, "peak": peak, "ratio": ratio,
                                          "transition_ratio": tr_ratio})
    print(f"成片自检完成：程序查出的问题 {len(problems)} 条 → {run / '成片自检.md'}")
    for p in problems:
        print("  - " + p)


if __name__ == "__main__":
    main()
