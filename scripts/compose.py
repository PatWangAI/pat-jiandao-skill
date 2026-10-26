"""阶段 3 配素材和第 7 步预览：按 Claude 写的 work/shots.json，把每句话配上素材和动效，写出成片表，出预览图。

用法：uv run python scripts/compose.py <视频文件夹> [runs/<时间> 文件夹，默认最新一次] [--draft]

work/shots.json（Claude 看过转写稿、讲稿和素材后写）：
  {"shots": [{"sentence": 1, "from_token": 0, "to_token": 32, "asset": "s01", "effect": "截图聚焦",
              "params": {"target": "highlight"}, "desc": "看到的画面，一句话"}]}
  - sentence 是讲稿第几句；from_token、to_token 是这句话在转写稿里从第几个字到第几个字（含两端）。
    每个要进字幕的字恰好属于一句，顺序不变（和字幕同一个检查）；讲稿里没说的句子不写。
  - asset 是素材清单里的 id（s01），或 素材/ 里的文件名。用户手动补的文件优先，其次是自动拿到的（主方案或备选）。
  - effect：图片可用 无、推拉、模糊垫底；网页整页截图还可以用 截图聚焦（params.target 是 highlight 或 headline）；
    视频只能用 无（铺满）。推拉的 params.direction 是 in（推近，默认）或 out；
    params.keep 是推拉时一定要留在画面里的区域 [x0, y0, x1, y1]（占图宽高的比例，比如自己画的框），可以不写：
    不写时图里有黄框（和截图聚焦一样的样子）就自动找出来当 keep，整页截图按推近的那句。没有小窗（无人脸、竖屏）时推近不往右偏。
  - desc：Claude 看过素材后写的一句描述，查配错时对照。
  - transition（可以不写，阶段 6）：进入这一句用哪种转场，只能是 溶解、滑入、擦除、圆形展开、柔和推移、无（硬切）。
    Claude 按段落指定，同一段落写同一种；不写就按这个顺序逐句轮换（不轮到“无”）。第一句没有转场，写了也不用。
    竖屏里前后两句都是“人脸”、又没写的，默认硬切（画面本来就连着）。
无人脸模式（preprocess.json 的 no_face，2026-10-18 加）：不叠小窗，不读 work/pip.json，不出 小窗样帧.jpg，
  也不用 video_concat.mp4；缺素材的句子放黑底“待补素材”。
竖屏（inputs.json 的 vertical，第二期，2026-10-24 选的摆法 C）：成片 1080×1920，不叠小窗。asset 写“人脸”的句子
  人脸全屏（effect 写“无”，无人脸模式不能用）；竖的图片、竖的视频铺满竖屏（竖图写的“模糊垫底”按“无”铺满）；横的素材照 16:9
  的动效出帧，放进竖屏中间、上下垫模糊的同一画面。成片表记 vertical（几何）、每句记 vfill（是不是铺满）。
work/pip.json（可以没有）：{"crop": [x, y, 边长]}，人脸小窗从工作视频里取哪一块正方形（像素）；
  还可以有 "track": [[粗剪秒数, x, y], …]（scripts/pip_track.py 自动跟踪写的），小窗跟着脸走，边长用 crop 的。

每句的起点：这句第一个字在粗剪里开始前 0.1 秒（字的开始落在静音里时，先挪到静音结束，和字幕同一规则），
对齐到画面帧；第一句从 0 开始，最后一句到片尾，所以整片每一帧都有画面。
转场（阶段 6）：每句进入时的转场写进成片表，长度是 config.yaml 的 transition_seconds 按帧取整；
这一句短于两倍转场长度时，进入这句改成硬切，免得转场占掉大半句；transition_seconds 写 0 就整片硬切。
有句子缺素材时：不加 --draft 就停下列出缺哪几句；加 --draft 出草稿，缺的句子放全屏人脸（无人脸模式是黑底）、标“待补素材”。

输出在 runs/<时间>/：composition.json（成片表，render_final.py 只认它）、预览.jpg（每句一帧，带小窗）、小窗样帧.jpg（无人脸模式没有）。
"""

import argparse
import re
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import (CONFIG, FPS, ROOT, add_timing, analyze, archive, join_text, keep_segments, latest_run, read_json, sha256,
                    to_rough, write_json)
from composite import (crop_at, default_crop, overlay_pip, pip_geometry, todo_frame, vertical_bg, vertical_face,
                       vertical_layout, vertical_wrap, video_frames, video_info)
from effects import Still, find_marks, make
from fetch_assets import VIEW_H, used
from make_subtitles import check_lines, line_tokens, subtitle_tokens
from prepare import sentences

IMG = {".jpg", ".jpeg", ".png", ".webp"}
VID = {".mp4", ".mov", ".m4v", ".webm"}
EFFECTS = {"无", "推拉", "模糊垫底", "截图聚焦"}
LEAD = 0.1  # 换画面比这句话早 0.1 秒
# 转场（SPEC 第 16 节）：中文名 → FFmpeg xfade 的类型；不用 fadeblack、fadewhite 这类会出黑帧白帧的
TRANSITIONS = {"溶解": "fade", "滑入": "slideleft", "擦除": "wipeleft", "圆形展开": "circleopen", "柔和推移": "smoothleft"}
NO_TRANSITION = "无"


def source_of(part: dict | None) -> dict | None:
    """一份素材结果的来源（媒体、日期、链接）；图库记 Pixabay 和作者；自己做的（AI 图、录屏）没有来源。
    链接用 plan.json 里写的网址（source_url）：截图时地址栏的地址（final_url）会变，比如往下滚自动换成下一篇、
    被拦到验证页、跳到英文版（第 07 期，2026-10-25）。图库没有 plan 里的网址，记 Pixabay 的页面。"""
    if not part:
        return None
    if (part.get("license") or "").startswith("Pixabay"):
        return {"media": "Pixabay", "author": part.get("author"), "url": part.get("final_url")}
    cr = part.get("credit")
    if cr:
        return {"media": cr["media"], "date": cr["date"], "url": part.get("source_url") or part.get("final_url")}
    return None


def manual_source_of(r: dict) -> dict | None:
    """用户手动补的文件的来源：plan.json 这一项写了 manual_source 就照它记，没写就当是从主方案那家拿的。"""
    ms = r.get("manual_source")
    if ms:
        return {"media": ms["media"], "date": ms["date"], "url": ms["url"]}
    return source_of(r)


FACE = "人脸"  # 竖屏里这一句放人脸全屏（第二期）


def portrait(r: dict, mat: Path) -> bool:
    """这项素材是竖的（高大于宽）：竖屏里铺满。网页整页长图不算（按网页的宽度排的，当横的放）。"""
    if r["kind"] == "image":
        with Image.open(mat / r["file"]) as im:
            return im.height > im.width
    if r["kind"] == "video":
        w, h, _ = video_info(mat / r["file"])
        return h > w
    return False


def vertical_preview(sc: dict, n: int, lay: dict, mat: Path, face_video: Path | None) -> tuple[int, Image.Image]:
    """竖屏预览取哪一帧、那一帧长什么样：和 render_final.py 竖屏的出帧同一套（composite 的 vertical_face、vertical_wrap）。"""
    k = n // 2
    if sc["kind"] == "missing":
        face = next(video_frames(face_video, sc["start_frame"] + k, 1)) if face_video else None
        return k, todo_frame(sc["sentence"], face, lay)
    if sc["kind"] == "face":
        return k, vertical_face(next(video_frames(face_video, sc["start_frame"] + k, 1)), lay)
    src, size = mat / sc["file"], tuple(lay["size"])
    if sc["kind"] == "video":
        _, _, vdur = video_info(src)
        local = round(((k / FPS) % vdur) * FPS) if vdur > 0 else 0  # 和成片一样：不够长就循环
        img = next(video_frames(src, local, 1), None) or next(video_frames(src, 0, 1))
        if sc["vfill"]:
            return k, Still(img, "无", size=size).frame(0)
        return k, vertical_wrap(Still(img, "无").frame(0), lay)
    if sc["vfill"]:  # 竖的图片铺满竖屏：推拉照推，别的都按“无”铺满
        return k, Still(src, "推拉" if sc["effect"] == "推拉" else "无", sc["params"], n, FPS, size=size).frame(k)
    fx = make(sc["kind"], src, sc["effect"], sc["params"], n, FPS, sc.get("boxes"), sc.get("scale", 2))
    if sc["effect"] == "截图聚焦":
        k = min(n - 1, round((fx.t_hold + fx.t_push + fx.t_mark + 0.2) * FPS))
    return k, vertical_wrap(fx.frame(k), lay, vertical_bg(fx.frame(0), lay))


def focus_as_page(u: dict) -> dict:
    """长图里的字和量的位置对不上（fetch_assets.py 的 page_check）：截图聚焦改用单屏图。单屏图正好一屏 16:9，
    从整屏推近到那句；位置用单屏图的，只留整个落在这一屏里的（不在这一屏的标题、高亮句当没量到）。"""
    f = u["boxes"]["focus"]
    inside = lambda b: b if b and b["y"] >= 0 and b["y"] + b["h"] <= VIEW_H else None
    page = {"headline": inside(f.get("headline")), "highlight": inside(f.get("highlight")), "article": None}
    return {"kind": "page", "file": u["files"]["focus"], "focus": u["files"]["focus"], "page_image": "单屏图",
            "boxes": {"page": page, "focus": f}, "scale": u["scale"], "outside": False, "source": source_of(u)}


def resolve(asset: str, by_id: dict, mat: Path) -> dict:
    """素材 id 或文件名 → 实际用哪个文件、是什么类型、来源是谁（按实际用的那个文件记）。"""
    src = None
    if asset == FACE:
        return {"kind": "face", "file": None, "source": None}
    if asset in by_id:
        r = by_id[asset]
        if r.get("manual_file"):  # 用户手动补的：来源照 manual_source，没写就是主方案写的那家
            f, src = r["manual_file"], manual_source_of(r)
        elif (u := used(r)) is not None:
            if "page" in u["files"] and (u.get("page_check") or {}).get("use") == "focus":
                return focus_as_page(u)
            if "page" in u["files"]:
                return {"kind": "page", "file": u["files"]["page"], "focus": u["files"]["focus"],
                        "boxes": u["boxes"], "scale": u["scale"], "outside": bool(u.get("highlight_outside_page")),
                        "source": source_of(u)}
            f, src = u["files"]["stock"], source_of(u)
        else:
            return {"kind": "missing", "file": None, "source": None}
    elif (mat / asset).is_file():
        f = asset
        m = re.match(r"(s\d{2}[a-z]?)_(fallback_|manual\.)?", asset)  # s07_fallback_focus.jpg 这类：反查是哪一项的哪一份
        if m and m.group(1) in by_id:
            r = by_id[m.group(1)]
            src = manual_source_of(r) if m.group(2) == "manual." else source_of(r.get("fallback") if m.group(2) else r)
    else:
        raise ValueError(f"素材“{asset}”既不是素材清单里的 id，也不在 素材/ 里")
    ext = Path(f).suffix.lower()
    if ext not in IMG | VID:
        raise ValueError(f"素材文件 {f} 的格式认不出（图片要 jpg、png、webp，视频要 mp4、mov）")
    return {"kind": "image" if ext in IMG else "video", "file": f, "source": src}


def check_shots(shots: list, edl: dict, tok: list, sent_ids: list, by_id: dict, mat: Path,
                face_ok: bool = False) -> tuple[list, list]:
    errs, resolved = [], []
    lines = [{"text": "x", "from_token": s.get("from_token"), "to_token": s.get("to_token")} for s in shots]
    errs += [e.replace("行", "句", 1).replace("要进字幕的字", "要进字幕的字（说出来的字）") for e in check_lines(lines, edl, len(tok))
             if "个字，超过" not in e and "没有文字" not in e]
    prev = 0
    for k, s in enumerate(shots, 1):
        name = f"第 {k} 条（讲稿第 {s.get('sentence')} 句）"
        if s.get("sentence") not in sent_ids:
            errs.append(f"{name}：sentence 不是讲稿里的句号")
        elif s["sentence"] <= prev:
            errs.append(f"{name}：句号要从小到大、不重复")
        else:
            prev = s["sentence"]
        if s.get("effect") not in EFFECTS:
            errs.append(f"{name}：effect “{s.get('effect')}” 不是 {'、'.join(sorted(EFFECTS))} 之一")
        if not s.get("desc"):
            errs.append(f"{name}：没写 desc")
        if "transition" in s and s["transition"] not in (*TRANSITIONS, NO_TRANSITION):
            errs.append(f"{name}：transition “{s['transition']}” 不是 {'、'.join([*TRANSITIONS, NO_TRANSITION])} 之一")
        try:
            r = resolve(str(s.get("asset", "")), by_id, mat)
        except ValueError as e:
            errs.append(f"{name}：{e}")
            r = {"kind": "missing", "file": None}
        eff = s.get("effect")
        if r["kind"] == "face" and not face_ok:
            errs.append(f"{name}：素材“{FACE}”（人脸全屏）只能用在竖屏、有人脸的视频里")
        if r["kind"] == "face" and eff != "无":
            errs.append(f"{name}：人脸全屏只能用“无”，不能用“{eff}”")
        if r["kind"] == "video" and eff != "无":
            errs.append(f"{name}：视频素材只能用“无”（铺满），不能用“{eff}”")
        keep = (s.get("params") or {}).get("keep")
        if keep is not None and not (isinstance(keep, list) and len(keep) == 4 and all(isinstance(v, (int, float)) for v in keep)
                                     and 0 <= keep[0] < keep[2] <= 1 and 0 <= keep[1] < keep[3] <= 1):
            errs.append(f"{name}：params.keep 要写成 [x0, y0, x1, y1]，都是 0 到 1 之间占图宽高的比例，x0 < x1、y0 < y1")
        if eff == "截图聚焦" and r["kind"] not in ("page", "missing"):
            errs.append(f"{name}：截图聚焦只能用在自动截的网页整页图上，这项是{r['kind']}")
        if eff == "截图聚焦" and r["kind"] == "page":
            tgt = (s.get("params") or {}).get("target", "highlight")
            if tgt not in ("highlight", "headline"):
                errs.append(f"{name}：截图聚焦的 target 只能是 highlight 或 headline")
            elif not r["boxes"]["page"].get(tgt) and r.get("page_image") == "单屏图":
                errs.append(f"{name}：这张截图的长图和量的位置对不上，改用了单屏图，单屏图里没有 {tgt}；"
                            f"改用另一个 target、别的动效，或者 fetch_assets.py 加 --only 重截")
            elif not r["boxes"]["page"].get(tgt):
                errs.append(f"{name}：这张截图没记下 {tgt} 的位置，改用另一个 target 或别的动效")
            elif tgt == "highlight" and r.get("outside"):
                errs.append(f"{name}：高亮句在整页长图外面（页面太长），推近会推到图外；改用 target headline 或别的动效")
        resolved.append(r)
    return errs, resolved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("run", nargs="?")
    ap.add_argument("--draft", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    folder = Path(a.folder).expanduser().resolve()
    run = Path(a.run).resolve() if a.run else latest_run(folder)
    inputs = read_json(folder / "inputs.json")
    pre = read_json(folder / "work" / "preprocess.json")
    edl = read_json(run / "edl_final.json")
    tok = read_json(folder / edl["transcript"])["tokens"]
    mat = folder / "素材"
    assets = read_json(mat / "assets.json") if (mat / "assets.json").exists() else {"items": []}
    by_id = {r["id"]: r for r in assets["items"]}
    sents = sentences(Path(inputs["script"]))
    shots_path = folder / "work" / "shots.json"
    if not shots_path.exists():
        sys.exit(f"没有 {shots_path}：先由 Claude 对照转写稿、讲稿和素材写分句和配素材")
    shots = read_json(shots_path)["shots"]
    no_face = bool(pre.get("no_face"))
    vertical = bool(inputs.get("vertical"))
    face_video = None if no_face else run / "video_concat.mp4"
    for need in [p for p in (face_video, run / "粗剪.wav", run / "subtitles.json") if p]:
        if not need.exists():
            sys.exit(f"缺 {need.name}：先跑完阶段 1 的 cut_audio.py、render_rough.py")

    errs, res = check_shots(shots, edl, tok, sorted(sents), by_id, mat, face_ok=vertical and not no_face)
    if errs:
        print("shots.json 校验不通过：")
        for e in errs:
            print("  - " + e)
        sys.exit(1)
    missing = [s["sentence"] for s, r in zip(shots, res) if r["kind"] == "missing"]
    if missing and not a.draft:
        sys.exit(f"这几句还没有素材：第 {'、'.join(map(str, missing))} 句。补好后重跑；想先看效果就加 --draft 出草稿")

    # 每句的起止帧
    import soundfile as sf
    a16, _ = sf.read(folder / "work" / "audio.wav", dtype="float32")
    loud = analyze(a16)
    segs = keep_segments(edl)
    total = sum(b - a_ for a_, b in segs)
    subs = subtitle_tokens(edl, len(tok))
    starts = []
    for s in shots:
        first = line_tokens(s, subs)[0]
        t = tok[first]["start"]
        for qa, qb in loud.quiet:
            if qa <= t <= qb:
                t = qb
                break
        rt = to_rough(t, segs, "next")
        starts.append(round(((rt if rt is not None else total / FPS) - LEAD) * FPS))
    starts[0] = 0
    for k in range(1, len(starts)):
        starts[k] = min(max(starts[k], starts[k - 1] + 1), total - (len(starts) - k))
    ends = starts[1:] + [total]

    # 人脸小窗取景（无人脸模式没有小窗）
    pip_path = folder / "work" / "pip.json"
    pip = None
    if not no_face:
        fw, fh = pre["proxy"]["width"], pre["proxy"]["height"]
        pipj = read_json(pip_path) if pip_path.exists() else {"crop": default_crop(fw, fh)}
        crop, track = pipj["crop"], pipj.get("track") or []
        x, y, side = crop
        for t_, x_, y_ in [(None, x, y)] + [tuple(p) for p in track]:
            if not (0 <= x_ and 0 <= y_ and side > 0 and x_ + side <= fw and y_ + side <= fh):
                sys.exit(f"work/pip.json 的取景 {[x_, y_, side]}{f'（track {t_} 秒）' if t_ is not None else ''} 超出了工作视频的画面（{fw}×{fh}）")
        if any(b[0] <= a_[0] for a_, b in zip(track, track[1:])):
            sys.exit("work/pip.json 的 track 时间要从小到大")
        g = pip_geometry()
        pip = {**g, "crop": crop, **({"track": track} if track else {})}
    lay = None
    if vertical:  # 竖屏不叠小窗；人脸按 pip.json 的取景对准（横的工作视频要裁一条）
        lay = vertical_layout(None if no_face else (pre["proxy"]["width"], pre["proxy"]["height"]),
                              pip["crop"] if pip and pip_path.exists() else None)
        pip = None
        for r in res:  # 竖的图片、视频铺满竖屏，横的放进中间
            r["vfill"] = portrait(r, mat)

    # 每句进入时的转场：写了就用写的，没写按顺序逐句轮换；第一句没有，太短的句子硬切
    tf = max(0, round(CONFIG["transition_seconds"] * FPS))  # 0 表示整片硬切
    order = list(TRANSITIONS)
    scenes = []
    for k, (s, r, a_, b) in enumerate(zip(shots, res, starts, ends)):
        idx = line_tokens(s, subs)
        label = None if k == 0 else s.get("transition") or order[(k - 1) % len(order)]
        note = "第一句" if k == 0 else ("写的" if s.get("transition") else "默认轮换")
        if label and not s.get("transition") and s["asset"] == FACE and shots[k - 1]["asset"] == FACE:
            label, note = NO_TRANSITION, "人脸接人脸，画面本来就连着，硬切"  # 竖屏整片人脸时不逐句轮换转场
        if label and tf == 0:
            label, note = NO_TRANSITION, "config.yaml 的 transition_seconds 是 0，整片硬切"
        elif label and label != NO_TRANSITION and b - a_ < 2 * tf:
            label, note = NO_TRANSITION, f"这句只有 {(b - a_) / FPS:.2f} 秒，短于两倍转场，硬切"
        params = dict(s.get("params") or {})
        if s["effect"] == "推拉" and r["kind"] in ("image", "page"):
            if pip is None:  # 没有小窗（无人脸、竖屏）：推近不往右上偏
                params.setdefault("center", [0.5, 0.5])
            if r["kind"] == "image" and not params.get("keep") and (mk := find_marks(mat / r["file"])):
                params.update(keep=mk, keep_from="图里的黄框")  # 推近时黄框始终在画面里
        sc = {"sentence": s["sentence"], "script_text": sents[s["sentence"]],
              "spoken": join_text([tok[i]["text"] for i in idx]), "from_token": s["from_token"], "to_token": s["to_token"],
              "asset": s["asset"], "desc": s["desc"], "effect": s["effect"], "params": params,
              "start_frame": a_, "end_frame": b, "transition": TRANSITIONS.get(label), "transition_label": label or "",
              "transition_note": note, **r}
        scenes.append(sc)
    comp = {"version": 1, "size": lay["size"] if lay else [1920, 1080], "fps": FPS, "total_frames": total, "draft": bool(missing),
            "edl": "edl_final.json", "edl_sha256": sha256(run / "edl_final.json"),
            "subtitles": "subtitles.json", "subtitles_sha256": sha256(run / "subtitles.json"),
            "shots_sha256": sha256(shots_path), "face_video": face_video.name if face_video else None, "audio": "粗剪.wav",
            "loudness_lufs": CONFIG["loudness_lufs"], "no_face": no_face, "pip": pip, "materials": str(mat), "transition_frames": tf,
            **({"vertical": lay} if lay else {}),
            "scenes": scenes, "missing_sentences": missing,
            "not_spoken": [k for k in sorted(sents) if k not in {s["sentence"] for s in shots}]}

    # 预览：每句取一帧（截图聚焦取画好高亮框之后），和成片用同一套出帧和小窗
    font = ImageFont.truetype(str(ROOT / CONFIG["font_file"]), 22)
    tiles, picks = [], []
    tw, th, cols = (360, 640, 4) if lay else (640, 360, 3)  # 竖屏的预览格是竖的
    for sc in scenes:
        n = sc["end_frame"] - sc["start_frame"]
        if lay:
            k, img = vertical_preview(sc, n, lay, mat, face_video)
        elif sc["kind"] == "missing":
            k = n // 2
            img = todo_frame(sc["sentence"], None if no_face else next(video_frames(face_video, sc["start_frame"] + k, 1)))
        else:
            src = mat / sc["file"]
            if sc["kind"] == "video":
                k = n // 2
                _, _, vdur = video_info(src)
                # 和成片一样：视频不够长就循环，取循环后这一刻的那一帧
                local = round(((k / FPS) % vdur) * FPS) if vdur > 0 else 0
                img = next(video_frames(src, local, 1), None) or next(video_frames(src, 0, 1))
                tmp = run / "_preview_v.jpg"
                img.save(tmp)
                img = Still(tmp, "无").frame(0)
                tmp.unlink()
            else:
                eff = make(sc["kind"], src, sc["effect"], sc["params"], n, FPS, sc.get("boxes"), sc.get("scale", 2))
                k = min(n - 1, round((eff.t_hold + eff.t_push + eff.t_mark + 0.2) * FPS)) if sc["effect"] == "截图聚焦" else n // 2
                img = eff.frame(k)
            if pip:
                face = next(video_frames(face_video, sc["start_frame"] + k, 1))
                img = overlay_pip(img, face, crop_at(pip, (sc["start_frame"] + k) / FPS), pip)
        if sc["transition"] and k < tf:  # 预览要取转场结束以后的画面
            sys.exit(f"第 {sc['sentence']} 句的预览帧落在转场里（第 {k} 帧，转场 {tf} 帧），compose.py 有问题")
        picks.append({"sentence": sc["sentence"], "frame": sc["start_frame"] + k})
        t = img.resize((tw, th), Image.BICUBIC)
        tr = f" ←{sc['transition_label']}" if sc["transition_label"] else ""
        tiles.append((t, f"第{sc['sentence']}句{tr} {sc['asset']} {sc['effect']} {sc['start_frame'] / FPS:.1f}—{sc['end_frame'] / FPS:.1f}秒"))
    comp["preview"] = picks
    pad, lab = 10, (84 if lay else 34)  # 竖屏格子窄，说明按格子宽折行，最多三行

    def wrap(label: str) -> str:
        """竖屏：说明按空格分开，一行放不下就换行。"""
        out, cur = [], ""
        for w in label.split(" "):
            if cur and d.textlength(cur + " " + w, font=font) > tw - 4:
                out.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        return "\n".join(out + [cur])
    if lay:  # 竖屏的排法记进成片表，成片自检按它取格子
        comp["preview_layout"] = {"tile": [tw, th], "cols": cols, "pad": pad, "label": lab}
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + lab + pad) + pad), (30, 32, 36))
    d = ImageDraw.Draw(sheet)
    for k, (t, label) in enumerate(tiles):
        x0, y0 = pad + (k % cols) * (tw + pad), pad + (k // cols) * (th + lab + pad)
        sheet.paste(t, (x0, y0))
        d.text((x0, y0 + th + 2), wrap(label) if lay else label, fill=(235, 235, 235), font=font)
    if old := archive(run, ["composition.json", "预览.jpg", "小窗样帧.jpg"]):
        print(f"上一版的成片表和预览挪进了 {old.name}/")
    sheet.save(run / "预览.jpg", quality=88)
    if pip:
        # 小窗样帧：工作视频的一帧上画出取景框，Claude 看人脸在不在框里
        face = next(video_frames(face_video, total // 2, 1))
        smp = face.copy()
        sx, sy, side = crop_at(pip, total // 2 / FPS)
        ImageDraw.Draw(smp).rectangle([sx, sy, sx + side, sy + side], outline=(255, 0, 0), width=6)
        smp.save(run / "小窗样帧.jpg", quality=85)
    write_json(run / "composition.json", comp)
    add_timing(run, "compose（配素材和预览）", time.time() - t0)
    print(f"成片表写好了：{len(scenes)} 句，{'草稿，缺素材的第 ' + '、'.join(map(str, missing)) + (' 句放黑底' if no_face else ' 句放全屏人脸') if missing else '素材齐全'}"
          f"{'，无人脸模式（不叠小窗）' if no_face else ''}；预览 → {run / '预览.jpg'}")
    if comp["not_spoken"]:
        print(f"注意：讲稿里这几句口播里没说：第 {'、'.join(map(str, comp['not_spoken']))} 句")
    if pip and not pip_path.exists():
        print(f"注意：没有 work/pip.json，人脸小窗用默认取景 {pip['crop']}，看 小窗样帧.jpg 确认人脸在框里")


if __name__ == "__main__":
    main()
