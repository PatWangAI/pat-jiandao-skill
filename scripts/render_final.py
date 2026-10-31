"""第 8 步 渲染成片（阶段 4 合成、阶段 5 动效）：只认 runs/<时间>/composition.json。

用法：uv run python scripts/render_final.py <视频文件夹> [runs/<时间> 文件夹，默认最新一次] [--no-effects] [--no-transitions]

按段渲染再合成（技术说明第 7 节）：
1. 每句一段，渲染成 1920×1080、30 帧的短片段，放在 final_segs/：
   图片按动效出帧（effects.py，和预览图同一套）；视频铺满画面、不够长就循环；缺素材的草稿句放全屏人脸。
2. 转场（阶段 6，SPEC 第 16 节）：下一句要转场时，这一句多渲染一条转场长度的尾巴（视频接着放，图片动效停在最后一帧）。
   每句切成“开头（参与转场）、正文、尾巴”几小段；上一句的尾巴和这一句的开头用 FFmpeg xfade 做成转场小段，
   按“正文、转场、正文……”的顺序拼起来。转场从后一句的起点开始，整片帧数、每句起点都不变。
3. 最后一遍叠左下角圆形小窗（人脸来自阶段 1 剪好的 video_concat.mp4，和声音同一份保留段），
   烧 16:9 版的字幕，配上统一到 -16 LUFS 的声音（整段乘同一个增益，超过 -2 dBFS 的峰用限幅器压住）。
   缺素材的句子不叠小窗（人脸已经是大画面）。
   16:9 整屏人脸的句子（shots.json 的 asset 写“人脸”，2026-10-29 加）：从 video_concat.mp4 接着这句的帧号取，16:9 的
   缩放到 1920×1080，别的比例照缺素材草稿的样子模糊垫底；不叠小窗。和叠小窗的句子之间有转场时，小窗在转场里
   跟着那一句的画面淡出、淡入（不透明度等于转场里那一句画面占的比例，FFmpeg sendcmd 逐帧改 colorchannelmixer 的 aa）。
   无人脸模式（成片表的 no_face，2026-10-18 加）：不叠小窗、不读 video_concat.mp4，最后一遍只烧字幕、配声音；
   缺素材的草稿句放黑底“待补素材”；字幕照旧水平居中，不用再查躲不躲小窗。
   竖屏（成片表的 vertical，第二期，2026-10-24 选的摆法 C）：1080×1920，不叠小窗。人脸句从 video_concat.mp4 裁一块
   放大铺满；横的素材照 16:9 的动效出帧，再放进竖屏中间、上下垫模糊的同一画面（composite.vertical_wrap，视频用同样做法的
   FFmpeg 滤镜）；竖的素材铺满；字幕按竖屏的字号和位置，躲开平台界面（config.yaml 的 vertical_safe）。
4. 背景音乐（阶段 6）：素材/bgm.<扩展名> 加 素材/bgm.json（来源）；几首轮流放时 bgm.json 写 tracks（格式见 find_bgm），
   再写 switch_seconds 就按时间轮流（每首放这么多秒换下一首，下次接着上次停的地方放）。
   音乐去掉首尾静音，比片子短就循环（同一首接头交叉淡化 0.5 秒，几首换歌 2 秒，几首先各自调到一样响），长就截掉；按人声的响度调到比人声低 16 dB（config.yaml 的 loudness_lufs 减 bgm_lufs），开头 1 秒淡入、结尾 2 秒淡出，
   和人声相加后再统一响度。没有 bgm 文件就不加音乐，照旧出片。
--no-effects 把推拉、截图聚焦都换成不动的图，只用来比渲染时间（阶段 5 要求有动效时不超过无动效的 2 倍）。
--no-transitions 全部硬切，只用来比渲染时间（阶段 6 要求加转场不超过不加的 1.5 倍）。

输出：成片.mp4（草稿是 成片_草稿.mp4，无动效是 成片_无动效.mp4，无转场是 成片_无转场.mp4）、final.ass、和成片同名的 .srt（时间和 final.ass 一样，有英文就两行，开源首发加）、final_audio.wav、
bgm_track.wav（只有音乐那一轨，音量和成片里一样，成片自检用）、render_report.json。
对比版本（无动效、无转场）的声音、音乐轨、字幕写在它自己的 final_segs_* 文件夹里，不覆盖正式版的。
同一个运行目录重渲时，上一版先挪进 旧版_<时间>/。
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from concurrent.futures import ProcessPoolExecutor

from common import (CONFIG, FFMPEG, FPS, ROOT, SR_OUT, add_timing, analyze, archive, check_font, latest_run, read_json,
                    run_ffmpeg, sha256, text_width, write_json)
from composite import crop_at, full_face, save_masks, todo_frame, vertical_bg, vertical_wrap, video_frames, video_info
from effects import H, W, Still, make
from render_rough import FONT_EM_RATIO, ass_escape, ass_time, time_lines, write_srt
from PIL import ImageFont

REAL_EM = 1.452  # 思源黑体粗体：字号 / 单字宽，实测（SPEC 第 13 节第 7 条）
WORKERS = 3
X264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
# 每个小段都转成同一种颜色标记（BT.709、有限范围）。图片段、图库视频段原来标记各不相同，拼起来以后最后一遍
# 中途换标记，FFmpeg 会重建滤镜，后面的画面时间戳往后跳 2—3 帧，音画就错开了（10-08 的成片晚了约 0.1 秒）
NORM = ("scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
        "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709")
AUDIO_EXT = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".aiff", ".aif"}
BGM_FIELDS = {"title": "曲名", "author": "作者", "url": "链接", "license": "许可", "date": "获取日期"}
BGM_FADE_IN, BGM_FADE_OUT = 1.0, 2.0   # 秒
BGM_XFADE = 0.5                        # 同一首循环时接头交叉淡化（秒）
BGM_SWITCH_XFADE = 2.0                 # 几首轮流放时，换歌的交叉淡化（秒）
BGM_LEVEL = -23.0                      # 几首轮流放时，先各自调到这个响度（LUFS）再接起来，最后整体再按人声调
BGM_SILENCE_DB = -50                   # 音乐首尾低于 -50 dBFS 的算静音，循环前去掉
BGM_MIN_S = 5                          # 去掉首尾静音后短于 5 秒的音乐不收
BGM_SWITCH_MIN = 10                    # 按时间轮流时，每首至少放 10 秒再换
LIMIT_DB = -2.0                        # 统一响度后，超过 -2 dBFS 的峰由限幅器压住（成片要求峰值不超过 -1 dBFS）


def encode_parts(inp: list[str], pre: str, parts: list[tuple[Path, int, int]], frames=None) -> None:
    """一次解码、按帧号切成几小段分别编码。parts 是 (输出文件, 起帧, 止帧)，止帧不含。
    inp 是 FFmpeg 的输入参数；frames 给了就是一帧一帧的 PIL 图，从标准输入喂进去。pre 是切之前的滤镜（可以为空）。"""
    m = len(parts)
    head = f"[0:v]{pre + ',' if pre else ''}{NORM}" + (f",split={m}" + "".join(f"[s{i}]" for i in range(m)) if m > 1 else "[s0]")
    graph = ";".join([head] + [f"[s{i}]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[o{i}]"
                               for i, (_, a, b) in enumerate(parts)])
    outs = []
    for i, (out, a, b) in enumerate(parts):
        outs += ["-map", f"[o{i}]", *X264, "-frames:v", str(b - a), str(out)]
    cmd = [FFMPEG, "-v", "error", "-y", *inp, "-filter_complex", graph, *outs]
    if frames is None:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"FFmpeg 编码 {parts[0][0].name} 时报错：\n{r.stderr.strip()}")
        return
    n = max(b for _, _, b in parts)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    last, buf = None, None
    try:
        for k, img in enumerate(frames):
            if k >= n:
                break
            if img is not last:  # 同一张图（停住的画面）只转一次字节
                last, buf = img, img.tobytes()
            p.stdin.write(buf)
    except BrokenPipeError:
        pass
    p.stdin.close()
    err = p.stderr.read().decode(errors="replace")
    if p.wait() != 0:
        raise SystemExit(f"FFmpeg 编码 {parts[0][0].name} 时报错：\n{err.strip()}")


RAW_IN = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]


def raw_in(size) -> list[str]:
    return ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-"]


def vwrap_filter(lay: dict) -> str:
    """FFmpeg 版的 composite.vertical_wrap：16:9 的画面放进竖屏中间，背后垫同一画面放大、缩小模糊再放大、压暗。"""
    vw, vh = lay["size"]
    x, y, w, h = lay["material_box"]
    return (f"split=2[vb][vf];[vb]scale=-2:{vh},crop={vw}:{vh},scale={vw // 4}:{vh // 4}:flags=bilinear,gblur=sigma=10,"
            f"scale={vw}:{vh}:flags=bilinear,colorchannelmixer=rr=0.55:gg=0.55:bb=0.55[vbg];"
            f"[vf]scale={w}:{h}:flags=bicubic[vfg];[vbg][vfg]overlay={x}:{y}")


def render_scene_vertical(sc: dict, k: int, run: Path, mat: Path, parts: list, total: int, eff: str, no_face: bool,
                          lay: dict, seg_dir: Path) -> None:
    """竖屏的一句（第二期，摆法 C）。和 compose.vertical_preview 出的预览帧是同一套做法。"""
    size, kind = tuple(lay["size"]), sc["kind"]
    if kind == "missing" and no_face:
        png = seg_dir / f"{k:03d}.png"
        todo_frame(sc["sentence"], None, lay).save(png)
        encode_parts(["-loop", "1", "-framerate", str(FPS), "-t", f"{(total + 1) / FPS:.4f}", "-i", str(png)], "", parts)
    elif kind == "missing":
        frames = (todo_frame(sc["sentence"], face, lay) for face in video_frames(run / "video_concat.mp4", sc["start_frame"], total))
        encode_parts(raw_in(size), "", parts, frames)
    elif kind == "face":  # 人脸全屏：从粗剪的人脸视频里接着这一句的帧号取，尾巴也是往下放的画面
        x, y, w, h = lay["face_crop"]
        encode_parts(["-ss", f"{max(0.0, (sc['start_frame'] - 0.5) / FPS):.6f}", "-i", str(run / "video_concat.mp4"), "-an"],
                     f"crop={w}:{h}:{x}:{y},scale={size[0]}:{size[1]}:flags=bicubic,setsar=1", parts)
    elif kind == "video":
        fit = (f"fps={FPS},scale={size[0]}:{size[1]}:force_original_aspect_ratio=increase,crop={size[0]}:{size[1]},setsar=1"
               if sc["vfill"] else f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,{vwrap_filter(lay)}")
        encode_parts(["-stream_loop", "-1", "-i", str(mat / sc["file"]), "-an"], fit, parts)
    elif sc["vfill"]:  # 竖的图片铺满竖屏：推拉照推，别的按“无”铺满
        n = sc["end_frame"] - sc["start_frame"]
        fx = Still(mat / sc["file"], "推拉" if eff == "推拉" else "无", sc.get("params") or {}, n, FPS, size=size)
        if eff == "推拉":
            encode_parts(raw_in(size), "", parts, (fx.frame(min(i, n - 1)) for i in range(total)))
        else:
            png = seg_dir / f"{k:03d}.png"
            fx.frame(0).save(png)
            encode_parts(["-loop", "1", "-framerate", str(FPS), "-t", f"{(total + 1) / FPS:.4f}", "-i", str(png)], "", parts)
    elif eff in ("无", "模糊垫底"):
        png = seg_dir / f"{k:03d}.png"
        f = make(kind, mat / sc["file"], eff, sc.get("params") or {}, 1, FPS, sc.get("boxes"), sc.get("scale", 2)).frame(0)
        vertical_wrap(f, lay).save(png)
        encode_parts(["-loop", "1", "-framerate", str(FPS), "-t", f"{(total + 1) / FPS:.4f}", "-i", str(png)], "", parts)
    else:  # 推拉、截图聚焦：16:9 的动效一帧一帧放进竖屏，背景用这句第一帧做的，不跟着动
        n = sc["end_frame"] - sc["start_frame"]
        fx = make(kind, mat / sc["file"], eff, sc.get("params") or {}, n, FPS, sc.get("boxes"), sc.get("scale", 2))
        bg = vertical_bg(fx.frame(0), lay)
        encode_parts(raw_in(size), "", parts, (vertical_wrap(fx.frame(min(i, n - 1)), lay, bg) for i in range(total)))


def render_scene(sc: dict, k: int, run: Path, mat: Path, seg_dir: Path, no_fx: bool, head: int, tail: int,
                 no_face: bool = False, lay: dict | None = None) -> dict:
    """渲染一句，切成 开头（head 帧，参与进入这句的转场）、正文、尾巴（tail 帧，这句画面延长出来参与下一个转场）。
    没有转场的那头就不切。返回各小段的文件名。"""
    n = sc["end_frame"] - sc["start_frame"]
    total = n + tail
    names = {"body": f"{k:03d}.mp4"}
    parts = [(seg_dir / names["body"], head, n)]
    if head:
        names["head"] = f"{k:03d}_head.mp4"
        parts.insert(0, (seg_dir / names["head"], 0, head))
    if tail:
        names["tail"] = f"{k:03d}_tail.mp4"
        parts.append((seg_dir / names["tail"], n, total))
    kind, eff = sc["kind"], sc["effect"]
    if no_fx and eff in ("推拉", "截图聚焦"):
        eff = "无"
    if lay:  # 竖屏（第二期）另走一套，16:9 的照下面原样
        render_scene_vertical(sc, k, run, mat, parts, total, eff, no_face, lay, seg_dir)
    elif kind == "missing" and no_face:  # 无人脸模式：黑底“待补素材”，一张图
        png = seg_dir / f"{k:03d}.png"
        todo_frame(sc["sentence"]).save(png)
        encode_parts(["-loop", "1", "-framerate", str(FPS), "-t", f"{(total + 1) / FPS:.4f}", "-i", str(png)], "", parts)
    elif kind == "missing":
        frames = (todo_frame(sc["sentence"], face) for face in video_frames(run / "video_concat.mp4", sc["start_frame"], total))
        encode_parts(RAW_IN, "", parts, frames)
    elif kind == "face":  # 16:9 整屏人脸（2026-10-29）：接着这一句的帧号取，尾巴也是往下放的画面
        face_video = run / "video_concat.mp4"
        fw, fh, _ = video_info(face_video)
        if abs(fw / fh - W / H) < 0.01 * W / H:  # 16:9 的（宽高比差不到 1%）直接缩放到整屏
            encode_parts(["-ss", f"{max(0.0, (sc['start_frame'] - 0.5) / FPS):.6f}", "-i", str(face_video), "-an"],
                         f"scale={W}:{H}:flags=bicubic,setsar=1", parts)
        else:  # 竖的、4:3 的：和预览一样用 full_face 模糊垫底
            encode_parts(RAW_IN, "", parts, (full_face(face) for face in video_frames(face_video, sc["start_frame"], total)))
    elif kind == "video":  # 视频接着放，尾巴也是往下放的画面
        encode_parts(["-stream_loop", "-1", "-i", str(mat / sc["file"]), "-an"],
                     f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1", parts)
    elif eff in ("无", "模糊垫底"):
        png = seg_dir / f"{k:03d}.png"
        make(kind, mat / sc["file"], eff, sc.get("params") or {}, 1, FPS, sc.get("boxes"), sc.get("scale", 2)).frame(0).save(png)
        encode_parts(["-loop", "1", "-framerate", str(FPS), "-t", f"{(total + 1) / FPS:.4f}", "-i", str(png)], "", parts)
    else:  # 推拉、截图聚焦：动效按这句的长度算，尾巴停在最后一帧
        fx = make(kind, mat / sc["file"], eff, sc.get("params") or {}, n, FPS, sc.get("boxes"), sc.get("scale", 2))
        encode_parts(RAW_IN, "", parts, (fx.frame(min(i, n - 1)) for i in range(total)))
    return names


def face_pip(scenes: list, heads: list, tails: list) -> tuple[list, list]:
    """16:9 整屏人脸的句子（2026-10-29）不叠小窗。返回 (不叠小窗的帧范围 [起, 止), 小窗不透明度 [(帧号, 0—1), …])。
    和叠小窗的句子之间有转场时，转场里小窗跟着那一句的画面淡出、淡入：xfade 第 i 帧里前一句占 1 - i/转场帧数，
    小窗的不透明度就取这个比例，转场结束那一帧正好没了（或正好全出来）。硬切、人脸接人脸、接缺素材的句子，小窗直接不叠。
    不透明度只在有淡出淡入时用（sendcmd 改 colorchannelmixer 的 aa）；整段不叠由 overlay 的 enable 按成片的帧算，一帧不差。
    sendcmd 按人脸视频那一路的时间发命令，video_concat.mp4 的时间戳和成片会差一帧（横拍测试片实测，命令晚一帧生效），
    所以改不透明度的命令都放在藏着的那几帧里：淡入前先归零、硬切出来前先设回 1，早一帧晚一帧都看不出来。"""
    pip_on = lambda sc: sc["kind"] not in ("face", "missing")
    hide, pts = [], []
    for k, sc in enumerate(scenes):
        if sc["kind"] != "face":
            continue
        a, b, h, t = sc["start_frame"], sc["end_frame"], heads[k], tails[k]
        if h and k > 0 and pip_on(scenes[k - 1]):
            pts += [(a + i, 1 - i / h) for i in range(h)]
            a += h
        nxt = k + 1 < len(scenes) and pip_on(scenes[k + 1])
        if t and nxt:
            pts += [(a + 1 if b - a > 1 else a, 0.0)]  # 藏着的时候归零，淡入从 0 开始
            pts += [(b + i, i / t) for i in range(t)] + [(b + t, 1.0)]
        else:
            b += t
            if nxt:
                pts.append((max(a, b - 1), 1.0))  # 藏着的最后一帧设回 1，硬切出来就是整个小窗
        hide.append((a, b))
    fading = any(0 < v < 1 for _, v in pts)
    return hide, sorted(pts) if fading else []


def make_transition(seg_dir: Path, a: str, b: str, kind: str, frames: int, out: str) -> str:
    """上一句的尾巴 a 和这一句的开头 b（都是 frames 帧）做成一段 frames 帧的转场。"""
    run_ffmpeg(["-i", a, "-i", b, "-filter_complex",
                f"[0:v][1:v]xfade=transition={kind}:duration={frames / FPS:.6f}:offset=0,{NORM}[v]",
                "-map", "[v]", *X264, "-frames:v", str(frames), out], cwd=seg_dir)
    return out


def normalize(x: np.ndarray, dst: Path, target: float) -> dict:
    """统一响度：整段乘同一个增益到 target LUFS，再用限幅器只压住超过 -2 dBFS 的峰（人声的爆破音之类）。
    不用 loudnorm：它在增益加上去峰值超标时会退回动态模式，背景音乐会跟着人声起伏 1—2 dB（阶段 6 审查实测）。"""
    i = measure_lufs(x)
    gain = target - i
    y = np.ascontiguousarray(x * np.float32(10 ** (gain / 20)), np.float32)
    peak = float(20 * np.log10(np.abs(y).max() + 1e-12))
    ch = y.shape[1]
    # latency=1：补上限幅器预读带来的 5 毫秒延迟，声音不和画面错开；level=0：不自动拉回 0 dB
    r = subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "f32le", "-ar", str(SR_OUT), "-ac", str(ch), "-i", "-",
                        "-af", f"alimiter=limit={10 ** (LIMIT_DB / 20):.6f}:attack=5:release=50:level=0:latency=1",
                        "-c:a", "pcm_s16le", str(dst)], input=y.tobytes(), capture_output=True)
    if r.returncode != 0:
        raise SystemExit("限幅失败：\n" + r.stderr.decode(errors="replace")[-500:])
    z, sr = sf.read(dst, dtype="int16", always_2d=True)
    n = len(x)
    z = np.concatenate([z, np.zeros((max(0, n - len(z)), ch), z.dtype)])[:n]
    sf.write(dst, z if ch > 1 else z[:, 0], sr, subtype="PCM_16")
    return {"measured_input_i": i, "gain_db": round(gain, 2), "peak_before_limit_db": round(peak, 2),
            "limited": peak > LIMIT_DB, "target": target}


def measure_lufs(x: np.ndarray) -> float:
    """一段 48 kHz 的声音（float32，单声道或双声道）的整体响度（LUFS），用 FFmpeg 的 ebur128 量。
    注意单声道和复制成双声道的同一段声音，量出来差 3 dB（BS.1770 按声道相加），所以要按混音时的声道量。"""
    ch = 1 if x.ndim == 1 else x.shape[1]
    r = subprocess.run([FFMPEG, "-hide_banner", "-f", "f32le", "-ar", str(SR_OUT), "-ac", str(ch), "-i", "-",
                        "-af", "ebur128", "-f", "null", "-"], input=np.ascontiguousarray(x, np.float32).tobytes(),
                       capture_output=True)
    err = r.stderr.decode(errors="replace")
    m = re.search(r"I:\s+(-?[\d.]+) LUFS", err[err.rfind("Summary:"):])
    if not m:
        raise SystemExit("量响度失败：\n" + err[-500:])
    return float(m.group(1))


def find_bgm(mat: Path) -> tuple[list[tuple[Path, dict]], float | None] | None:
    """素材/ 里的背景音乐和 bgm.json（文件名不分大小写）。都没有就返回 None（不加音乐）。
    一首：bgm.<扩展名> 加 bgm.json {"title", "author", "url", "license", "date"}。
    几首按顺序循环（比如两首交替）：bgm.json 写 {"tracks": [{"file": "bgm_1.mp3", "title": …}, …]}，
    文件名以 bgm 开头。整首放完再换下一首；再写 "switch_seconds": 30 就是每首放 30 秒换下一首，
    写成一串 [120, 120, 120, 60] 就是依次每段放这么久、写完了按最后一个一直轮，
    下次轮到它时接着上次停的地方放。只有一样、文件和 bgm.json 对不上、扩展名认不出、来源缺项都停下。
    返回 ([(文件, 来源), …], 每首放几秒换，整首放完再换是 None)。"""
    if not mat.is_dir():  # 竖屏整片人脸、没放音乐时可以没有 素材/（开源首发测试片 10-25 撞到）
        return None
    named = [f for f in mat.iterdir() if f.is_file() and f.name.lower().startswith("bgm")]
    metas = [f for f in named if f.name.lower() == "bgm.json"]
    files = sorted(f for f in named if f not in metas)
    odd = [f.name for f in files if f.suffix.lower() not in AUDIO_EXT]
    if odd:
        sys.exit(f"素材/ 里的 {'、'.join(odd)} 格式认不出，背景音乐要 {'、'.join(sorted(AUDIO_EXT))} 之一")
    if not files:
        if metas:
            sys.exit(f"有 {metas[0].name} 但 素材/ 里没有音乐文件（bgm.mp3、bgm_1.mp3 之类）")
        return None
    if not metas:
        sys.exit(f"有 {files[0].name} 但没有 bgm.json：要写曲名、作者、链接、许可、获取日期，来源清单要用")
    try:
        info = read_json(metas[0])
    except ValueError as e:
        sys.exit(f"bgm.json 不是正确的 JSON：{e}")
    if not isinstance(info, dict):
        sys.exit('bgm.json 要写成 {"title": …, "author": …, "url": …, "license": …, "date": …} 这样的一个对象，'
                 '几首就写 {"tracks": [{"file": …, "title": …}, …]}')

    def lacks(meta: dict, name: str) -> None:
        lack = [zh for k, zh in BGM_FIELDS.items() if not str(meta.get(k) or "").strip()]
        if lack:
            sys.exit(f"bgm.json 里{name}缺 {'、'.join(lack)}（键名 {'、'.join(BGM_FIELDS)}）")

    if "tracks" not in info:
        if len(files) > 1:
            sys.exit(f"素材/ 里有 {len(files)} 个背景音乐：{'、'.join(f.name for f in files)}。只用一首就只留一个；"
                     '几首循环就在 bgm.json 里写 "tracks"，按顺序列出每首的 file 和来源')
        if "switch_seconds" in info:
            sys.exit('bgm.json 的 "switch_seconds" 只用在几首轮流放（"tracks"）的时候')
        lacks(info, "")
        return [(files[0], info)], None
    tracks = info["tracks"]
    if not isinstance(tracks, list) or not tracks or not all(isinstance(t, dict) for t in tracks):
        sys.exit('bgm.json 的 "tracks" 要写成一串 {"file": …, "title": …} ')
    by_name = {f.name.lower(): f for f in files}
    out = []
    for k, t in enumerate(tracks, 1):
        f = by_name.get(str(t.get("file") or "").lower())
        if not f:
            sys.exit(f"bgm.json 第 {k} 首的 file “{t.get('file')}” 在 素材/ 里找不到（文件名要以 bgm 开头）")
        lacks(t, f"第 {k} 首")
        out.append((f, t))
    extra = sorted(set(f.name for f in files) - {f.name for f, _ in out})
    if extra:
        sys.exit(f"素材/ 里的 {'、'.join(extra)} 没写进 bgm.json 的 tracks：要用就写进去，不用就挪走")
    sw = info.get("switch_seconds")
    if sw is not None:
        seq = sw if isinstance(sw, list) else [sw]
        if not seq or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v < BGM_SWITCH_MIN for v in seq):
            sys.exit(f'bgm.json 的 "switch_seconds" 要写秒数，或者一串秒数（依次每段放多久，写完了按最后一个一直轮），'
                     f'每个至少 {BGM_SWITCH_MIN}（现在是 {sw!r}）')
        if len(out) < 2:
            sys.exit('bgm.json 的 "switch_seconds" 要两首以上才有用')
    return out, sw


def decode_music(path: Path) -> np.ndarray:
    """一首音乐解码成 48 kHz 双声道，去掉首尾静音。解码失败、全是静音、太短，都停下。"""
    r = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-vn", "-f", "f32le", "-ac", "2", "-ar", str(SR_OUT), "-"],
                       capture_output=True)
    if r.returncode != 0:
        sys.exit(f"背景音乐 {path.name} 解码失败：\n{r.stderr.decode(errors='replace').strip()}")
    m = np.frombuffer(r.stdout, np.float32).reshape(-1, 2)
    if len(m) < BGM_MIN_S * SR_OUT:
        sys.exit(f"背景音乐 {path.name} 只有 {len(m) / SR_OUT:.1f} 秒，太短（至少 {BGM_MIN_S} 秒），循环起来会像嗡嗡声")
    hop = SR_OUT // 100
    rms = np.sqrt(np.mean(m[: len(m) // hop * hop].reshape(-1, hop, 2) ** 2, axis=(1, 2)) + 1e-12)
    loud = np.nonzero(20 * np.log10(rms) > BGM_SILENCE_DB)[0]
    if not len(loud):
        sys.exit(f"背景音乐 {path.name} 全是静音")
    m = m[loud[0] * hop:(loud[-1] + 1) * hop]
    if len(m) < BGM_MIN_S * SR_OUT:
        sys.exit(f"背景音乐 {path.name} 去掉首尾静音后只有 {len(m) / SR_OUT:.1f} 秒，太短（至少 {BGM_MIN_S} 秒）")
    return m


def loop_music(m: np.ndarray, n: int) -> np.ndarray:
    """一首音乐从头循环到至少 n 个采样，接头 0.5 秒等功率交叉淡化。"""
    x = int(BGM_XFADE * SR_OUT)
    ph = np.linspace(0, np.pi / 2, x, dtype=np.float32)[:, None]
    pieces, tail, length = [m[:-x]], m[-x:], len(m) - x
    while length < n:
        pieces += [tail * np.cos(ph) + m[:x] * np.sin(ph), m[x:-x]]
        tail, length = m[-x:], length + len(m) - x
    return np.concatenate(pieces + [tail])


def bgm_bed(tracks: list[tuple[Path, dict]], n: int, switch: float | None = None) -> tuple[np.ndarray, dict]:
    """几首音乐按顺序轮流放、循环或截到 n 个采样。几首先各自调到一样的响度，免得换歌时忽大忽小。
    接头用等功率交叉淡化：同一首循环 0.5 秒，换歌 2 秒。在各句渲染之前调用，音乐有问题先停下。
    switch 给了秒数就按时间轮流：每首放 switch 秒换下一首；再轮到它时接着上次停的地方放，放到头就从头循环。
    switch 也可以是一串秒数（比如“先各放 2 分钟，之后各放 1 分钟”，写 [120, 120, 120, 60]）：
    第 k 段放第 k 个秒数，写完了一直按最后一个。"""
    ms = [decode_music(f) for f, _ in tracks]
    if len(ms) > 1:
        ms = [m * np.float32(10 ** ((BGM_LEVEL - measure_lufs(m)) / 20)) for m in ms]
    if switch:
        seq = switch if isinstance(switch, list) else [switch]
        x = int(BGM_SWITCH_XFADE * SR_OUT)
        ph = np.linspace(0, np.pi / 2, x, dtype=np.float32)[:, None]
        streams = [loop_music(m, n + int(max(seq) * SR_OUT) + x) for m in ms]  # 每首最多用掉片长加一段，够了
        pos, plays, out, k = [0] * len(ms), [0] * len(ms), None, 0
        # 每段 s + x 个采样，前 x 个和上一段的后 x 个叠成换歌的交叉淡化，所以每 s 个采样换一次
        while out is None or len(out) < n:
            i = k % len(ms)
            s = int(seq[min(k, len(seq) - 1)] * SR_OUT)
            seg = streams[i][pos[i]:pos[i] + s + x]
            pos[i] += s + x
            plays[i] += 1
            out = seg if out is None else np.concatenate([out[:-x], out[-x:] * np.cos(ph) + seg[:x] * np.sin(ph), seg[x:]])
            k += 1
        info = {"tracks": [{**{key: meta[key] for key in BGM_FIELDS}, "file": f.name, "sha256": sha256(f),
                            "music_seconds": round(len(m) / SR_OUT, 2), "plays": c}
                           for (f, meta), m, c in zip(tracks, ms, plays)],
                "crossfade": x / SR_OUT, "switch_seconds": switch}
        info["sha256"] = hashlib.sha256(("".join(t["sha256"] for t in info["tracks"]) + f"|{switch}").encode()).hexdigest()
        info["title"] = " / ".join(t["title"] for t in info["tracks"])
        return out[:n].copy(), info
    x = int((BGM_XFADE if len(ms) == 1 else BGM_SWITCH_XFADE) * SR_OUT)
    ph = np.linspace(0, np.pi / 2, x, dtype=np.float32)[:, None]
    # 第一遍去掉最后 x 个采样，之后每遍是“上一遍的结尾和这一遍的开头叠在一起的 x 个采样 + 这一遍的中段”
    pieces, tail, length, plays, k = [ms[0][:-x]], ms[0][-x:], len(ms[0]) - x, [1] + [0] * (len(ms) - 1), 1
    while length < n:
        m = ms[k % len(ms)]
        pieces += [tail * np.cos(ph) + m[:x] * np.sin(ph), m[x:-x]]
        tail, length = m[-x:], length + len(m) - x
        plays[k % len(ms)] += 1
        k += 1
    pieces.append(tail)
    info = {"tracks": [{**{key: meta[key] for key in BGM_FIELDS}, "file": f.name, "sha256": sha256(f),
                        "music_seconds": round(len(m) / SR_OUT, 2), "plays": c}
                       for (f, meta), m, c in zip(tracks, ms, plays)],
            "crossfade": x / SR_OUT}
    info["sha256"] = info["tracks"][0]["sha256"] if len(ms) == 1 else \
        hashlib.sha256("".join(t["sha256"] for t in info["tracks"]).encode()).hexdigest()
    info["title"] = " / ".join(t["title"] for t in info["tracks"])
    return np.concatenate(pieces)[:n].copy(), info


def mix_bgm(v: np.ndarray, bed: np.ndarray, info: dict, target: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """音乐按人声的响度调到低 (loudness_lufs - bgm_lufs) dB，加淡入淡出，和人声相加。
    人声单声道时先复制成双声道再量响度（混音就是这么混的），否则会差 3 dB。返回混好的声音、调好的音乐轨和报告。"""
    n = len(v)
    vv = v if v.shape[1] == 2 else np.repeat(v, 2, axis=1)
    lv, lm = measure_lufs(vv), measure_lufs(bed)
    below = target - CONFIG["bgm_lufs"]
    gain = lv - below - lm
    bed = bed * np.float32(10 ** (gain / 20))
    fi, fo = min(int(BGM_FADE_IN * SR_OUT), n // 2), min(int(BGM_FADE_OUT * SR_OUT), n // 2)
    bed[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None]
    bed[n - fo:] *= np.linspace(1, 0, fo, dtype=np.float32)[:, None]
    info = {**info, "voice_lufs": lv, "music_lufs_before": lm, "gain_db": round(gain, 2), "target_below_voice_db": below,
            "fade_in": BGM_FADE_IN, "fade_out": BGM_FADE_OUT}
    return vv + bed, bed, info


def english_of(lines: list) -> bool:
    """字幕有没有英文（中文下面加一行英文，10-11 加）：每行都写了 en 才算有；只写了一部分就停下列出来。"""
    has = [bool(str(x.get("en") or "").strip()) for x in lines]
    if any(has) and not all(has):
        miss = [str(k) for k, h in enumerate(has, 1) if not h]
        sys.exit(f"subtitles.json 里有的行写了英文 en、有的没写：第 {'、'.join(miss[:20])} 行没写。要英文就每行都写")
    return all(has) and bool(has)


def write_ass(path: Path, lines: list, pip: dict | None, lay: dict | None = None) -> dict:
    """16:9 成片的字幕，水平居中（左右边距一样）。有英文时，英文用小一号的字放在最下面，中文抬高一行放在它上面。
    pip 是 None（无人脸模式）时没有小窗，不查挡不挡。
    竖屏（lay 是成片表的 vertical，第二期）：字号按 vertical_subtitle_char_px，左右各让出右侧按钮那么宽，
    放在底部界面（账号、文案）上面 40 像素，另查有没有进平台界面挡住的地方。"""
    vertical = lay is not None
    W_, H_ = lay["size"] if vertical else (W, H)
    size = round(CONFIG["vertical_subtitle_char_px" if vertical else "subtitle_char_px"] * FONT_EM_RATIO)
    outline = max(3, round(size * 0.06))
    margin_v = lay["safe"]["bottom"] + 40 if vertical else 56
    side = lay["safe"]["right"] if vertical else 80
    en = english_of(lines)
    en_size = CONFIG.get("vertical_subtitle_en_size", 40) if vertical else CONFIG.get("subtitle_en_size", 48)
    en_outline = max(2, round(en_size * 0.06))
    zh_margin = margin_v + round(en_size * 1.3) if en else margin_v
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W_}
PlayResY: {H_}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{CONFIG['font_name']},{size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{outline},0,2,{side},{side},{zh_margin},1
Style: En,{CONFIG['font_name']},{en_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{en_outline},0,2,{side},{side},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    rows = [f"Dialogue: 0,{ass_time(x['start'])},{ass_time(x['end'])},Sub,,0,0,0,,{ass_escape(x['text'])}" for x in lines]
    if en:
        # 英文放第 1 层：同一层里同时出现的两行，libass 会为了避让把后一行往上推，英文就跑到中文上面去了（10-11 实测）
        rows += [f"Dialogue: 1,{ass_time(x['start'])},{ass_time(x['end'])},En,,0,0,0,,{ass_escape(x['en'].strip())}" for x in lines]
    path.write_text(head + "\n".join(rows) + "\n", encoding="utf-8")
    # 小窗挡不挡字幕：最宽那行字的左边，要在小窗右边再往右 20 像素以外
    widest = max((text_width(x["text"]) for x in lines), default=0)
    w_px = widest * size / REAL_EM + 2 * outline
    left = W_ / 2 - w_px / 2
    pip_right = pip["x"] + pip["diameter"] if pip else None
    info = {"font_size": size, "widest_chars": widest, "widest_px": round(w_px), "left_px": round(left),
            "pip_right_px": pip_right, "clear_of_pip": not pip or left >= pip_right + 20, "bottom_margin_px": zh_margin,
            "english": en, "no_face": not pip}
    if en:
        # 英文按字体实际宽度量（字号 / REAL_EM 是一个 em 的像素数，和中文同一套换算）
        font = ImageFont.truetype(str(ROOT / CONFIG["font_file"]), max(1, round(en_size / REAL_EM)))
        top = max(lines, key=lambda x: font.getlength(x["en"].strip()))
        en_px = font.getlength(top["en"].strip()) + 2 * en_outline
        en_left = W_ / 2 - en_px / 2
        info.update(en_font_size=en_size, en_widest_px=round(en_px), en_left_px=round(en_left), en_widest_text=top["en"].strip(),
                    en_clear_of_pip=not pip or en_left >= pip_right + 20, en_bottom_margin_px=margin_v)
        info["clear_of_pip"] = info["clear_of_pip"] and info["en_clear_of_pip"]
    if vertical:  # 竖屏：字幕左右都在右侧按钮那一列以内，最上面一行在顶部界面以下
        sf = lay["safe"]
        widest_all = max(w_px, info.get("en_widest_px", 0))
        top = H_ - zh_margin - size
        info.update(vertical=True, right_px=round(W_ / 2 + widest_all / 2), right_limit_px=W_ - sf["right"],
                    top_px=top, bottom_px=H_ - margin_v,
                    clear_of_ui=W_ / 2 + widest_all / 2 <= W_ - sf["right"] and top >= sf["top"])
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("run", nargs="?")
    ap.add_argument("--no-effects", dest="no_fx", action="store_true")
    ap.add_argument("--no-transitions", dest="no_tr", action="store_true")
    a = ap.parse_args()
    if a.no_fx and a.no_tr:
        sys.exit("--no-effects 和 --no-transitions 一次只用一个（各自和正式版比渲染时间）")
    t0 = time.time()
    folder = Path(a.folder).expanduser().resolve()
    run = Path(a.run).resolve() if a.run else latest_run(folder)
    comp = read_json(run / "composition.json")
    no_face = bool(comp.get("no_face"))
    lay = comp.get("vertical")  # 竖屏（第二期）
    stale = [n for n, f in (("剪辑表", "edl"), ("字幕", "subtitles")) if sha256(run / comp[f]) != comp[f + "_sha256"]]
    if sha256(folder / "work" / "shots.json") != comp["shots_sha256"]:
        stale.append("分句配素材（work/shots.json）")
    if stale:
        sys.exit(f"{'、'.join(stale)} 在出成片表之后改过，先重跑 compose.py")
    edl = read_json(run / comp["edl"])
    tok = read_json(folder / edl["transcript"])["tokens"]
    mat = Path(comp["materials"])
    total = comp["total_frames"]
    n_samples = total * SR_OUT // FPS
    found = find_bgm(mat)  # 音乐文件、来源、解码有问题，都在各句渲染和挪旧版之前停下
    bgm, switch = found if found else (None, None)
    bed_raw, bed_info = bgm_bed(bgm, n_samples, switch) if bgm else (None, None)
    if a.no_tr and "transition_frames" not in comp:
        sys.exit("这份成片表是阶段 6 之前出的，没有转场，不用 --no-transitions")
    trial = a.no_fx or a.no_tr  # 只为比渲染时间出的版本
    name = "成片_无动效.mp4" if a.no_fx else "成片_无转场.mp4" if a.no_tr else ("成片_草稿.mp4" if comp["draft"] else "成片.mp4")
    srt = Path(name).with_suffix(".srt").name  # 和成片同名，播放器会自己带上
    keep = [name] if trial else [name, "final.ass", srt, "final_audio.wav", "bgm_track.wav", "pip_mask.png", "pip_ring.png", "pip_track.cmd",
                                 "pip_fade.cmd", "ring_fade.cmd"]
    if old := archive(run, keep, copy=["render_report.json"]):
        print(f"上一版 {name} 和它的字幕、声音挪进了 {old.name}/")

    # 每次渲染的分段放进带时间的新文件夹，旧的不删（失败的渲染也留着）
    seg_dir = run / f"final_segs{'_nofx' if a.no_fx else '_notr' if a.no_tr else ''}_{time.strftime('%H%M%S')}"
    seg_dir.mkdir()
    # 只为比渲染时间出的版本，声音、音乐轨、字幕都写在自己的分段文件夹里，不碰正式版配套的文件
    work = seg_dir if trial else run
    t1 = time.time()
    scenes = comp["scenes"]
    tf = comp.get("transition_frames", 0)
    kinds = [None if a.no_tr else sc.get("transition") for sc in scenes]  # 进入每句的转场（xfade 类型），None 是硬切
    heads = [tf if kd else 0 for kd in kinds]
    tails = heads[1:] + [0]
    for sc, kd in zip(scenes, kinds):
        tr = f"，进入时{sc['transition_label']}" if kd else ""
        print(f"第 {sc['sentence']} 句：{sc['kind']} {sc['effect']}，{(sc['end_frame'] - sc['start_frame']) / FPS:.1f} 秒{tr}", flush=True)
    # 各句分段同时渲染（3 个进程，每个约 0.3—0.5 GB 内存）；各种对比版本用同样的并行，比时间才公平
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(render_scene, sc, k, run, mat, seg_dir, a.no_fx, heads[k], tails[k], no_face, lay)
                for k, sc in enumerate(scenes)]
        parts = [f.result() for f in futs]
        trans = {k: pool.submit(make_transition, seg_dir, parts[k - 1]["tail"], parts[k]["head"], kinds[k], tf,
                                f"{k:03d}_trans.mp4") for k in range(1, len(scenes)) if kinds[k]}
        trans = {k: f.result() for k, f in trans.items()}
    order = []
    for k, pt in enumerate(parts):
        order += ([trans[k]] if k in trans else []) + [pt["body"]]
    (seg_dir / "list.txt").write_text("".join(f"file '{n}'\n" for n in order), encoding="utf-8")
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "scenes.mp4"], cwd=seg_dir)
    t_scenes = time.time() - t1

    audio = work / "final_audio.wav"
    target = comp["loudness_lufs"]
    v, _ = sf.read(run / comp["audio"], dtype="float32", always_2d=True)
    v = np.concatenate([v, np.zeros((max(0, n_samples - len(v)), v.shape[1]), np.float32)])[:n_samples]
    bgm_info = None
    if bgm:
        mix, track, bgm_info = mix_bgm(v, bed_raw, bed_info, target)
        loud = normalize(mix, audio, target)
        # 音乐轨乘上同一个增益存下来，就是成片里音乐的实际音量；再量一遍，报告写量到的人声和音乐差多少
        track *= np.float32(10 ** (loud["gain_db"] / 20))
        sf.write(work / "bgm_track.wav", track, SR_OUT, subtype="PCM_16")
        ml = measure_lufs(track)
        bgm_info.update(track="bgm_track.wav", track_sha256=sha256(work / "bgm_track.wav"), music_lufs_final=ml,
                        voice_lufs_final=round(bgm_info["voice_lufs"] + loud["gain_db"], 2),
                        below_voice_db=round(bgm_info["voice_lufs"] + loud["gain_db"] - ml, 1))
    else:
        print("素材/ 里没有 bgm.*，这次不加背景音乐")
        loud = normalize(v, audio, target)
    a16, _ = sf.read(folder / "work" / "audio.wav", dtype="float32")
    lines = time_lines(read_json(run / comp["subtitles"])["lines"], edl, tok, analyze(a16))
    check_font([x["text"] for x in lines] + [str(x.get("en") or "") for x in lines] + ["待补素材：第 0 句"], W, H, run)
    sub_info = write_ass(work / "final.ass", lines, None if no_face else comp["pip"], lay)
    write_srt(work / srt, lines, english_of(lines))
    fonts = (ROOT / CONFIG["font_file"]).parent
    if no_face or lay:  # 无人脸模式、竖屏：不叠小窗，只烧字幕、配声音
        ins = ["-i", str(seg_dir / "scenes.mp4"), "-i", audio.name]
        graph, a_map = f"[0:v]subtitles=final.ass:fontsdir={fonts}[v]", "1:a"
    else:
        pm, pr = save_masks(comp["pip"], work)
        g = comp["pip"]
        x, y, s = g["crop"]
        D, X, Y = g["diameter"], g["x"], g["y"]
        # 缺素材的句子（草稿）不叠小窗。有转场时：进入这句的转场里还看得到上一句，小窗留着；
        # 离开这句的转场里还看得到全屏人脸，小窗晚一个转场再出来（16:9 整屏人脸的句子见 face_pip，转场里淡出淡入）
        gaps = [(sc["start_frame"] + heads[k], sc["end_frame"] + tails[k]) for k, sc in enumerate(scenes) if sc["kind"] == "missing"]
        hide, fade = face_pip(scenes, heads, tails)  # 16:9 整屏人脸的句子也不叠小窗，转场里淡出淡入
        gaps += hide
        en = ""
        if gaps:
            en = ":enable='not(" + "+".join(f"between(t,{(a_ - 0.5) / FPS:.4f},{(b - 0.5) / FPS:.4f})" for a_, b in gaps) + ")'"
        crop_f = f"crop={s}:{s}:{x}:{y}"
        if g.get("track"):  # pip_track.py 跟踪取景：每 0.1 秒按插值改一次 crop 的 x、y（和预览用同一个 crop_at）
            cmds = [f"{i / 10:.1f} crop@pip x {c[0]}, crop@pip y {c[1]};"
                    for i in range(int(total / FPS * 10) + 1) for c in [crop_at(g, i / 10)]]
            (work / "pip_track.cmd").write_text("\n".join(cmds) + "\n", encoding="utf-8")
            x0, y0, _ = crop_at(g, 0)
            crop_f = f"sendcmd=f=pip_track.cmd,crop@pip={s}:{s}:{x0}:{y0}"
        c_fade = r_fade = ""
        ring_in = ["-loop", "1"]
        if fade:  # 小窗和白边各自一条 sendcmd，紧挨着要改的 colorchannelmixer，按经过的那一帧的时间改
            for f_name, target in (("pip_fade.cmd", "colorchannelmixer@pipa"), ("ring_fade.cmd", "colorchannelmixer@ringa")):
                (work / f_name).write_text("".join(f"{max(0.0, (k - 0.5) / FPS):.4f} {target} aa {v:.4f};\n" for k, v in fade),
                                           encoding="utf-8")
            c_fade = ",sendcmd=f=pip_fade.cmd,colorchannelmixer@pipa=aa=1"
            r_fade = ",sendcmd=f=ring_fade.cmd,colorchannelmixer@ringa=aa=1"
            ring_in = ["-loop", "1", "-framerate", str(FPS)]  # 白边也按 30 帧走，和小窗同一帧改
        graph = (f"[1:v]{crop_f},scale={D}:{D},format=rgba[f];[2:v]format=gray,scale={D}:{D}[m];"
                 f"[f][m]alphamerge{c_fade}[c];[3:v]format=rgba{r_fade}[r];[0:v][c]overlay={X}:{Y}{en}[o1];[o1][r]overlay={X}:{Y}{en}[o2];"
                 f"[o2]subtitles=final.ass:fontsdir={fonts}[v]")
        ins = ["-i", str(seg_dir / "scenes.mp4"), "-i", str(run / comp["face_video"]), "-loop", "1", "-i", pm.name,
               *ring_in, "-i", pr.name, "-i", audio.name]
        a_map = "4:a"
    t2 = time.time()
    run_ffmpeg([*ins, "-filter_complex", graph, "-map", "[v]", "-map", a_map,
                "-frames:v", str(total), "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(run / name)], cwd=work)
    secs = time.time() - t0
    rep_path = run / "render_report.json"
    rep = read_json(rep_path) if rep_path.exists() else {}
    key = "no_effects" if a.no_fx else "no_transitions" if a.no_tr else "effects"
    rep[key] = {
        "file": name, "seconds": round(secs, 1), "scene_seconds": round(t_scenes, 1),
        "composition_sha256": sha256(run / "composition.json"), "bgm_sha256": bgm_info["sha256"] if bgm_info else None,
        "bgm_lufs": CONFIG["bgm_lufs"] if bgm_info else None,
        "transitions": len(trans),
        "final_pass_seconds": round(time.time() - t2, 1), "at": time.strftime("%Y-%m-%d %H:%M")}
    if not trial:
        rep.update(subtitles=sub_info, loudness=loud, draft=comp["draft"], bgm=bgm_info,
                   transition_frames=tf, transition_count=len(trans))
    write_json(rep_path, rep)
    label = {"no_effects": "无动效", "no_transitions": "无转场"}.get(key, "正式")
    add_timing(run, f"render_final（{label}）", secs)
    music = f"，背景音乐《{bgm_info['title']}》比人声低 {bgm_info['below_voice_db']:g} dB（量的）" if bgm_info else "，没加背景音乐"
    print(f"成片渲染完成：{len(scenes)} 句，转场 {len(trans)} 处{music}{'，无人脸模式（不叠小窗）' if no_face else ''}，"
          f"用时 {secs:.1f} 秒（各句 {t_scenes:.1f} 秒）→ {run / name}")
    if lay and not sub_info["clear_of_ui"]:
        print(f"注意：竖屏字幕进了平台界面挡住的地方（右边到 {sub_info['right_px']} 像素，限 {sub_info['right_limit_px']}；"
              f"最上面在 {sub_info['top_px']} 像素）")
    if not sub_info["clear_of_pip"]:
        print(f"注意：最宽那行字幕的左边在 {sub_info['left_px']} 像素"
              + (f"、最宽那行英文的左边在 {sub_info['en_left_px']} 像素" if sub_info.get("english") else "") + "，离小窗太近")


if __name__ == "__main__":
    main()
