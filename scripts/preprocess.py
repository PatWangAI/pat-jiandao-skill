"""第 3 步 预处理：读一次原片，出三个文件。

用法：uv run python scripts/preprocess.py <视频文件夹>

  work/proxy.mp4     固定 30 帧的工作视频，按旋转信息摆正，长边不超过 1920，没有声音
  work/audio48.wav   48 kHz 声音，成片用
  work/audio.wav     16 kHz 单声道声音，转写和找静音用
  work/preprocess.json  原片和三个文件的信息（帧数、尺寸、方向），后面的脚本都读它

三者都从 0 开始；做完核对时长，相差不到 1 帧才算对齐。声音比画面长或短一点（不到 0.1 秒），
按画面的帧数补静音或截掉，并在 preprocess.json 里记一笔；差得更多就停下。
原片是 HDR 就停（SPEC 第 8 节第 1 条）：请用户关掉 iPhone 的“HDR 视频”重录。

无人脸模式（inputs.json 的 no_face，或原片只有声音；2026-10-18 加）：不用原片的画面，也不查 HDR。
proxy.mp4 换成 1280×720 的黑画面，帧数按声音长度算，后面剪辑表、粗剪、自检照旧按帧走；粗剪就是黑底加字幕。
preprocess.json 记 "no_face": true，compose.py、render_final.py 据此不叠小窗。
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from common import FPS, SR, SR_OUT, ffprobe, picture_stream, read_json, run_ffmpeg, write_json

HDR_TRANSFER = {"arib-std-b67", "smpte2084"}  # iPhone 的 HDR 是 HLG，也可能是 PQ
BLANK_SIZE = (1280, 720)  # 无人脸模式的黑画面工作视频


def probe_audio_only(path: Path) -> dict:
    """无人脸模式只看声音。方向记横屏：工作视频是 16:9 的黑画面，粗剪自检拿它比方向。"""
    info = ffprobe(path)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if a is None:
        sys.exit(f"原片里没有声音：{path}")
    return {
        "duration": float(info["format"]["duration"]), "has_picture": picture_stream(info) is not None,
        "orientation": "横屏", "hdr": False, "start_gap": 0.0,
        "audio": {"codec": a["codec_name"], "sample_rate": int(a.get("sample_rate", 0)),
                  "channels": int(a.get("channels", 1))},
    }


def probe_source(path: Path) -> dict:
    info = ffprobe(path)
    v = picture_stream(info)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if v is None or a is None:
        sys.exit(f"原片里缺{'画面' if v is None else '声音'}：{path}")
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = int(v.get("tags", {}).get("rotate", rot)) % 360
    w, h = v["width"], v["height"]
    if rot in (90, 270):
        w, h = h, w
    return {
        "duration": float(info["format"]["duration"]),
        "codec": v["codec_name"], "coded_size": [v["width"], v["height"]], "rotation": rot,
        "display_size": [w, h], "orientation": "竖屏" if h > w else "横屏",
        "avg_frame_rate": v.get("avg_frame_rate"), "color_transfer": v.get("color_transfer", ""),
        # iPhone 的 HDR 视频还带杜比视界配置（side data 里的 DOVI），有它也算
        "start_gap": round(float(a.get("start_time", 0)) - float(v.get("start_time", 0)), 3),
        "hdr": v.get("color_transfer", "") in HDR_TRANSFER
               or any("DOVI" in sd.get("side_data_type", "") for sd in v.get("side_data_list", [])),
        "audio": {"codec": a["codec_name"], "sample_rate": int(a.get("sample_rate", 0)),
                  "channels": int(a.get("channels", 1))},
    }


def fit_samples(path: Path, sr: int, frames: int) -> float:
    """把声音补静音或截到正好等于画面的长度，返回原来差了多少秒（声音减画面）。"""
    x, got_sr = sf.read(path, dtype="int16", always_2d=True)
    assert got_sr == sr
    want = int(round(frames * sr / FPS))
    diff = (len(x) - want) / sr
    if len(x) < want:
        x = np.concatenate([x, np.zeros((want - len(x), x.shape[1]), dtype=x.dtype)])
    x = x[:want]
    sf.write(path, x if x.shape[1] > 1 else x[:, 0], sr, subtype="PCM_16")
    return diff


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    t0 = time.time()
    folder = Path(sys.argv[1]).expanduser().resolve()
    inputs = read_json(folder / "inputs.json")
    src = folder / inputs["source"]
    work = folder / "work"
    work.mkdir(exist_ok=True)

    # 无人脸模式：inputs.json 写了 no_face（prepare.py --no-face，纯录音自动写），或者原片根本没有画面
    no_face = bool(inputs.get("no_face")) or picture_stream(ffprobe(src)) is None
    src_info = probe_audio_only(src) if no_face else probe_source(src)
    if src_info["hdr"]:
        sys.exit(f"原片是 HDR 视频（{src_info['color_transfer']}），先停下。"
                 "请在 iPhone 的 设置 → 相机 → 录制视频 里关掉“HDR 视频”再重录。")

    ch = min(2, src_info["audio"]["channels"])
    af = "aresample=async=1:first_pts=0"
    audio_outs = [
        "-map", "0:a:0", "-af", af, "-ar", str(SR_OUT), "-ac", str(ch), "-c:a", "pcm_s16le", str(work / "audio48.wav"),
        "-map", "0:a:0", "-af", af, "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", str(work / "audio.wav"),
    ]
    if no_face:
        # 先出声音，按声音长度（取整到帧）出同样帧数的黑画面
        run_ffmpeg(["-i", str(src), *audio_outs])
        n = sf.info(work / "audio48.wav").frames
        bw, bh = BLANK_SIZE
        run_ffmpeg(["-f", "lavfi", "-i", f"color=c=black:s={bw}x{bh}:r={FPS}", "-frames:v", str(max(1, round(n * FPS / SR_OUT))),
                    "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-g", "30",
                    str(work / "proxy.mp4")])
    else:
        # 一条命令读一次原片出三个文件。fps 的 start_time=0 和 aresample 的 first_pts=0 让画面和声音都从 0 开始，
        # 哪边起步晚就在开头补一点；FFmpeg 默认按旋转信息把画面摆正
        vf = ("fps=30:start_time=0,scale=w='min(1920,iw)':h='min(1920,ih)':"
              "force_original_aspect_ratio=decrease:force_divisible_by=2,format=yuv420p")
        run_ffmpeg([
            "-i", str(src),
            "-map", "0:v:0", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-g", "30",
            "-an", str(work / "proxy.mp4"), *audio_outs,
        ])

    p = ffprobe(work / "proxy.mp4")
    v = p["streams"][0]
    start = float(v.get("start_time", 0))
    frames = int(v["nb_frames"])
    notes = []
    # 输出的三个文件起点都是 0（滤镜对齐的），真正要看的是原片里画面和声音起点差了多少，差的部分在开头补上了
    if abs(src_info["start_gap"]) >= 1 / FPS:
        notes.append(f"原片的声音比画面{'晚' if src_info['start_gap'] > 0 else '早'} {abs(src_info['start_gap']):.3f} 秒开始，"
                     "已在晚的一边开头补齐")
    if abs(start) > 1e-3:
        sys.exit(f"工作视频没有从 0 开始（start_time = {start}），对不齐，先停下")
    for name, sr in (("audio48.wav", SR_OUT), ("audio.wav", SR)):
        a = ffprobe(work / name)
        if abs(float(a["streams"][0].get("start_time", 0))) > 1e-3:
            sys.exit(f"{name} 没有从 0 开始，对不齐，先停下")
        diff = fit_samples(work / name, sr, frames)
        if abs(diff) >= 0.1:
            sys.exit(f"{name} 比画面{'长' if diff > 0 else '短'} {abs(diff):.3f} 秒，差得太多，先停下")
        if abs(diff) >= 1 / FPS:
            notes.append(f"{name} 原来比画面{'长' if diff > 0 else '短'} {abs(diff):.3f} 秒，已按画面长度补齐或截掉")
    w, h = v["width"], v["height"]
    orient = "竖屏" if h > w else "横屏"
    if orient != src_info["orientation"]:
        sys.exit(f"工作视频是{orient}，原片显示是{src_info['orientation']}，方向不对，先停下")

    if no_face:
        notes.append("无人脸模式：不用原片的画面，工作视频是黑画面，成片不叠人脸小窗")
    write_json(work / "preprocess.json", {
        "no_face": no_face,
        "source": src_info,
        "proxy": {"width": w, "height": h, "frames": frames, "fps": FPS, "duration": frames / FPS,
                  "orientation": orient},
        "notes": notes,
        "seconds": round(time.time() - t0, 1),
    })
    print(f"预处理完成：{w}×{h}（{orient}），{frames} 帧，{frames / FPS:.2f} 秒，用时 {time.time() - t0:.1f} 秒")
    for n in notes:
        print("注意：" + n)


if __name__ == "__main__":
    main()
