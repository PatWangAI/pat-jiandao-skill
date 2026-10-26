"""人脸小窗跟踪取景：每 0.5 秒找一次脸（macOS 自带的 Vision，scripts/face_track.swift），平滑以后写进 work/pip.json 的 track，
小窗跟着脸走，脸一直在圆框正中。录的时候前后坐姿不一样（比如开头坐得低，45 秒以后坐高了），固定取景顾不了两头。

用法：uv run python scripts/pip_track.py <视频文件夹> [runs/<时间> 文件夹，默认最新一次]
输出 work/pip.json：{"crop": [x, y, 边长], "track": [[粗剪秒数, x, y], …]}，边长全程不变；再出 <运行>/小窗跟踪.jpg（12 帧画上取景圆）给 Claude 核对。
之后从 compose.py 起重跑。要回到固定取景，删掉 pip.json 里的 track 就行。
"""

import argparse
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from common import FFMPEG, ROOT, latest_run, read_json, write_json
from composite import video_info

STEP = 0.5          # 每 0.5 秒找一次脸
SIZE_K = 1.8        # 取景边长 = 脸框高的 1.8 倍（中位数）：整个头加一点肩膀
UP = 0.40           # 圆心放在脸框从上往下 40% 处：脸框只到眉毛，往上挪一点把头发也框进去
MEDIAN_N, MEAN_N = 5, 7   # 先 5 个点取中位数去掉偶尔认错的，再 7 个点（3.5 秒）求平均，小窗不跟着小晃动抖


def smooth(v: list[float]) -> list[float]:
    n = len(v)
    med = [statistics.median(v[max(0, i - MEDIAN_N // 2): i + MEDIAN_N // 2 + 1]) for i in range(n)]
    return [statistics.fmean(med[max(0, i - MEAN_N // 2): i + MEAN_N // 2 + 1]) for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("run", nargs="?")
    a = ap.parse_args()
    folder = Path(a.folder)
    if read_json(folder / "work" / "preprocess.json").get("no_face"):
        sys.exit("这条是无人脸模式（只用声音），成片不叠小窗，不用跟踪取景")
    run = Path(a.run) if a.run else latest_run(folder)
    face_video = run / "video_concat.mp4"
    if not face_video.exists():
        sys.exit(f"没有 {face_video}，先跑完 render_rough.py")
    fw, fh, _ = video_info(face_video)
    tmp = Path(tempfile.mkdtemp(prefix="pip-track-"))
    subprocess.run([FFMPEG, "-v", "error", "-i", str(face_video), "-vf", f"fps={1 / STEP},scale=540:-2", "-q:v", "4",
                    str(tmp / "f_%05d.jpg")], check=True)
    out = subprocess.run(["swift", str(ROOT / "scripts" / "face_track.swift"), str(tmp)], capture_output=True, text=True)
    if out.returncode:
        sys.exit("找脸出错：" + out.stderr[-2000:])
    rows = [ln.split(",") for ln in out.stdout.split()]
    found = [(k, [float(x) for x in r[1:]]) for k, r in enumerate(rows) if len(r) == 5]
    if len(found) < len(rows) * 0.8:
        sys.exit(f"只在 {len(found)}/{len(rows)} 帧里找到脸，不够做跟踪，先用固定取景")
    # 找不到脸的帧用前后找到的那帧补
    cx, cy, hs = [], [], []
    j = 0
    for k in range(len(rows)):
        while j + 1 < len(found) and abs(found[j + 1][0] - k) <= abs(found[j][0] - k):
            j += 1
        x, y, w, h = found[j][1]
        cx.append((x + w / 2) * fw)
        cy.append((y + h * UP) * fh)
        hs.append(h * fh)
    side = min(fw, fh, round(statistics.median(hs) * SIZE_K))
    track = []
    for k, (x, y) in enumerate(zip(smooth(cx), smooth(cy))):
        px = min(max(0, round(x - side / 2)), fw - side)
        py = min(max(0, round(y - side / 2)), fh - side)
        track.append([round(k * STEP, 2), px, py])
    write_json(folder / "work" / "pip.json", {"crop": [track[0][1], track[0][2], side], "track": track,
                                               "note": f"pip_track.py 自动跟踪：每 {STEP} 秒一点，粗剪时间；边长 {side}"})
    # 核对图：均匀取 12 帧，画上取景圆
    tiles = []
    for k in [round(i * (len(rows) - 1) / 11) for i in range(12)]:
        im = Image.open(tmp / f"f_{k + 1:05d}.jpg").convert("RGB")
        sc = im.width / fw
        _, x, y = track[k]
        d = ImageDraw.Draw(im)
        d.ellipse([x * sc, y * sc, (x + side) * sc, (y + side) * sc], outline=(255, 0, 0), width=4)
        d.text((8, 8), f"{k * STEP:.1f}s", fill=(255, 255, 0))
        tiles.append(im.resize((270, round(270 * im.height / im.width))))
    sheet = Image.new("RGB", (270 * 6, tiles[0].height * 2), (0, 0, 0))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 6) * 270, (i // 6) * t.height))
    sheet.save(run / "小窗跟踪.jpg", quality=85)
    print(f"小窗跟踪取景：{len(found)}/{len(rows)} 帧找到脸，边长 {side}，{len(track)} 个点 → {folder / 'work' / 'pip.json'}；核对图 {run / '小窗跟踪.jpg'}")


if __name__ == "__main__":
    main()
