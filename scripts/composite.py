"""成片画面的公用部分：左下角圆形小窗、缺素材时的全屏人脸（无人脸模式是黑底）、从视频里取帧。预览图和成片用同一套几何和遮罩。"""

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from common import CONFIG, FFMPEG, FPS, ROOT
from effects import H, W, cover_box

SS = 4  # 遮罩先画 4 倍大再缩小，圆边不起锯齿


def pip_geometry() -> dict:
    d, m, b = CONFIG["pip_diameter"], CONFIG["pip_margin"], CONFIG["pip_border"]
    return {"diameter": d, "margin": m, "border": b, "x": m, "y": H - m - d}


def default_crop(fw: int, fh: int) -> list:
    """没给 work/pip.json 时的人脸取景：竖屏取画面宽的 80%、从上往下 18% 处开始的正方形；横屏取中间的正方形。"""
    if fh > fw:
        s = int(fw * 0.8)
        return [(fw - s) // 2, int(fh * 0.18), s]
    s = int(fh * 0.8)
    return [(fw - s) // 2, (fh - s) // 2, s]


def crop_at(pip: dict, t: float) -> list:
    """t 秒（粗剪时间）时的人脸取景。pip.json 写了 track（pip_track.py 自动跟踪）时在前后两个点之间按直线插值，
    边长不变；没写 track 就是固定的 crop。成片里 render_final.py 按同样的点用 FFmpeg sendcmd 移动取景。"""
    s = pip["crop"][2]
    tr = pip.get("track")
    if not tr:
        return list(pip["crop"])
    if t <= tr[0][0]:
        return [tr[0][1], tr[0][2], s]
    for (t0, x0, y0), (t1, x1, y1) in zip(tr, tr[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return [round(x0 + (x1 - x0) * f), round(y0 + (y1 - y0) * f), s]
    return [tr[-1][1], tr[-1][2], s]


def masks(g: dict) -> tuple[Image.Image, Image.Image]:
    """圆形遮罩（L）和白边（RGBA），都是 diameter×diameter。"""
    d, b = g["diameter"], g["border"]
    big = d * SS
    m = Image.new("L", (big, big), 0)
    ImageDraw.Draw(m).ellipse([0, 0, big - 1, big - 1], fill=255)
    ring = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ring)
    dr.ellipse([0, 0, big - 1, big - 1], fill=(255, 255, 255, 255))
    dr.ellipse([b * SS, b * SS, big - 1 - b * SS, big - 1 - b * SS], fill=(0, 0, 0, 0))
    return m.resize((d, d), Image.LANCZOS), ring.resize((d, d), Image.LANCZOS)


def save_masks(g: dict, folder: Path) -> tuple[Path, Path]:
    m, ring = masks(g)
    pm, pr = folder / "pip_mask.png", folder / "pip_ring.png"
    m.save(pm)
    ring.save(pr)
    return pm, pr


def overlay_pip(frame: Image.Image, face: Image.Image, crop: list, g: dict) -> Image.Image:
    """预览用：把人脸帧按 crop 裁成正方形，缩成小窗，叠上白边，贴到左下角。成片里同样的事由 FFmpeg 做。"""
    x, y, s = crop
    d = g["diameter"]
    m, ring = masks(g)
    f = face.crop((x, y, x + s, y + s)).resize((d, d), Image.BICUBIC)
    out = frame.copy()
    out.paste(f, (g["x"], g["y"]), m)
    out.paste(ring, (g["x"], g["y"]), ring)
    return out


def full_face(face: Image.Image) -> Image.Image:
    """缺素材的句子（草稿）：人脸放成大画面，竖屏就模糊垫底。"""
    fw, fh = face.size
    small = face.resize((W // 4, H // 4), Image.BILINEAR, box=cover_box(fw, fh))
    bg = small.filter(ImageFilter.GaussianBlur(10)).resize((W, H), Image.BILINEAR)
    bg = Image.blend(bg, Image.new("RGB", (W, H)), 0.45)
    sc = min(W / fw, H / fh)
    fg = face.resize((round(fw * sc), round(fh * sc)), Image.BICUBIC)
    bg.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2))
    return bg


def todo_frame(sentence: int, face: Image.Image | None = None, vertical: dict | None = None) -> Image.Image:
    """缺素材的句子（草稿）那一帧：有人脸就放全屏人脸，无人脸模式放黑底；左上角标“待补素材：第 N 句”。
    竖屏（vertical 是 vertical_layout() 的结果）：人脸铺满竖屏，字标在顶部界面下面。"""
    if vertical:
        vw, vh = vertical["size"]
        img = vertical_face(face, vertical) if face is not None else Image.new("RGB", (vw, vh))
        xy = (60, vertical["safe"]["top"] + 30)
    else:
        img = full_face(face) if face is not None else Image.new("RGB", (W, H))
        xy = (60, 50)
    font = ImageFont.truetype(str(ROOT / CONFIG["font_file"]), 64)
    ImageDraw.Draw(img).text(xy, f"待补素材：第 {sentence} 句", fill=(255, 220, 0), font=font)
    return img


# ---------- 竖屏（第二期，2026-10-24 选的 C：人脸全屏，配了素材的句子整屏换成素材） ----------

def vertical_layout(face_size: tuple | None, pip_crop: list | None) -> dict:
    """竖屏的几何：成片尺寸、平台界面挡住的地方、横的素材放在哪、人脸从工作视频里取哪一块。
    人脸：竖的工作视频整幅放大铺满；横的工作视频按 9:16 裁一条，中心对着 pip.json 的取景（没有就取正中）。"""
    vw, vh = CONFIG["vertical_size"]
    mh = round(vw * H / W)  # 横的素材缩到竖屏宽时的高度
    my = min(max(round(vh * CONFIG["vertical_material_y"] - mh / 2), 0), vh - mh)
    lay = {"size": [vw, vh], "safe": dict(CONFIG["vertical_safe"]), "material_box": [0, my, vw, mh]}
    if face_size:
        fw, fh = face_size
        if fw / fh > vw / vh:  # 横的工作视频：按 9:16 裁一条
            cw = round(fh * vw / vh) // 2 * 2
            cx = pip_crop[0] + pip_crop[2] / 2 if pip_crop else fw / 2
            x = int(min(max(cx - cw / 2, 0), fw - cw)) // 2 * 2
            lay["face_crop"] = [x, 0, cw, fh]
        else:  # 竖的（或比 9:16 更窄的）：裁上下多出来的
            ch = round(fw * vh / vw) // 2 * 2
            lay["face_crop"] = [0, (fh - ch) // 2 // 2 * 2, fw, ch]
    return lay


def vertical_face(face: Image.Image, lay: dict) -> Image.Image:
    """预览用：一帧人脸按 face_crop 裁好、放大到竖屏。成片里同样的事由 FFmpeg 做（render_final 的 face_filter）。"""
    x, y, w, h = lay["face_crop"]
    return face.crop((x, y, x + w, y + h)).resize(tuple(lay["size"]), Image.BICUBIC)


def vertical_bg(frame16x9: Image.Image, lay: dict) -> Image.Image:
    """横的素材放进竖屏时垫在后面的那张：同一画面放大盖满竖屏、缩小模糊再放大、压暗（和 16:9 的模糊垫底一样的做法）。"""
    vw, vh = lay["size"]
    small = frame16x9.resize((vw // 4, vh // 4), Image.BILINEAR, box=cover_box(*frame16x9.size, aspect=vw / vh))
    bg = small.filter(ImageFilter.GaussianBlur(10)).resize((vw, vh), Image.BILINEAR)
    return Image.blend(bg, Image.new("RGB", (vw, vh)), 0.45)


def vertical_wrap(frame16x9: Image.Image, lay: dict, bg: Image.Image | None = None) -> Image.Image:
    """横的素材（16:9 的一帧）放进竖屏：缩到竖屏宽，放在 material_box，上下是 vertical_bg。
    bg 给了就用它（一句里背景不动，只算一次）。"""
    x, y, w, h = lay["material_box"]
    out = (bg if bg is not None else vertical_bg(frame16x9, lay)).copy()
    out.paste(frame16x9.resize((w, h), Image.BICUBIC), (x, y))
    return out


def video_info(path: Path) -> tuple[int, int, float]:
    """按旋转信息摆正后的宽、高，以及时长（秒）。FFmpeg 解码时会自动摆正，所以拼帧要用摆正后的宽高。"""
    import json
    p = subprocess.run([CONFIG["ffprobe"], "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,duration:stream_side_data=rotation:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True)
    d = json.loads(p.stdout)
    st = d["streams"][0]
    w, h = st["width"], st["height"]
    rot = next((int(sd["rotation"]) for sd in st.get("side_data_list", []) if "rotation" in sd), 0)
    if rot % 180:
        w, h = h, w
    dur = float(st.get("duration") or d.get("format", {}).get("duration") or 0)
    return w, h, dur


def video_frames(path: Path, start_frame: int, n: int, size: tuple | None = None):
    """从固定 30 帧的视频里按帧号取 n 帧（PIL 图）。"""
    w, h, _ = video_info(path)
    cmd = [FFMPEG, "-v", "error", "-ss", f"{max(0.0, (start_frame - 0.5) / FPS):.6f}", "-i", str(path),
           "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size_b = w * h * 3
    try:
        for _ in range(n):
            buf = proc.stdout.read(size_b)
            if len(buf) < size_b:
                break
            yield Image.fromarray(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
    finally:
        proc.stdout.close()
        proc.wait()
