"""阶段 5 的三种图片动效，以及不带动效时的铺满画面。预览图和成片都调这里的同一个函数，保证两边画面一致。

每种动效都是“第几帧 → 一张 1920×1080 的图”。镜头位置用小数算，PIL 按小数坐标取样（resize 的 box 参数），
推拉时不会一像素一像素地跳，所以不抖；取景框始终在图片以内，所以没有黑边。

  无        铺满：按比例放大到盖住画面，多出来的裁掉
  推拉      缓慢推近（或拉远），整段约放大 1.12 倍，向画面中间偏右移一点（左下角是人脸小窗）；
            params.center 改推近的中心（没有小窗时 compose.py 写 [0.5, 0.5]，不往右偏），
            params.keep 是一定要留在画面里的区域 [x0, y0, x1, y1]（占图宽高的比例）：取景框挪到装得下它，
            装不下就少推一点（第 07 期框贴着左边、底边的图被裁掉一截，2026-10-25 加）
  模糊垫底  图片完整放在中间，背后垫一张放大、模糊、压暗的同一张图；竖图、长图用它
  截图聚焦  网页整页长图：先停在网页首屏，再推近到要高亮的那句话，然后画上高亮框
"""

import math

from PIL import Image, ImageDraw, ImageFilter

W, H = 1920, 1080
ASPECT = W / H

# 截图聚焦的节奏：先停 0.8 秒看整个网页首屏，用 1.4 秒推近，再用 0.3 秒画出高亮框，之后停住到这句结束
HOLD_S, PUSH_S, MARK_S = 0.8, 1.4, 0.3
MIN_WINDOW_CSS = 880      # 推得最近时，取景框至少 880 CSS 像素宽：整行正文都在画面里，16 像素的字放大后约 35 像素高
TARGET_SPAN = 0.62        # 推近后，高亮句的宽度约占画面的 62%
MARK_FILL = (255, 214, 0, 70)     # 高亮：半透明黄底
MARK_LINE = (255, 196, 0, 255)    # 高亮：黄框
PUSH_ZOOM = 1.12
PUSH_CENTER = (0.54, 0.46)  # 有小窗时推近往中间偏右上一点，让开左下角的小窗
KEEP_PAD = 0.015            # params.keep 四边再多留图宽高的 1.5%，框的边线不贴着画面边


def ease(x: float) -> float:
    """慢起慢停（smoothstep）。"""
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def lerp(a, b, t):
    return tuple(p + (q - p) * t for p, q in zip(a, b))


def cover_box(iw: int, ih: int, zoom: float = 1.0, cx: float = 0.5, cy: float = 0.5, aspect: float = ASPECT) -> tuple:
    """在 iw×ih 的图里取一个 16:9 的框（竖屏铺满时 aspect 是 9:16）：zoom=1 时正好盖满，zoom 越大框越小；
    框的中心尽量放在 (cx, cy)，但不出图。"""
    if iw / ih > aspect:
        bh = ih / zoom
        bw = bh * aspect
    else:
        bw = iw / zoom
        bh = bw / aspect
    x = min(max(cx * iw - bw / 2, 0), iw - bw)
    y = min(max(cy * ih - bh / 2, 0), ih - bh)
    return (x, y, x + bw, y + bh)


def fit_box(iw: int, ih: int, zoom: float, cx: float, cy: float, aspect: float, keep: list | None = None) -> tuple:
    """cover_box 再加一条：keep（[x0, y0, x1, y1]，占图宽高的比例）整个在框里。放大 zoom 倍装不下 keep 就少放大一点
    （不小于 1，也就是不推）；框先放在 (cx, cy)，再挪到刚好装下 keep。没有 keep 时和 cover_box 一样。"""
    if not keep:
        return cover_box(iw, ih, zoom, cx, cy, aspect)
    kx0, ky0 = max(0.0, (keep[0] - KEEP_PAD) * iw), max(0.0, (keep[1] - KEEP_PAD) * ih)
    kx1, ky1 = min(float(iw), (keep[2] + KEEP_PAD) * iw), min(float(ih), (keep[3] + KEEP_PAD) * ih)
    full = cover_box(iw, ih, 1.0, 0.5, 0.5, aspect)
    fw, fh = full[2] - full[0], full[3] - full[1]
    zoom = max(1.0, min(zoom, fw / max(1.0, kx1 - kx0), fh / max(1.0, ky1 - ky0)))
    bw, bh = fw / zoom, fh / zoom

    def place(c: float, lo: float, hi: float, size: float, total: float) -> float:
        p = c - size / 2
        p = min(max(p, hi - size), lo) if hi - lo <= size else (lo + hi - size) / 2  # 装不下（图比画面窄长）就按 keep 居中
        return min(max(p, 0.0), total - size)
    x, y = place(cx * iw, kx0, kx1, bw, iw), place(cy * ih, ky0, ky1, bh, ih)
    return (x, y, x + bw, y + bh)


def find_marks(path) -> list | None:
    """图里已经画好的高亮框（和截图聚焦一样的黄框 MARK_LINE；手动补的图常照这个样子画）在哪：
    返回把所有黄框包住的范围 [x0, y0, x1, y1]（占图宽高的比例），没有就 None。
    只认横、竖都连成直线、而且横线和竖线围得上的那种颜色，照片里零星的黄色不算。"""
    import numpy as np
    im = path if isinstance(path, Image.Image) else Image.open(path)
    if im.width >= 2 * W:
        im.draft("RGB", (im.width // 2, im.height // 2))  # JPEG 解码时直接缩一半，快
    if im.width > 2 * W:  # 别的格式的大图：取最近的像素缩，颜色不混
        im = im.resize((W, max(1, round(im.height * W / im.width))), Image.NEAREST)
    a = np.asarray(im.convert("RGB"), dtype=np.int16)
    m = (a[..., 0] >= 225) & (a[..., 1] >= 160) & (a[..., 1] <= 225) & (a[..., 2] <= 80)
    h, w = m.shape

    def segments(M, min_len):
        """每一行里连续 min_len 个以上的段：[(行号, 起, 止)]。"""
        out = []
        for i in np.nonzero(M.sum(1) >= min_len)[0]:
            d = np.diff(np.concatenate(([0], M[i].astype(np.int8), [0])))
            out += [(i, s, e) for s, e in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]) if e - s >= min_len]
        return out
    hs, vs = segments(m, max(20, 0.03 * w)), segments(m.T, max(8, 0.02 * h))
    if not hs or not vs:
        return None
    by_h = (min(s for _, s, _ in hs), min(i for i, _, _ in hs), max(e for _, _, e in hs), max(i for i, _, _ in hs) + 1)
    by_v = (min(i for i, _, _ in vs), min(s for _, s, _ in vs), max(i for i, _, _ in vs) + 1, max(e for _, _, e in vs))
    tol = (0.02 * w, 0.02 * h, 0.02 * w, 0.02 * h)
    if any(abs(p - q) > t + 3 for p, q, t in zip(by_h, by_v, tol)):  # 横线和竖线围不成框
        return None
    x0, y0 = min(by_h[0], by_v[0]), min(by_h[1], by_v[1])
    x1, y1 = max(by_h[2], by_v[2]), max(by_h[3], by_v[3])
    inner = m[y0 + 6:y1 - 6, x0 + 6:x1 - 6]
    if inner.size == 0 or inner.mean() > 0.35:  # 里面也是这个颜色：是一块实心的黄（比如标志），不是框
        return None
    return [round(float(v), 4) for v in (x0 / w, y0 / h, x1 / w, y1 / h)]


def render_box(img: Image.Image, box: tuple, size: tuple = (W, H)) -> Image.Image:
    return img.resize(size, Image.BICUBIC, box=box, reducing_gap=2.0)


class Still:
    """一张图，按动效出每一帧。"""

    def __init__(self, path, effect: str = "无", params: dict | None = None, frames: int = 1, fps: int = 30,
                 size: tuple = (W, H)):
        """size 是出帧的尺寸，默认 16:9 成片；竖屏里竖的图片直接铺满竖屏时给 (1080, 1920)（第二期），只用“无”和“推拉”。"""
        self.effect, self.params, self.frames, self.fps = effect, params or {}, max(1, frames), fps
        self.size, self.aspect = size, size[0] / size[1]
        self.img = (path if isinstance(path, Image.Image) else Image.open(path)).convert("RGB")
        iw, ih = self.img.size
        if size != (W, H) and effect not in ("无", "推拉"):
            raise ValueError(f"铺满竖屏的图片只能用“无”或“推拉”，不能用“{effect}”")
        if effect == "模糊垫底":
            self.static = self._blur_fill()
        elif effect == "推拉":
            # 先缩到比成片略大（放大 1.12 倍时仍够清楚）再推，取样的面积小很多，看起来一样，算得快
            bx = cover_box(iw, ih, aspect=self.aspect)
            k = (size[0] * PUSH_ZOOM * 1.05) / (bx[2] - bx[0])
            if k < 1:
                self.img = self.img.resize((round(iw * k), round(ih * k)), Image.LANCZOS, reducing_gap=2.0)
                iw, ih = self.img.size
            out = self.params.get("direction", "in") == "out"
            # 推近时向中间偏右上移一点，让开左下角的小窗（没有小窗时 center 是正中）；keep 那块始终在画面里
            cx, cy = self.params.get("center") or PUSH_CENTER
            keep = self.params.get("keep")
            self.a = fit_box(iw, ih, PUSH_ZOOM if out else 1.0, 0.5, 0.5, self.aspect, keep)
            self.b = fit_box(iw, ih, 1.0 if out else PUSH_ZOOM, cx, cy, self.aspect, keep)
        elif effect == "无":
            self.static = render_box(self.img, cover_box(iw, ih, aspect=self.aspect), size)
        else:
            raise ValueError(f"图片不支持动效“{effect}”")

    def _blur_fill(self) -> Image.Image:
        iw, ih = self.img.size
        # 先缩到四分之一再模糊、再放大：样子一样，快十倍左右
        small = self.img.resize((W // 4, H // 4), Image.BILINEAR, box=cover_box(iw, ih))
        bg = small.filter(ImageFilter.GaussianBlur(10)).resize((W, H), Image.BILINEAR)
        bg = Image.blend(bg, Image.new("RGB", (W, H), (0, 0, 0)), 0.45)
        s = min(W / iw, H / ih)
        fg = self.img.resize((max(1, round(iw * s)), max(1, round(ih * s))), Image.BICUBIC, reducing_gap=2.0)
        bg.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2))
        return bg

    def frame(self, k: int) -> Image.Image:
        if self.effect in ("无", "模糊垫底"):
            return self.static
        t = k / max(1, self.frames - 1)
        return render_box(self.img, lerp(self.a, self.b, t), self.size)  # 推拉匀速，看着更平稳


class PageFocus:
    """截图聚焦：page 是整页长图（两倍清晰度），boxes 是 fetch_assets.py 记下的位置（CSS 像素）。"""

    def __init__(self, path, boxes: dict, scale: float, frames: int, fps: int = 30, target: str = "highlight"):
        self.frames, self.fps = max(1, frames), fps
        img = Image.open(path).convert("RGB")
        iw, ih = img.size
        page = boxes.get("page", {})
        tb = page.get(target) or page.get("headline")
        # 高亮框：推近哪句就框哪句（标题也框），跨行的每行一个框
        self.marks = []
        if tb:
            self.marks = [{k: v * scale for k, v in b.items() if k in "xywh"} for b in (tb.get("lines") or [tb])]
        # 起点：正文的开头（标题和头几段），左右各留 80 CSS 像素；没认出正文就用网页首屏（整页宽）
        art = page.get("article")
        if art:
            aw = min(iw, (art["w"] + 160) * scale)
            ax = min(max((art["x"] + art["w"] / 2) * scale - aw / 2, 0), iw - aw)
            ay = min(max((art["y"] - 40) * scale, 0), max(0.0, ih - aw / ASPECT))
            self.a = (ax, ay, ax + aw, ay + aw / ASPECT)
        else:
            self.a = (0.0, 0.0, float(iw), iw / ASPECT)
        if self.a[3] > ih:
            self.a = cover_box(iw, ih, 1.0, 0.5, 0.0)
        # 终点：高亮句放大到约 62% 画面宽，居中略偏右上
        if tb:
            x, y, w, h = (tb[k] * scale for k in ("x", "y", "w", "h"))
            blk = tb.get("block")
            # 取景框够宽装下整段（左右各留 40 CSS 像素），水平按整段居中；没记段落就按这句居中
            need = (blk["w"] + 80) * scale if blk else 0
            bw = min(iw, max(MIN_WINDOW_CSS * scale, w / TARGET_SPAN, need))
            bh = bw / ASPECT
            cx = (blk["x"] + blk["w"] / 2) * scale if blk else x + w / 2
            cy = y + h / 2 + bh * 0.04
            bx = min(max(cx - bw / 2, 0), iw - bw)
            by = min(max(cy - bh / 2, 0), max(0.0, ih - bh))
            if bh > ih:  # 页面比框还矮：退回铺满
                bx, by, bw, bh = cover_box(iw, ih)[0], 0.0, ih * ASPECT, float(ih)
            self.b = (bx, by, bx + bw, by + bh)
        else:
            self.b = self.a
        self.img = img
        self.marked = self._draw_marks(img) if self.marks else None
        # 这一句太短时按比例压缩，保证高亮框在这句结束前画完（留两帧）
        total = self.frames / fps
        k = min(1.0, max(0.01, (total - 2 / fps) / (HOLD_S + PUSH_S + MARK_S + 0.6)))
        self.t_hold, self.t_push, self.t_mark = HOLD_S * k, PUSH_S * k, MARK_S * k

    def _draw_marks(self, img: Image.Image) -> Image.Image:
        """高亮框画在原图上，再跟原图一起缩放，所以框和字一起移动、不会单独抖动。"""
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        pad = 6
        for b in self.marks:
            r = [b["x"] - pad, b["y"] - pad, b["x"] + b["w"] + pad, b["y"] + b["h"] + pad]
            d.rectangle(r, fill=MARK_FILL, outline=MARK_LINE, width=6)
        return Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")

    def window(self, t: float) -> tuple:
        if t <= self.t_hold:
            return self.a
        return lerp(self.a, self.b, ease((t - self.t_hold) / self.t_push))

    def mark_alpha(self, t: float) -> float:
        if not self.marked:
            return 0.0
        return ease((t - self.t_hold - self.t_push) / self.t_mark)

    def frame(self, k: int) -> Image.Image:
        t = k / self.fps
        box = self.window(t)
        a = self.mark_alpha(t)
        key = (box, a)
        if getattr(self, "_last", (None,))[0] == key:  # 停住的那几秒画面不变，直接用上一帧
            return self._last[1]
        if a <= 0:
            img = render_box(self.img, box)
        elif a >= 1:
            img = render_box(self.marked, box)
        else:
            img = Image.blend(render_box(self.img, box), render_box(self.marked, box), a)
        self._last = (key, img)
        return img


def make(kind: str, path, effect: str, params: dict, frames: int, fps: int = 30, boxes=None, scale=2.0):
    """按素材和动效造一个出帧的对象，出 16:9 的帧。kind：page（整页长图）、image 或 video 以外的图片。
    整页长图不用截图聚焦时，取截图聚焦推近后那一框（整段正文、居中）当一张普通图片用；推拉时那句话始终在画面里。
    竖屏（第二期）里横的素材也用它出 16:9 的帧，再由 composite.vertical_wrap 放进竖屏中间。"""
    if effect == "截图聚焦":
        if kind != "page" or not boxes:
            raise ValueError("截图聚焦只能用在网页整页截图上")
        return PageFocus(path, boxes, scale, frames, fps, params.get("target", "highlight"))
    if kind == "page":
        pf = PageFocus(path, boxes, scale, 1, 30, params.get("target", "highlight"))
        x0, y0, x1, y1 = pf.b
        if effect == "推拉" and not params.get("keep") and pf.marks:  # 推近那句话（或标题）始终在画面里
            ms = pf.marks
            params = {**params, "keep": [(min(m["x"] for m in ms) - x0) / (x1 - x0), (min(m["y"] for m in ms) - y0) / (y1 - y0),
                                         (max(m["x"] + m["w"] for m in ms) - x0) / (x1 - x0),
                                         (max(m["y"] + m["h"] for m in ms) - y0) / (y1 - y0)]}
        return Still(pf.img.crop(tuple(round(v) for v in pf.b)), effect, params, frames, fps)
    return Still(path, effect, params, frames, fps)
