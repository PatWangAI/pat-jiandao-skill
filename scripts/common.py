"""各个脚本共用的部分：路径和设置、算音量、找静音、对齐到画面帧、拼字、算字宽，
以及“从剪辑表算出保留段”。声音、画面、字幕、自检都调同一个 keep_segments()，
保证四处按同一份保留段走（SPEC 第 2 节）。

两个词：分析帧是算音量用的 10 毫秒一格；画面帧是视频的一帧，1/30 秒。
剪辑表里的刀点一律是画面帧号，秒数只用来显示。
"""

import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    """设置分两份：config.yaml（仓库里，大家都一样的参数和默认路径）和 config.local.yaml（本机的路径，不进仓库，
    样例是 config.local.example.yaml）。本机那份写了的项，覆盖 config.yaml 的同名项。"""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    local = ROOT / "config.local.yaml"
    if local.exists():
        cfg.update(yaml.safe_load(local.read_text(encoding="utf-8")) or {})
    return cfg


CONFIG = load_config()
FFMPEG = CONFIG["ffmpeg"]
FFPROBE = CONFIG["ffprobe"]

FPS = 30
SR = 16000          # 转写和找静音用的声音
SR_OUT = 48000      # 成片用的声音，一帧正好 1600 个采样

# 剪切规则（SPEC 第 4 节）
HOP = 160           # 分析帧 10 毫秒一格
WIN = 320           # 每格看 20 毫秒
MIN_QUIET = 5       # 连续 5 个分析帧（50 毫秒）以上低于门限才算一段静音
EDGE_PAD = 0.005    # 静音起止按首尾两个分析帧的中心各往外推 5 毫秒
CUT_MARGIN = 0.008  # 刀点离静音两端至少 8 毫秒
BEFORE = 0.15       # 找静音的窗口：交界点往前 0.15 秒
AFTER = 0.6         # 往后 0.6 秒（FunASR 的时间常常偏早）
MIN_SOUND = 5       # 两刀之间至少要有 5 个有声的分析帧，否则说明跳过了要删的字
BLANK_MARGIN = 0.1  # blank_cut 当静音的范围离留下的字至少 0.1 秒，免得转写的字边界不准、把字尾当杂音
PENDING_WIN = 0.12  # 连读待定用户说删时，在交界点前后 0.12 秒里找音量最小的帧边界
LONG_PAUSE = 0.45   # 长于 0.45 秒的停顿才压
PAUSE_KEEP = 0.15   # 压停顿时两边各留 0.15 秒
SEAM_KEEP = 0.3     # 删除处接缝两边一共留约 0.3 秒
EDGE_KEEP = 0.15    # 片头片尾各留 0.15 秒
# 节奏（开源首发 10-26 加）：“标准”就是上面这几个数，和以前一样；“紧”的剪口、压过的停顿合起来只留约 0.15 秒，
# 片头片尾各留 0.1 秒，第一个字前面隔着长停顿的声音（按录制键、吸气）不算片头。剪辑表记 pace，复核挪刀时照同一个节奏算
PACES = {
    "标准": {"pause_keep": PAUSE_KEEP, "seam_keep": SEAM_KEEP, "edge_keep": EDGE_KEEP, "skip_lead_noise": False},
    "紧": {"pause_keep": 0.075, "seam_keep": 0.15, "edge_keep": 0.1, "skip_lead_noise": True},
}
FADE = 0.008        # 每个保留段头尾淡入淡出 8 毫秒，防爆音
MAX_CHARS = 16      # 字幕每行不超过 16 个字


# ---------- 路径和小工具 ----------

def videos_root() -> Path:
    p = Path(CONFIG["videos_root"]).expanduser()
    return p if p.is_absolute() else ROOT / p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def run_ffmpeg(args: list[str], cwd: Path | None = None) -> None:
    """跑一条 FFmpeg 命令；出错时把 FFmpeg 的原文带出来（SPEC 第 3 节第 8 步）。"""
    r = subprocess.run([FFMPEG, "-v", "error", "-y", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FFmpeg 报错（退出码 {r.returncode}）：\n{r.stderr.strip()}\n命令：ffmpeg {' '.join(args)}")


def check_font(texts: list[str], w: int, h: int, workdir: Path) -> None:
    """确认这些文字都由 fonts/ 里的思源黑体渲染：用 FFmpeg 的详细日志看 libass 选了哪个字体文件。
    字体名写错或缺字时，libass 会悄悄换成系统字体（10-07 审查时发现过），所以选不中就停下。"""
    probe = workdir / "_font_check.ass"
    rows = [f"Dialogue: 0,0:00:00.00,0:00:01.00,S,,0,0,0,,{t}" for t in texts]
    probe.write_text(f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: S,{CONFIG['font_name']},20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,2,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""" + "\n".join(rows) + "\n", encoding="utf-8")
    fonts = (ROOT / CONFIG["font_file"]).parent
    r = subprocess.run([FFMPEG, "-v", "verbose", "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r=30",
                        "-vf", f"subtitles={probe.name}:fontsdir={fonts}", "-frames:v", "1", "-f", "null", "-"],
                       cwd=workdir, capture_output=True, text=True)
    probe.unlink()
    picks = {ln.split("->", 1)[1].strip() for ln in r.stderr.splitlines() if "fontselect:" in ln and "->" in ln}
    want = Path(CONFIG["font_file"]).stem
    bad = [p for p in picks if want not in p]
    if r.returncode != 0 or not picks or bad:
        raise SystemExit(f"字幕没有全部用上 {want}：libass 选的是 {sorted(picks) or '（没看到选字记录）'}。"
                         "检查 config.yaml 的 font_name，或者字幕里有思源黑体没有的字")


def ffprobe(path: Path) -> dict:
    r = subprocess.run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"读不了这个文件：{path}\n{r.stderr.strip()}")
    return json.loads(r.stdout)


def picture_stream(info: dict) -> dict | None:
    """ffprobe 结果里的画面流。mp3、m4a 带的封面图（attached_pic）不算画面。"""
    return next((s for s in info["streams"] if s["codec_type"] == "video"
                 and not s.get("disposition", {}).get("attached_pic")), None)


def has_picture(path: Path) -> bool:
    return picture_stream(ffprobe(path)) is not None


def fmt(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:05.2f}"


def new_run_dir(video_dir: Path) -> Path:
    run = video_dir / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    return run


def latest_run(video_dir: Path) -> Path:
    runs = sorted(p for p in (video_dir / "runs").glob("*") if p.is_dir())
    if not runs:
        raise SystemExit(f"{video_dir / 'runs'} 里还没有剪过的结果，先跑 cut_audio.py")
    return runs[-1]


def archive(run_dir: Path, names: list[str], copy: list[str] = ()) -> Path | None:
    """重跑前把上一版的这些文件挪进 旧版_<时间>/（copy 里的只复制），运行结果都不删。没有旧文件就什么也不做。"""
    there = [n for n in names if (run_dir / n).exists()]
    if not there:
        return None
    import shutil
    old = run_dir / time.strftime("旧版_%H%M%S")
    old.mkdir(exist_ok=True)
    for n in there:
        (run_dir / n).rename(old / n)
    for n in copy:
        if (run_dir / n).exists():
            shutil.copy2(run_dir / n, old / n)
    return old


def add_timing(run_dir: Path, step: str, seconds: float) -> None:
    """每个脚本把自己的用时追加进 runs/<时间>/timing.json，运行记录的“机器耗时”从这里加总。"""
    p = run_dir / "timing.json"
    data = read_json(p) if p.exists() else []
    data.append({"step": step, "seconds": round(seconds, 1)})
    write_json(p, data)


# ---------- 文字 ----------

def join_text(words: list[str]) -> str:
    """中文字直接相连，英文词前后加空格。"""
    s = ""
    for w in words:
        if s and w and (w[0].isascii() or s[-1].isascii()):
            s += " "
        s += w
    return s


def text_width(s: str) -> float:
    """字幕字宽：英文字母、数字、半角符号两个算一个字，其余一个算一个。"""
    return sum(0.5 if ch.isascii() else 1.0 for ch in s)


def normalize(s: str) -> str:
    """复核对字用：去空格和标点，转小写。"""
    return "".join(ch.lower() for ch in s if ch.isalnum())


# ---------- 声音和静音 ----------

def frame_center(k) -> float:
    return k * HOP / SR + WIN / 2 / SR


@dataclass
class Loudness:
    """一段 16 kHz 声音的音量分析：每个分析帧的 dBFS、静音门限、所有静音段（秒）。"""
    db: np.ndarray
    thr: float
    quiet: list  # [(起, 止)]，按时间排好

    @property
    def sound(self) -> np.ndarray:
        return self.db >= self.thr

    def db_at(self, t: float) -> float:
        k = int(round((t - WIN / 2 / SR) * SR / HOP))
        return float(self.db[max(0, min(k, len(self.db) - 1))])

    def in_quiet(self, t: float) -> bool:
        return any(a <= t <= b for a, b in self.quiet)

    def sound_frames(self, t0: float, t1: float) -> int:
        """中心落在 [t0, t1) 里的有声分析帧个数。"""
        c = frame_center(np.arange(len(self.db)))
        return int(np.sum(self.sound & (c >= t0) & (c < t1)))

    def sound_end_before(self, t: float) -> float:
        """t 之前最后一个有声分析帧的结束（中心 + 5 毫秒）；前面没有声音就返回 0。"""
        c = frame_center(np.arange(len(self.db)))
        idx = np.where(self.sound & (c < t))[0]
        return float(c[idx[-1]] + EDGE_PAD) if len(idx) else 0.0

    def sound_start_after(self, t: float) -> float:
        c = frame_center(np.arange(len(self.db)))
        idx = np.where(self.sound & (c > t))[0]
        return float(c[idx[0]] - EDGE_PAD) if len(idx) else float(c[-1] + EDGE_PAD)

    def speech_span(self) -> tuple[float, float]:
        """第一个和最后一个有声分析帧的起止。只认连续 5 帧以上的声音，免得按录制键的咔哒声算进片头片尾。"""
        s = list(self.sound)
        runs, start = [], None
        for k, v in enumerate(s + [False]):
            if v and start is None:
                start = k
            elif not v and start is not None:
                if k - start >= MIN_QUIET:
                    runs.append((start, k - 1))
                start = None
        if not runs:
            raise SystemExit("整段声音都低于静音门限，找不到说话的部分")
        return frame_center(runs[0][0]) - EDGE_PAD, frame_center(runs[-1][1]) + EDGE_PAD


def analyze(audio: np.ndarray) -> Loudness:
    """SPEC 第 4 节第 1 条：算音量、定门限、找静音段。"""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = 1 + max(0, len(audio) - WIN) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(audio, WIN)[::HOP][:n]
    db = 20 * np.log10(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-10)
    floor, loud = np.percentile(db, 10), np.percentile(db, 95)
    thr = float(floor + max(8.0, 0.25 * (loud - floor)))
    return Loudness(db=db, thr=thr, quiet=quiet_runs(db, thr))


def quiet_runs(db: np.ndarray, thr: float) -> list:
    """低于门限、连续 MIN_QUIET 个分析帧以上的静音段（秒）。"""
    quiet, start = [], None
    for k, q in enumerate(list(db < thr) + [False]):
        if q and start is None:
            start = k
        elif not q and start is not None:
            if k - start >= MIN_QUIET:
                quiet.append((frame_center(start) - EDGE_PAD, frame_center(k - 1) + EDGE_PAD))
            start = None
    return quiet


def blank_spans(blank_cut: list | None, tok: list) -> list:
    """decisions.json 的 blank_cut（“起-止”字序号）→ 原片秒数：起字结束到止字开始，
    扣掉中间（要删的）字说话的时间，离留下的两个字各让出 BLANK_MARGIN。"""
    spans = []
    for key in blank_cut or []:
        i, j = (int(x) for x in key.split("-"))
        edges = [tok[i]["end"] + BLANK_MARGIN]
        edges += [t for x in tok[i + 1:j] for t in (x["start"], x["end"])]
        edges.append(tok[j]["start"] - BLANK_MARGIN)
        spans += [(edges[k], edges[k + 1]) for k in range(0, len(edges), 2) if edges[k + 1] > edges[k]]
    return spans


def quiet_over(loud: Loudness, spans: list) -> Loudness:
    """用户说某处空白太长（decisions.json 的 blank_cut，2026-10-26 加，第 08 期）：spans 里的小杂音
    （碰麦、咔哒、呼吸）当静音，和两边的静音连成一整段；删除处接缝、压停顿都按这一整段算，剩约 0.3 秒。
    只给定刀用（plan_cuts.py、cut_audio.py）；没写 blank_cut 时原样返回，和以前一样。"""
    if not spans:
        return loud
    c = frame_center(np.arange(len(loud.db)))
    db = loud.db.copy()
    for a, b in spans:
        m = (c > a) & (c < b)
        db[m] = np.minimum(db[m], loud.thr - 1.0)
    return Loudness(db=db, thr=loud.thr, quiet=quiet_runs(db, loud.thr))


def frame_in_quiet(q: tuple[float, float]) -> int | None:
    """静音段里离两端都至少 8 毫秒的画面帧边界中，离静音中点最近的一个。"""
    a, b = q
    lo, hi = math.ceil((a + CUT_MARGIN) * FPS - 1e-9), math.floor((b - CUT_MARGIN) * FPS + 1e-9)
    if lo > hi:
        return None
    mid = (a + b) / 2 * FPS
    return int(min(range(lo, hi + 1), key=lambda f: (abs(f - mid), f)))


def nearest_quiet(loud: Loudness, junction: float) -> int | None:
    """和 [交界点 − 0.15, 交界点 + 0.6] 有重叠的静音段里，离交界点最近的一段（返回序号）。"""
    best, best_d = None, None
    for i, (a, b) in enumerate(loud.quiet):
        if b < junction - BEFORE or a > junction + AFTER:
            continue
        d = 0.0 if a <= junction <= b else min(abs(a - junction), abs(b - junction))
        if best is None or d < best_d:
            best, best_d = i, d
    return best


def min_volume_frame(loud: Loudness, left: dict, right: dict) -> int:
    """交界点前后 0.12 秒内（不越过左字开头和右字结尾）音量最小的画面帧边界。"""
    j = (left["end"] + right["start"]) / 2
    lo = max(j - PENDING_WIN, left["start"])
    hi = min(j + PENDING_WIN, right["end"])
    frames = list(range(math.ceil(lo * FPS), math.floor(hi * FPS) + 1)) or [int(round(j * FPS))]
    return int(min(frames, key=lambda f: (loud.db_at(f / FPS), abs(f / FPS - j))))


def splice(audio: np.ndarray, sr: int, segs: list[tuple[int, int]]) -> np.ndarray:
    """按保留段（画面帧号）把声音拼成一段，每段头尾淡入淡出 8 毫秒。采样号 = 帧号 × sr / 30 再取整。"""
    parts = []
    n_fade = int(FADE * sr)
    for a, b in segs:
        x = audio[int(round(a * sr / FPS)):int(round(b * sr / FPS))].astype(np.float32).copy()
        n = min(n_fade, len(x) // 2)
        if n > 0:
            ramp = np.linspace(0, 1, n, dtype=np.float32)
            if x.ndim > 1:
                ramp = ramp[:, None]
            x[:n] *= ramp
            x[-n:] *= ramp[::-1]
        parts.append(x)
    return np.concatenate(parts) if parts else audio[:0]


# ---------- 剪辑表 ----------

def removed_ranges(edl: dict) -> list[tuple[int, int]]:
    """剪辑表里所有要去掉的范围（帧号，左闭右开），合并好、排好序。"""
    dur = edl["duration_frames"]
    rs = [(0, edl["edges"]["start_frame"]), (edl["edges"]["end_frame"], dur)]
    rs += [(d["out_frame"], d["in_frame"]) for d in edl["deletions"]]
    rs += [(p["out_frame"], p["in_frame"]) for p in edl["pauses"]]
    merged = []
    for a, b in sorted(r for r in rs if r[1] > r[0]):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def keep_segments(edl: dict) -> list[tuple[int, int]]:
    """从剪辑表算出保留段（帧号，左闭右开）。声音、画面、字幕、自检都只用这一个函数。"""
    segs, pos = [], 0
    for a, b in removed_ranges(edl):
        if a > pos:
            segs.append((pos, a))
        pos = max(pos, b)
    if pos < edl["duration_frames"]:
        segs.append((pos, edl["duration_frames"]))
    return segs


def to_rough(t: float, segs: list[tuple[int, int]], mode: str = "exact") -> float | None:
    """把原片时间换成粗剪时间。落在去掉的范围里时：exact 返回 None，next 推到下一个保留段开头，prev 收到上一个保留段结尾。"""
    clock = 0.0
    prev_end = None
    for a, b in segs:
        ta, tb = a / FPS, b / FPS
        if t < ta:
            if mode == "next":
                return clock
            if mode == "prev":
                return prev_end
            return None
        if t <= tb:
            return clock + (t - ta)
        clock += tb - ta
        prev_end = clock
    return prev_end if mode == "prev" else None


def seam_times(edl: dict) -> list[dict]:
    """每个接缝在粗剪里的时间，以及它属于哪一处删除或压停顿。"""
    segs = keep_segments(edl)
    owners = [(d["out_frame"], d["in_frame"], "删除", d["id"]) for d in edl["deletions"]]
    owners += [(p["out_frame"], p["in_frame"], "长停顿", p["id"]) for p in edl["pauses"]]
    out, clock = [], 0.0
    for i, (a, b) in enumerate(segs):
        clock += (b - a) / FPS
        if i + 1 < len(segs):
            gap = (b, segs[i + 1][0])
            who = [(k, n) for (o, inn, k, n) in owners if o < gap[1] and inn > gap[0]]
            out.append({"rough_s": clock, "out_frame": gap[0], "in_frame": gap[1], "owners": who})
    return out


def apply_pauses(edl: dict, loud: Loudness, pauses_keep: list | None = None) -> dict:
    """SPEC 第 4 节第 5 条：刀都定了以后算片头片尾、删除处接缝、长停顿。

    每条删除的 out_snap、in_snap 是按规则找到的刀点；这里算出实际下刀的 out_frame、in_frame。
    复核每挪一次刀（改 snap），就重新调一次这个函数。
    """
    dur = edl["duration_frames"]
    pauses_keep = pauses_keep or []
    pace = PACES[edl.get("pace", "标准")]
    pause_keep, seam_keep, edge_keep = pace["pause_keep"], pace["seam_keep"], pace["edge_keep"]
    first, last = loud.speech_span()
    if pace["skip_lead_noise"] and "first_word_s" in edl:
        w = loud.sound_start_after(max(0.0, edl["first_word_s"] - 0.3))  # 第一个字的声音从哪起（字的时间常偏早）
        if w - loud.sound_end_before(w) > LONG_PAUSE:  # 前面隔着长停顿的声音不是说话
            first = max(first, w)
    start = max(0, math.floor((first - edge_keep) * FPS + 1e-9))
    end = min(dur, math.ceil((last + edge_keep) * FPS - 1e-9))
    zones = []  # 每处删除碰到的范围：左边声音结束到右边声音开始，长停顿不能碰
    for d in edl["deletions"]:
        if d["out_how"] == "片头":
            rs = loud.sound_start_after(d["in_snap"] / FPS)
            d["out_frame"] = 0
            d["in_frame"] = max(d["in_snap"], math.floor((rs - edge_keep) * FPS + 1e-9))
            start = max(start, d["in_frame"])
            zones.append((0.0, rs))
        elif d["in_how"] == "片尾":
            ls = loud.sound_end_before(d["out_snap"] / FPS)
            d["out_frame"] = min(d["out_snap"], math.ceil((ls + edge_keep) * FPS - 1e-9))
            d["in_frame"] = dur
            end = min(end, d["out_frame"])
            zones.append((ls, dur / FPS))
        else:
            out_t, in_t = d["out_snap"] / FPS, d["in_snap"] / FPS
            ls, rs = loud.sound_end_before(out_t), loud.sound_start_after(in_t)
            gl, gr = max(0.0, out_t - ls), max(0.0, rs - in_t)
            d["out_frame"], d["in_frame"] = d["out_snap"], d["in_snap"]
            if gl + gr > LONG_PAUSE:
                kl = min(gl, pause_keep)
                kr = min(gr, seam_keep - kl)
                if kl + kr < seam_keep:
                    kl = min(gl, seam_keep - kr)
                d["out_frame"] = min(d["out_snap"], math.ceil((ls + kl) * FPS - 1e-9))
                d["in_frame"] = max(d["in_snap"], math.floor((rs - kr) * FPS + 1e-9))
            zones.append((ls, rs))
        d["out_s"], d["in_s"] = round(d["out_frame"] / FPS, 3), round(d["in_frame"] / FPS, 3)
    edl["edges"] = {"start_frame": start, "end_frame": end}

    pauses = []
    for a, b in loud.quiet:
        if b - a <= LONG_PAUSE or a < start / FPS or b > end / FPS:
            continue
        if any(a < zb and b > za for za, zb in zones):
            continue
        if any(a < kb and b > ka for ka, kb in pauses_keep):
            continue
        o = math.ceil((a + pause_keep) * FPS - 1e-9)
        i = math.floor((b - pause_keep) * FPS + 1e-9)
        if i - o < 1:
            continue
        pauses.append({"id": len(pauses) + 1, "type": "长停顿", "out_frame": o, "in_frame": i,
                       "quiet": [round(a, 3), round(b, 3)],
                       "reason": f"{b - a:.2f} 秒的停顿压到约 {format(round(2 * pause_keep, 2), 'g')} 秒"})
    edl["pauses"] = pauses
    return edl
