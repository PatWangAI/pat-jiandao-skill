"""环境检查：剪辑要用的东西装没装好，缺什么、怎么装（立项卡 v1.1 第二期“装得上”）。

用法：
  uv run python scripts/doctor.py          逐项报“有、缺、可选缺”，缺必需的退出码 1
  uv run python scripts/doctor.py --json   同样的结果按 JSON 打印
  python3 scripts/doctor.py                Python 环境还没装好（uv sync 之前）时用这个，只用自带的库

查的是：Mac 和芯片、uv、Python 环境和固定版本的库、ffmpeg-full（带不带 libass、要用的滤镜和编码器，
再真烧一行字幕看选没选中思源黑体）、字体文件、三个转写模型、本机设置文件和里面的文件夹、磁盘空间；
可选的：Google Chrome（找素材截网页）、swift（小窗跟脸）、Pixabay 密钥（图库找图）、Claude Code。
逐项报“有、缺、可选缺”的做法参考 kajisho5/ffmpeg-skill 的 doctor（MIT），代码是自己写的。
"""

import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {  # download_models.py 下到 ~/.cache/modelscope/models/ 下面这三个文件夹
    "识别 SeACo-Paraformer": "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "断句 FSMN-VAD": "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "标点 CT-Punc": "iic--punc_ct-transformer_cn-en-common-vocab471067-large",
}
PINNED = {"torch": "2.8.0", "torchaudio": "2.8.0", "funasr": "1.4.16", "modelscope": "1.40.1",
          "pyyaml": "6.0.3", "playwright": "1.55.0"}  # 和 pyproject.toml 一致
OTHER_LIBS = {"soundfile": "soundfile", "numpy": "numpy", "PIL": "pillow"}
FILTERS = ["subtitles", "ass", "xfade", "overlay", "alphamerge", "ebur128", "alimiter", "loudnorm", "scale", "crop"]
ENCODERS = ["libx264", "aac"]
MODEL_HINT = "在仓库里运行：MODELSCOPE_DOWNLOAD_PARALLELS=8 uv run python scripts/download_models.py（约 2.1 GB，从 ModelScope 下）"

results: list[dict] = []


def report(name: str, status: str, detail: str, fix: str = "") -> None:
    """status：有、缺（必需的）、可选缺。"""
    results.append({"item": name, "status": status, "detail": detail, "fix": fix})


def load_settings() -> tuple[dict, str]:
    """和 common.py 一样：config.yaml，再用 config.local.yaml 覆盖。没装 pyyaml 时返回空设置，后面用默认值。"""
    try:
        import yaml
    except ImportError:
        return {}, "没装 pyyaml，读不了设置文件，按默认值查"
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    local = ROOT / "config.local.yaml"
    note = "没有 config.local.yaml，用 config.yaml 里的默认值"
    if local.exists():
        cfg.update(yaml.safe_load(local.read_text(encoding="utf-8")) or {})
        note = "config.yaml + config.local.yaml"
    return cfg, note


def check_mac() -> None:
    if sys.platform != "darwin":
        report("系统", "缺", f"这是 {platform.system()}，只支持 macOS")
        return
    chip = platform.machine()
    mem = 0
    r = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip().isdigit():
        mem = int(r.stdout) / 2 ** 30
    detail = f"macOS {platform.mac_ver()[0]}，{'Apple 芯片' if chip == 'arm64' else chip}，内存 {mem:.0f} GB"
    if chip != "arm64":
        report("系统", "有", detail + "；只在 Apple 芯片上测过，Intel 的 Homebrew 装在 /usr/local，要改设置里 ffmpeg 的路径")
    else:
        report("系统", "有", detail + ("；8 GB 能跑，剪片时别同时开太多程序" if mem and mem < 12 else ""))


def check_uv() -> None:
    uv = shutil.which("uv") or next((p for p in ("/opt/homebrew/bin/uv", str(Path.home() / ".local/bin/uv"))
                                     if Path(p).exists()), None)
    if uv:
        v = subprocess.run([uv, "--version"], capture_output=True, text=True).stdout.strip()
        report("uv（管 Python 环境）", "有", v)
    else:
        report("uv（管 Python 环境）", "缺", "找不到 uv", "brew install uv")


def check_python() -> bool:
    venv = ROOT / ".venv"
    in_venv = Path(sys.prefix).resolve() == venv.resolve() if venv.exists() else False
    ver = ".".join(map(str, sys.version_info[:3]))
    if not in_venv or sys.version_info[:2] != (3, 12):
        how = "在仓库里先运行 uv sync（会自动下 Python 3.12 和固定版本的库），再用 uv run python scripts/doctor.py 检查"
        report("Python 环境", "缺", f"现在用的是 {sys.executable}（{ver}），不是仓库的 .venv 里的 3.12", how)
        return False
    bad = []
    for name, want in PINNED.items():
        try:
            got = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            got = None
        if got != want:
            bad.append(f"{name} 要 {want}，现在{'没装' if got is None else '是 ' + got}")
    for mod, pkg in OTHER_LIBS.items():
        if importlib.util.find_spec(mod) is None:
            bad.append(f"{pkg} 没装")
    if bad:
        report("Python 环境", "缺", "；".join(bad), "在仓库里运行 uv sync")
        return False
    report("Python 环境", "有", f"{ver}，{sys.prefix}；torch、funasr 等版本和 pyproject.toml 一致")
    return True


def check_ffmpeg(cfg: dict) -> None:
    ff = cfg.get("ffmpeg", "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg")
    fp = cfg.get("ffprobe", "/opt/homebrew/opt/ffmpeg-full/bin/ffprobe")
    fix = "brew install ffmpeg-full（Homebrew 默认的 ffmpeg 不带 libass，烧不了字幕；ffmpeg-full 不挂到 PATH 上，设置里写全路径）"
    missing = [p for p in (ff, fp) if not Path(p).exists()]
    if missing:
        other = shutil.which("ffmpeg")
        extra = f"；PATH 里有 {other}，但默认版多半不带 libass" if other else ""
        report("ffmpeg-full", "缺", f"找不到 {'、'.join(missing)}{extra}", fix)
        return
    ver = subprocess.run([ff, "-hide_banner", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
    flt = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    enc = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    have_f = {ln.split()[1] for ln in flt.splitlines() if len(ln.split()) > 2 and re.fullmatch(r"[TSC.]{2,3}", ln.split()[0])}
    have_e = {ln.split()[1] for ln in enc.splitlines() if len(ln.split()) > 2 and re.fullmatch(r"[VAS][A-Z.]{5}", ln.split()[0])}
    lack = [f for f in FILTERS if f not in have_f] + [e for e in ENCODERS if e not in have_e]
    if lack:
        why = "（subtitles、ass 是 libass 的，没有就烧不了字幕）" if {"subtitles", "ass"} & set(lack) else ""
        report("ffmpeg-full", "缺", f"{ff} 缺 {'、'.join(lack)}{why}", fix)
        return
    report("ffmpeg-full", "有", f"{ver}；libass、xfade、libx264、aac 等都在")
    check_burn(ff, cfg)


def check_burn(ff: str, cfg: dict) -> None:
    """真烧一行中文字幕，看 libass 选的是不是仓库里的思源黑体（字体名写错时它会悄悄换成系统字体）。"""
    font = ROOT / cfg.get("font_file", "fonts/SourceHanSansCN-Bold.otf")
    if not font.exists():
        report("字幕字体", "缺", f"找不到 {font}", "从仓库重新取一份 fonts/（git checkout -- fonts）")
        return
    name = cfg.get("font_name", "Source Han Sans CN")
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "t.ass").write_text(f"""[Script Info]
ScriptType: v4.00+
PlayResX: 640
PlayResY: 360

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: S,{name},40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,0,0,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,S,,0,0,0,,剪辑环境检查 Claude Code
""", encoding="utf-8")
        r = subprocess.run([ff, "-v", "verbose", "-f", "lavfi", "-i", "color=c=black:s=640x360:r=30",
                            "-vf", f"subtitles=t.ass:fontsdir={font.parent}", "-frames:v", "1", "-f", "null", "-"],
                           cwd=td, capture_output=True, text=True)
    picks = {ln.split("->", 1)[1].strip() for ln in r.stderr.splitlines() if "fontselect:" in ln and "->" in ln}
    if r.returncode != 0 or not picks or any(font.stem not in p for p in picks):
        report("烧字幕试一行", "缺", f"libass 选的字体是 {sorted(picks) or '（没看到选字记录）'}，不是 {font.stem}",
               "检查 config.yaml 的 font_name 和 font_file")
    else:
        report("烧字幕试一行", "有", f"中文和英文都由 {font.name} 渲染")


def check_models() -> None:
    base = Path.home() / ".cache" / "modelscope" / "models"
    lack, size = [], 0
    for label, d in MODELS.items():
        files = list((base / d).glob("snapshots/*/model.pt"))
        if not files:
            lack.append(label)
        else:
            size += files[0].stat().st_size
    if lack:
        report("转写模型", "缺", f"{base} 里没有：{'、'.join(lack)}", MODEL_HINT)
    else:
        report("转写模型", "有", f"三个都在（{base}，模型文件共 {size / 1e9:.1f} GB）")


def check_settings(cfg: dict, note: str) -> None:
    status = "有" if (ROOT / "config.local.yaml").exists() else "可选缺"
    fix = "" if status == "有" else "要改视频放哪、去哪找原片和讲稿：cp config.local.example.yaml config.local.yaml，再照里面的说明改"
    report("本机设置", status, note, fix)
    if not cfg:
        return
    root = Path(str(cfg.get("videos_root", "~/视频制作"))).expanduser()
    root = root if root.is_absolute() else ROOT / root
    parent = next((p for p in [root, *root.parents] if p.exists()), None)
    if parent is None or not os.access(parent, os.W_OK):
        report("视频文件夹", "缺", f"{root} 建不了（{parent} 不能写）", "改 config.local.yaml 的 videos_root")
    else:
        report("视频文件夹", "有", f"{root}{'' if root.exists() else '（还没有，第一次剪的时候自动建）'}")
    dirs = [Path(str(d)).expanduser() for d in cfg.get("video_dirs") or []]
    gone = [str(d) for d in dirs if not d.is_dir()]
    if gone:
        report("找原片的文件夹", "可选缺", f"这些不存在：{'、'.join(gone)}", "改 config.local.yaml 的 video_dirs，或剪的时候直接给原片路径")
    else:
        report("找原片的文件夹", "有", "、".join(map(str, dirs)) or "没设，剪的时候直接给原片路径")
    sr = cfg.get("scripts_root")
    if sr and not Path(str(sr)).expanduser().is_dir():
        report("找讲稿的文件夹", "可选缺", f"{sr} 不存在", "改 config.local.yaml 的 scripts_root，或剪的时候直接给讲稿路径")
    elif sr:
        report("找讲稿的文件夹", "有", str(sr))
    free = shutil.disk_usage(parent or Path.home()).free / 1e9
    report("磁盘空间", "有" if free >= 10 else "缺", f"还剩 {free:.0f} GB" + ("" if free >= 10 else "，一条几分钟的片子中间文件要几个 GB"),
           "" if free >= 10 else "清出 10 GB 以上")


def check_optional() -> None:
    chrome = Path("/Applications/Google Chrome.app")
    report("Google Chrome（找素材时截网页）", "有" if chrome.exists() else "可选缺",
           str(chrome) if chrome.exists() else "没装，找素材时截不了网页，其余照常", "" if chrome.exists() else "装 Google Chrome")
    sw = shutil.which("swift")
    report("swift（小窗跟着脸走）", "有" if sw else "可选缺", sw or "没装，小窗只能用固定取景",
           "" if sw else "xcode-select --install（装 Xcode 命令行工具）")
    key = bool(os.environ.get("PIXABAY_API_KEY"))
    if not key:
        rc = Path.home() / ".zshrc"
        key = rc.exists() and "PIXABAY_API_KEY" in rc.read_text(encoding="utf-8", errors="ignore")
    report("Pixabay 密钥（图库找图）", "有" if key else "可选缺", "环境变量或 ~/.zshrc 里有" if key else "没有，图库的图要自己找",
           "" if key else "在 pixabay.com 登录后打开 API 文档页取免费密钥，写进 ~/.zshrc：export PIXABAY_API_KEY=…（不要贴进聊天）")
    cc = shutil.which("claude")
    report("Claude Code", "有" if cc else "可选缺", cc or "命令行里没找到（用桌面版也行）",
           "" if cc else "装 Claude Code 桌面版或命令行版，在仓库文件夹里打开")


def main() -> None:
    as_json = "--json" in sys.argv
    cfg, note = load_settings()
    check_mac()
    check_uv()
    check_python()
    check_ffmpeg(cfg)
    check_models()
    check_settings(cfg, note)
    check_optional()
    lack = [r for r in results if r["status"] == "缺"]
    if as_json:
        print(json.dumps({"ok": not lack, "items": results}, ensure_ascii=False, indent=1))
    else:
        def cols(s: str) -> int:  # 终端里汉字占两格
            return sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)

        width = max(cols(r["item"]) for r in results)
        for r in results:
            mark = {"有": "✓ 有    ", "缺": "✗ 缺    ", "可选缺": "- 可选缺"}[r["status"]]
            print(f"{mark}  {r['item']}{' ' * (width - cols(r['item']))}  {r['detail']}")
            if r["fix"]:
                print(f"{' ' * (cols(mark) + 2 + width)}  怎么装：{r['fix']}")
        print()
        print("必需的都齐了，可以剪。" if not lack else f"缺 {len(lack)} 样必需的：{'、'.join(r['item'] for r in lack)}。照上面“怎么装”装好再跑一遍。")
    sys.exit(1 if lack else 0)


if __name__ == "__main__":
    main()
