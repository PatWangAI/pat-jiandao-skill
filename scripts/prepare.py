"""第 1 步 准备：找原片和讲稿，建视频文件夹，复制原片（原来的文件不动）。

用法：
  uv run python scripts/prepare.py --list
      列出最新的 3 个原片候选和找到的讲稿，给用户选
  uv run python scripts/prepare.py --video <原片> [--script <讲稿> | --no-script] [--name <名称>] [--no-face] [--vertical]
      建 <videos_root>/<日期_名称>/，复制原片，写 inputs.json。同一条原片（sha256 相同）已经有文件夹的，接着用。
      原片可以是视频，也可以是纯录音（m4a、wav、mp3 等，手机录音或语音备忘录）。
      --no-face：无人脸模式（2026-10-18 加），只用声音，成片不叠人脸小窗；inputs.json 记 "no_face": true。
      纯录音自动按无人脸模式；带画面的视频不想露脸就加这个参数
      --no-script：没有讲稿（讲稿可选，第二期，2026-10-24 验收）。只靠转写稿判断语气词、说断和重说；inputs.json 记 "no_script": true，
      script 是空的。默认关：没加它又找不到讲稿时照旧停下，不自己改成无讲稿
      --vertical：出竖屏成片 1080×1920（第二期，2026-10-24 选的摆法 C：没配素材的句子人脸全屏，配了素材的句子
      整屏换成素材，不叠小窗）；inputs.json 记 "vertical": true。默认关，不说竖屏就照旧出 16:9。已经建好的文件夹补跑一次也能改上
  uv run python scripts/prepare.py --script-only [--script <讲稿>] [--name <名称>]
      还没录原片时，只按讲稿建 <videos_root>/<日期_名称>/（第 2 步找素材用）；原片来了用同一个名称接着用
  uv run python scripts/prepare.py --script-text <视频文件夹>
      打印讲稿里“讲稿全文”一节（讲稿不复制进仓库）
  uv run python scripts/prepare.py --shot-table <视频文件夹>
      打印讲稿全文、镜头表、生图提示词和图库关键词、来源几节（第 2 步写素材清单用）
  uv run python scripts/prepare.py --script-from-cut <视频文件夹>
      没有讲稿的视频，粗剪定了以后，按最新一次运行的 subtitles.json 生成一份讲稿 work/讲稿_按转写.md
      （句子按转写稿的句号、问号断，太长的在字幕行之间切开），写进 inputs.json 的 script，后面找素材、配素材照常用它

设置在 config.yaml，本机的路径在 config.local.yaml（覆盖同名项）。
找原片：设置里 video_dirs 的几个文件夹里扩展名为 .mov、.mp4、.m4v 的视频和 .m4a、.wav、.mp3 等录音，按修改时间取最新的 3 个。
找讲稿：scripts_root（没设就不自动找）下期号最大的“第NN期_…”文件夹里，文件名含“讲稿”、版本号最高的一份。
"""

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

from common import CONFIG, has_picture, read_json, sha256, videos_root, write_json

VIDEO_EXT = {".mov", ".mp4", ".m4v"}
AUDIO_EXT = {".m4a", ".wav", ".mp3", ".aac", ".caf", ".aiff", ".aif", ".flac"}  # 手机录音、语音备忘录


def video_candidates(n: int = 3) -> list[Path]:
    files = []
    for d in CONFIG.get("video_dirs") or []:
        d = Path(d).expanduser()
        if d.is_dir():
            files += [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXT | AUDIO_EXT]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:n]


def find_script() -> Path | None:
    if not CONFIG.get("scripts_root"):
        return None
    root = Path(CONFIG["scripts_root"]).expanduser()
    if not root.is_dir():
        return None
    eps = [(int(m.group(1)), p) for p in root.iterdir()
           if p.is_dir() and (m := re.match(r"第(\d+)期", p.name))]
    if not eps:
        return None
    folder = max(eps)[1]

    def version(p: Path) -> tuple:
        m = re.search(r"_v(\d+)\.(\d+)", p.name)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    scripts = [p for p in folder.glob("*.md") if "讲稿" in p.name]
    return max(scripts, key=version) if scripts else None


def script_text(path: Path) -> str:
    """只取“## 讲稿全文”这一节。"""
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(r"^## 讲稿全文\s*$(.*?)(?=^## |\Z)", text, re.S | re.M)
    if not m:
        raise SystemExit(f"讲稿里没有“## 讲稿全文”一节：{path}")
    return m.group(1).strip()


def script_sections(path: Path, titles: tuple) -> str:
    """取讲稿里标题以这些字开头的几节，原样拼起来。"""
    text = Path(path).read_text(encoding="utf-8")
    parts = re.findall(r"^(## .*?)(?=^## |\Z)", text, re.S | re.M)
    return "\n".join(x.strip() + "\n" for x in parts if x[3:].startswith(titles))


def sentences(path: Path) -> dict[int, str]:
    """讲稿全文里按“1. ……”编号的句子。"""
    return {int(m.group(1)): m.group(2).strip()
            for m in re.finditer(r"^(\d+)\.\s*(.+)$", script_text(path), re.M)}


def default_name(script: Path | None) -> str:
    """名称默认取讲稿主题，比如“自媒体_第03期_Astra锐评讲稿与镜头表_v0.2”→ Astra锐评。"""
    if script:
        m = re.search(r"第\d+期_(.+?)讲稿", script.name)
        if m:
            return m.group(1)
    return "未命名"


def find_existing(digest: str) -> Path | None:
    root = videos_root()
    for p in root.glob("*/inputs.json"):
        if read_json(p).get("sha256") == digest:
            return p.parent
    return None


def find_waiting(name: str) -> Path | None:
    """只按讲稿建好、还在等原片的文件夹（同一个名称）。"""
    for p in sorted(videos_root().glob(f"*_{name}/inputs.json")):
        if not read_json(p).get("sha256"):
            return p.parent
    return None


def new_folder(name: str) -> Path:
    folder = videos_root() / f"{time.strftime('%Y-%m-%d')}_{name}"
    if folder.exists():
        folder = folder.with_name(folder.name + time.strftime("_%H%M%S"))
    for sub in ("原片", "素材", "work", "runs", "成片"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    return folder


SENT_MAX = 24  # 按转写生成讲稿时，一句攒到这么多字就在字幕行之间切开（一句一个画面）


def script_from_cut(folder: Path) -> Path:
    """没有讲稿的视频：按剪好的字幕（最新一次运行的 subtitles.json）生成讲稿，句号给找素材、配素材用。"""
    from common import latest_run, text_width
    inputs = read_json(folder / "inputs.json")
    if inputs.get("script") and not inputs.get("script_from_cut"):
        sys.exit(f"这条视频有讲稿（{inputs['script']}），不用生成")
    run = latest_run(folder)
    if not (run / "subtitles.json").exists():
        sys.exit(f"{run / 'subtitles.json'} 还没有：粗剪的字幕定了再生成讲稿")
    lines = read_json(run / "subtitles.json")["lines"]
    tr = read_json(folder / "work" / "transcript.json")
    ends = {x["last_token"] for x in tr["sentences"] if x["text"][-1:] in "。？！"}
    sents, cur = [], []
    for ln in lines:
        cur.append(ln)
        width = sum(text_width(x["text"]) for x in cur)
        if any(ln["from_token"] <= e <= ln["to_token"] for e in ends) or width >= SENT_MAX or ln is lines[-1]:
            # 字幕里句中的逗号写成了空格：两边都是汉字的空格换回逗号，英文词之间的空格不动
            text = "，".join(re.sub(r"(?<=[^\x00-\x7f]) (?=[^\x00-\x7f])", "，", x["text"]) for x in cur)
            sents.append((text if text.endswith("？") else text + "。", cur[0]["from_token"], cur[-1]["to_token"]))
            cur = []
    out = folder / "work" / "讲稿_按转写.md"
    L = [f"# 讲稿（按转写生成）：{inputs.get('name', folder.name)}", "",
         f"没有讲稿。这份是 prepare.py --script-from-cut 按 {run.name} 的字幕生成的（{time.strftime('%Y-%m-%d %H:%M')}），"
         "只给找素材、配素材用：句号就是一句一个画面。改了字幕要重新生成。", "", "## 讲稿全文", ""]
    L += [f"{k}. {t}" for k, (t, _, _) in enumerate(sents, 1)]
    L += ["", "## 句子在转写稿里的字序号（写 shots.json 用）", ""]
    L += [f"- 第 {k} 句：from_token {f}，to_token {t}" for k, (_, f, t) in enumerate(sents, 1)]
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    write_json(folder / "inputs.json", {**inputs, "script": str(out), "script_from_cut": True})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--script")
    ap.add_argument("--name")
    ap.add_argument("--script-text", dest="script_text")
    ap.add_argument("--shot-table", dest="shot_table")
    ap.add_argument("--script-only", dest="script_only", action="store_true")
    ap.add_argument("--new-script", dest="new_script", action="store_true")
    ap.add_argument("--no-face", dest="no_face", action="store_true")
    ap.add_argument("--no-script", dest="no_script", action="store_true")
    ap.add_argument("--vertical", action="store_true")
    ap.add_argument("--script-from-cut", dest="script_from_cut")
    a = ap.parse_args()

    if a.script_from_cut:
        out = script_from_cut(Path(a.script_from_cut).expanduser().resolve())
        print(f"按粗剪的字幕生成了讲稿：{out}（{len(sentences(out))} 句）")
        return
    if a.script_text or a.shot_table:
        inputs = read_json(Path(a.script_text or a.shot_table) / "inputs.json")
        if not inputs.get("script"):
            sys.exit("这条视频没有讲稿（无讲稿模式）：判删除看 work/阅读稿.md；要找素材先在粗剪定了以后跑 --script-from-cut")
        if a.script_text:
            print(script_text(Path(inputs["script"])))
        else:
            print(script_sections(Path(inputs["script"]), ("讲稿全文", "镜头表", "AI 生图", "来源")))
        return
    if a.script_only:
        script = Path(a.script).expanduser().resolve() if a.script else find_script()
        if script is None or not script.is_file():
            sys.exit("缺文件：讲稿")
        script_text(script)
        name = a.name or default_name(script)
        folder = find_waiting(name)
        if folder:
            print(f"这份讲稿已经有文件夹，接着用：{folder}")
            return
        folder = new_folder(name)
        write_json(folder / "inputs.json", {"script": str(script), "name": name, "date": time.strftime("%Y-%m-%d")})
        print(f"建好了（还没有原片）：{folder}")
        return

    if a.list or not a.video:
        vids = video_candidates()
        print("原片候选（最新的在前）：")
        for i, p in enumerate(vids, 1):
            mb = p.stat().st_size / 1e6
            kind = "，录音" if p.suffix.lower() in AUDIO_EXT else ""
            print(f"  {i}. {p}  （{time.strftime('%m-%d %H:%M', time.localtime(p.stat().st_mtime))}，{mb:.0f} MB{kind}）")
        if not vids:
            print("  没找到。请把原片放进“下载”文件夹，或给出 --video 路径")
        s = find_script()
        print(f"讲稿：{s or '没找到'}")
        print(f"默认名称：{default_name(s)}")
        return

    src = Path(a.video).expanduser().resolve()
    if a.no_script and a.script:
        sys.exit("--no-script 和 --script 只能用一个")
    script = None if a.no_script else Path(a.script).expanduser().resolve() if a.script else find_script()
    missing = [x for x, ok in (("原片 " + str(src), src.is_file()),
                                ("讲稿", a.no_script or (script is not None and script.is_file()))) if not ok]
    if missing:
        hint = ("。没有讲稿的话，问用户后加 --no-script（讲稿可选：只靠转写稿判断语气词、说断和重说）"
                if "讲稿" in missing else "")
        sys.exit("缺文件：" + "；".join(missing) + hint)
    if script:
        script_text(script)  # 先确认讲稿里有“讲稿全文”一节
    # 无人脸模式：纯录音自动打开；带画面的视频要 --no-face
    no_face = a.no_face or not has_picture(src)

    digest = sha256(src)
    folder = find_existing(digest)
    if folder:
        print(f"这条原片已经有文件夹，接着用：{folder}")
        inputs = read_json(folder / "inputs.json")
        if no_face and not inputs.get("no_face"):
            inputs = {**inputs, "no_face": True}
            write_json(folder / "inputs.json", inputs)
            print("改成无人脸模式（inputs.json 记了 no_face），从 preprocess.py 起重跑")
        if a.vertical and not inputs.get("vertical"):
            write_json(folder / "inputs.json", {**inputs, "vertical": True})
            print("改成出竖屏成片（inputs.json 记了 vertical），从 compose.py 起重跑")
        return
    name = a.name or default_name(script)
    folder = find_waiting(name) if script else None  # 只按讲稿建的文件夹，没讲稿时不会有
    if folder:
        old_script = read_json(folder / "inputs.json").get("script")
        if old_script != str(script) and not a.new_script:
            sys.exit(f"{folder.name} 找素材时用的讲稿是 {Path(old_script).name}，现在是 {script.name}。"
                     "素材按旧讲稿的句号排，先问用户：要换新讲稿就加 --new-script，再按新讲稿检查素材清单")
        print(f"接着用只按讲稿建好的文件夹（素材可能已经找过）：{folder}")
    else:
        folder = new_folder(name)
        # 同一份讲稿重录了一条原片：素材只依赖讲稿，把旧文件夹里找好的素材复制过来（旧的不动）
        olds = [p.parent for p in sorted(videos_root().glob(f"*_{name}/inputs.json"))
                if script and p.parent != folder and read_json(p).get("script") == str(script) and (p.parent / "素材" / "plan.json").exists()]
        if olds:
            shutil.copytree(olds[-1] / "素材", folder / "素材", dirs_exist_ok=True)
            print(f"同一份讲稿在 {olds[-1].name} 里已经找过素材，复制了一份过来")
    copied = folder / "原片" / src.name
    shutil.copy2(src, copied)
    if sha256(copied) != digest:
        sys.exit("复制后的原片和原来的不一样，请重试")
    old = read_json(folder / "inputs.json") if (folder / "inputs.json").exists() else {}
    write_json(folder / "inputs.json", {
        **old,
        "source_original": str(src),
        "source": str(copied.relative_to(folder)),
        "sha256": digest,
        "script": str(script) if script else None,
        "name": name,
        "date": old.get("date", time.strftime("%Y-%m-%d")),
        **({"no_face": True} if no_face else {}),
        **({"no_script": True} if a.no_script else {}),
        **({"vertical": True} if a.vertical else {}),
    })
    print(f"建好了：{folder}" + ("（无人脸模式：只用声音，成片不叠小窗）" if no_face else "")
          + ("（没有讲稿：只靠转写稿判删除）" if a.no_script else "") + ("（出竖屏成片 1080×1920）" if a.vertical else ""))


if __name__ == "__main__":
    main()
