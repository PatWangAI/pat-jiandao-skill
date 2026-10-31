"""按成片（或粗剪）里的秒数，找出那一刻是第几句、用的哪项素材、哪行字幕，以及要改的话改哪个文件、从哪步重跑。

用法：uv run python scripts/locate.py <视频文件夹> <时间> [<时间> ...] [--run runs/<时间> 文件夹，默认最新一次]
时间写秒数或分:秒，比如 23、23.5、1:05、01:05.2。

用户说“第 23 秒那张图换掉”“1 分 05 秒那行字幕错了”时先跑它，只读它指出的那一条再改，不用通读 shots.json 和 subtitles.json。
成片的时间和粗剪一样（成片的声音就是粗剪的声音），所以出成片以前也能用，那时只报字幕。
出片以后又改过 shots.json 或 subtitles.json 但还没重跑的，会提示，时间按上次出片算。
"""

import argparse
import re
from pathlib import Path

from common import fmt, latest_run, read_json, sha256


def parse_time(s: str) -> float:
    m = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)", s.strip())
    if not m:
        raise SystemExit(f"看不懂时间“{s}”，写秒数或分:秒，比如 23、1:05")
    return int(m.group(1) or 0) * 60 + float(m.group(2))


def sec(t: str) -> float:
    h, m, s = t.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def read_subs(path: Path) -> list[tuple[float, float, str]]:
    """字幕的时间：.srt 取每块第一行字（中文），.ass 只取 Sub 样式那几行（En 是英文行）。"""
    out = []
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".srt":
        for block in re.split(r"\n\s*\n", text.strip()):
            rows = block.strip().splitlines()
            if len(rows) >= 3 and "-->" in rows[1]:
                a, b = (x.strip() for x in rows[1].split("-->"))
                out.append((sec(a), sec(b), rows[2].strip()))
        return out
    for row in text.splitlines():
        if row.startswith("Dialogue:"):
            f = row[len("Dialogue:"):].split(",", 9)
            if f[3].strip() == "Sub":
                out.append((sec(f[1].strip()), sec(f[2].strip()), re.sub(r"\{[^}]*\}", "", f[9]).replace("\\N", "").strip()))
    return out


def subs_file(run: Path, final: bool) -> Path | None:
    """10-26 以前的运行目录没有 .srt，退回读 .ass。"""
    names = ["成片.srt", "final.ass", "粗剪.srt", "subtitles.ass"] if final else ["粗剪.srt", "subtitles.ass"]
    return next((run / n for n in names if (run / n).exists()), None)


def same_hash(path: Path, recorded: str | None) -> bool:
    return not recorded or not path.exists() or sha256(path) == recorded


def describe_scene(sc: dict, nth: int, fps: int, by_id: dict) -> list[str]:
    a, b = sc["start_frame"] / fps, sc["end_frame"] / fps
    if sc.get("kind") == "face":
        picture = f"人脸（{sc.get('desc', '')}）"
    else:
        picture = f"素材 {sc['asset']}，{sc.get('file') or '没有文件'} — {sc.get('desc', '')}"
        src = sc.get("source")
        if src:
            picture += f"；来源 {src.get('media', '')} {src.get('date', '')}".rstrip()
    trans = sc.get("transition_label") or "无"
    lines = [
        f"  第 {sc['sentence']} 句 {fmt(a)}—{fmt(b)}（画面：{picture}）",
        f"    说的：{sc.get('spoken', '')}",
        f"    讲稿：{sc.get('script_text', '')}",
        f"    动效：{sc.get('effect', '无')}；进场转场：{trans}",
        f"    换画面、动效、转场 → work/shots.json 第 {nth} 条（sentence {sc['sentence']}），改完从 compose.py 起重跑",
    ]
    if sc["asset"] in by_id:
        lines.append(f"    换这项素材本身 → 素材/plan.json 里 id {sc['asset']} 那项，"
                     f"重取用 fetch_assets.py --only {sc['asset']}，再从 compose.py 起重跑")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("times", nargs="+")
    ap.add_argument("--run")
    a = ap.parse_args()
    times = [parse_time(raw) for raw in a.times]
    folder = Path(a.folder).expanduser().resolve()
    run = Path(a.run).resolve() if a.run else latest_run(folder)
    lines = read_json(run / "subtitles.json")["lines"] if (run / "subtitles.json").exists() else []

    comp = read_json(run / "composition.json") if (run / "composition.json").exists() else None
    notes = []
    if comp:
        fps = comp["fps"]
        total = comp["total_frames"] / fps
        sub_path = subs_file(run, final=True)
        shots = read_json(folder / "work" / "shots.json")["shots"] if (folder / "work" / "shots.json").exists() else []
        nth_of = {s["sentence"]: i + 1 for i, s in enumerate(shots)}
        plan = folder / "素材" / "plan.json"
        by_id = {it["id"]: it for it in read_json(plan).get("items", [])} if plan.exists() else {}
        if not same_hash(folder / "work" / "shots.json", comp.get("shots_sha256")):
            notes.append("work/shots.json 在上次出成片以后改过、还没重跑，下面的画面按上次成片的样子报")
        if not same_hash(run / "subtitles.json", comp.get("subtitles_sha256")):
            notes.append("subtitles.json 在上次出成片以后改过、还没重跑，字幕时间按上次出片算")
        where = f"成片（{run.name}，共 {fmt(total)}）"
    else:
        sub_path = subs_file(run, final=False)
        total = None
        where = f"粗剪（{run.name}，还没出成片，只报字幕）"
    srt = read_subs(sub_path) if sub_path else []
    if not sub_path:
        notes.append("运行目录里没有字幕文件（.srt 或 .ass），报不了字幕")
    elif lines and (len(srt) != len(lines) or any(s[2].replace(" ", "") != l["text"].replace(" ", "") for s, l in zip(srt, lines))):
        notes.append(f"{sub_path.name} 和 subtitles.json 对不上（字幕改过还没重渲），字幕时间按 {sub_path.name} 报")

    print(f"在{where}里找：")
    for n in notes:
        print(f"  注意：{n}")
    for t in times:
        print(f"\n{fmt(t)}：")
        if total is not None and t >= total:
            print(f"  超出成片长度 {fmt(total)}")
            continue
        if comp:
            f = int(t * fps)
            scenes = comp["scenes"]
            hit = next((i for i, sc in enumerate(scenes) if sc["start_frame"] <= f < sc["end_frame"]), None)
            if hit is None:
                print("  这一帧不在任何一句里")
            else:
                sc = scenes[hit]
                for row in describe_scene(sc, nth_of.get(sc["sentence"], "?"), fps, by_id):
                    print(row)
                # 转场从后一句的起点开始，占后一句开头的 transition_frames 帧（render_final.py 开头的说明）
                tf = comp.get("transition_frames", 0)
                k = f - sc["start_frame"]
                if hit > 0 and sc.get("transition") and k < tf:
                    print(f"    这一刻还在从第 {scenes[hit - 1]['sentence']} 句转进来的转场（{sc['transition_label']}）里，"
                          f"第 {k + 1}/{tf} 帧，画面里两句都有；改转场就改第 {sc['sentence']} 句那条")
        sub = next((i for i, s in enumerate(srt) if s[0] <= t < s[1]), None)
        if not srt:
            continue
        if sub is None:
            before = [i for i, s in enumerate(srt) if s[1] <= t]
            print("  这一刻没有字幕" + (f"（前一行是第 {before[-1] + 1} 行「{srt[before[-1]][2]}」）" if before else ""))
        else:
            s0, s1, text = srt[sub]
            en = lines[sub].get("en", "") if sub < len(lines) else ""
            print(f"  字幕：第 {sub + 1} 行 {fmt(s0)}—{fmt(s1)}「{text}」" + (f" / {en}" if en else ""))
            tok = f"，from_token {lines[sub]['from_token']}" if sub < len(lines) else ""
            redo = "改完从 render_rough.py 起重跑，再跑 check.py" + ("；出过成片的再从 compose.py 起重跑" if comp else "")
            print(f"    改这行字幕 → {run.name}/subtitles.json 第 {sub + 1} 行{tok}，{redo}")


if __name__ == "__main__":
    main()
