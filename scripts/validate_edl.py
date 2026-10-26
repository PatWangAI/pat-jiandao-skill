"""第 6 步 校验剪辑表（SPEC 第 7 节）。

用法：uv run python scripts/validate_edl.py <edl.json 或 edl_final.json>

查的是：帧号是整数、都在 0 到 duration_frames 之间；每条删除 out_frame 小于 in_frame；
删除和压停顿互不重叠，都在片头片尾之内；from_token 不大于 to_token，都在转写稿范围内；
每条都有 reason；transcript_sha256 和转写稿对得上；source 文件存在，source_sha256 对得上。
work/edl.json 另查有没有被手改过（和 plan_cuts.py 上次写出的对不上就不通过）。
通过就打印“校验通过”，不通过列出每条错误，退出码 1。
"""

import sys
from functools import lru_cache
from pathlib import Path

from common import read_json, sha256

DEL_TYPES = {"语气词", "说断", "重说", "改稿"}  # 改稿：用户要求删掉的内容，不是口误（比如发某个平台的版本要去掉的词句）


@lru_cache(maxsize=8)
def _sha(path: str, mtime: float) -> str:
    return sha256(Path(path))


def file_sha(path: Path) -> str:
    return _sha(str(path), path.stat().st_mtime)


def find_folder(path: Path) -> Path:
    for p in [path.parent, *path.parents]:
        if (p / "inputs.json").exists():
            return p
    raise SystemExit(f"{path} 不在视频文件夹里（往上找不到 inputs.json）")


def edl_touched(folder: Path) -> str | None:
    """work/edl.json 和 plan_cuts.py 上次写出的不一样（有人手改过）时，返回说明；没改过返回 None。"""
    edl, state = folder / "work" / "edl.json", folder / "work" / ".plan_cuts_last.json"
    if not edl.exists():
        return "work/edl.json 还没有，先跑 plan_cuts.py"
    if not state.exists() or read_json(state).get("edl_sha256") != sha256(edl):
        return ("work/edl.json 和 plan_cuts.py 上次写出的不一样，可能有人手改过。edl.json 不手改："
                "先问用户，把改动换成 deletions.json 或 decisions.json 里的条目，从 plan_cuts.py 重跑")
    return None


def validate(edl: dict, folder: Path) -> list[str]:
    errs = []
    dur = edl.get("duration_frames")
    if not isinstance(dur, int) or dur <= 0:
        return ["duration_frames 不是正整数"]

    def frame_ok(x, what):
        if not isinstance(x, int) or isinstance(x, bool):
            errs.append(f"{what} 不是整数：{x!r}")
            return False
        if not 0 <= x <= dur:
            errs.append(f"{what} = {x} 不在 0 到 {dur} 之间")
            return False
        return True

    e = edl.get("edges", {})
    s0, s1 = e.get("start_frame"), e.get("end_frame")
    if frame_ok(s0, "片头 start_frame") and frame_ok(s1, "片尾 end_frame") and s0 >= s1:
        errs.append(f"片头 {s0} 不早于片尾 {s1}")

    tr_path = folder / edl.get("transcript", "")
    n_tokens = None
    if not tr_path.is_file():
        errs.append(f"转写稿不存在：{tr_path}")
    else:
        if file_sha(tr_path) != edl.get("transcript_sha256"):
            errs.append("transcript_sha256 和转写稿对不上（转写稿改过，剪辑表要重新生成）")
        n_tokens = len(read_json(tr_path)["tokens"])
    src = folder / edl.get("source", "")
    if not src.is_file():
        errs.append(f"原片不存在：{src}")
    elif file_sha(src) != edl.get("source_sha256"):
        errs.append("source_sha256 和原片对不上")

    ranges = []
    for kind, items in (("删除", edl.get("deletions", [])), ("压停顿", edl.get("pauses", [])),
                        ("连读待定", edl.get("pending", []))):
        for d in items:
            name = f"{kind}第 {d.get('id')} 条"
            if not d.get("reason"):
                errs.append(f"{name} 没写 reason")
            if kind == "删除" and d.get("type") not in DEL_TYPES:
                errs.append(f"{name} 的 type “{d.get('type')}” 不是语气词、说断、重说、改稿之一")
            if kind != "压停顿":
                f, t = d.get("from_token"), d.get("to_token")
                if not isinstance(f, int) or not isinstance(t, int) or f > t:
                    errs.append(f"{name} 的 from_token {f!r} 大于 to_token {t!r} 或不是整数")
                elif n_tokens is not None and not (0 <= f and t < n_tokens):
                    errs.append(f"{name} 的字序号 {f}—{t} 超出转写稿范围（共 {n_tokens} 个字）")
            if kind == "连读待定":
                continue
            o, i = d.get("out_frame"), d.get("in_frame")
            if frame_ok(o, f"{name} out_frame") and frame_ok(i, f"{name} in_frame"):
                if o >= i:
                    errs.append(f"{name} 的 out_frame {o} 不小于 in_frame {i}")
                head_tail = d.get("out_how") == "片头" or d.get("in_how") == "片尾"
                if not head_tail and isinstance(s0, int) and isinstance(s1, int) and (o < s0 or i > s1):
                    errs.append(f"{name}（{o}—{i} 帧）不在片头片尾（{s0}—{s1} 帧）之内")
                if not head_tail:
                    ranges.append((o, i, name))
    ranges.sort()
    for (a0, a1, an), (b0, b1, bn) in zip(ranges, ranges[1:]):
        if b0 < a1:
            errs.append(f"{an}（{a0}—{a1} 帧）和{bn}（{b0}—{b1} 帧）重叠")
    return errs


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1]).expanduser().resolve()
    folder = find_folder(path)
    errs = validate(read_json(path), folder)
    if path == folder / "work" / "edl.json" and (msg := edl_touched(folder)):
        errs.insert(0, msg)
    if errs:
        print(f"校验不通过，共 {len(errs)} 条：")
        for e in errs:
            print("  - " + e)
        sys.exit(1)
    print(f"校验通过：{path.name}")


if __name__ == "__main__":
    main()
