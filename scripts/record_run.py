"""第 10 步 记录：在运行记录表里插一行。记在设置里 run_log 指的文件（相对路径从仓库根目录算）；
没设就是 videos_root 下的 运行记录.md，没有这个文件就新建一份（开源首发起，各人的记录不放进仓库）。

用法：uv run python scripts/record_run.py <视频文件夹> [runs/<时间> 文件夹，默认最新一次]

插在运行记录表最后一行之后（不是文件末尾）。能算的字段填上；误删、漏删、手动操作次数写“待填”，
用户验收后由 Claude 补。配错句数出过成片表的写“待填”（用户看成片后定），没出过写“—”；素材两列从 素材/assets.json 取，没有就写“—”。
同一次运行只记一次。
"""

import sys
from pathlib import Path

import yaml

from common import CONFIG, FPS, ROOT, fmt, keep_segments, latest_run, read_json, videos_root

TABLE_HEAD = "## 运行记录"
NEW_LOG = """# 运行记录

skill 每剪一条，record_run.py 在下面的表里插一行。误删、漏删、配错句数、手动操作次数先写“待填”，看过成片以后补上。

## 运行记录

| 日期 | skill 版本 | 原片时长 | 成片时长 | 剪切点数 | 误删 | 漏删 | 配错句数 | 自动获取素材 | 手动补齐素材 | 自检轮数 | 机器耗时 | 手动操作次数 | 现金支出 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""


def log_path() -> Path:
    p = CONFIG.get("run_log")
    if not p:
        return videos_root() / "运行记录.md"
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    folder = Path(sys.argv[1]).expanduser().resolve()
    run = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else latest_run(folder)
    done = run / "record.done"
    if done.exists():
        print(f"这次运行已经记过了：{done.read_text(encoding='utf-8').strip()}")
        return
    inputs = read_json(folder / "inputs.json")
    pre = read_json(folder / "work" / "preprocess.json")
    tr = read_json(folder / "work" / "transcript.json")
    edl = read_json(run / "edl_final.json")
    chk = read_json(run / "check.json")
    timing = read_json(run / "timing.json")
    version = yaml.safe_load((ROOT / "pipeline.yaml").read_text(encoding="utf-8"))["version"]

    total = sum(b - a for a, b in keep_segments(edl)) / FPS
    assets = folder / "素材" / "assets.json"
    if assets.exists():  # 阶段 2 起：自动获取 = 主方案自动 + 自动备选
        sm = read_json(assets)["summary"]
        auto, manual = f"{sm['自动'] + sm['自动（备选）']}（备选 {sm['自动（备选）']}）", str(sm["手动补齐"])
    else:
        auto = manual = "—"
    machine = pre["seconds"] + tr["elapsed_seconds"] + sum(t["seconds"] for t in timing)
    wrong = "待填" if (run / "composition.json").exists() else "—"  # 配错几句由用户看成片后定
    date = run.name[:8]
    row = (f"| {date[:4]}-{date[4:6]}-{date[6:]}（{inputs['name']}，runs/{run.name}） | {version} | "
           f"{fmt(pre['proxy']['duration'])} | {fmt(total)} | {chk['cuts']} | 待填 | 待填 | {wrong} | {auto} | {manual} | "
           f"复核 {chk['recheck_rounds']} 轮 | {machine:.0f} 秒（含预处理和首次转写） | 待填 | 0 |")

    log = log_path()
    if not log.exists():
        log.write_text(NEW_LOG, encoding="utf-8")
        print(f"新建了运行记录：{log}")
    lines = log.read_text(encoding="utf-8").split("\n")
    try:
        k = next(i for i, ln in enumerate(lines) if ln.startswith(TABLE_HEAD))
    except StopIteration:
        sys.exit(f"{log} 里找不到“{TABLE_HEAD}”一节")
    # 跳到表格开始，再找到表格最后一行
    i = k + 1
    while i < len(lines) and not lines[i].startswith("|"):
        i += 1
    while i < len(lines) and lines[i].startswith("|"):
        i += 1
    lines.insert(i, row)
    log.write_text("\n".join(lines), encoding="utf-8")
    done.write_text(row + "\n", encoding="utf-8")
    print(f"运行记录插了一行（{log}）：\n" + row)


if __name__ == "__main__":
    main()
