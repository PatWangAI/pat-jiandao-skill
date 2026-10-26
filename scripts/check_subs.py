"""只校验字幕，不渲染：<运行>/subtitles.json 的每个字是不是恰好在一行里、顺序对不对、每行有没有超过 16 个字。

用法：uv run python scripts/check_subs.py <视频文件夹> [runs/<时间> 文件夹，默认最新一次]

和 render_rough.py 渲染前的检查是同一个（make_subtitles.check_lines）。写完 subtitles.json 先用它查，
不用等出粗剪；没有讲稿时从 subtitles_terms.json 改出 subtitles.json 以后也用它。
"""

import sys
from pathlib import Path

from common import latest_run, read_json
from make_subtitles import check_lines


def main() -> None:
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    folder = Path(sys.argv[1]).expanduser().resolve()
    run = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else latest_run(folder)
    edl = read_json(run / "edl_final.json")
    tok = read_json(folder / edl["transcript"])["tokens"]
    lines = read_json(run / "subtitles.json")["lines"]
    errs = check_lines(lines, edl, len(tok))
    if errs:
        sys.exit("subtitles.json 校验不通过：\n  - " + "\n  - ".join(errs))
    print(f"subtitles.json 校验通过：{len(lines)} 行")


if __name__ == "__main__":
    main()
