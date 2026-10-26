# Pat 剪刀 · AI 剪辑 skill（pat-jiandao-skill）

文科生 Pat 的文具盒 skills 第 1 件。
把手机录的中文口播（或只录声音）剪成带中英字幕的成片。说“剪成成片”或输入 /jianji 启用；skill 在 .claude/skills/jianji/SKILL.md（TRAE 读 .trae/skills/jianji/ 里同一份的副本），步骤清单在 pipeline.yaml。

## 运行
- 安装：docs/安装说明.md。装好后跑环境检查 `uv run python scripts/doctor.py`。
- Python 依赖：`uv sync`，版本固定在 pyproject.toml 和 uv.lock。脚本一律用 `uv run python scripts/...` 跑，都以视频文件夹为参数。
- FFmpeg 用 ffmpeg-full（带 libass），路径写在设置的 ffmpeg、ffprobe（默认 /opt/homebrew/opt/ffmpeg-full/bin/，不在 PATH 里）。
- 转写模型：`MODELSCOPE_DOWNLOAD_PARALLELS=8 uv run python scripts/download_models.py`。转写结果按原片 sha256 缓存在 cache/。
- 设置：config.yaml（字体、小窗、响度、转场这些参数和默认路径）；config.local.yaml（本机的路径，不进仓库，样例 config.local.example.yaml），写了的项覆盖 config.yaml。

## 规矩
- 代码和设置的注释里提到的 SPEC、PRD、立项卡、技术说明、阶段 N、第 NN 期，指作者没公开的开发文档和往期视频，不用去找。
- 不动系统 Python，用 uv 另建 3.12 的环境。
- 密钥（Pixabay 的 PIXABAY_API_KEY）只放环境变量，不写进仓库、不贴进聊天。
- 视频、音频、素材、runs/、config.local.yaml 不提交。
- 删什么只由用户确认过的方案决定；刀点、时间全由脚本算（见 SKILL.md“硬规则”）。
