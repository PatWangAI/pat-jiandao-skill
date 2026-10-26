# 第三方说明

这个仓库自己的代码按 MIT 许可公开（见 LICENSE）。下面列的是仓库里带着的、装的时候要下载的、运行时会调用的别人的东西，各按各自的许可。

## 一、仓库里带着的

### 思源黑体 Source Han Sans CN Bold

- 文件：`fonts/SourceHanSansCN-Bold.otf`，用来烧字幕。
- 来源：Adobe，https://github.com/adobe-fonts/source-han-sans （SubsetOTF/CN）。
- 许可：SIL Open Font License 1.1，原文在 `licenses/SourceHanSans-OFL.txt`。可以随软件一起分发；不能单独拿去卖；改过的字体不能再用保留名“Source”。本仓库没有改动字体文件。

## 二、装的时候自己下载的（不在仓库里）

### 转写模型（FunASR）

`scripts/download_models.py` 从 ModelScope 下载这三个模型，存在你自己的电脑上：

| 用途 | 模型 | ModelScope 页面标的许可 |
| --- | --- | --- |
| 识别 | iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch | Apache License 2.0 |
| 断句 | iic/speech_fsmn_vad_zh-cn-16k-common-pytorch | Apache License 2.0 |
| 标点 | iic/punc_ct-transformer_cn-en-common-vocab471067-large | Apache License 2.0 |

- 作者：FunASR 团队（Alibaba Group），https://github.com/modelscope/FunASR
- FunASR 的模型协议（FunASR Model Open Source License Agreement）要求使用、复制、修改、分享时注明来源和作者、保留模型名字。本项目不分发模型文件，也不改模型。

### Python 库

`uv sync` 按 `pyproject.toml` 和 `uv.lock` 装，本仓库不带这些库：

| 库 | 许可 |
| --- | --- |
| torch、torchaudio | BSD |
| funasr | MIT |
| modelscope | Apache-2.0 |
| soundfile | BSD-3-Clause |
| numpy | BSD-3-Clause 等 |
| pyyaml | MIT |
| playwright | Apache-2.0 |
| pillow | MIT-CMU |
| pyJianYingDraft（导剪映草稿用，https://github.com/GuanYixuan/pyJianYingDraft ） | Apache-2.0 |
| pymediainfo、imageio（pyJianYingDraft 用的） | MIT、BSD-2-Clause |

它们各自还依赖别的库，许可以各自的包为准。本仓库调用这些库，没有复制它们的代码。

## 三、运行时调用的程序和服务

- FFmpeg（ffmpeg-full，用 Homebrew 装）：作为单独的程序调用，本仓库不带 FFmpeg。它的许可取决于编译方式（LGPL 或 GPL），见 https://ffmpeg.org/legal.html 。
- Google Chrome（可选，找素材时截网页）：用户自己装。
- Pixabay 图库接口（可选）：每个用户用自己申请的密钥，放在环境变量里，仓库不带密钥。用接口要守 Pixabay 的条款（https://pixabay.com/api/docs/ ），比如注明素材来源、不大批量自动下载；下到的图片和视频按 Pixabay Content License 用。

## 四、素材的版权

skill 会按你的讲稿截网页、下图库素材、放背景音乐，并写一份 来源清单.md。这些素材的版权属于原作者，用在视频里是否合适，由用的人自己判断；发布前请看来源清单。遇到网站拦截程序访问，skill 不绕过。仓库里不带任何背景音乐、图库素材和网页截图。

## 五、商标

Claude、Claude Code 是 Anthropic 的商标；TRAE、剪映是字节跳动旗下的产品；FFmpeg、Pixabay、ModelScope 等名称属于各自的所有者。本项目与这些公司和项目没有隶属或背书关系，提到它们只是为了说明怎么配合使用。
