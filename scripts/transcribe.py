"""第 4 步 转写：把 work/audio.wav 转成逐字带时间戳的转写稿。

用法：uv run python scripts/transcribe.py <视频文件夹>

输出 work/transcript.json（每个字，英文是每个词，的开始和结束时间，以及按标点分好的句子）
和 work/transcript.md（给人看的转写稿，每句前面标时间）。
转写结果按原片 sha256 缓存在 cache/transcripts/，同一条原片不重复转写；
缓存里另记 audio.wav 的 sha256，预处理改过、声音变了就重新转写。
load_model() 和 transcribe() 也给 cut_audio.py 的复核用，同一个模型只加载一次。
"""

import json
import sys
import time
from pathlib import Path

from common import ROOT, fmt, join_text, read_json, sha256, write_json

CACHE_DIR = ROOT / "cache" / "transcripts"
MODEL_TAG = "paraformer-zh+fsmn-vad+ct-punc / funasr 1.4.16"
MODEL_DIRS = [  # download_models.py 下到这里
    "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "iic--punc_ct-transformer_cn-en-common-vocab471067-large",
]


def check_models() -> None:
    base = Path.home() / ".cache" / "modelscope" / "models"
    missing = [m for m in MODEL_DIRS if not any((base / m).glob("snapshots/*/*"))]
    if missing:
        sys.exit("转写模型不在：" + "、".join(missing) +
                 "\n请先运行：MODELSCOPE_DOWNLOAD_PARALLELS=8 uv run python scripts/download_models.py")


def load_model():
    from funasr import AutoModel

    return AutoModel(
        model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc",
        device="cpu", disable_update=True,
    )


def transcribe(wav: Path, model) -> dict:
    """返回 FunASR 的原始输出（raw_text 是空格隔开的逐字结果，timestamp 是毫秒）。"""
    try:
        res = model.generate(
            input=str(wav), batch_size_s=300, sentence_timestamp=True, return_raw_text=True,
        )
    except Exception as e:  # FunASR 报错：显示原文，停（SPEC 第 3 节第 4 步）
        sys.exit(f"FunASR 报错：{type(e).__name__}: {e}")
    return res[0]


def build_transcript(raw: dict) -> dict:
    """把 FunASR 的输出整理成：逐字列表 + 句子列表（句子记下自己包含哪几个字）。"""
    words = raw["raw_text"].split()
    stamps = raw["timestamp"]
    if len(words) != len(stamps):
        sys.exit(f"字数 {len(words)} 和时间戳数 {len(stamps)} 对不上，转写结果不能用")
    tokens = [
        {"i": i, "text": w, "start": s / 1000, "end": e / 1000}
        for i, (w, (s, e)) in enumerate(zip(words, stamps))
    ]

    # FunASR 1.4.16 拼出的句子文字会整体错开一个字（2026-10-07 测试片上看到），
    # 但每句的时间戳个数是对的，所以句子文字用逐字结果重拼，只借它句末的标点
    sentences = []
    cursor = 0
    for k, sent in enumerate(raw.get("sentence_info", [])):
        n = len(sent["timestamp"])
        tail = sent["text"].strip()[-1:]
        text = join_text(words[cursor:cursor + n]) + (tail if tail in "，。？！、" else "")
        sentences.append({
            "i": k,
            "text": text,
            "start": sent["start"] / 1000,
            "end": sent["end"] / 1000,
            "first_token": cursor,
            "last_token": cursor + n - 1,
        })
        cursor += n
    if sentences and cursor != len(tokens):
        print(f"注意：句子里的字数合计 {cursor}，逐字列表 {len(tokens)}，分句可能有偏差")
    return {"text": raw["text"], "tokens": tokens, "sentences": sentences}


def write_markdown(tr: dict, path: Path) -> None:
    lines = [
        f"# 转写稿：{tr['source_name']}",
        "",
        f"原片时长 {fmt(tr['duration'])}｜模型 {tr['model']}｜转写用时 {tr['elapsed_seconds']:.1f} 秒",
        "",
        "## 分句",
        "",
        "| 句 | 时间 | 文字 |",
        "| --- | --- | --- |",
    ]
    for s in tr["sentences"]:
        lines.append(f"| {s['i'] + 1} | {fmt(s['start'])}—{fmt(s['end'])} | {s['text']} |")
    lines += ["", "## 逐字时间（秒）", ""]
    for s in tr["sentences"]:
        toks = tr["tokens"][s["first_token"]: s["last_token"] + 1]
        cells = " ".join(f"{t['text']}`{t['start']:.2f}-{t['end']:.2f}`" for t in toks)
        lines.append(f"{s['i'] + 1}. {cells}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    folder = Path(sys.argv[1]).expanduser().resolve()
    inputs = read_json(folder / "inputs.json")
    work = folder / "work"
    wav = work / "audio.wav"
    if not wav.exists():
        sys.exit(f"没有 {wav}，先跑 preprocess.py")
    pre = read_json(work / "preprocess.json")

    digest, audio_digest = inputs["sha256"], sha256(wav)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{digest}.json"
    # 缓存只存模型的原始输出，整理成转写稿这一步每次重做，改整理方式不用重新转写
    cached = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else None
    if cached and cached.get("audio_sha256") == audio_digest:
        print(f"用缓存的转写结果：{cache_file.name}")
        raw, elapsed = cached["funasr_raw"], cached["elapsed_seconds"]
    else:
        check_models()
        t0 = time.time()
        raw = transcribe(wav, load_model())
        elapsed = round(time.time() - t0, 1)
        write_json(cache_file, {"model": MODEL_TAG, "elapsed_seconds": elapsed,
                                "audio_sha256": audio_digest, "funasr_raw": raw})

    transcript = build_transcript(raw)
    transcript.update({
        "source_name": Path(inputs["source"]).name,
        "source_sha256": digest,
        "audio_sha256": audio_digest,
        "model": MODEL_TAG,
        "elapsed_seconds": elapsed,
        "duration": pre["proxy"]["duration"],
        "funasr_raw": raw,
    })
    write_json(work / "transcript.json", transcript)
    write_markdown(transcript, work / "transcript.md")
    print(f"转写完成：{len(transcript['tokens'])} 个字词，{len(transcript['sentences'])} 句，"
          f"用时 {transcript['elapsed_seconds']} 秒 → {work / 'transcript.md'}")


if __name__ == "__main__":
    main()
