"""从 ModelScope 下载转写用的三个 FunASR 模型（共约 2.1 GB），存在 ~/.cache/modelscope。

用法：uv run python scripts/download_models.py
已下载的文件不会重复下载。
"""

from modelscope import snapshot_download

MODELS = {
    "识别 SeACo-Paraformer": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "断句 FSMN-VAD": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "标点 CT-Punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
}

for label, model_id in MODELS.items():
    path = snapshot_download(model_id)
    print(f"{label}: {path}", flush=True)
