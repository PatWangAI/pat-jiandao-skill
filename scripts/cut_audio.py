"""第 8 步之一 剪声音并复核（SPEC 第 4 节第 6—8 条）。

用法：uv run python scripts/cut_audio.py <视频文件夹>

按 work/edl.json（用户确认的方案）剪声音，然后复核：用同一个模型把剪好的 16 kHz 声音再转写一遍，
和“原转写去掉删掉的字”逐字对照，只看每个删除接缝前后 3 个字以内的差异。
四种情况自动挪刀，每次只挪到相邻的一段静音：
  前一刀少删（多出的字对得上被删内容的开头）→ 前一刀挪到前一段静音
  后一刀少删（多出的字对得上被删内容的结尾）→ 后一刀挪到后一段静音
  前一刀多删（接缝前的保留字整字缺失）     → 前一刀挪到后一段静音
  后一刀多删（接缝后的保留字整字缺失）     → 后一刀挪到前一段静音
认法不同造成的替换不挪，列出来给人听。最多 3 轮，第 3 轮只复核不再挪；仍有问题时，回到没有多删、
少删最少的那一轮（先比多删，再比少删，一样好取较晚的一轮）。
复核只挪刀，不改删什么。

输出在新建的 runs/<时间>/ 里：edl_final.json（挪过的刀写在这里）、粗剪.wav（48 kHz）、
recheck.json（每轮转写、发现、挪刀）、recheck_第N轮.wav、timing.json。
"""

import copy
import difflib
import sys
import time
from pathlib import Path

import soundfile as sf

from common import (AFTER, BEFORE, FPS, MIN_SOUND, SR, SR_OUT, add_timing, analyze, apply_pauses,
                    blank_spans, frame_in_quiet, quiet_over, keep_segments, new_run_dir, normalize, read_json, seam_times,
                    splice, write_json)
from transcribe import check_models, load_model, transcribe
from validate_edl import edl_touched, validate

MAX_ROUNDS = 3
NEAR = 3  # 只看接缝前后 3 个字以内


def expected_string(tok: list, edl: dict):
    """原转写去掉删掉的字，逐字拼成一串；同时记下每个删除接缝在这串里的位置。"""
    gone = set()
    for d in edl["deletions"]:
        gone.update(range(d["from_token"], d["to_token"] + 1))
    s, seams = "", {}
    starts = {d["from_token"]: d for d in edl["deletions"]}
    for i, t in enumerate(tok):
        if i in starts and starts[i]["out_how"] != "片头" and starts[i]["in_how"] != "片尾":
            seams[starts[i]["id"]] = len(s)
        if i not in gone:
            s += normalize(t["text"])
    return s, seams


def actual_string(raw: dict):
    """复核转写拼成一串，每个字记下它在粗剪里的时间（秒）。"""
    words = raw.get("raw_text", "").split()
    stamps = raw.get("timestamp", [])
    s, times = "", []
    for w, (a, b) in zip(words, stamps):
        n = normalize(w)
        s += n
        times += [(a / 1000 + b / 1000) / 2] * len(n)
    return s, times


def compare(edl: dict, tok: list, raw: dict) -> tuple[list, str]:
    exp, seams = expected_string(tok, edl)
    act, times = actual_string(raw)
    seam_t = {}
    for st in seam_times(edl):
        for kind, n in st["owners"]:
            if kind == "删除":
                seam_t[n] = st["rough_s"]
    dels = {d["id"]: d for d in edl["deletions"]}
    found = []
    ops = difflib.SequenceMatcher(None, exp, act, autojunk=False).get_opcodes()
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        near = []
        for did, p in seams.items():
            hit = (p - NEAR <= i1 <= p + NEAR) if i1 == i2 else (i1 < p + NEAR and i2 > p - NEAR)
            if hit:
                near.append((min(abs(i1 - p), abs(i2 - p)), did, p))
        if not near:
            continue
        _, did, p = min(near)
        d = dels[did]
        gone = normalize(d["text"])
        extra, missing = act[j1:j2], exp[i1:i2]
        item = {"deletion": did, "seam_rough_s": round(seam_t.get(did, 0), 2),
                "expected": exp[max(0, p - 6):p] + "｜" + exp[p:p + 6], "extra": extra, "missing": missing}
        if tag == "insert":
            pre, suf = gone.startswith(extra), gone.endswith(extra)
            if pre and suf:  # 比如单字“嗯”：看它在接缝前还是接缝后
                mid = sum(times[j1:j2]) / len(extra)
                pre, suf = mid < seam_t.get(did, 0), mid >= seam_t.get(did, 0)
            if pre:
                item.update(kind="少删", side="out", note=f"接缝旁多出“{extra}”，是被删内容的开头")
            elif suf:
                item.update(kind="少删", side="in", note=f"接缝旁多出“{extra}”，是被删内容的结尾")
            else:
                item.update(kind="替换", side=None, note=f"接缝旁多出“{extra}”，对不上被删的内容")
        elif tag == "delete":
            if i2 <= p:
                item.update(kind="多删", side="out", note=f"接缝前的“{missing}”整字缺失")
            elif i1 >= p:
                item.update(kind="多删", side="in", note=f"接缝后的“{missing}”整字缺失")
            else:
                item.update(kind="替换", side=None, note=f"接缝两边的“{missing}”缺失")
        else:
            item.update(kind="替换", side=None, note=f"“{missing}”认成了“{extra}”")
        found.append(item)
    # 缺的字旁边有多出的字或替换（比如“能大约是”认成“呢能约是”），缺的字可能是被认成了别的字，
    # 多出的字对得上被删内容、旁边却有纯缺字的，也可能是缺的那个字认错了。这两种分不清，按替换列出来给人听，不挪刀
    for did in {f["deletion"] for f in found}:
        here = [f for f in found if f["deletion"] == did]
        has_extra = any(f["extra"] for f in here)
        has_pure_missing = any(f["missing"] and not f["extra"] for f in here)
        for f in here:
            if (f["kind"] == "多删" and has_extra) or (f["kind"] == "少删" and has_pure_missing):
                f["note"] += "；同一接缝旁还有别的字对不上，按替换处理"
                f["kind"], f["side"] = "替换", None
    return found, act


def quiet_index(loud, t: float, direction: int) -> int | None:
    """t 所在的静音段往 direction（-1 前、+1 后）数一段的序号。t 不在静音里时，取它前面或后面的第一段。"""
    for k, (a, b) in enumerate(loud.quiet):
        if a <= t <= b:
            k2 = k + direction
            return k2 if 0 <= k2 < len(loud.quiet) else None
    if direction < 0:
        ks = [k for k, (a, b) in enumerate(loud.quiet) if b < t]
        return ks[-1] if ks else None
    ks = [k for k, (a, b) in enumerate(loud.quiet) if a > t]
    return ks[0] if ks else None


def move_cuts(edl: dict, findings: list, loud, tok: list, rnd: int, dur: int) -> list:
    moved, done = [], set()
    dels = sorted(edl["deletions"], key=lambda d: d["from_token"])
    for f in findings:
        if f["kind"] not in ("少删", "多删"):
            continue
        d = next(x for x in dels if x["id"] == f["deletion"])
        side = f["side"]
        if (d["id"], side) in done:
            continue
        done.add((d["id"], side))
        snap = d[f"{side}_snap"]
        # 少删：前一刀往前、后一刀往后；多删：前一刀往后、后一刀往前
        direction = {("少删", "out"): -1, ("少删", "in"): 1, ("多删", "out"): 1, ("多删", "in"): -1}[(f["kind"], side)]
        k = quiet_index(loud, snap / FPS, direction)
        new = frame_in_quiet(loud.quiet[k]) if k is not None else None
        why = None
        if new is None:
            why = "那个方向没有别的静音段"
        else:
            o = new if side == "out" else d["out_snap"]
            i = new if side == "in" else d["in_snap"]
            idx = dels.index(d)
            if side == "in" and new / FPS > tok[d["to_token"] + 1]["end"] + AFTER:
                why = "后一刀会越过保留的第一个字结束后 0.6 秒"
            elif side == "out" and new / FPS < tok[d["from_token"] - 1]["start"] - BEFORE:
                why = "前一刀会越过保留的最后一个字开始前 0.15 秒"
            elif o >= i or loud.sound_frames(o / FPS, i / FPS) < MIN_SOUND:
                why = "挪完两刀之间没有声音了"
            elif side == "out" and idx > 0 and dels[idx - 1]["in_snap"] is not None and new <= dels[idx - 1]["in_snap"]:
                why = "会和上一处删除重叠"
            elif side == "in" and idx + 1 < len(dels) and dels[idx + 1]["out_snap"] is not None and new >= dels[idx + 1]["out_snap"]:
                why = "会和下一处删除重叠"
        if why:
            f["unfixed"] = why
            continue
        d["moves"].append({"round": rnd, "cut": "前一刀" if side == "out" else "后一刀",
                           "from_frame": snap, "to_frame": new, "reason": f["note"]})
        d[f"{side}_snap"] = new
        d[f"{side}_how"] = "复核挪刀"
        d[f"{side}_quiet"] = [round(x, 3) for x in loud.quiet[k]]
        moved.append(f"第 {d['id']} 处{'前' if side == 'out' else '后'}一刀从第 {snap} 帧挪到第 {new} 帧（{new / FPS:.3f} 秒）")
    return moved


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    t0 = time.time()
    folder = Path(sys.argv[1]).expanduser().resolve()
    work = folder / "work"
    if msg := edl_touched(folder):
        sys.exit(msg)
    edl = read_json(work / "edl.json")
    errs = validate(edl, folder)
    if errs:
        sys.exit("edl.json 校验不通过，先重跑 plan_cuts.py：\n  - " + "\n  - ".join(errs))
    dec = read_json(work / "decisions.json") if (work / "decisions.json").exists() else {}
    tok = read_json(folder / edl["transcript"])["tokens"]
    a16, sr = sf.read(work / "audio.wav", dtype="float32")
    a48, sr48 = sf.read(work / "audio48.wav", dtype="float32")
    assert sr == SR and sr48 == SR_OUT
    loud = quiet_over(analyze(a16), blank_spans(dec.get("blank_cut"), tok))  # 用户说太长的空白，和 plan_cuts.py 一样当一整段静音
    dur = edl["duration_frames"]
    run = new_run_dir(folder)

    check_models()
    print("加载转写模型……", flush=True)
    model = load_model()
    cur = copy.deepcopy(edl)
    rounds = []
    for rnd in range(1, MAX_ROUNDS + 1):
        apply_pauses(cur, loud, dec.get("pauses_keep"))
        errs = validate(cur, folder)
        if errs:
            sys.exit(f"第 {rnd} 轮挪刀后校验不通过：\n  - " + "\n  - ".join(errs))
        wav = run / f"recheck_第{rnd}轮.wav"
        sf.write(wav, splice(a16, SR, keep_segments(cur)), SR, subtype="PCM_16")
        tr0 = time.time()
        raw = transcribe(wav, model)
        findings, text = compare(cur, tok, raw)
        over = sum(f["kind"] == "多删" for f in findings)
        under = sum(f["kind"] == "少删" for f in findings)
        info = {"round": rnd, "transcribe_seconds": round(time.time() - tr0, 1), "text": text,
                "findings": findings, "moves": [], "edl": copy.deepcopy(cur), "over": over, "under": under}
        rounds.append(info)
        print(f"第 {rnd} 轮：多删 {over} 处，少删 {under} 处，替换 {len(findings) - over - under} 处", flush=True)
        if not over + under:
            break
        if rnd == MAX_ROUNDS:  # 已经 3 轮，不再挪刀：挪了也没法复核
            for f in findings:
                if f["kind"] in ("少删", "多删"):
                    f["unfixed"] = f"已到 {MAX_ROUNDS} 轮，没再挪"
            break
        nxt = copy.deepcopy(cur)
        info["moves"] = move_cuts(nxt, findings, loud, tok, rnd, dur)
        for m in info["moves"]:
            print("  挪刀：" + m)
        if not info["moves"]:
            break
        cur = nxt

    # 回到没有多删、少删最少的那一轮：先比多删（切到字），再比少删；一样好时取较晚的一轮
    best = min(rounds, key=lambda r: (r["over"], r["under"], -r["round"]))
    final = best["edl"]
    write_json(run / "edl_final.json", final)
    errs = validate(final, folder)
    if errs:
        sys.exit("edl_final.json 校验不通过：\n  - " + "\n  - ".join(errs))
    rough = splice(a48, SR_OUT, keep_segments(final))
    sf.write(run / "粗剪.wav", rough, SR_OUT, subtype="PCM_16")
    write_json(run / "recheck.json", {
        "final_round": best["round"],
        "rounds": [{k: v for k, v in r.items() if k != "edl"} for r in rounds],
    })
    add_timing(run, f"cut_audio（剪声音和复核 {len(rounds)} 轮）", time.time() - t0)
    left = [f for f in best["findings"]]
    print(f"用第 {best['round']} 轮的结果：粗剪 {len(rough) / SR_OUT:.2f} 秒，"
          f"剩下要人听的 {len(left)} 处 → {run}")


if __name__ == "__main__":
    main()
