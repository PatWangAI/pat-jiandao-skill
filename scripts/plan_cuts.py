"""第 5 步 方案：按 Claude 写的删除清单算剪切点，写出剪辑表和给用户看的方案。

用法：uv run python scripts/plan_cuts.py <视频文件夹> [--force]

读 work/ 里的 deletions.json（Claude 写）、decisions.json（Claude 按用户的话写，可以没有；
blank_cut 见 common.quiet_over）、
transcript.json、audio.wav，写出 work/edl.json 和 work/plan.md，接着马上校验。

没有讲稿时（inputs.json 的 no_script，讲稿可选，第二期）：每条还可以写 risk（低、高），说断和重说必须写
keep（留下的那一遍的字序号 [起, 止]）。plan.md 多两样：每条后面写“剩下读作”（删完接缝两边读起来是什么）和风险，
另加一张重复句子表（说了几遍、保了哪遍、删了哪几遍）；另写 work/删后全文.md，按删完的样子把全文排出来，
接缝处标 ｜，给 Claude 自己通读。有讲稿时 plan.md 和以前一样。两张表和自读借 chengfeng-videocut-skills 的做法（Apache-2.0）。

做的事（SPEC 第 4 节）：
1. 相邻或重叠的删除先合成一处；
2. 每处找两刀：在交界点附近最近的静音段里，取离静音中点最近的画面帧边界；
3. 找不到静音、或两刀之间没声音（跳过了要删的字），这处不剪，进“连读待定”；
   用户说删的，改在交界处音量最小的帧边界下刀；
4. 刀都定了再压停顿：片头片尾、删除处接缝、长停顿。

edl.json 只由本脚本生成，不手改。发现它和本脚本上次写出的不一样（有人改过），就停下来问，
确认要覆盖时加 --force。
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf

from common import (CONFIG, FPS, MIN_SOUND, PACES, SR, analyze, apply_pauses, blank_spans, fmt, frame_in_quiet, join_text,
                    keep_segments, min_volume_frame, nearest_quiet, quiet_over, read_json, sha256, write_json)
from validate_edl import DEL_TYPES, edl_touched, validate

TYPE_RANK = {"语气词": 0, "说断": 1, "重说": 2, "改稿": 3}


def check_deletions(dels: list, n: int, no_script: bool = False) -> list[str]:
    errs = []
    if not isinstance(dels, list):
        return ["deletions.json 应是一个列表"]
    for k, d in enumerate(dels, 1):
        f, t = d.get("from_token"), d.get("to_token")
        if d.get("type") not in DEL_TYPES:
            errs.append(f"第 {k} 条 type “{d.get('type')}” 不是语气词、说断、重说、改稿之一")
        if not isinstance(f, int) or not isinstance(t, int) or f > t or f < 0 or t >= n:
            errs.append(f"第 {k} 条字序号 {f}—{t} 不对（应是整数，from 不大于 to，在 0—{n - 1} 之间）")
        if not d.get("reason"):
            errs.append(f"第 {k} 条没写 reason")
        if "risk" in d and d["risk"] not in ("低", "高"):
            errs.append(f"第 {k} 条 risk “{d['risk']}” 只能是“低”或“高”")
        keep = d.get("keep")
        if keep is not None:
            if not (isinstance(keep, list) and len(keep) == 2 and all(isinstance(x, int) for x in keep)
                    and 0 <= keep[0] <= keep[1] < n):
                errs.append(f"第 {k} 条 keep {keep!r} 应写成 [起, 止]，留下的那一遍的字序号，在 0—{n - 1} 之间")
            elif isinstance(f, int) and isinstance(t, int) and keep[0] <= t and keep[1] >= f:
                errs.append(f"第 {k} 条 keep {keep} 和要删的 {f}—{t} 重叠了")
        elif no_script and d.get("type") in ("说断", "重说"):
            errs.append(f"第 {k} 条是{d.get('type')}：没有讲稿时要写 keep（留下的那一遍的字序号 [起, 止]），重复句子表靠它")
    return errs


def check_decisions(dec: dict) -> list[str]:
    errs = []
    for part in ("pending", "deletions"):
        for key, v in dec.get(part, {}).items():
            if v not in ("删", "不删"):
                errs.append(f"decisions.json 的 {part}[{key}] 取值“{v}”，只能是“删”或“不删”")
            a, _, b = key.partition("-")
            if not (a.isdigit() and b.isdigit()):
                errs.append(f"decisions.json 的键“{key}”应写成“起-止”字序号")
    for r in dec.get("pauses_keep", []):
        if not (isinstance(r, list) and len(r) == 2 and r[0] < r[1]):
            errs.append(f"pauses_keep 里的 {r} 应写成 [起秒, 止秒]")
    for key in dec.get("blank_cut", []):
        a, _, b = str(key).partition("-")
        if not (a.isdigit() and b.isdigit() and int(a) < int(b)):
            errs.append(f"blank_cut 里的“{key}”应写成“起-止”字序号：空白前最后一个字、空白后第一个字")
    return errs


def silence_cut(loud, left: dict, right: dict) -> dict | None:
    qi = nearest_quiet(loud, (left["end"] + right["start"]) / 2)
    if qi is None:
        return None
    f = frame_in_quiet(loud.quiet[qi])
    if f is None:
        return None
    return {"snap": f, "how": "最近的静音段", "quiet": [round(x, 3) for x in loud.quiet[qi]]}


def quietest_cut(loud, left: dict, right: dict) -> dict:
    return {"snap": min_volume_frame(loud, left, right), "how": "交界处最小音量", "quiet": None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    folder = Path(a.folder).expanduser().resolve()
    work = folder / "work"
    inputs = read_json(folder / "inputs.json")
    pre = read_json(work / "preprocess.json")
    tr = read_json(work / "transcript.json")
    tok = tr["tokens"]
    n = len(tok)
    dur = pre["proxy"]["frames"]

    # edl.json 被手改过就停下来问
    edl_path, state_path = work / "edl.json", work / ".plan_cuts_last.json"
    if edl_path.exists() and not a.force and (msg := edl_touched(folder)):
        sys.exit(msg + "。用户同意覆盖就加 --force 重跑")

    dels = read_json(work / "deletions.json")
    dec_path = work / "decisions.json"
    dec = read_json(dec_path) if dec_path.exists() else {}
    no_script = bool(inputs.get("no_script"))
    pace = inputs.get("pace") or CONFIG.get("pace") or "标准"  # 这条视频的节奏（prepare.py --pace），没写就照设置
    errs = check_deletions(dels, n, no_script) + check_decisions(dec)
    if pace not in PACES:
        errs.append(f"节奏只能是 {'、'.join(PACES)}，现在写的是“{pace}”")
    if errs:
        print("deletions.json 或 decisions.json 有错：")
        for e in errs:
            print("  - " + e)
        sys.exit(1)

    audio, sr = sf.read(work / "audio.wav", dtype="float32")
    assert sr == SR
    loud = analyze(audio)

    # plan.md 里的编号：按开始的字排序后从 1 数
    items = sorted(dels, key=lambda d: (d["from_token"], d["to_token"]))
    for k, d in enumerate(items, 1):
        d["item"] = k
    keys = {f"{d['from_token']}-{d['to_token']}" for d in items}
    stray = [k for k in dec.get("deletions", {}) if k not in keys]
    if stray:  # 对不上的“不删”会被悄悄照删，算误删，所以停下
        sys.exit(f"decisions.json 的 deletions 里这些字序号对不上 deletions.json 的任何一条：{'、'.join(stray)}。"
                 "键要和 plan.md 里那一条的字序号完全一样")
    kept_by_pat = [d for d in items if dec.get("deletions", {}).get(f"{d['from_token']}-{d['to_token']}") == "不删"]
    active = [d for d in items if d not in kept_by_pat]
    # 用户说某处空白太长：两个字中间只能有要删的字，否则会把留下的字当杂音压掉
    gone = {k for d in active for k in range(d["from_token"], d["to_token"] + 1)}
    for key in dec.get("blank_cut", []):
        i, j = (int(x) for x in key.split("-"))
        kept = [k for k in range(i + 1, j) if k not in gone]
        if j >= n or kept:
            sys.exit(f"decisions.json 的 blank_cut“{key}”不对：" + ("字序号超出转写稿" if j >= n else
                     f"中间有要留的字（{'、'.join(map(str, kept))}）") + "。两个字序号是空白前最后一个字、空白后第一个字")
    loud = quiet_over(loud, blank_spans(dec.get("blank_cut"), tok))

    # 1. 相邻或重叠的删除合成一处（中间没有保留的字）
    groups = []
    for d in active:
        if groups and d["from_token"] <= groups[-1]["to_token"] + 1:
            g = groups[-1]
            g["to_token"] = max(g["to_token"], d["to_token"])
            g["members"].append(d)
        else:
            groups.append({"from_token": d["from_token"], "to_token": d["to_token"], "members": [d]})

    deletions, pending, used_keys = [], [], set()
    for g in groups:
        f, t = g["from_token"], g["to_token"]
        key = f"{f}-{t}"
        head, tail = f == 0, t == n - 1
        out = {"snap": None, "how": "片头", "quiet": None} if head else silence_cut(loud, tok[f - 1], tok[f])
        inn = {"snap": None, "how": "片尾", "quiet": None} if tail else silence_cut(loud, tok[t], tok[t + 1])

        def sound_between(o, i):
            t0 = 0.0 if o["snap"] is None else o["snap"] / FPS
            t1 = dur / FPS if i["snap"] is None else i["snap"] / FPS
            return t1 > t0 and loud.sound_frames(t0, t1) >= MIN_SOUND

        ok = out is not None and inn is not None and sound_between(out, inn)
        if not ok:
            if dec.get("pending", {}).get(key) == "删":
                used_keys.add(key)
                mo = out if head else quietest_cut(loud, tok[f - 1], tok[f])
                mi = inn if tail else quietest_cut(loud, tok[t], tok[t + 1])
                tries = [(out or mo, inn or mi), (mo, inn or mi), (out or mo, mi), (mo, mi)]
                pick = next(((o, i) for o, i in tries if sound_between(o, i)), None)
                if pick:
                    out, inn = pick
                    ok = True
            if not ok:
                # 说明为什么没切：哪边和要删的字之间没有停顿
                gap_l = tok[f]["start"] - tok[f - 1]["end"] if not head else 9
                gap_r = tok[t + 1]["start"] - tok[t]["end"] if not tail else 9
                why = (f"和前面的“{tok[f - 1]['text']}”连着说，没有停顿" if gap_l <= gap_r
                       else f"和后面的“{tok[t + 1]['text']}”连着说，没有停顿")
                if dec.get("pending", {}).get(key) == "删":
                    why += "；说了要删，但在交界处也找不到能下刀的地方"
                pending.append({"id": len(pending) + 1, "type": "连读待定",
                                "orig_type": g["members"][0]["type"],
                                "items": [m["item"] for m in g["members"]], "from_token": f, "to_token": t,
                                "text": join_text([x["text"] for x in tok[f:t + 1]]), "reason": why})
                continue
        typ = max((m["type"] for m in g["members"]), key=TYPE_RANK.get)
        deletions.append({
            "id": len(deletions) + 1, "type": typ, "items": [m["item"] for m in g["members"]],
            "from_token": f, "to_token": t, "text": join_text([x["text"] for x in tok[f:t + 1]]),
            "out_frame": None, "in_frame": None, "out_s": None, "in_s": None,
            "out_how": out["how"], "in_how": inn["how"], "out_quiet": out["quiet"], "in_quiet": inn["quiet"],
            "out_snap": out["snap"], "in_snap": inn["snap"],
            "reason": "；".join(m["reason"] for m in g["members"]), "moves": [],
        })
    for key in dec.get("pending", {}):
        if key not in used_keys and dec["pending"][key] == "删" and not any(f"{p['from_token']}-{p['to_token']}" == key for p in pending):
            print(f"注意：decisions.json 里的连读待定“{key}”对不上任何一处，没用上")

    edl = {
        "version": 1, "fps": FPS, "duration_frames": dur,
        "source": inputs["source"], "source_sha256": inputs["sha256"],
        "transcript": "work/transcript.json", "transcript_sha256": sha256(work / "transcript.json"),
        "edges": {}, "deletions": deletions, "pauses": [], "pending": pending,
        **({"no_script": True} if no_script else {}),  # 没讲稿：字幕要用（make_subtitles.subtitle_tokens），有讲稿时不写这一项
        # 节奏“紧”才写这两项（common.PACES）；标准节奏不写，剪辑表和以前一样
        **({"pace": pace, "first_word_s": round(tok[0]["start"], 3)} if pace != "标准" else {}),
    }
    apply_pauses(edl, loud, dec.get("pauses_keep"))
    errs = validate(edl, folder)
    write_json(edl_path, edl)
    write_json(state_path, {"edl_sha256": sha256(edl_path)})
    write_plan(work / "plan.md", inputs, edl, items, kept_by_pat, tok, loud)
    if no_script:
        write_after(work / "删后全文.md", inputs, edl, tr, folder)
    if errs:
        print("剪辑表校验不通过：")
        for e in errs:
            print("  - " + e)
        sys.exit(1)
    rough = sum(b - a for a, b in keep_segments(edl)) / FPS
    print(f"方案写好了，校验通过：删 {len(deletions)} 处，压停顿 {len(edl['pauses'])} 处，"
          f"连读待定 {len(pending)} 处；原片 {fmt(dur / FPS)}，粗剪约 {fmt(rough)} → {work / 'plan.md'}")


def context(tok: list, f: int, t: int, k: int = 6) -> str:
    left = join_text([x["text"] for x in tok[max(0, f - k):f]])
    mid = join_text([x["text"] for x in tok[f:t + 1]])
    right = join_text([x["text"] for x in tok[t + 1:t + 1 + k]])
    return f"……{left}【{mid}】{right}……"


def reads_as(tok: list, gone: set, f: int, t: int, k: int = 6) -> str:
    """删完以后接缝两边读起来是什么：往左、往右各取 6 个留下的字（别处删掉的字也跳过），接缝标 ｜。"""
    left = [i for i in range(f - 1, -1, -1) if i not in gone][:k][::-1]
    right = [i for i in range(t + 1, len(tok)) if i not in gone][:k]
    return f"……{join_text([tok[i]['text'] for i in left])}｜{join_text([tok[i]['text'] for i in right])}……"


def retake_table(tok: list, dels: list, pending: set = frozenset()) -> list[str]:
    """重复句子表：按 keep（留下的那一遍）把删掉的几遍归到一起。逐条表看不出“录了几遍、保哪遍”，这张表给用户核对。
    进了连读待定的那几遍也列上，标“连读待定，默认不删”。"""
    groups = {}
    for d in dels:
        groups.setdefault(tuple(d["keep"]), []).append(d)
    L = ["", f"## 重复句子表（{len(groups)} 句说了不止一遍）", "",
         "| 句子（留下那遍的开头） | 说了几遍 | 保了哪遍 | 删了哪几遍 | 风险 |", "| --- | --- | --- | --- | --- |"]
    for (k0, k1), ms in sorted(groups.items()):
        takes = sorted([(m["from_token"], m) for m in ms] + [(k0, None)], key=lambda x: x[0])
        pos = {f: n for n, (f, _) in enumerate(takes, 1)}
        text = join_text([x["text"] for x in tok[k0:k1 + 1]])
        dropped = "、".join(f"第 {pos[m['from_token']]} 遍 {fmt(tok[m['from_token']]['start'])}（条 {m['item']}"
                            f"{'，连读待定，默认不删' if m['item'] in pending else ''}）" for _, m in takes if m)
        risk = "高" if any(m.get("risk") == "高" for m in ms) else "低"
        L.append(f"| {text[:14]}{'…' if len(text) > 14 else ''} | {len(takes)} | 第 {pos[k0]} 遍 {fmt(tok[k0]['start'])} | {dropped} | {risk} |")
    if not groups:
        L.append("| — | | | 没有 | |")
    return L


def write_after(path: Path, inputs: dict, edl: dict, tr: dict, folder: Path) -> None:
    """删后全文：按删完的样子排出全文，一句一行（句号、问号、叹号断），接缝处标 ｜，给 Claude 自己通读。
    连读待定的字（默认不删）用〔〕框上；术语表改得到的字按改过的显示（只是显示，转写稿不改）。"""
    from terms import find_fixes, glossary_files, load_glossary
    tok = tr["tokens"]
    gone = {i for d in edl["deletions"] for i in range(d["from_token"], d["to_token"] + 1)}
    pend = {i for d in edl["pending"] for i in range(d["from_token"], d["to_token"] + 1)}
    kept = [i for i in range(len(tok)) if i not in gone]
    fixes = {f["from_token"]: f for f in find_fixes(tok, load_glossary(glossary_files(folder)), kept)}
    shown, skip_to = {}, -1
    for i in kept:
        if i <= skip_to:
            shown[i] = ""
        elif i in fixes:
            shown[i], skip_to = fixes[i]["right"], fixes[i]["to_token"]
        else:
            shown[i] = tok[i]["text"]
    tails = {x["last_token"]: x["text"][-1] for x in tr["sentences"] if x["text"][-1:] in "，。？！、"}
    lines, cur, cut = [], [], False
    for i, t in enumerate(tok):
        if i in gone:
            cut = True
            continue
        if cut and (cur or lines):  # 前面刚删过：标接缝（落在句首也标）
            cur.append("｜")
        cut = False
        word = shown[i] + tails.get(i, "")
        if i in pend and i - 1 not in pend:
            word = "〔" + word
        if i in pend and i + 1 not in pend:
            word += "〕"
        if not shown[i] and cur:  # 术语表合掉的字：它后面的标点接到上一个词上
            cur[-1] += word
        elif word:
            cur.append(word)
        if tails.get(i, "") in ("。", "？", "！"):
            lines.append(join_text(cur).replace(" ｜ ", "｜"))
            cur = []
    if cur:
        lines.append(join_text(cur).replace(" ｜ ", "｜"))
    head = [f"# 删后全文：{inputs['name']}", "", f"照 deletions.json 删完的样子，一句一行，｜是删过的接缝（{len(edl['deletions'])} 处）。"
            "〔〕里是连读待定、默认不删的字，选删的话这几个字就没了，也要读一遍看通不通。"
            "从头读一遍：读不通、意思变了的那几条撤掉；拿不准的不删。", ""]
    path.write_text("\n".join(head + lines) + "\n", encoding="utf-8")


def write_plan(path: Path, inputs: dict, edl: dict, items: list, kept: list, tok: list, loud) -> None:
    dur = edl["duration_frames"] / FPS
    rough = sum(b - a for a, b in keep_segments(edl)) / FPS
    by_item = {}
    for d in edl["deletions"]:
        for it in d["items"]:
            by_item[it] = d
    pend_items = {it for p in edl["pending"] for it in p["items"]}
    filler = [d for d in items if d["type"] == "语气词" and d["item"] in by_item]
    redo = [d for d in items if d["type"] in ("说断", "重说") and d["item"] in by_item]
    edit = [d for d in items if d["type"] == "改稿" and d["item"] in by_item]

    no_script = bool(inputs.get("no_script"))
    gone = {i for d in edl["deletions"] for i in range(d["from_token"], d["to_token"] + 1)}

    def row(d):
        f, t = d["from_token"], d["to_token"]
        if no_script:  # 没讲稿：多写删完读作什么、风险
            return (f"| {d['item']} | {fmt(tok[f]['start'])} | {context(tok, f, t)} | {reads_as(tok, gone, f, t)} | "
                    f"{d.get('risk', '低')} | {d['reason']} | {f}-{t} |")
        return f"| {d['item']} | {fmt(tok[f]['start'])} | {context(tok, f, t)} | {d['reason']} | {f}-{t} |"

    head = (["| 条 | 时间 | 原话 | 剩下读作 | 风险 | 理由 | 字序号 |", "| --- | --- | --- | --- | --- | --- | --- |"] if no_script
            else ["| 条 | 时间 | 原话 | 理由 | 字序号 |", "| --- | --- | --- | --- | --- |"])

    L = [f"# 剪辑方案：{inputs['name']}", "",
         f"原片 {fmt(dur)}，按这个方案剪完约 {fmt(rough)}。删 {len(filler) + len(redo) + len(edit)} 处"
         f"（语气词 {len(filler)}、重说 {len(redo)}、改稿 {len(edit)}），压长停顿 {len(edl['pauses'])} 处，"
         f"连读待定 {len(edl['pending'])} 处默认不删。【】里是要删的字。", "",
         "可以就回“可以”；要改就说第几条删不删、哪段停顿别压。", ""]
    L += [f"## 语气词（{len(filler)} 处）", "", *head]
    L += [row(d) for d in filler] or ["| — | | 没有 | | |"]
    L += ["", f"## 重说（{len(redo)} 处，说断的也算在这里，只留最后一遍完整的）", "", *head]
    L += [row(d) for d in redo] or ["| — | | 没有 | | |"]
    if no_script:
        L += retake_table(tok, [d for d in items if d.get("keep") and (d["item"] in by_item or d["item"] in pend_items)],
                          pend_items)
    if edit:
        L += ["", f"## 改稿（{len(edit)} 处，你要求删掉的内容，不是口误）", "", *head]
        L += [row(d) for d in edit]
    keep_s = format(round(2 * PACES[edl.get("pace", "标准")]["pause_keep"], 2), "g")  # 标准是 0.3，节奏紧是 0.15
    L += ["", f"## 长停顿（{len(edl['pauses'])} 处，压到约 {keep_s} 秒）", "",
          "| 停顿 | 时间 | 位置 | 原长 |", "| --- | --- | --- | --- |"]
    for p in edl["pauses"]:
        a, b = p["quiet"]
        left = [x for x in tok if x["start"] < a][-6:]
        right = [x for x in tok if x["start"] >= a][:6]
        where = f"……{join_text([x['text'] for x in left])} ⏸ {join_text([x['text'] for x in right])}……"
        L.append(f"| {p['id']} | {fmt(a)} | {where} | {b - a:.2f} 秒 |")
    if not edl["pauses"]:
        L.append("| — | | 没有 | |")
    L += ["", f"## 连读待定（{len(edl['pending'])} 处，默认不删，请你逐处决定）", ""]
    L += (["| 条 | 时间 | 原话 | 选删的话读作 | 风险 | 为什么没切 | 字序号 |", "| --- | --- | --- | --- | --- | --- | --- |"] if no_script
          else ["| 条 | 时间 | 原话 | 为什么没切 | 字序号 |", "| --- | --- | --- | --- | --- |"])
    by_no = {d["item"]: d for d in items}
    for p in edl["pending"]:
        f, t = p["from_token"], p["to_token"]
        if no_script:  # 没讲稿：这几条的风险、删了读作什么也写上，选删之前看得到
            risk = "高" if any(by_no[i].get("risk") == "高" for i in p["items"]) else "低"
            L.append(f"| {'、'.join(map(str, p['items']))} | {fmt(tok[f]['start'])} | {context(tok, f, t)} | "
                     f"{reads_as(tok, gone | set(range(f, t + 1)), f, t)} | {risk} | {p['reason']} | {f}-{t} |")
        else:
            L.append(f"| {'、'.join(map(str, p['items']))} | {fmt(tok[f]['start'])} | {context(tok, f, t)} | "
                     f"{p['reason']} | {f}-{t} |")
    if not edl["pending"]:
        L.append("| — | | 没有 | | |" + (" | |" if no_script else ""))
    if kept:
        L += ["", "## 你决定不删的", ""]
        L += [f"- 第 {d['item']} 条（字序号 {d['from_token']}-{d['to_token']}）：{context(tok, d['from_token'], d['to_token'])}" for d in kept]
    merged = [d for d in edl["deletions"] if len(d["items"]) > 1]
    if merged:
        L += ["", "相邻的几条合成一处下刀：" + "；".join(
            "第 " + "、".join(map(str, d["items"])) + " 条" for d in merged) + "。"]
    L += ["", f"合计：删 {len(edl['deletions'])} 处（按下刀算），压停顿 {len(edl['pauses'])} 处；"
          f"原片 {fmt(dur)}，粗剪约 {fmt(rough)}。时间是原片里的位置，转写给的字时间常常偏早零点几秒。", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
