"""第 2 步 找素材（SPEC 第 14 节）：按 Claude 写的 素材/plan.json，自动截网页、搜图库，列出要用户手动补的。

用法：uv run python scripts/fetch_assets.py <视频文件夹> [--force]

plan.json 每条一项素材，id 是 s 加句号（不到两位前面补 0），同一句多项再加一个小写字母：s01、s04b、s100、s127b；
sentence 是第几句，desc 写画面。kind 是下面之一：
  截图      url、highlight（要高亮的原文，可以没有）、credit（媒体名和日期）
  图库图片  query（英文关键词，可以是几组，按顺序试）
  图库视频  query
  AI生图    prompt
  手动视频  url、time_range，第三方的要 credit
  录屏      note
  现成图片  note（从哪来、怎么拿）；别人的图（下载的照片、自己截的网页、访谈画面截图）还写 url（出处那一页）和 credit，
            来源清单照它记；自己做的（封面、自己画的图）可以不写（2026-10-31 加）
主方案要手动、又有能自动拿的备选时，写 fallback（一项素材，格式同上）。
手动补的文件不是从 url 那一页拿的（比如官方 PDF、X 帖的配图、视频的帖子页），写 manual_source：
  {"media": 媒体名, "date": 日期, "url": 实际来源的网址}，来源清单照它写；改它不会重取自动的那份。
来源清单的链接用 plan.json 里写的 url（用上的是主方案就写主方案的，是备选就写备选的），不用截图时地址栏的地址
（往下滚会自动换成下一篇、被拦到验证页、跳到别的语言版，第 07 期都遇到过）；地址栏变了在素材清单里注明。
竖屏有人脸时（inputs.json 记 vertical、没有 no_face）：放人脸全屏的句子不写，不算缺；items 可以是空的（整片人脸）。
16:9 有人脸时，整屏放人脸（不配素材、不叠小窗）的句子要在顶上明写 "face": [1, 2, 41]（句号列表，和 items 平级，
  2026-10-29 加）：写了的不算缺，items 里不能再给它们配素材；不写就和以前一样，每句都要有素材。竖屏也可以写，无人脸模式不能写。

自动的三种：
  截图：本机 Chrome 无头打开网页，窗口 1920×1080、两倍清晰度，出 <id>_page.jpg（整页长图，最长 8000 像素高）
        和 <id>_focus.jpg（要高亮的句子滚到画面中间的一屏），记下标题和高亮句在两张图里的位置。
        打不开、返回 400 以上、遇到人机验证页就标“需手动”，不绕过。
        截完拿单屏图（量完马上截的）比一比长图里那句（高亮句，没有就标题）在不在量的位置上（page_check）；
        对不上就回到顶上重量、重截一次长图，还对不上，截图聚焦改用单屏图，素材清单里写明。
        以前截的没比过的，重跑时用留下的两张图补比（不重截）。
  图库：Pixabay 搜索，取第一个够大的结果，记页面链接和作者。要环境变量 PIXABAY_API_KEY，没有就标“需手动”。
手动的四种标“需手动”，在素材清单里写清楚怎么拿、存成 素材/<id>_manual.<扩展名>；放进去以后重跑，记为“手动补齐”。
重跑时，plan.json 里没改过、文件还在的项不重取；加 --force 全部重取，--only s08,s14 只重取这几项。

输出（都在 素材/）：assets.json、素材清单.md、预览.jpg。
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageFont

from common import CONFIG, FFMPEG, ROOT, read_json, write_json
from prepare import sentences

AUTO = {"截图", "图库图片", "图库视频"}
MANUAL = {"AI生图", "手动视频", "录屏", "现成图片"}
VIEW_W, VIEW_H, SCALE = 1920, 1080, 2
MAX_PAGE_H = 8000
READABLE_PX = 20          # 标题字号不小于 20 CSS 像素：整页宽放进 1920 宽的画面时就是 20 像素高的字
CHALLENGE = re.compile(r"请稍候|Just a moment|Attention Required|安全验证|人机验证|verify you are human|Request Access", re.I)
PIXABAY_CACHE = ROOT / "cache" / "pixabay"
CACHE_HOURS = 24          # Pixabay 要求搜索结果缓存 24 小时
MAX_VIDEO_BYTES = 12_000_000

# 在网页里找一句话的位置：去掉空白、统一各种横线后逐字比对，返回整页坐标（CSS 像素）
FIND_JS = r"""
(phrase) => {
  const norm = s => s.replace(/[‐-―−]/g, '-').toLowerCase();
  const target = norm(phrase.replace(/\s+/g, ''));
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const pos = []; let text = '';
  while (walker.nextNode()) {
    const n = walker.currentNode, el = n.parentElement;
    if (!el) continue;
    const st = getComputedStyle(el);
    if (st.display === 'none' || st.visibility === 'hidden') continue;
    for (let i = 0; i < n.data.length; i++) {
      const c = n.data[i];
      if (/\s/.test(c)) continue;
      pos.push([n, i]); text += norm(c);
    }
  }
  const k = text.indexOf(target);
  if (k < 0 || !target) return null;
  const [n0, o0] = pos[k], [n1, o1] = pos[k + target.length - 1];
  const r = document.createRange(); r.setStart(n0, o0); r.setEnd(n1, o1 + 1);
  n0.parentElement.setAttribute('data-jianji-hl', '1');
  const rs = Array.from(r.getClientRects()).filter(x => x.width > 0 && x.height > 0);
  if (!rs.length) return null;
  const x0 = Math.min(...rs.map(x => x.left)), y0 = Math.min(...rs.map(x => x.top));
  const x1 = Math.max(...rs.map(x => x.right)), y1 = Math.max(...rs.map(x => x.bottom));
  // 句子跨行时每行一个框（同一行的小框合并），高亮时贴着字走
  const lines = [];
  for (const x of rs.sort((a, b) => a.top - b.top || a.left - b.left)) {
    const l = lines.find(l => Math.abs(l.top - x.top) < 4);
    if (l) { l.left = Math.min(l.left, x.left); l.right = Math.max(l.right, x.right); l.bottom = Math.max(l.bottom, x.bottom); }
    else lines.push({left: x.left, top: x.top, right: x.right, bottom: x.bottom});
  }
  // 高亮句所在的那一段（往上找第一个够宽的块级元素）：推近时按整段居中，整行正文都在画面里
  let el = n0.parentElement, block = null;
  while (el && el !== document.body) {
    const st = getComputedStyle(el), r = el.getBoundingClientRect();
    if ((st.display === 'block' || st.display === 'list-item' || st.display === 'flex') && r.width >= 200) {
      block = {x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height}; break;
    }
    el = el.parentElement;
  }
  return {x: x0 + scrollX, y: y0 + scrollY, w: x1 - x0, h: y1 - y0, block,
          lines: lines.map(l => ({x: l.left + scrollX, y: l.top + scrollY, w: l.right - l.left, h: l.bottom - l.top})),
          font_px: parseFloat(getComputedStyle(n0.parentElement).fontSize)};
}
"""
# 挡住画面的固定浮层：盖住窗口四分之一以上的。弹窗广告、下载 App 提示这类就隐藏，记下是什么；
# 文字里有登录、订阅、会员、付费、验证这类词，或者里面套了 iframe（验证码常这样），算访问限制，
# 不隐藏、不改滚动，交回 Python 标“需手动”。正文本身就在固定容器里的（含标题或大半正文），不算浮层
OVERLAY_JS = r"""
() => {
  const ACCESS = /登录|登陆|注册|订阅|会员|付费|验证|滑块|解锁|继续阅读|captcha|sign ?in|log ?in|subscri|paywall|verify|robot/i;
  const hidden = [], blocked = [], area = innerWidth * innerHeight;
  const h1 = document.querySelector('h1'), total = (document.body.innerText || '').length;
  for (const el of document.querySelectorAll('body *')) {
    const st = getComputedStyle(el);
    if (st.position !== 'fixed' && st.position !== 'sticky') continue;
    if (st.display === 'none' || st.visibility === 'hidden') continue;
    const r = el.getBoundingClientRect();
    if (r.width * r.height < area * 0.25) continue;
    const text = (el.innerText || '').trim().replace(/\s+/g, ' ');
    if ((h1 && el.contains(h1)) || text.length > total * 0.5) continue;
    const label = (text || el.className || el.tagName).toString().slice(0, 30);
    if (ACCESS.test(text) || el.querySelector('iframe')) { blocked.push(label); continue; }
    hidden.push(label);
    el.style.setProperty('display', 'none', 'important');
  }
  if (hidden.length && !blocked.length) {
    for (const e of [document.documentElement, document.body]) {
      if (getComputedStyle(e).overflow === 'hidden') e.style.setProperty('overflow', 'visible', 'important');
    }
  }
  return {hidden, blocked};
}
"""
# 只留正文：标题、要高亮的句子、正文第一段共同所在的那一块是正文；正文以外的（导航、别的新闻、侧栏、二维码、
# 浮动工具条）每一层的兄弟都设成看不见（版面不动，位置不变）；正文里的二维码、分享、评分、相关推荐、下载 App、
# 广告这类小块也设成看不见。返回正文的范围和正文最后一段的底边，整页图只截到那里
ISOLATE_JS = r"""
() => {
  const vis = e => e && e.getBoundingClientRect().height > 0;
  const h1 = Array.from(document.querySelectorAll('h1')).find(vis);
  const hl = document.querySelector('[data-jianji-hl]');
  const paras = Array.from(document.querySelectorAll('p')).filter(p => vis(p) && p.innerText.trim().length >= 40);
  const first = paras.find(p => !h1 || (h1.compareDocumentPosition(p) & Node.DOCUMENT_POSITION_FOLLOWING));
  const anchors = [h1, hl, first].filter(Boolean);
  if (!anchors.length) return null;
  const chain = el => { const a = []; for (let e = el; e; e = e.parentElement) a.push(e); return a; };
  let root = anchors[0];
  for (const a of anchors.slice(1)) { const c = new Set(chain(a)); root = chain(root).find(e => c.has(e)); }
  if (!root || root === document.body || root === document.documentElement) return null;
  const label = e => ((e.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 16) || e.tagName.toLowerCase());
  const hide = (e, list) => { e.style.setProperty('visibility', 'hidden', 'important'); list.push(label(e)); };
  const outside = [], inside = [];
  for (let e = root; e && e.parentElement && e !== document.body; e = e.parentElement) {
    for (const sib of e.parentElement.children) {
      if (sib !== e && !['SCRIPT', 'STYLE', 'LINK', 'META', 'NOSCRIPT'].includes(sib.tagName)) hide(sib, outside);
    }
  }
  const TXT = /二维码|扫码|扫一扫|下载.{0,6}(APP|App|app|客户端)|分享到|打赏|评分|相关文章|相关阅读|推荐阅读|延伸阅读|大家在看|热门|好书精选|精彩视频|猜你喜欢|广告|关注我们|返回顶部|发表评论/;
  const ATTR = /(^|[\s_-])(qr|qrcode|erweima|share|social|recommend|related|comment|rank|hot|ad|ads|adv|advert|banner|download|toolbar|float|sidebar|aside|crumb|breadcrumb|nav|tags?)([\s_-]|$)/i;
  const keep = el => el.contains(h1) || (hl && el.contains(hl)) || (h1 && h1.contains(el)) || (hl && hl.contains(el));
  for (const el of root.querySelectorAll('*')) {
    if (keep(el) || /^(P|H[1-6]|BR|SPAN|STRONG|EM|B|A)$/.test(el.tagName)) continue;
    const t = (el.innerText || '').trim();
    const attr = (typeof el.className === 'string' ? el.className : '') + ' ' + (el.id || '');
    const qrImg = el.tagName === 'IMG' && /qr|erweima|二维码/i.test((el.currentSrc || el.src || '') + ' ' + (el.alt || ''));
    if (el.tagName === 'IFRAME' || el.tagName === 'ASIDE' || el.tagName === 'NAV' || qrImg || ATTR.test(attr)
        || (t.length > 0 && t.length < 120 && TXT.test(t))) hide(el, inside);
  }
  const r = root.getBoundingClientRect();
  const ps = Array.from(root.querySelectorAll('p')).filter(p => vis(p) && p.innerText.trim().length >= 20
    && getComputedStyle(p).visibility !== 'hidden');
  const last = ps.length ? ps[ps.length - 1].getBoundingClientRect() : r;
  const top = Math.min(r.top, h1 ? h1.getBoundingClientRect().top : r.top);
  return {article: {x: r.left + scrollX, y: top + scrollY, w: r.width, h: last.bottom - top},
          content_bottom: last.bottom + scrollY, hidden_outside: outside.length, hidden_inside: inside.slice(0, 30)};
}
"""
# 正文里的图片等加载完再截（最多等 5 秒）
WAIT_IMG_JS = r"""
async () => {
  const imgs = Array.from(document.images).filter(i => !i.complete);
  await Promise.race([Promise.all(imgs.map(i => new Promise(r => { i.onload = i.onerror = r; }))),
                      new Promise(r => setTimeout(r, 5000))]);
  return imgs.length;
}
"""
HEADLINE_JS = r"""
() => {
  const h = Array.from(document.querySelectorAll('h1')).find(e => e.getBoundingClientRect().height > 0);
  if (!h) return null;
  const r = h.getBoundingClientRect();
  return {x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height,
          font_px: parseFloat(getComputedStyle(h).fontSize), text: h.innerText.trim().slice(0, 80)};
}
"""


def check_plan(plan: dict, sent_ids: list[int], face_ok: bool = False, has_face: bool = True) -> list[str]:
    """校验 Claude 写的 plan.json：每句至少一项、kind 合法、各类必填项齐全。
    face_ok（竖屏、有人脸）：没写素材的句子放人脸全屏，不算缺，items 也可以是空的（整片人脸）。
    顶上的 face（句号列表，16:9 要整屏放人脸时明写）：这几句不算缺、不能再配素材；has_face 是 False（无人脸模式）时不能写。"""
    errs, listed = [], set()
    face = plan.get("face")
    if face is not None:
        if not has_face:
            errs.append("无人脸模式（只用声音）没有人脸可放，plan.json 不能写 face")
        elif not isinstance(face, list) or not all(isinstance(k, int) and not isinstance(k, bool) for k in face):
            errs.append('plan.json 的 face 要写成句号的列表，比如 "face": [1, 2, 41]')
        else:
            bad = [k for k in face if k not in sent_ids]
            if bad:
                errs.append(f"plan.json 的 face 里第 {'、'.join(map(str, bad))} 句不是讲稿里的句号（讲稿有第 {sent_ids[0]}—{sent_ids[-1]} 句）")
            if len(set(face)) != len(face):
                errs.append("plan.json 的 face 里有重复的句号")
            listed = set(face) & set(sent_ids)
    items = plan.get("items")
    if not isinstance(items, list) or (not items and not face_ok and not listed):
        return errs + ["plan.json 里没有 items"]

    def check_one(it, name, allow_fallback=True):
        k = it.get("kind")
        if k not in AUTO | MANUAL:
            errs.append(f"{name} 的 kind “{k}” 不是 {'、'.join(sorted(AUTO | MANUAL))} 之一")
            return
        need = {"截图": ["url", "credit"], "图库图片": ["query"], "图库视频": ["query"],
                "AI生图": ["prompt"], "手动视频": ["url", "time_range"], "录屏": ["note"], "现成图片": ["note"]}[k]
        for f in need:
            if not it.get(f):
                errs.append(f"{name}（{k}）缺 {f}")
        if k == "现成图片" and it.get("url") and not it.get("credit"):  # 不写 credit，来源清单就没有这张图的来源
            errs.append(f"{name}（现成图片）写了 url 就要写 credit（媒体名和日期）")
        cr = it.get("credit")
        if cr is not None and not (isinstance(cr, dict) and cr.get("media") and cr.get("date")):
            errs.append(f"{name} 的 credit 要写成 {{\"media\": 媒体名, \"date\": 日期}}")
        ms = it.get("manual_source")
        if ms is not None and not (isinstance(ms, dict) and ms.get("media") and ms.get("date")
                                   and str(ms.get("url", "")).startswith(("http://", "https://"))):
            errs.append(f"{name} 的 manual_source 要写成 {{\"media\": 媒体名, \"date\": 日期, \"url\": 网址}}")
        if "fallback" in it:
            if not allow_fallback:
                errs.append(f"{name} 的备选里不能再套备选")
            elif not isinstance(it["fallback"], dict):
                errs.append(f"{name} 的 fallback 应是一项素材")
            else:
                check_one(it["fallback"], name + " 的备选", allow_fallback=False)

    seen = set()
    for it in items:
        iid = it.get("id", "")
        name = f"素材 {iid or '（没写 id）'}"
        m = re.fullmatch(r"s(\d{2,})[a-z]?", str(iid))  # 100 句以上的讲稿是 s100、s127b（2026-10-31 放开，以前只到 s99）
        if not m:
            errs.append(f"{name} 的 id 要写成 s01、s04b、s127 这样")
        if iid in seen:
            errs.append(f"{name} 的 id 重复")
        seen.add(iid)
        sn = it.get("sentence")
        if not isinstance(sn, int) or sn not in sent_ids:
            errs.append(f"{name} 的 sentence {sn!r} 不是讲稿里的句号（讲稿有第 {sent_ids[0]}—{sent_ids[-1]} 句）")
        elif m and m.group(1) != f"{sn:02d}":  # 数字就是句号：第 10 句写 s100、s010 都不行
            errs.append(f"{name} 的 id 和句号对不上：第 {sn} 句写 s{sn:02d}、s{sn:02d}b 这样")
        if sn in listed:
            errs.append(f"{name}：第 {sn} 句写在 face 里（整屏放人脸），又配了素材，二选一")
        if not it.get("desc"):
            errs.append(f"{name} 没写 desc（画面）")
        check_one(it, name)
    covered = {it.get("sentence") for it in items}
    missing = [k for k in sent_ids if k not in covered and k not in listed]
    if missing and not face_ok:
        errs.append(f"这几句没有素材：第 {'、'.join(map(str, missing))} 句")
    return errs


def item_sha(it: dict) -> str:
    """plan.json 一项的指纹，变了才重取。manual_source 只管来源清单，不算在里面。"""
    it = {k: v for k, v in it.items() if k != "manual_source"}
    return hashlib.sha256(json.dumps(it, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def moved(part: dict) -> str | None:
    """截图时地址栏的地址和 plan.json 写的不是同一页（不算 http/https、末尾斜杠、问号后面的参数）：返回地址栏的地址。"""
    a, b = part.get("source_url"), part.get("final_url")
    if part.get("kind") != "截图" or not a or not b:
        return None
    key = lambda u: re.sub(r"^https?://", "", u.split("#")[0].split("?")[0]).rstrip("/").lower()
    return b if key(a) != key(b) else None


class Browser:
    """用到截图时才启动 Chrome，全程共用一个。"""

    def __init__(self):
        self.pw = self.browser = self.ctx = None

    def context(self):
        if self.ctx is None:
            from playwright.sync_api import sync_playwright
            self.pw = sync_playwright().start()
            self.browser = self.pw.chromium.launch(channel="chrome", headless=True)
            self.ctx = self.browser.new_context(viewport={"width": VIEW_W, "height": VIEW_H},
                                                device_scale_factor=SCALE, locale="zh-CN")
        return self.ctx

    def close(self):
        if self.browser:
            self.browser.close()
            self.pw.stop()


def shoot(br: Browser, it: dict, out: Path, prefix: str) -> dict:
    url = it["url"]
    pg = br.context().new_page()
    try:
        try:
            resp = pg.goto(url, wait_until="domcontentloaded", timeout=45000)
        except Exception as e:
            return {"ok": False, "reason": f"打不开：{type(e).__name__}"}
        pg.wait_for_timeout(2500)
        status = resp.status if resp else None
        title = pg.title()
        if (status and status >= 400) or CHALLENGE.search(title):
            return {"ok": False, "reason": f"网站拦下了程序访问（状态 {status}，标题“{title[:20]}”），不绕过"}
        height = min(pg.evaluate("() => document.documentElement.scrollHeight"), MAX_PAGE_H)
        for y in range(0, height, 700):  # 往下慢慢滚一遍，让懒加载的图片出来
            pg.evaluate(f"window.scrollTo(0, {y})")
            pg.wait_for_timeout(250)
        pg.evaluate("window.scrollTo(0, 0)")
        pg.wait_for_timeout(600)
        ov = pg.evaluate(OVERLAY_JS)
        if ov["blocked"]:  # 访问限制浮层：重新载入一次再看，还在就交给用户，不绕过
            pg.reload(wait_until="domcontentloaded", timeout=45000)
            pg.wait_for_timeout(2500)
            ov = pg.evaluate(OVERLAY_JS)
        if ov["blocked"]:
            return {"ok": False, "reason": f"页面有登录、订阅或验证之类的浮层挡住（“{ov['blocked'][0]}”），不绕过"}
        hidden = ov["hidden"]
        head = pg.evaluate(HEADLINE_JS)
        hl = pg.evaluate(FIND_JS, it["highlight"]) if it.get("highlight") else None
        iso = pg.evaluate(ISOLATE_JS)  # 只留正文
        pg.evaluate(WAIT_IMG_JS)
        page_file = out / f"{prefix}_page.jpg"

        def page_shot(head, hl, iso):
            """再量一次标题和高亮句的位置，截整页长图（截到正文最后一段下面一点）。"""
            pg.wait_for_timeout(300)
            head = pg.evaluate(HEADLINE_JS) or head
            if it.get("highlight"):
                hl = pg.evaluate(FIND_JS, it["highlight"]) or hl
            full = pg.evaluate("() => document.documentElement.scrollHeight")
            bottom = iso["content_bottom"] if iso else full
            if hl:
                bottom = max(bottom, hl["y"] + hl["h"])
            height = int(min(full, bottom + 120, MAX_PAGE_H))
            pg.screenshot(path=str(page_file), type="jpeg", quality=90, full_page=True,
                          clip={"x": 0, "y": 0, "width": VIEW_W, "height": height})
            return head, hl, height

        head, hl, height = page_shot(head, hl, iso)
        # 推近用的一屏：高亮句放到画面中间；没有高亮句就截标题那一屏
        target = hl or head
        top = 0 if target is None else max(0, target["y"] + target["h"] / 2 - VIEW_H / 2)
        pg.evaluate(f"window.scrollTo(0, {top})")
        pg.wait_for_timeout(800)
        ov = pg.evaluate(OVERLAY_JS)
        if ov["blocked"]:
            return {"ok": False, "reason": f"页面有登录、订阅或验证之类的浮层挡住（“{ov['blocked'][0]}”），不绕过"}
        hidden += [h for h in ov["hidden"] if h not in hidden]
        sy = pg.evaluate("() => scrollY")
        # 单屏图用截之前马上量的位置（标题和高亮句都重量）；长图用截长图之前量的，两张图各对各的
        # （第 07 期路透转载页往下滚以后上面少了 44 像素，以前标题只量一次，单屏图上的标题框就偏了）
        head_f = pg.evaluate(HEADLINE_JS) or head
        hl_f = (pg.evaluate(FIND_JS, it["highlight"]) or hl) if it.get("highlight") else None
        focus_file = out / f"{prefix}_focus.jpg"
        pg.screenshot(path=str(focus_file), type="jpeg", quality=92)

        def box(b, dy=0):
            if b is None:
                return None
            out = {k: round(b[k] - (dy if k == "y" else 0), 1) for k in ("x", "y", "w", "h")}
            if "lines" in b:
                out["lines"] = [box(l, dy) for l in b["lines"]]
            if b.get("block"):
                out["block"] = box(b["block"], dy)
            return out

        page_boxes = {"headline": box(head), "highlight": box(hl), "article": box((iso or {}).get("article"))}
        focus_boxes = {"headline": box(head_f, sy), "highlight": box(hl_f, sy), "scroll_y": sy}
        # 长图里的字和量的位置对不对得上：截长图时页面还会动（整页截图时懒加载的广告冒出来，第 07 期路透转载页的框落到了上一行），
        # 拿单屏图（量完马上截的，一屏截图不会触发这些）比一比；对不上就回到顶上重量、重截一次长图，还对不上，截图聚焦改用单屏图
        tgt = "highlight" if hl and hl_f else "headline"
        chk = align_check(page_file, focus_file, page_boxes[tgt], focus_boxes[tgt])
        first_dy = chk.get("dy")
        if chk["state"] == "偏了":
            pg.evaluate("window.scrollTo(0, 0)")
            pg.wait_for_timeout(1000)
            iso2 = pg.evaluate(ISOLATE_JS) or iso
            head, hl, height = page_shot(head, hl, iso2)
            page_boxes = {"headline": box(head), "highlight": box(hl), "article": box((iso2 or {}).get("article"))}
            chk = align_check(page_file, focus_file, page_boxes[tgt], focus_boxes[tgt])
            chk["retried_from_dy"] = first_dy
        page_check = {**chk, "target": tgt, "use": "focus" if chk["state"] == "偏了" else "page"}

        outside = bool(hl and hl["y"] + hl["h"] > height)  # 页面太长，高亮句在长图（最长 8000 CSS 像素）外面
        return {"ok": True, "files": {"page": page_file.name, "focus": focus_file.name},
                "status_code": status, "title": title, "final_url": pg.url, "page_height_css": height,
                "scale": SCALE, "headline": head and {**box(head), "font_px": head["font_px"], "text": head["text"]},
                "highlight_found": hl is not None if it.get("highlight") else None,
                "boxes": {"page": page_boxes, "focus": focus_boxes}, "page_check": page_check,
                "readable": head is not None and head["font_px"] >= READABLE_PX, "hidden_overlays": hidden,
                "article_only": iso is not None, "hidden_clutter": (iso or {}).get("hidden_inside", []),
                "highlight_outside_page": outside}
    finally:
        pg.close()


ALIGN_SEARCH = 400   # 在长图里记下的位置上下各找 400 CSS 像素
ALIGN_SCORE = 0.8    # 相似度（-1 到 1）不到这个数，算没找到
ALIGN_TOL = 3        # 差 3 CSS 像素以内算对得上（JPEG 压缩、亚像素位置）


def align_check(page_path: Path, focus_path: Path, pb: dict | None, fb: dict | None, scale: float = SCALE) -> dict:
    """长图里的字是不是在量的位置上。单屏图是量完马上截的一屏，当准的：从单屏图取这句（高亮句或标题）连同上下各一行左右，
    在长图里量的位置上下 ALIGN_SEARCH CSS 像素内逐个位置比，找最像的地方。差 0 就是两个框框着同一块字。
    返回 {"state": 对得上 / 偏了 / 比不了, "dy": 长图里的字实际比量的低多少 CSS 像素, "score": 最像处的相似度, "reason": …}。"""
    import numpy as np
    if not pb or not fb:
        return {"state": "比不了", "reason": "没量到这句的位置"}
    if fb["y"] < 0 or fb["y"] + fb["h"] > VIEW_H:
        return {"state": "比不了", "reason": "这句不在单屏图里"}

    def gray(p: Path):
        im = Image.open(p)
        im.draft("L", (im.width // round(scale), im.height // round(scale)))  # JPEG 解码时直接缩成 CSS 像素，快、省内存
        im = im.convert("L")
        if im.width != VIEW_W:
            im = im.resize((VIEW_W, round(im.height * VIEW_W / im.width)))
        return np.asarray(im, dtype=np.float32)

    P, F = gray(page_path), gray(focus_path)
    blk = fb.get("block") or fb
    x0 = max(0, int(min(blk["x"], fb["x"]) - 20))
    x1 = min(VIEW_W, int(max(blk["x"] + blk["w"], fb["x"] + fb["w"]) + 20))
    y0, y1 = max(0, int(fb["y"] - 40)), min(F.shape[0], int(fb["y"] + fb["h"] + 40))
    T = F[y0:y1, x0:x1]
    if T.size == 0 or T.std() < 3:
        return {"state": "比不了", "reason": "单屏图里这一块是空白"}
    T = T - T.mean()
    tn = float(np.sqrt((T * T).sum()))
    base = round(pb["y"] - fb["y"] + y0)  # 位置对的话，这一块在长图里的上边
    scores = {}
    for dy in range(-ALIGN_SEARCH, ALIGN_SEARCH + 1):
        a = base + dy
        if a < 0 or a + T.shape[0] > P.shape[0]:
            continue
        Wd = P[a:a + T.shape[0], x0:x1]
        Wd = Wd - Wd.mean()
        d = float(np.sqrt((Wd * Wd).sum()))
        scores[dy] = float((T * Wd).sum()) / (tn * d) if d > 0 else 0.0
    if not scores:
        return {"state": "比不了", "reason": "量的位置在长图外面"}
    dy = max(scores, key=scores.get)
    best, at0 = scores[dy], scores.get(0, -1.0)
    out = {"dy": dy, "score": round(best, 3), "score_at_0": round(at0, 3)}
    if at0 >= ALIGN_SCORE and (abs(dy) <= ALIGN_TOL or at0 >= best - 0.02):
        return {"state": "对得上", **out}
    if best >= ALIGN_SCORE:
        return {"state": "偏了", **out}
    return {"state": "比不了", "reason": "长图里没找到和单屏图一样的那一块", **out}


def pixabay_search(kind: str, q: str, key: str) -> dict:
    url = "https://pixabay.com/api/" if kind == "图库图片" else "https://pixabay.com/api/videos/"
    params = {"q": q[:100], "safesearch": "true", "per_page": 20}
    if kind == "图库图片":
        params.update(image_type="photo", orientation="horizontal", min_width=1280)
    tag = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()[:16]
    cache = PIXABAY_CACHE / f"{tag}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < CACHE_HOURS * 3600:
        return read_json(cache)
    try:
        r = requests.get(url, params={**params, "key": key}, timeout=30)
    except requests.RequestException as e:  # 原文里带着请求网址和密钥，只留异常类型
        raise RuntimeError(f"连不上 Pixabay（{type(e).__name__}）") from None
    if r.status_code == 429:
        raise RuntimeError("Pixabay 调用太频繁（每分钟 100 次），过一分钟再跑")
    if r.status_code != 200:
        raise RuntimeError(f"Pixabay 返回 {r.status_code}：{r.text[:80].replace(key, '***')}")
    data = r.json()
    PIXABAY_CACHE.mkdir(parents=True, exist_ok=True)
    write_json(cache, data)
    return data


def pixabay_key() -> str | None:
    """密钥放在环境变量里（写在 ~/.zshrc）。Claude Code 是在写进去之前启动的话环境里还没有，就让 zsh 读一遍 ~/.zshrc 取出来，不用重开。"""
    key = os.environ.get("PIXABAY_API_KEY")
    if key:
        return key
    r = subprocess.run(["zsh", "-c", 'source ~/.zshrc >/dev/null 2>&1; printf %s "$PIXABAY_API_KEY"'],
                       capture_output=True, text=True, timeout=20)
    return r.stdout.strip() or None


def stock(it: dict, out: Path, prefix: str) -> dict:
    key = pixabay_key()
    if not key:
        return {"ok": False, "reason": "没设 PIXABAY_API_KEY"}
    kind = it["kind"]
    queries = it["query"] if isinstance(it["query"], list) else [it["query"]]
    for q in queries:
        try:
            data = pixabay_search(kind, q, key)
        except Exception as e:
            return {"ok": False, "reason": str(e).replace(key, "***")}
        for hit in data.get("hits", []):
            if kind == "图库图片":
                src, ext = hit.get("largeImageURL"), ".jpg"
            else:
                vs = hit.get("videos", {})
                # 优先 1920 宽（large 常是 4K，文件大、拖慢），没有再取 1280—1920 的，最后才用 large
                # 只要横向的（宽至少是高的 1.3 倍），竖视频放进 16:9 会被裁掉大半
                ok = [vs[k] for k in ("large", "medium", "small") if vs.get(k, {}).get("url") and vs[k].get("width", 0) >= 1280
                      and vs[k]["width"] >= 1.3 * vs[k].get("height", 1)]
                # 1920 宽且不超过 12 MB 的优先；太大就取 1280—1920 里最小的文件（网慢时一段 28 MB 要下好几分钟）
                v = next((x for x in ok if x["width"] == 1920 and x.get("size", 0) <= MAX_VIDEO_BYTES), None) \
                    or min((x for x in ok if x["width"] <= 1920), key=lambda x: x.get("size", 0), default=None) \
                    or (ok[0] if ok else None)
                if v is None:
                    continue
                src, ext = v["url"], ".mp4"
            if not src:
                continue
            f = out / f"{prefix}_stock{ext}"
            try:
                with requests.get(src, timeout=120, stream=True) as r:
                    if r.status_code != 200:
                        continue
                    with open(f, "wb") as fh:
                        for chunk in r.iter_content(1 << 16):
                            fh.write(chunk)
            except requests.RequestException as e:
                return {"ok": False, "reason": f"下载 Pixabay 文件失败（{type(e).__name__}）"}
            size = media_size(f)
            return {"ok": True, "files": {"stock": f.name}, "query": q, "size": size,
                    "final_url": hit.get("pageURL"), "author": hit.get("user"), "tags": hit.get("tags"),
                    "license": "Pixabay Content License：免费商用，不强制署名"}
    return {"ok": False, "reason": f"Pixabay 上没搜到合适的（关键词：{'；'.join(queries)}）"}


def media_size(f: Path) -> list | None:
    """下载好的文件实际多大（宽、高）。"""
    if f.suffix == ".jpg":
        try:
            with Image.open(f) as im:
                return list(im.size)
        except Exception:
            return None
    r = subprocess.run([CONFIG["ffprobe"], "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", str(f)], capture_output=True, text=True)
    try:
        return [int(x) for x in r.stdout.strip().split(",")[:2]]
    except ValueError:
        return None


def used(r: dict) -> dict | None:
    """这一项实际用的那份自动结果：主方案成功用主方案，否则用成功的备选。"""
    if r.get("ok"):
        return r
    fb = r.get("fallback") or {}
    return fb if fb.get("ok") else None


def fetch_one(br: Browser, it: dict, out: Path, prefix: str) -> dict:
    # 一项出错不影响其余各项（PRD 第 4 节第 2 步：标“需手动”，继续其余条目）
    try:
        if it["kind"] == "截图":
            r = shoot(br, it, out, prefix)
        elif it["kind"] in ("图库图片", "图库视频"):
            r = stock(it, out, prefix)
        else:
            r = {"ok": False, "reason": "这一项要手动"}
    except Exception as e:
        key = os.environ.get("PIXABAY_API_KEY") or "\0"
        r = {"ok": False, "reason": f"出错：{type(e).__name__}：{str(e)[:80].replace(key, '***')}"}
    r.update(kind=it["kind"], source_url=it.get("url"), credit=it.get("credit"),
             fetched_at=time.strftime("%Y-%m-%d %H:%M"))
    return r


def manual_hint(it: dict) -> str:
    k = it["kind"]
    if k == "AI生图":
        return f"用这条提示词生成：{it['prompt']}"
    if k == "手动视频":
        return f"打开 {it['url']}，取 {it['time_range']}，下载或录屏"
    if k == "录屏":
        return it["note"]
    if k == "现成图片":
        return it["note"] + (f"（出处 {it['url']}）" if it.get("url") else "")
    if k == "截图":
        return f"在浏览器里打开 {it['url']}，截一张整页图（要高亮：{it.get('highlight') or '无'}）"
    q = it["query"][0] if isinstance(it["query"], list) else it["query"]
    sub = "images" if k == "图库图片" else "videos"
    return f"在 Pixabay 搜“{q}”：https://pixabay.com/{sub}/search/{quote(q)}/"


def thumb_of(folder: Path, files: dict) -> Image.Image | None:
    f = files.get("manual") or files.get("focus") or files.get("stock")
    if not f:
        return None
    p = folder / f
    if p.suffix.lower() in (".mp4", ".mov", ".m4v", ".webm"):
        tmp = folder / "_thumb.jpg"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", "1", "-i", str(p), "-frames:v", "1", str(tmp)])
        if not tmp.exists():
            return None
        im = Image.open(tmp).convert("RGB")
        tmp.unlink()
        return im
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


def contact_sheet(folder: Path, results: list, path: Path) -> None:
    """自动拿到和手动补齐的素材缩略图拼成一张，截图上用红框标出高亮句，一眼看清不清楚、框得对不对。"""
    tw, th, cols, pad, lab = 480, 270, 4, 12, 40
    rows = -(-len(results) // cols)
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + lab + pad) + pad), (32, 36, 40))
    font = ImageFont.truetype(str(ROOT / CONFIG["font_file"]), 20)
    d = ImageDraw.Draw(sheet)
    for k, r in enumerate(results):
        x, y = pad + (k % cols) * (tw + pad), pad + (k // cols) * (th + lab + pad)
        use = r if r.get("files") else (r.get("fallback") or {})
        if r.get("manual_file"):
            use = {"files": {"manual": r["manual_file"]}}
        im = thumb_of(folder, use.get("files", {})) if r["status"] != "需手动" or use.get("files") else None
        if im is None:
            d.rectangle([x, y, x + tw, y + th], fill=(70, 70, 70))
            d.text((x + 20, y + th // 2 - 12), "需手动", fill=(230, 230, 230), font=font)
        else:
            iw, ih = im.size
            s = min(tw / iw, th / ih)
            small = im.resize((max(1, int(iw * s)), max(1, int(ih * s))))
            ox, oy = x + (tw - small.width) // 2, y + (th - small.height) // 2
            sheet.paste(small, (ox, oy))
            hb = (use.get("boxes") or {}).get("focus", {}).get("highlight")
            if hb and "focus" in use.get("files", {}):
                f = s * use.get("scale", SCALE)
                for b in hb.get("lines") or [hb]:
                    d.rectangle([ox + b["x"] * f, oy + b["y"] * f, ox + (b["x"] + b["w"]) * f, oy + (b["y"] + b["h"]) * f],
                                outline=(255, 60, 60), width=3)
        d.text((x, y + th + 6), f"{r['id']} 第{r['sentence']}句 {r['kind']}｜{r['status']}", fill=(240, 240, 240), font=font)
    sheet.save(path, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", help="只重取这几项，逗号隔开，比如 s08,s14")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    t0 = time.time()
    folder = Path(a.folder).expanduser().resolve()
    inputs = read_json(folder / "inputs.json")
    out = folder / "素材"
    out.mkdir(exist_ok=True)
    sents = sentences(Path(inputs["script"]))
    plan_path = out / "plan.json"
    if not plan_path.exists():
        sys.exit(f"没有 {plan_path}：先由 Claude 照讲稿的镜头表写素材清单")
    plan = read_json(plan_path)
    face_ok = bool(inputs.get("vertical")) and not inputs.get("no_face")  # 竖屏有人脸：没写素材的句子放人脸
    errs = check_plan(plan, sorted(sents), face_ok, not inputs.get("no_face"))
    if errs:
        print("plan.json 校验不通过：")
        for e in errs:
            print("  - " + e)
        sys.exit(1)
    listed = sorted(set(plan.get("face") or []))  # 16:9 明写的整屏人脸句（竖屏没写的句子本来就放人脸）
    face = sorted(set(sents) - {it["sentence"] for it in plan["items"]}) if face_ok else listed
    if not plan["items"]:
        print(f"plan.json 一项素材也没有：竖屏整片放人脸全屏（{len(sents)} 句），不用找素材，配素材时每句 asset 写“人脸”" if face_ok else
              f"plan.json 一项素材也没有：face 写了全部 {len(sents)} 句，整片放人脸全屏，不用找素材，配素材时每句 asset 写“人脸”")
        return

    prev_path = out / "assets.json"
    prev_all = read_json(prev_path) if prev_path.exists() else {}
    prev = {r["id"]: r for r in prev_all.get("items", [])}
    sent_now = {str(k): v for k, v in sents.items()}
    if prev_all and (prev_all.get("script") != inputs["script"] or prev_all.get("sentences", sent_now) != sent_now):
        changed = [k for k in sent_now if prev_all.get("sentences", {}).get(k) != sent_now[k]]
        if not a.force:
            sys.exit(f"讲稿和上次找素材时不一样（上次 {Path(prev_all.get('script', '')).name}，现在 {Path(inputs['script']).name}；"
                     f"变了的句子：第 {'、'.join(changed) or '？'} 句）。先按新讲稿检查 plan.json 的句号和网址，确认后加 --force 重跑")
        prev = {}
    br = Browser()
    results, fetched = [], 0
    try:
        for it in plan["items"]:
            iid, sha = it["id"], item_sha(it)
            old = prev.get(iid)
            files_there = old and all((out / f).exists() for f in old.get("files", {}).values()) and \
                all((out / f).exists() for f in (old.get("fallback") or {}).get("files", {}).values())
            redo = a.force or (only is not None and iid in only)
            if only is not None and iid not in only and old:
                r = old  # 只重取指定的几项，其余照旧
                r.pop("manual_file", None)
                r.pop("manual_source", None)
            elif not redo and old and old.get("item_sha") == sha and files_there and used(old) is not None:
                r = old  # plan.json 里这一项没改过、自动拿到的文件还在，不重取
                r.pop("manual_file", None)
                r.pop("manual_source", None)
            else:
                fetched += 1
                print(f"{iid}（第 {it['sentence']} 句，{it['kind']}）……", flush=True)
                r = fetch_one(br, it, out, iid)
                if not r["ok"] and it.get("fallback"):
                    fb = fetch_one(br, it["fallback"], out, f"{iid}_fallback")
                    r["fallback"] = fb
                r.update(id=iid, sentence=it["sentence"], desc=it["desc"], effect=it.get("effect"), item_sha=sha)
                if not r["ok"]:
                    r["how"] = manual_hint(it)
                state = "自动" if r["ok"] else "自动（备选）" if used(r) else "需手动"
                print(f"  → {state}" + (f"：{r.get('reason')}" if not r["ok"] else ""), flush=True)
            r["status"] = "自动" if r.get("ok") else "自动（备选）" if used(r) else "需手动"
            for part in (r, r.get("fallback")):  # 以前截的没比过长图和单屏图：用留下的两张图补比一次（不重截）
                if part and part.get("ok") and "page" in part.get("files", {}) and "page_check" not in part:
                    b = part["boxes"]
                    tgt = "highlight" if b["focus"].get("highlight") else "headline"
                    chk = align_check(out / part["files"]["page"], out / part["files"]["focus"], b["page"].get(tgt),
                                      b["focus"].get(tgt), part.get("scale", SCALE))
                    # 以前的写法：高亮句两张图都用截单屏图前量的（长图那份可能偏），标题都用截长图前量的（单屏图那份可能偏）
                    part["page_check"] = {**chk, "target": tgt, "later": True,
                                          "use": "focus" if chk["state"] == "偏了" and tgt == "highlight" else "page"}
            manual = sorted(out.glob(f"{iid}_manual.*"))
            if manual:  # 用户手动放进来的，剪辑时优先用
                r["status"] = "手动补齐"
                r["manual_file"] = manual[0].name
                if it.get("manual_source"):
                    r["manual_source"] = it["manual_source"]
            results.append(r)
    finally:
        br.close()

    n = len(results)
    # 自动 = 主方案自动拿到；自动（备选）= 主方案没拿到、备选拿到了；手动补齐 = 用户放了手动文件（可能和自动重叠）；
    # 需手动 = 既没有自动文件也没有手动文件。用户补了手动文件，不会把“自动”算少
    cnt = {"自动": sum(bool(r.get("ok")) for r in results),
           "自动（备选）": sum(not r.get("ok") and used(r) is not None for r in results),
           "手动补齐": sum(bool(r.get("manual_file")) for r in results),
           "需手动": sum(used(r) is None and not r.get("manual_file") for r in results)}
    auto_used = [(r["id"], used(r)) for r in results if used(r) is not None]
    no_link = [i for i, u in auto_used if not u.get("final_url")]
    unreadable = [i for i, u in auto_used if u.get("kind") == "截图" and not u.get("readable")]
    no_hl = [i for i, u in auto_used if u.get("highlight_found") is False]
    outside = [i for i, u in auto_used if u.get("highlight_outside_page")]
    jumped = [i for i, u in auto_used if moved(u)]
    page_off = [i for i, u in auto_used if (u.get("page_check") or {}).get("use") == "focus"]
    seconds = round(time.time() - t0, 1)
    runs = prev_all.get("runs", []) if prev_all.get("script") == inputs["script"] else []
    runs.append({"at": time.strftime("%Y-%m-%d %H:%M"), "seconds": seconds, "fetched": fetched, **cnt})
    summary = {"sentences": len(sents), "items": n, **cnt, "face_sentences": face,
               "auto_ratio": round(cnt["自动"] / n, 3), "auto_ratio_with_fallback": round((cnt["自动"] + cnt["自动（备选）"]) / n, 3),
               "missing_source_link": no_link, "headline_too_small": unreadable, "highlight_not_found": no_hl,
               "highlight_outside_page": outside, "url_changed": jumped, "page_uses_focus": page_off,
               "seconds": seconds, "fetched": fetched,
               "longest_run_seconds": max(x["seconds"] for x in runs)}
    write_json(prev_path, {"script": inputs["script"], "sentences": sent_now, "summary": summary, "runs": runs, "items": results,
                           **({"face": listed} if listed else {})})  # compose.py 拿它核对 shots.json 的“人脸”句
    contact_sheet(out, results, out / "预览.jpg")
    write_list(out / "素材清单.md", inputs, sents, results, summary)
    print(f"素材：{n} 项，自动 {cnt['自动']}、自动备选 {cnt['自动（备选）']}、手动补齐 {cnt['手动补齐']}、还要手动 {cnt['需手动']}；"
          f"自动占 {summary['auto_ratio']:.0%}（含备选 {summary['auto_ratio_with_fallback']:.0%}）；这次取了 {fetched} 项、用时 {seconds} 秒，"
          f"历次最长 {summary['longest_run_seconds']} 秒 → {out / '素材清单.md'}")
    if face:
        print(f"没写素材的第 {'、'.join(map(str, face))} 句放人脸全屏（竖屏），配素材时 asset 写“人脸”" if face_ok else
              f"plan.json 的 face 写的第 {'、'.join(map(str, face))} 句整屏放人脸（不叠小窗），配素材时 asset 写“人脸”")
    for label, ids in (("标题字号小于 20 像素", unreadable), ("没找到要高亮的句子", no_hl), ("自动获取却没记来源链接", no_link),
                       ("高亮句在整页长图外面（页面超过 8000 CSS 像素）", outside),
                       ("截图时地址栏变了（来源照 plan.json 记，核对截的是不是这一页）", jumped),
                       ("长图里的字和量的位置对不上，截图聚焦改用单屏图", page_off)):
        if ids:
            print(f"注意：{label}：{'、'.join(ids)}")


def page_check_note(part: dict) -> str:
    """素材清单里写长图和量的位置比的结果：对上了不写；重截才对上、改用单屏图、比不了都写明。"""
    c = part["page_check"]
    what = "高亮句" if c.get("target") == "highlight" else "标题"
    if c["state"] == "对得上":
        if c.get("retried_from_dy") is not None:
            return f"；长图第一次截时{what}比量的位置偏了 {abs(c['retried_from_dy'])} 像素，重量、重截一次对上了"
        return ""
    if c["state"] == "偏了" and c["use"] == "page":  # 以前截的标题：长图那份是截长图前量的，对；单屏图上的标题框偏了
        return f"；单屏图上的标题框偏了 {abs(c['dy'])} 像素（重跑时补比的），截图聚焦用长图，不受影响"
    if c["state"] == "偏了":
        how = "重截也没对上" if c.get("retried_from_dy") is not None else "重跑时拿留下的两张图补比的"
        return (f"；长图里的{what}比量的位置{'低' if c['dy'] > 0 else '高'} {abs(c['dy'])} 像素（{how}），"
                f"截图聚焦改用单屏图 {part['files']['focus']}；想重截长图就加 --only")
    if c.get("target") == "highlight":
        return f"；长图和单屏图没比出来（{c.get('reason')}），看预览里框得对不对"
    return ""


def write_list(path: Path, inputs: dict, sents: dict, results: list, sm: dict) -> None:
    L = [f"# 素材清单：{inputs['name']}", "",
         f"讲稿：{Path(inputs['script']).name}。共 {sm['sentences']} 句、{sm['items']} 项素材："
         f"自动获取 {sm['自动']} 项，主方案没拿到、用了自动备选 {sm['自动（备选）']} 项，手动补齐 {sm['手动补齐']} 项（可能和自动的重叠），"
         f"还要手动 {sm['需手动']} 项。自动获取占 {sm['auto_ratio']:.0%}（算上备选 {sm['auto_ratio_with_fallback']:.0%}）。"
         f"这次取了 {sm['fetched']} 项、用时 {sm['seconds']} 秒，历次最长 {sm['longest_run_seconds']} 秒（验收要求不超过 10 分钟）。"
         f"缩略图见 预览.jpg（截图上的红框是要高亮的句子）。"
         + ((f"没写素材的第 {'、'.join(map(str, sm['face_sentences']))} 句放人脸全屏（竖屏）。" if inputs.get("vertical") else
             f"第 {'、'.join(map(str, sm['face_sentences']))} 句整屏放人脸（plan.json 的 face，不叠小窗）。")
            if sm.get("face_sentences") else ""), "",
         "| 句 | 素材 | 画面 | 方式 | 状态 | 文件 | 来源 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        use = r if r.get("files") else (r.get("fallback") or {})
        files = "、".join(list(use.get("files", {}).values()) + ([r["manual_file"]] if r.get("manual_file") else [])) or "—"
        src = (use.get("source_url") or use.get("final_url")) if use.get("files") else (r.get("source_url") or "—")
        src = src or "—"
        note = ""
        if r["kind"] == "截图" and r.get("ok"):
            hd = r.get("headline") or {}
            note = f"；标题 {hd.get('font_px', '?')} 像素" + ("" if r.get("highlight_found") is not False else "；没找到高亮句")
            if r.get("ok") and r.get("article_only") is False:
                note += "；没认出正文，整页都截了"
            if r.get("highlight_outside_page"):
                note += "；高亮句在整页长图外面"
            if r.get("hidden_overlays"):
                note += "；隐藏了挡画面的浮层“" + "”“".join(r["hidden_overlays"]) + "”"
        if use.get("files") and (to := moved(use)):
            note += f"；截图时地址栏变成了 {to}（往下滚换了文章、被拦或跳转），来源照 plan.json 的网址记，核对截的是不是这一页"
        if use.get("files") and use.get("page_check"):
            note += page_check_note(use)
        cr = use.get("credit") if use.get("files") else r.get("credit")
        credit = f"（{cr['media']} {cr['date']}）" if cr else ""
        if r.get("manual_source"):  # 手动补的文件剪辑时优先用，来源照它写的
            ms = r["manual_source"]
            credit += f"；手动补的来自 {ms['url']}（{ms['media']} {ms['date']}）"
        L.append(f"| {r['sentence']} | {r['id']} | {r['desc']} | {r['kind']} | {r['status']}{note} | {files} | {src}{credit} |")
    todo = [r for r in results if not r.get("manual_file") and not r.get("ok")]
    L += ["", f"## 要你手动补的（{len(todo)} 项）", ""]
    if not todo:
        L.append("没有。")
    for k, r in enumerate(todo, 1):
        line = f"{k}. {r['id']}（第 {r['sentence']} 句，{r['kind']}）：{r['desc']}。{r.get('how', '').rstrip('。')}。存成 素材/{r['id']}_manual.<扩展名>。"
        if r.get("reason") and r["reason"] != "这一项要手动":
            line += f"自动没拿到的原因：{r['reason']}。"
        fb = r.get("fallback")
        if fb and fb.get("ok"):
            line += f"备选已自动下好：{'、'.join(fb['files'].values())}，不补也能用备选。"
        L.append(line)
    L += ["", "放好以后说“素材放好了”，重跑一遍就会记成“手动补齐”。", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
