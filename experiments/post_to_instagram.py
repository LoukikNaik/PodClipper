#!/usr/bin/env python3
"""Post a rendered reel to Instagram, with LLM-generated caption + hashtags.

Pulls the reel's cached transcript (`words.json`) and sidecar metadata
(`reel_*.txt`), asks the configured LLM for a hook caption + relevant
hashtags, then drives Instagram via Chrome DevTools Protocol (Playwright
attached to a Chrome already running with --remote-debugging-port=9222).

Posting flow adapted from anime-reel-maker/backend/post_to_instagram.py.
Selectors discovered by step-by-step CDP exploration of instagram.com
and are subject to change when Instagram rebuilds its UI.

Requires:
  * Chrome already running with --remote-debugging-port=9222, logged into
    Instagram (or INSTAGRAM_USER/INSTAGRAM_PASS in .env for auto-login)
  * `playwright` installed in the environment

Usage:
    python post_to_instagram.py outputs/<run>/reel_NN_<slug>.mp4
    python post_to_instagram.py <reel.mp4> --dry-run
    python post_to_instagram.py <reel.mp4> --caption "custom caption"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.llm import LLMError, build_provider
from src.logging_util import setup_logging

DEBUG_PORT = 9222
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_ROOT = SCRIPT_DIR / ".cache"
INSTAGRAM_URL = "https://www.instagram.com/"
LOGIN_URL = "https://www.instagram.com/accounts/login/"
MAX_HASHTAGS = 15
INSTAGRAM_CAPTION_LIMIT = 2200


CAPTION_SYSTEM = """You write Instagram Reel captions and hashtags for short vertical clips cut from podcast videos.

You will receive the reel's title, the editor's note about why it was picked, and the transcript of what is actually said in the clip.

Output STRICT JSON with exactly two fields:
  {
    "caption": "<1-3 short lines, hook-style, no hashtags inside>",
    "hashtags": ["#one", "#two", ...]
  }

Rules:
- Caption is 1-3 lines. Newlines separate lines. No hashtags inside the caption — they go in the array.
- Open with a scroll-stopping hook grounded in what was ACTUALLY said. Address the viewer ("you/your") when natural.
- Do not paraphrase the whole clip; tease the payoff, leave the answer in the video.
- 8-15 hashtags. Mix broad (#mindset) with niche (#stoicfounder). Lowercase. No spaces. No duplicates.
- Hashtags must be relevant to the transcript content, not generic filler.
- No emojis unless the transcript itself has a strongly emotional beat that justifies one.
- Output JSON ONLY. No prose, no code fence, no trailing comma.
"""


def _read_sidecar(sidecar_path: Path) -> dict:
    """Parse a reel_*.txt sidecar into a flat dict. Missing file → empty dict."""
    if not sidecar_path.exists():
        return {}
    data: dict = {}
    for raw in sidecar_path.read_text().splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        data[key.strip()] = val.strip()
    return data


def _locate_words_json(reel_slug: str) -> Optional[Path]:
    """Glob the cache for `<video>/<reel_slug>/words.json`.

    Reels carry their slug into the cache verbatim (see pipeline.py), so a
    single glob lands the right transcript without needing the source-video
    hash. If multiple matches exist (same slug across runs), pick the
    most recently modified.
    """
    if not CACHE_ROOT.exists():
        return None
    matches = sorted(
        CACHE_ROOT.glob(f"*/{reel_slug}/words.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _transcript_from_words(words_path: Path) -> str:
    words = json.loads(words_path.read_text())
    parts = [w.get("text", "") for w in words if isinstance(w, dict)]
    return "".join(parts).strip()


def _extract_json_object(text: str) -> dict:
    """Pull a JSON object out of LLM output. Tolerant of code fences or prose."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last < first:
        raise ValueError(f"no JSON object in LLM output: {text[:200]!r}")
    return json.loads(text[first : last + 1])


def _sanitize_hashtag(tag: str) -> str:
    tag = tag.strip()
    if not tag.startswith("#"):
        tag = "#" + tag
    # Strip whitespace + punctuation Instagram won't accept inside a tag
    tag = "#" + re.sub(r"[^A-Za-z0-9_]", "", tag[1:])
    return tag if len(tag) > 1 else ""


def _format_caption(caption: str, hashtags: list[str]) -> str:
    seen: set[str] = set()
    deduped: list[str] = []
    for h in hashtags:
        clean = _sanitize_hashtag(h)
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            deduped.append(clean)
    deduped = deduped[:MAX_HASHTAGS]
    full = caption.strip()
    if deduped:
        full = f"{full}\n\n{' '.join(deduped)}"
    return full[:INSTAGRAM_CAPTION_LIMIT]


def generate_caption(reel_path: Path, cfg, log) -> str:
    """Build caption + hashtags from the reel's sidecar + cached transcript."""
    slug = reel_path.stem  # e.g. reel_01_confidence-is-an-output-not-input
    sidecar = reel_path.with_suffix(".txt")
    meta = _read_sidecar(sidecar)
    title = meta.get("title") or slug
    reason = meta.get("reason", "")

    words_path = _locate_words_json(slug)
    if words_path is None:
        log.warning(f"no words.json found for slug {slug!r}; caption will lean on title alone")
        transcript = ""
    else:
        transcript = _transcript_from_words(words_path)
        log.info(f"transcript loaded from {words_path.relative_to(SCRIPT_DIR)} ({len(transcript)} chars)")

    user_prompt = (
        f"Title: {title}\n"
        f"Editor's note: {reason}\n\n"
        f"Transcript:\n{transcript or '(transcript unavailable — work from the title)'}\n"
    )

    provider = build_provider(cfg.llm)
    log.info(f"asking {provider.name} for caption + hashtags...")
    try:
        response = provider.complete(
            user_prompt=user_prompt,
            system_prompt=CAPTION_SYSTEM,
            max_tokens=600,
        )
    except LLMError as e:
        log.error(f"LLM call failed: {e}")
        raise

    parsed = _extract_json_object(response)
    caption = str(parsed.get("caption", "")).strip()
    hashtags = parsed.get("hashtags", []) or []
    if not isinstance(hashtags, list):
        hashtags = []
    if not caption:
        raise ValueError(f"LLM returned empty caption; raw: {response[:300]!r}")

    return _format_caption(caption, [str(h) for h in hashtags])


# -------------------- Instagram posting (raw CDP over WebSocket) --------------
# Playwright's connect_over_cdp enumerates every target the browser holds
# (pages + iframes + service workers + extensions) before returning, which
# can take many minutes on a long-running daily Chrome. We bypass that by
# attaching directly to the single Instagram page target.
#
# All UI interaction goes through Runtime.evaluate (JS executed in the
# page's main frame). File upload is the only exception — that needs
# DOM.setFileInputFiles against the <input type="file"> node.


class CDPClient:
    """Minimal JSON-RPC-over-WebSocket Chrome DevTools Protocol client.

    Attached to one page target. Send commands with .send(method, params).
    Calls block on the matching response — there's no async / event loop here.
    """

    def __init__(self, ws_url: str, log):
        import websocket  # type: ignore
        self.log = log
        self.ws = websocket.create_connection(ws_url, timeout=30)
        self._id = 0

    def send(self, method: str, params: dict | None = None, timeout: float = 30.0):
        import websocket  # type: ignore
        self._id += 1
        msg_id = self._id
        payload = {"id": msg_id, "method": method}
        if params is not None:
            payload["params"] = params
        self.ws.send(json.dumps(payload))
        # Drain events until our response arrives. Chrome interleaves
        # event notifications (no "id" key) with command responses.
        # Events we care about get appended to self._pending_events.
        self.ws.settimeout(timeout)
        while True:
            try:
                raw = self.ws.recv()
            except websocket.WebSocketTimeoutException as e:
                raise TimeoutError(f"CDP {method} timed out after {timeout}s") from e
            data = json.loads(raw)
            if data.get("id") == msg_id:
                if "error" in data:
                    raise RuntimeError(f"CDP {method} error: {data['error']}")
                return data.get("result", {})
            if "method" in data:
                getattr(self, "_pending_events", []).append(data)
            # else: ignore

    def collect_events(self, methods: tuple[str, ...]) -> list[dict]:
        """Drain queued CDP events matching any of `methods`, returning them
        oldest-first and dropping them from the pending queue.
        """
        if not hasattr(self, "_pending_events"):
            return []
        keep, taken = [], []
        for ev in self._pending_events:
            (taken if ev.get("method") in methods else keep).append(ev)
        self._pending_events = keep
        return taken

    def wait_for_event(self, method: str, timeout: float = 10.0) -> dict | None:
        """Block until a CDP event with matching method arrives, or timeout.

        Recv-loop variant of send(): drains the websocket until we see the
        target event. Drops responses with no matching id (shouldn't happen
        since callers always pair send() with a response).
        """
        import websocket  # type: ignore
        if not hasattr(self, "_pending_events"):
            self._pending_events = []
        # Check queued events first
        for i, ev in enumerate(self._pending_events):
            if ev.get("method") == method:
                return self._pending_events.pop(i).get("params", {})
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.ws.settimeout(max(0.1, deadline - time.time()))
            try:
                raw = self.ws.recv()
            except websocket.WebSocketTimeoutException:
                break
            data = json.loads(raw)
            if data.get("method") == method:
                return data.get("params", {})
            if "method" in data:
                self._pending_events.append(data)
        return None

    def eval_js(self, expression: str, await_promise: bool = True, return_by_value: bool = True):
        """Run JS in the page; return the (deserialized) result value."""
        result = self.send("Runtime.evaluate", {
            "expression": expression,
            "awaitPromise": await_promise,
            "returnByValue": return_by_value,
            "userGesture": True,
        })
        if "exceptionDetails" in result:
            raise RuntimeError(f"JS exception: {result['exceptionDetails']}")
        return result.get("result", {}).get("value")

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass


def _find_ig_target() -> Optional[dict]:
    """Return the CDP target dict for an Instagram page, or None."""
    import urllib.request
    try:
        with urllib.request.urlopen(
            f"http://localhost:{DEBUG_PORT}/json/list", timeout=5,
        ) as r:
            targets = json.load(r)
    except Exception:
        return None
    # Prefer real pages (type=="page") whose URL is on instagram.com.
    for t in targets:
        if t.get("type") == "page" and "instagram.com" in (t.get("url") or ""):
            return t
    return None


# JS helpers reused by the post flow. Kept as Python strings so the file
# stays single-source — these get injected via Runtime.evaluate.

_JS_CLICK_BY_ARIA = r"""
((label) => {
    const nodes = document.querySelectorAll('svg[aria-label="' + label + '"]');
    for (const n of nodes) {
        const clickable = n.closest('[role="button"], button, a, div[tabindex]') || n;
        if (clickable.offsetParent !== null) { clickable.click(); return true; }
    }
    return false;
})
"""

_JS_CLICK_EXACT_TEXT = r"""
((target) => {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
        const node = walker.currentNode;
        if (node.textContent.trim() === target) {
            const el = node.parentElement;
            if (!el || el.tagName === 'title') continue;
            if (el.offsetParent === null && el.tagName !== 'BODY') continue;
            const clickable = el.closest('[role="button"], button, a, div[tabindex]') || el;
            clickable.click();
            return true;
        }
    }
    return false;
})
"""

_JS_CAPTION_BOX_RECT = r"""
(() => {
    const el = document.querySelector('[aria-label="Write a caption..."]');
    if (!el) return null;
    el.focus();
    const r = el.getBoundingClientRect();
    return {x: r.left + 12, y: r.top + r.height / 2};
})()
"""

_JS_VISIBLE_TEXT_PRESENT = r"""
((target) => {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
        const node = walker.currentNode;
        if (node.textContent.includes(target)) {
            const el = node.parentElement;
            if (el && el.offsetParent !== null) return true;
        }
    }
    return false;
})
"""


def _js_call(fn_literal: str, arg: str) -> str:
    """Wrap a JS arrow-fn literal as an immediately-invoked call with one arg."""
    return f"({fn_literal})({json.dumps(arg)})"


def _wait_for(client: CDPClient, js_predicate: str, label: str, log, timeout: float = 30.0, poll: float = 0.5) -> bool:
    """Poll a JS boolean predicate until true or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if client.eval_js(js_predicate, await_promise=False):
                return True
        except Exception as e:
            log.debug(f"wait_for({label}) eval failed: {e}")
        time.sleep(poll)
    log.warning(f"wait_for({label}) timed out after {timeout}s")
    return False


def _click_aria(client: CDPClient, label: str, log) -> bool:
    return bool(client.eval_js(_js_call(_JS_CLICK_BY_ARIA, label), await_promise=False))


def _click_text(client: CDPClient, text: str, log) -> bool:
    return bool(client.eval_js(_js_call(_JS_CLICK_EXACT_TEXT, text), await_promise=False))


def _screenshot(client: CDPClient, name: str, log) -> Optional[Path]:
    """Save a PNG screenshot of the current viewport to .cache/screenshots/."""
    try:
        result = client.send("Page.captureScreenshot", {"format": "png"})
        import base64
        out_dir = CACHE_ROOT / "post_screens"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{int(time.time())}_{name}.png"
        path.write_bytes(base64.b64decode(result["data"]))
        log.info(f"screenshot: {path.relative_to(SCRIPT_DIR)}")
        return path
    except Exception as e:
        log.warning(f"screenshot failed: {e}")
        return None


_JS_DOM_SUMMARY = r"""
((label) => {
    function summarize(el) {
        if (!el) return null;
        const rect = el.getBoundingClientRect();
        const text = (el.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 120);
        return {
            tag: el.tagName.toLowerCase(),
            id: el.id || null,
            cls: (el.className || '').toString().slice(0, 80) || null,
            ariaLabel: el.getAttribute('aria-label'),
            role: el.getAttribute('role'),
            type: el.getAttribute('type'),
            placeholder: el.getAttribute('placeholder'),
            contenteditable: el.getAttribute('contenteditable'),
            visible: rect.width > 0 && rect.height > 0 && el.offsetParent !== null,
            text: text || null,
        };
    }
    const dialog = document.querySelector('div[role="dialog"]');
    const root = dialog || document.body;
    const interesting = [];
    const selectors = [
        'div[role="dialog"]',
        'h1, h2, h3',
        '[aria-label]',
        '[role="button"]',
        '[role="textbox"]',
        '[contenteditable]',
        'textarea',
        'button',
    ];
    for (const sel of selectors) {
        for (const el of root.querySelectorAll(sel)) {
            interesting.push({selector: sel, ...summarize(el)});
        }
    }
    return {label, url: location.href, dialogPresent: !!dialog,
            count: interesting.length, items: interesting};
})
"""


def _dump_dom(client: CDPClient, label: str, log) -> None:
    """Dump a structured summary of buttons/textboxes/aria-labeled elements
    in the current dialog. Helps figure out which selectors actually match.
    """
    try:
        data = client.eval_js(_js_call(_JS_DOM_SUMMARY, label), await_promise=False)
    except Exception as e:
        log.warning(f"dom dump failed: {e}")
        return
    out_dir = CACHE_ROOT / "post_screens"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{int(time.time())}_{label}.dom.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info(f"dom dump: {path.relative_to(SCRIPT_DIR)} ({data.get('count', '?')} items)")


_KEY_CODES = {
    " ": ("Space", 32),
    "\n": ("Enter", 13),
    ".": ("Period", 190),
    ",": ("Comma", 188),
    "?": ("Slash", 191),
    "!": ("Digit1", 49),
    "'": ("Quote", 222),
    '"': ("Quote", 222),
    "-": ("Minus", 189),
    "#": ("Digit3", 51),
    "_": ("Minus", 189),
}


def _key_event(client: CDPClient, ch: str, evt: str) -> None:
    """Dispatch a single keyDown/keyUp event for character `ch`.

    Each character is sent as a separate keyDown/keyUp pair so the page's
    keydown / keypress / keyup listeners fire in order — that's the path
    Instagram's React state watches for draft caption persistence.
    """
    if ch == "\n":
        code, vk = "Enter", 13
        text = "\r"
    elif ch.isalpha():
        code = f"Key{ch.upper()}"
        vk = ord(ch.upper())
        text = ch
    elif ch.isdigit():
        code = f"Digit{ch}"
        vk = ord(ch)
        text = ch
    else:
        code, vk = _KEY_CODES.get(ch, ("Unidentified", 0))
        text = ch

    params = {
        "type": evt,
        "key": ch if evt == "keyDown" or ch != "\n" else "Enter",
        "code": code,
        "windowsVirtualKeyCode": vk,
        "nativeVirtualKeyCode": vk,
    }
    if evt == "keyDown":
        params["text"] = text
        if ch.isupper():
            params["modifiers"] = 8  # Shift
    client.send("Input.dispatchKeyEvent", params)


def _type_caption(client: CDPClient, caption: str, log) -> bool:
    """Type the caption character-by-character via real key events.

    Why not Input.insertText? IG's first-post composer also listens for
    keydown/keypress events (for draft persistence and React state sync),
    and Input.insertText only fires beforeinput/input. Real per-char
    dispatchKeyEvent fires the full keyboard event sequence so IG's
    captionState ends up containing the text — without this, the caption
    visibly renders in the contenteditable but the post goes out empty.
    """
    rect = client.eval_js(_JS_CAPTION_BOX_RECT, await_promise=False)
    if not rect:
        log.warning("caption box not found at type time")
        return False

    x, y = rect["x"], rect["y"]
    for event_type in ("mousePressed", "mouseReleased"):
        client.send("Input.dispatchMouseEvent", {
            "type": event_type, "x": x, "y": y,
            "button": "left", "clickCount": 1,
            "buttons": 1 if event_type == "mousePressed" else 0,
        })
    time.sleep(0.3)

    for ch in caption:
        _key_event(client, ch, "keyDown")
        _key_event(client, ch, "keyUp")
        # Tiny inter-key delay — fully serial keystrokes confuse React's
        # batching on some IG versions; ~5ms keeps it natural.
        time.sleep(0.005)

    # Trailing edit (space + backspace) — a no-op character-level mutation
    # that forces IG's caption-draft reducer to re-fire after the bulk
    # paste. Without this, the captionState we end up with sometimes lags
    # the last keystroke and the share goes out with a truncated body.
    for ch, code, vk in (
        (" ", "Space", 32),
        ("Backspace", "Backspace", 8),
    ):
        for evt in ("keyDown", "keyUp"):
            params = {
                "type": evt, "key": ch, "code": code,
                "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk,
            }
            if evt == "keyDown" and ch == " ":
                params["text"] = " "
            client.send("Input.dispatchKeyEvent", params)
        time.sleep(0.05)

    # Give IG's draft-persist API call ~3s to fly before we click Share.
    # We don't know which xhr is the draft save, so just sleep generously.
    log.info("waiting 3s for IG to persist caption state...")
    time.sleep(3.0)

    readback = client.eval_js(
        """(() => {
            const el = document.querySelector('[aria-label="Write a caption..."]');
            if (!el) return {found: false};
            return {found: true, text: el.innerText || el.textContent || '',
                    childLen: el.children.length};
        })()""",
        await_promise=False,
    )
    log.info(f"caption typed ({len(caption)} chars); readback: {readback}")
    return True


def _dismiss_modal(client: CDPClient, log) -> None:
    """Close any open Instagram modal so we start from a clean state.

    Idempotent — clicks the X button if present, then presses Escape, then
    sleeps. Safe to call when no modal is open.
    """
    closed = client.eval_js(
        """(() => {
            const dialog = document.querySelector('div[role="dialog"]');
            if (!dialog) return false;
            // Try the explicit close button first
            const closeBtn = dialog.querySelector('svg[aria-label="Close"]');
            if (closeBtn) {
                const clickable = closeBtn.closest('[role="button"], button, a, div[tabindex]') || closeBtn;
                clickable.click();
                return true;
            }
            return false;
        })()""",
        await_promise=False,
    )
    if closed:
        log.info("dismissed pre-existing modal via close button")
        time.sleep(1)
    # Also send Escape — covers confirmation popovers ("Discard post?")
    client.send("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Escape", "code": "Escape", "windowsVirtualKeyCode": 27})
    client.send("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Escape", "code": "Escape", "windowsVirtualKeyCode": 27})
    time.sleep(0.5)
    # If a "Discard post?" dialog appeared, confirm it
    discarded = client.eval_js(
        """(() => {
            const buttons = Array.from(document.querySelectorAll('button, div[role="button"]'));
            for (const b of buttons) {
                if (b.offsetParent === null) continue;
                const t = (b.textContent || '').trim();
                if (t === 'Discard' || t === 'Discard post' || t === 'Discard reel') {
                    b.click(); return true;
                }
            }
            return false;
        })()""",
        await_promise=False,
    )
    if discarded:
        log.info("confirmed discard")
        time.sleep(1)


def _set_file_input(client: CDPClient, video_path: Path, log) -> bool:
    """Upload via real mouse click + file chooser interception.

    Instagram's upload flow doesn't react to a change event on its <input
    type=file>; the flow is wired to the "Select from computer" button's
    click handler, which programmatically opens a file picker. Setting
    files on the input directly is silently ignored.

    The reliable path is:
      1. Page.setInterceptFileChooserDialog(enabled=true) — Chrome will
         redirect any native file picker to us as Page.fileChooserOpened.
      2. Get the "Select from computer" button's bounding box.
      3. Input.dispatchMouseEvent (mousePressed + mouseReleased) at the
         button center. A real synthetic click counts as a user gesture,
         unlike JS .click() which gets blocked.
      4. Wait for Page.fileChooserOpened — payload has backendNodeId of
         the input Instagram opened the chooser for.
      5. DOM.setFileInputFiles(backendNodeId, [path]) — fills the exact
         input Instagram is awaiting; React's bound handler fires.
    """
    abs_path = str(video_path.resolve())

    rect = client.eval_js(
        """(() => {
            const buttons = Array.from(document.querySelectorAll('button, div[role="button"]'));
            for (const b of buttons) {
                if (b.offsetParent === null) continue;
                const t = (b.textContent || '').trim();
                if (t === 'Select from computer' || t === 'Select From Computer') {
                    const r = b.getBoundingClientRect();
                    return {x: r.left + r.width / 2, y: r.top + r.height / 2,
                            w: r.width, h: r.height};
                }
            }
            return null;
        })()""",
        await_promise=False,
    )
    if not rect:
        log.error("couldn't find 'Select from computer' button")
        return False

    client.send("Page.setInterceptFileChooserDialog", {"enabled": True})
    try:
        x, y = rect["x"], rect["y"]
        for event_type in ("mousePressed", "mouseReleased"):
            client.send("Input.dispatchMouseEvent", {
                "type": event_type,
                "x": x,
                "y": y,
                "button": "left",
                "clickCount": 1,
                "buttons": 1 if event_type == "mousePressed" else 0,
            })

        event = client.wait_for_event("Page.fileChooserOpened", timeout=10)
        if event is None:
            log.error("Page.fileChooserOpened never fired after click")
            return False
        backend_node_id = event.get("backendNodeId")
        if not backend_node_id:
            log.error(f"fileChooserOpened event missing backendNodeId: {event}")
            return False

        client.send("DOM.setFileInputFiles", {
            "backendNodeId": backend_node_id,
            "files": [abs_path],
        })
        log.info("file attached via chooser intercept")
        return True
    finally:
        try:
            client.send("Page.setInterceptFileChooserDialog", {"enabled": False})
        except Exception:
            pass


def _post_reel_cdp(client: CDPClient, video_path: Path, caption: str, log, submit: bool = True) -> bool:
    log.info(f"uploading: {video_path.resolve()}")

    # Dismiss any leftover modal from a previous failed run
    _dismiss_modal(client, log)

    # Dismiss "Save your login info?" / notifications dialogs that sometimes
    # cover the New-post icon on a fresh session.
    for _ in range(3):
        if _click_text(client, "Not Now", log):
            time.sleep(1.5)
            continue
        break

    # Step 1: open the New-post dropdown
    if not _wait_for(
        client,
        "!!document.querySelector('svg[aria-label=\"New post\"]')",
        "New post icon",
        log,
        timeout=15,
    ):
        log.error("couldn't find New post icon — is this an Instagram page and logged in?")
        return False
    if not _click_aria(client, "New post", log):
        log.error("New post click failed")
        return False
    time.sleep(1.2)

    # Step 2: handle BOTH UI variants. Newer Instagram skips the create
    # dropdown — clicking "+" goes straight to the upload modal. Older
    # variants still show a "Post / Reel / Story" picker. We wait for
    # whichever appears first.
    file_input_js = "!!document.querySelector('input[type=\"file\"]')"
    dropdown_js = _js_call(_JS_VISIBLE_TEXT_PRESENT, "Create new post")
    if not _wait_for(
        client,
        f"({file_input_js}) || ({dropdown_js})",
        "upload modal or dropdown",
        log,
        timeout=15,
    ):
        log.error("neither file input nor dropdown appeared after New post click")
        _screenshot(client, "post_dropdown_missing", log)
        return False

    # If the dropdown variant rendered, click "Post" to advance.
    needs_post_click = not bool(client.eval_js(file_input_js, await_promise=False))
    if needs_post_click:
        log.info("create dropdown variant — clicking Post")
        if not _click_text(client, "Post", log):
            log.error("Post click failed")
            _screenshot(client, "post_click_failed", log)
            return False
        time.sleep(3)
    else:
        log.info("upload modal opened directly (no dropdown)")

    # Step 3: file input
    if not _wait_for(
        client,
        file_input_js,
        "file input",
        log,
        timeout=15,
    ):
        log.error("file input never appeared")
        _screenshot(client, "no_file_input", log)
        return False
    if not _set_file_input(client, video_path, log):
        return False
    log.info("file attached")
    time.sleep(8)
    _screenshot(client, "01_after_upload", log)
    _dump_dom(client, "01_after_upload", log)

    # Step 4: "Video posts shared as reels" → OK
    if _wait_for(
        client,
        _js_call(_JS_VISIBLE_TEXT_PRESENT, "OK"),
        "OK dialog",
        log,
        timeout=4,
        poll=0.3,
    ):
        _click_text(client, "OK", log)
        time.sleep(1.5)

    # Step 5: 9:16 crop (our reels are already 1080x1920 but force the aspect)
    if _click_aria(client, "Select crop", log):
        time.sleep(0.8)
        _click_aria(client, "Crop portrait icon", log)
        time.sleep(0.8)

    # Step 6: Next (Crop → Edit), then Next (Edit → Caption)
    for idx, label in enumerate(("crop", "edit"), start=2):
        if not _wait_for(
            client,
            _js_call(_JS_VISIBLE_TEXT_PRESENT, "Next"),
            f"Next ({label})",
            log,
            timeout=15,
        ):
            log.warning(f"Next button never appeared at {label} step")
            _screenshot(client, f"{idx:02d}_no_next_{label}", log)
            _dump_dom(client, f"{idx:02d}_no_next_{label}", log)
            continue
        _click_text(client, "Next", log)
        time.sleep(3)
        _screenshot(client, f"{idx:02d}_after_next_{label}", log)
        _dump_dom(client, f"{idx:02d}_after_next_{label}", log)

    # Step 7: caption. We can't just execCommand('insertText') here — that
    # mutates the DOM without firing the beforeinput event chain that React
    # listens for, so Instagram's controlled state stays empty and the post
    # goes out with no caption even though the box visually shows text.
    # Real keyboard injection via CDP's Input.insertText (after a real mouse
    # click to focus) does fire beforeinput and updates React state.
    if caption:
        if _wait_for(
            client,
            "!!document.querySelector('[aria-label=\"Write a caption...\"]')",
            "caption box",
            log,
            timeout=15,
        ):
            _type_caption(client, caption, log)
        else:
            log.warning("caption box never appeared")
            _screenshot(client, "no_caption_box", log)

    if not submit:
        _screenshot(client, "before_share", log)
        log.info(
            "no-submit: stopped before clicking Share. "
            "Review the modal in Chrome — close it manually to abort, or click Share to post."
        )
        return True

    # Step 8: Share
    if not _click_text(client, "Share", log):
        log.error("Share click failed")
        return False
    log.info("clicked Share")

    # Step 9: confirmation
    deadline = time.time() + 240
    while time.time() < deadline:
        for needle in ("Your reel has been shared", "Reel shared", "Post shared"):
            try:
                if client.eval_js(_js_call(_JS_VISIBLE_TEXT_PRESENT, needle), await_promise=False):
                    log.info("reel posted")
                    return True
            except Exception:
                pass
        time.sleep(2)
    log.info("Share clicked — verify in Instagram")
    return True


def _post_via_cdp(video_path: Path, caption: str, log, submit: bool = True) -> bool:
    target = _find_ig_target()
    if target is None:
        log.error(
            f"no Instagram tab found on CDP at port {DEBUG_PORT}. "
            f"Open instagram.com in the debug Chrome and re-run."
        )
        return False

    ws_url = target.get("webSocketDebuggerUrl")
    if not ws_url:
        log.error(f"target has no webSocketDebuggerUrl: {target}")
        return False
    log.info(f"attaching to Instagram tab: {target.get('url')}")

    client = CDPClient(ws_url, log)
    try:
        # Domains we need. Page is enabled so navigation events drain;
        # DOM is required for setFileInputFiles; Runtime for JS.
        client.send("Page.enable")
        client.send("DOM.enable")
        client.send("Runtime.enable")

        # Quick logged-in check. If we can see the New-post icon we're good.
        try:
            logged_in = client.eval_js(
                "!!document.querySelector('svg[aria-label=\"New post\"]') || "
                "!!document.querySelector('svg[aria-label=\"Home\"]')",
                await_promise=False,
            )
        except Exception:
            logged_in = False
        if not logged_in:
            log.error(
                "not logged into Instagram in this tab. Log in manually in the "
                "Chrome window, wait for the home feed, then re-run."
            )
            return False
        log.info("already logged in")

        return _post_reel_cdp(client, video_path, caption, log, submit=submit)
    finally:
        client.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate caption/hashtags from the reel transcript and post to Instagram."
    )
    ap.add_argument("reel", type=Path, help="Path to the rendered reel mp4 in outputs/")
    ap.add_argument(
        "-c", "--config", type=Path, default=Path("config/default.yaml"),
        help="YAML config (default: config/default.yaml) — only the llm.* section is read",
    )
    ap.add_argument(
        "--caption", default=None,
        help="Skip the LLM and post this caption verbatim",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print the generated caption and exit without touching Chrome",
    )
    ap.add_argument(
        "--no-submit", action="store_true",
        help="Walk through the Instagram modal up to the caption step but stop before clicking Share",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("DEBUG" if args.verbose else "INFO").getChild("post")

    if not args.reel.exists():
        log.error(f"reel not found: {args.reel}")
        return 2
    if args.reel.suffix.lower() != ".mp4":
        log.warning(f"expected an .mp4, got {args.reel.suffix} — continuing anyway")

    cfg = load_config(args.config)

    if args.caption:
        caption = args.caption
        log.info("using user-provided caption (LLM skipped)")
    else:
        try:
            caption = generate_caption(args.reel, cfg, log)
        except Exception as e:
            log.error(f"caption generation failed: {e}")
            return 1

    print("\n----- caption -----")
    print(caption)
    print("-------------------\n")

    if args.dry_run:
        log.info("dry-run: not posting")
        return 0

    ok = _post_via_cdp(args.reel, caption, log, submit=not args.no_submit)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
