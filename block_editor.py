"""
Block Editor - describe an app, get a wired block graph, generate Tkinter code,
then run it headless to test it and let the model fix what broke.

Run:   python block_editor.py        (opens http://127.0.0.1:5177)
Deps:  pip install flask   (no llama-cpp here - models are served by GGUFllama)
Needs: GGUFllama running on the Local backend with a model loaded and its HTTP API started
       (right-click the GGUFllama title bar -> Start HTTP API).
Build: pyinstaller --onefile --name BlockEditor block_editor.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import webbrowser

from flask import Flask, Response, jsonify, request

APP_PORT = 5177
DEFAULT_BRIDGE = "http://127.0.0.1:5178"     # GGUFllama's HTTP API

# ---------------------------------------------------------------------------
# Block library - the single source of truth.
# The editor, the model schema, the validator and the generator all read this.
# Port types: "ui" (parent/child), "flow" (triggers), "text" (data).
# A text input that isn't wired falls back to the setting with the same name.
# ---------------------------------------------------------------------------


def S(key, default="", options=None, label=None):
    return {"key": key, "default": default, "options": options, "label": label or key}


LIB = {
    "Window": {
        "cat": "ui",
        "desc": "The app window. Exactly one per graph.",
        "in": {},
        "out": {"children": "ui"},
        "settings": [S("title", "My App"), S("width", "420"), S("height", "320")],
    },
    "Button": {
        "cat": "ui",
        "desc": "A clickable button. Fires 'click' when pressed.",
        "in": {"parent": "ui"},
        "out": {"click": "flow"},
        "settings": [S("text", "Click me")],
    },
    "Label": {
        "cat": "ui",
        "desc": "Shows text. Wire a text output into 'text' to display it.",
        "in": {"parent": "ui", "text": "text"},
        "out": {},
        "settings": [S("text", ""), S("size", "12")],
    },
    "Entry": {
        "cat": "ui",
        "desc": "A text box the user types in. 'value' is what they typed.",
        "in": {"parent": "ui"},
        "out": {"value": "text"},
        "settings": [S("default", "")],
    },
    "HttpGet": {
        "cat": "io",
        "desc": "Fetches a URL when run. 'body' is the page text.",
        "in": {"run": "flow", "url": "text"},
        "out": {"body": "text", "done": "flow"},
        "settings": [S("url", "https://example.com")],
    },
    "ReadFile": {
        "cat": "io",
        "desc": "Reads a text file when run. 'content' is the file text.",
        "in": {"run": "flow", "path": "text"},
        "out": {"content": "text", "done": "flow"},
        "settings": [S("path", "input.txt")],
    },
    "WriteFile": {
        "cat": "io",
        "desc": "Writes text to a file when run.",
        "in": {"run": "flow", "content": "text", "path": "text"},
        "out": {"done": "flow"},
        "settings": [S("path", "output.txt"), S("content", "")],
    },
    "Timer": {
        "cat": "logic",
        "desc": "Fires 'tick' after a delay, once or repeatedly.",
        "in": {},
        "out": {"tick": "flow"},
        "settings": [S("seconds", "5"), S("repeat", "no", ["no", "yes"])],
    },
    "If": {
        "cat": "logic",
        "desc": "Checks a text value and fires 'then' or 'else'.",
        "in": {"run": "flow", "value": "text"},
        "out": {"then": "flow", "else": "flow"},
        "settings": [
            S("op", "contains", ["contains", "equals", "not_empty", "is_empty"]),
            S("compare", ""),
        ],
    },
    "Print": {
        "cat": "logic",
        "desc": "Prints text to the console when run.",
        "in": {"run": "flow", "text": "text"},
        "out": {"done": "flow"},
        "settings": [S("text", "")],
    },
    "Start": {
        "cat": "logic",
        "desc": "Fires 'start' once, as soon as the app opens.",
        "in": {},
        "out": {"start": "flow"},
        "settings": [],
    },
    "JsonGet": {
        "cat": "logic",
        "desc": "Reads one field out of JSON text when run. Field is a dotted path such as items.0.name.",
        "in": {"run": "flow", "json": "text"},
        "out": {"value": "text", "done": "flow"},
        "settings": [S("field", "name"), S("json", "", label="json (used if nothing is wired)")],
    },
    "TextTransform": {
        "cat": "logic",
        "desc": "Changes text when run: upper, lower, title or strip (trim spaces).",
        "in": {"run": "flow", "text": "text"},
        "out": {"result": "text", "done": "flow"},
        "settings": [S("op", "upper", ["upper", "lower", "title", "strip"]), S("text", "")],
    },
    "Join": {
        "cat": "logic",
        "desc": "Joins text 'a' and text 'b' with a separator when run.",
        "in": {"run": "flow", "a": "text", "b": "text"},
        "out": {"result": "text", "done": "flow"},
        "settings": [S("a", ""), S("b", ""), S("sep", " ", label="separator (\\n = new line)")],
    },
    "Now": {
        "cat": "logic",
        "desc": "Gets the current date and time as text when run, using a strftime format.",
        "in": {"run": "flow"},
        "out": {"value": "text", "done": "flow"},
        "settings": [S("format", "%Y-%m-%d %H:%M:%S")],
    },
    "MessageBox": {
        "cat": "ui",
        "desc": "Shows a popup message when run. Needs no parent.",
        "in": {"run": "flow", "text": "text"},
        "out": {"done": "flow"},
        "settings": [S("title", "Message"), S("text", "")],
    },
}

# Blocks that start a chain of flow: they have a flow output but no flow input.
TRIGGERS = {n for n, b in LIB.items()
            if "flow" in b["out"].values() and "flow" not in b["in"].values()}

DEMO = {
    "blocks": [
        {"id": "win", "type": "Window", "settings": {"title": "URL fetcher", "width": "420", "height": "320"}},
        {"id": "btn1", "type": "Button", "settings": {"text": "Fetch"}},
        {"id": "get1", "type": "HttpGet", "settings": {"url": "https://example.com"}},
        {"id": "lbl1", "type": "Label", "settings": {"text": "", "size": "11"}},
    ],
    "wires": [
        {"from_block": "win", "from_port": "children", "to_block": "btn1", "to_port": "parent"},
        {"from_block": "win", "from_port": "children", "to_block": "lbl1", "to_port": "parent"},
        {"from_block": "btn1", "from_port": "click", "to_block": "get1", "to_port": "run"},
        {"from_block": "get1", "from_port": "body", "to_block": "lbl1", "to_port": "text"},
    ],
}


CLOCK_DEMO = {
    "blocks": [
        {"id": "win", "type": "Window", "settings": {"title": "Clock", "width": "300", "height": "140"}},
        {"id": "tim1", "type": "Timer", "settings": {"seconds": "1", "repeat": "yes"}},
        {"id": "now1", "type": "Now", "settings": {"format": "%H:%M:%S"}},
        {"id": "lbl1", "type": "Label", "settings": {"text": "", "size": "24"}},
    ],
    "wires": [
        {"from_block": "win", "from_port": "children", "to_block": "lbl1", "to_port": "parent"},
        {"from_block": "tim1", "from_port": "tick", "to_block": "now1", "to_port": "run"},
        {"from_block": "now1", "from_port": "value", "to_block": "lbl1", "to_port": "text"},
    ],
}


def setting_of(block, key):
    v = (block.get("settings") or {}).get(key)
    if v is None:
        for s in LIB[block["type"]]["settings"]:
            if s["key"] == key:
                return s["default"]
        return ""
    return str(v)


def num(s, default):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Validation and repair
# ---------------------------------------------------------------------------


def validate(graph):
    errs = []
    ids = {}
    for b in graph.get("blocks") or []:
        t, bid = b.get("type"), b.get("id")
        if t not in LIB:
            errs.append(f"Unknown block type '{t}'.")
            continue
        if not bid:
            errs.append(f"A {t} block has no id.")
            continue
        if bid in ids:
            errs.append(f"Duplicate id '{bid}'.")
        ids[bid] = b

    if sum(1 for b in ids.values() if b["type"] == "Window") != 1:
        errs.append("The graph needs exactly one Window block.")

    wired_inputs = set()
    for w in graph.get("wires") or []:
        fb, fp = w.get("from_block"), w.get("from_port")
        tb, tp = w.get("to_block"), w.get("to_port")
        if fb not in ids or tb not in ids:
            errs.append(f"Wire {fb}.{fp} -> {tb}.{tp} points at a block that doesn't exist.")
            continue
        fo, ti = LIB[ids[fb]["type"]]["out"], LIB[ids[tb]["type"]]["in"]
        if fp not in fo:
            errs.append(f"'{fb}' ({ids[fb]['type']}) has no output '{fp}'. Its outputs: {', '.join(fo) or 'none'}.")
            continue
        if tp not in ti:
            errs.append(f"'{tb}' ({ids[tb]['type']}) has no input '{tp}'. Its inputs: {', '.join(ti) or 'none'}.")
            continue
        if fo[fp] != ti[tp]:
            errs.append(f"Type mismatch: {fb}.{fp} is {fo[fp]} but {tb}.{tp} is {ti[tp]}.")
            continue
        if ti[tp] != "flow":
            if (tb, tp) in wired_inputs:
                errs.append(f"{tb}.{tp} has more than one wire into it.")
            wired_inputs.add((tb, tp))

    for b in ids.values():
        if "parent" in LIB[b["type"]]["in"] and (b["id"], "parent") not in wired_inputs:
            errs.append(f"'{b['id']}' ({b['type']}) needs its parent wired from the Window's children output.")
    return errs


def repair(graph):
    """Fix the mistakes small models make so retries are spent on real problems."""
    out = {"blocks": [], "wires": []}
    idmap, used = {}, set()
    for b in graph.get("blocks") or []:
        if not isinstance(b, dict) or b.get("type") not in LIB:
            continue
        raw = str(b.get("id") or b["type"].lower())
        nid = re.sub(r"\W+", "_", raw).strip("_").lower() or b["type"].lower()
        if nid[0].isdigit():
            nid = "b_" + nid
        base, n = nid, 2
        while nid in used:
            nid = f"{base}{n}"
            n += 1
        used.add(nid)
        idmap.setdefault(raw, nid)
        given = b.get("settings") or {}
        settings = {}
        for s in LIB[b["type"]]["settings"]:
            v = str(given.get(s["key"], s["default"]))
            if s["options"] and v not in s["options"]:
                v = s["default"]
            settings[s["key"]] = v
        nb = {"id": nid, "type": b["type"], "settings": settings}
        for k in ("x", "y"):
            if k in b:
                nb[k] = b[k]
        out["blocks"].append(nb)

    # Keep exactly one Window.
    wins = [b for b in out["blocks"] if b["type"] == "Window"]
    if not wins:
        out["blocks"].insert(0, {
            "id": "window", "type": "Window",
            "settings": {s["key"]: s["default"] for s in LIB["Window"]["settings"]},
        })
        wins = [out["blocks"][0]]
    keep = wins[0]["id"]
    drop = {b["id"] for b in wins[1:]}
    out["blocks"] = [b for b in out["blocks"] if b["id"] not in drop]
    byid = {b["id"]: b for b in out["blocks"]}

    seen, wired_inputs = set(), set()
    for w in graph.get("wires") or []:
        if not isinstance(w, dict):
            continue
        fb = idmap.get(str(w.get("from_block")))
        tb = idmap.get(str(w.get("to_block")))
        fp, tp = w.get("from_port"), w.get("to_port")
        if fb in drop:
            fb = keep
        if tb in drop:
            continue
        if fb not in byid or tb not in byid:
            continue
        ft, tt = LIB[byid[fb]["type"]], LIB[byid[tb]["type"]]
        if fp not in ft["out"] and fp in ft["in"] and tp in tt["out"]:
            fb, fp, tb, tp = tb, tp, fb, fp  # model wired it backwards
            ft, tt = LIB[byid[fb]["type"]], LIB[byid[tb]["type"]]
        if fp not in ft["out"] or tp not in tt["in"] or ft["out"][fp] != tt["in"][tp]:
            continue
        key = (fb, fp, tb, tp)
        if key in seen:
            continue
        if tt["in"][tp] != "flow":
            if (tb, tp) in wired_inputs:
                continue
            wired_inputs.add((tb, tp))
        seen.add(key)
        out["wires"].append({"from_block": fb, "from_port": fp, "to_block": tb, "to_port": tp})

    for b in out["blocks"]:
        if "parent" in LIB[b["type"]]["in"] and (b["id"], "parent") not in wired_inputs:
            out["wires"].append({"from_block": keep, "from_port": "children",
                                 "to_block": b["id"], "to_port": "parent"})
    return out


# ---------------------------------------------------------------------------
# Code generator: graph -> Tkinter .py
# ---------------------------------------------------------------------------


TEST_MARKER = "@@BLOCK_TEST@@"

# Runs first in a test build: makes network, file and popup calls harmless, and records problems.
TEST_PRELUDE = """import json as _tj, io as _tio, traceback as _tb, urllib.request as _ur
_T = {'errors': [], 'stubbed': [], 'popups': []}
_VFS = {}
class _Resp:
    def __init__(self, b): self._b = b
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False
def _fake_urlopen(req, timeout=None):
    _T['stubbed'].append('GET ' + str(getattr(req, 'full_url', req)))
    return _Resp(b'{"title": "Example Domain", "name": "Test", "value": "42", "items": [{"name": "first"}]}')
_ur.urlopen = _fake_urlopen
class _VW(_tio.StringIO):
    def __init__(self, path, mode):
        super().__init__(); self._p = path; self._m = mode
    def close(self):
        if not self.closed:
            _VFS[self._p] = (_VFS.get(self._p, '') if 'a' in self._m else '') + self.getvalue()
            _T['stubbed'].append('WRITE ' + str(self._p))
        super().close()
def open(path, mode='r', *a, **k):
    if 'w' in mode or 'a' in mode:
        return _VW(path, mode)
    if path in _VFS:
        return _tio.StringIO(_VFS[path])
    _T['stubbed'].append('READ ' + str(path))
    return _tio.StringIO('sample file text')
try:
    messagebox.showinfo = lambda title='', msg='', **k: _T['popups'].append(str(msg))
except NameError:
    pass
def _who(tb):
    name = ''
    for fr in _tb.extract_tb(tb):
        head, _, rest = fr.name.partition('_')
        if rest and head in ('click', 'tick', 'run', 'refresh', 'start'):
            name = rest
    return _IDS.get(name, name)
def _fail(bid, et, ev):
    _T['errors'].append({'block': bid, 'error': getattr(et, '__name__', 'Error') + ': ' + str(ev)})
def _report(et, ev, tb):
    _fail(_who(tb), et, ev)
def _step(bid, kind, f):
    try:
        f()
    except Exception as e:
        _fail(_who(e.__traceback__) or bid, type(e), e)
    root.update()"""


def generate(graph, test=False):
    blocks_list = graph.get("blocks") or []
    blocks = {b["id"]: b for b in blocks_list}
    wires = [w for w in (graph.get("wires") or [])
             if w.get("from_block") in blocks and w.get("to_block") in blocks]
    wins = [b for b in blocks_list if b["type"] == "Window"]
    if len(wins) != 1:
        raise ValueError("The graph needs exactly one Window block.")
    win = wins[0]

    def ident(b):
        return re.sub(r"\W", "_", b["id"])

    def widget(b):
        return "w_" + ident(b)

    def fn(prefix, b):
        return prefix + "_" + ident(b)

    def incoming(bid, port):
        for w in wires:
            if w["to_block"] == bid and w["to_port"] == port:
                return w
        return None

    def outgoing(bid, port):
        bid = bid["id"] if isinstance(bid, dict) else bid
        return [w for w in wires if w["from_block"] == bid and w["from_port"] == port]

    def key(bid, port):
        return repr(f"{bid}.{port}")

    def expr(b, port):
        w = incoming(b["id"], port)
        if w:
            s = blocks[w["from_block"]]
            if s["type"] == "Entry" and w["from_port"] == "value":
                return widget(s) + ".get()"
            return "S.get(" + key(s["id"], w["from_port"]) + ", '')"
        return repr(setting_of(b, port))

    def refresh_calls(b, port):
        calls = []
        for w in outgoing(b, port):
            t = blocks[w["to_block"]]
            if t["type"] == "Label" and w["to_port"] == "text":
                calls.append(fn("refresh", t) + "()")
        return calls

    def flow_calls(b, port):
        calls = []
        for w in outgoing(b, port):
            t = blocks[w["to_block"]]
            if LIB[t["type"]]["in"].get(w["to_port"]) == "flow":
                calls.append(fn("run", t) + "()")
        return calls

    def ind(lines, n=1):
        return ["    " * n + ln for ln in lines]

    types = {b["type"] for b in blocks_list}
    width = int(num(setting_of(win, "width"), 420))
    height = int(num(setting_of(win, "height"), 320))
    L = ["import tkinter as tk"]
    if "HttpGet" in types:
        L.append("import urllib.request")
    if "JsonGet" in types:
        L.append("import json")
    if "Now" in types:
        L.append("import datetime")
    if "MessageBox" in types:
        L.append("from tkinter import messagebox")
    if test:
        L += TEST_PRELUDE.split("\n")
    L += ["", "S = {}  # values passed between blocks", "",
          "root = tk.Tk()", "root.title(" + repr(setting_of(win, "title")) + ")",
          f"root.geometry('{width}x{height}')"]
    if test:
        L += ["root.withdraw()", "root.report_callback_exception = _report"]
    L.append("")

    starts = []
    for b in blocks_list:
        t = b["type"]
        if t == "Button":
            L.append(f"def {fn('click', b)}():")
            L += ind(flow_calls(b, "click") or ["pass"])
            L.append("")
        elif t == "Timer":
            ms = int(max(0.1, num(setting_of(b, "seconds"), 5)) * 1000)
            name = fn("tick", b)
            body = flow_calls(b, "tick")
            if setting_of(b, "repeat") == "yes":
                body.append(f"root.after({ms}, {name})")
            L.append(f"def {name}():")
            L += ind(body or ["pass"])
            L.append("")
            starts.append(f"root.after({ms}, {name})")
        elif t == "HttpGet":
            k = key(b["id"], "body")
            L += [f"def {fn('run', b)}():",
                  f"    url = str({expr(b, 'url')}).strip()",
                  "    if not url.startswith(('http://', 'https://')):",
                  "        url = 'http://' + url",
                  "    try:",
                  "        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})",
                  "        with urllib.request.urlopen(req, timeout=15) as r:",
                  f"            S[{k}] = r.read().decode('utf-8', 'replace')",
                  "    except Exception as e:",
                  f"        S[{k}] = 'Error: ' + str(e)"]
            L += ind(refresh_calls(b, "body") + flow_calls(b, "done"))
            L.append("")
        elif t == "ReadFile":
            k = key(b["id"], "content")
            L += [f"def {fn('run', b)}():",
                  "    try:",
                  f"        with open(str({expr(b, 'path')}), encoding='utf-8', errors='replace') as f:",
                  f"            S[{k}] = f.read()",
                  "    except Exception as e:",
                  f"        S[{k}] = 'Error: ' + str(e)"]
            L += ind(refresh_calls(b, "content") + flow_calls(b, "done"))
            L.append("")
        elif t == "WriteFile":
            L += [f"def {fn('run', b)}():",
                  "    try:",
                  f"        with open(str({expr(b, 'path')}), 'w', encoding='utf-8') as f:",
                  f"            f.write(str({expr(b, 'content')}))",
                  "    except Exception as e:",
                  "        print('WriteFile failed:', e)"]
            L += ind(flow_calls(b, "done"))
            L.append("")
        elif t == "If":
            op = setting_of(b, "op")
            cond = {"contains": "cmp in v", "equals": "v == cmp",
                    "not_empty": "v.strip() != ''", "is_empty": "v.strip() == ''"}.get(op, "cmp in v")
            L += [f"def {fn('run', b)}():",
                  f"    v = str({expr(b, 'value')})",
                  f"    cmp = {setting_of(b, 'compare')!r}",
                  f"    if {cond}:"]
            L += ind(flow_calls(b, "then") or ["pass"], 2)
            L.append("    else:")
            L += ind(flow_calls(b, "else") or ["pass"], 2)
            L.append("")
        elif t == "Print":
            L.append(f"def {fn('run', b)}():")
            L.append(f"    print(str({expr(b, 'text')}), flush=True)")
            L += ind(flow_calls(b, "done"))
            L.append("")
        elif t == "Start":
            name = fn("start", b)
            L.append(f"def {name}():")
            L += ind(flow_calls(b, "start") or ["pass"])
            L.append("")
            starts.append(f"root.after(100, {name})")
        elif t == "JsonGet":
            k = key(b["id"], "value")
            L += [f"def {fn('run', b)}():",
                  f"    field = {setting_of(b, 'field')!r}",
                  "    try:",
                  f"        d = json.loads(str({expr(b, 'json')}))",
                  "        for part in field.split('.'):",
                  "            part = part.strip()",
                  "            if not part:",
                  "                continue",
                  "            d = d[int(part)] if isinstance(d, list) else d[part]",
                  f"        S[{k}] = d if isinstance(d, str) else json.dumps(d)",
                  "    except json.JSONDecodeError:",
                  f"        S[{k}] = 'Error: input is not valid JSON'",
                  "    except (KeyError, IndexError, ValueError, TypeError):",
                  f"        S[{k}] = 'Error: field not found: ' + field"]
            L += ind(refresh_calls(b, "value") + flow_calls(b, "done"))
            L.append("")
        elif t == "TextTransform":
            k = key(b["id"], "result")
            call = {"upper": "upper()", "lower": "lower()", "title": "title()",
                    "strip": "strip()"}.get(setting_of(b, "op"), "upper()")
            L += [f"def {fn('run', b)}():",
                  "    try:",
                  f"        S[{k}] = str({expr(b, 'text')}).{call}",
                  "    except Exception as e:",
                  f"        S[{k}] = 'Error: ' + str(e)"]
            L += ind(refresh_calls(b, "result") + flow_calls(b, "done"))
            L.append("")
        elif t == "Join":
            k = key(b["id"], "result")
            sep = setting_of(b, "sep").replace("\\n", "\n")
            L += [f"def {fn('run', b)}():",
                  "    try:",
                  f"        S[{k}] = str({expr(b, 'a')}) + {sep!r} + str({expr(b, 'b')})",
                  "    except Exception as e:",
                  f"        S[{k}] = 'Error: ' + str(e)"]
            L += ind(refresh_calls(b, "result") + flow_calls(b, "done"))
            L.append("")
        elif t == "Now":
            k = key(b["id"], "value")
            L += [f"def {fn('run', b)}():",
                  "    try:",
                  f"        S[{k}] = datetime.datetime.now().strftime({setting_of(b, 'format')!r})",
                  "    except Exception as e:",
                  f"        S[{k}] = 'Error: ' + str(e)"]
            L += ind(refresh_calls(b, "value") + flow_calls(b, "done"))
            L.append("")
        elif t == "MessageBox":
            L += [f"def {fn('run', b)}():",
                  f"    messagebox.showinfo({setting_of(b, 'title')!r}, str({expr(b, 'text')}))"]
            L += ind(flow_calls(b, "done"))
            L.append("")
        elif t == "Label":
            L.append(f"def {fn('refresh', b)}():")
            L.append(f"    {widget(b)}.config(text=str({expr(b, 'text')}))")
            L.append("")

    wrap = max(100, width - 30)
    for b in blocks_list:
        t, w = b["type"], widget(b)
        if t == "Button":
            L += [f"{w} = tk.Button(root, text={setting_of(b, 'text')!r}, command={fn('click', b)}, font=('Segoe UI', 11))",
                  f"{w}.pack(padx=10, pady=6)"]
        elif t == "Label":
            size = int(num(setting_of(b, "size"), 12))
            L += [f"{w} = tk.Label(root, text={setting_of(b, 'text')!r}, font=('Segoe UI', {size}), "
                  f"wraplength={wrap}, justify='left')",
                  f"{w}.pack(padx=10, pady=6, anchor='w')"]
        elif t == "Entry":
            L += [f"{w} = tk.Entry(root, font=('Segoe UI', 12))",
                  f"{w}.insert(0, {setting_of(b, 'default')!r})",
                  f"{w}.pack(fill='x', padx=10, pady=6)"]
            refreshers = refresh_calls(b, "value")
            if refreshers:
                L.append(f"{w}.bind('<KeyRelease>', lambda e: ({', '.join(refreshers)}))")
    L.append("")
    if not test:
        L += starts
        L.append("root.mainloop()")
        return "\n".join(L) + "\n"

    # Test harness: no mainloop. Fire every trigger once, then print a JSON report.
    idmap = {ident(b): b["id"] for b in blocks_list}
    L.append("_IDS = " + repr(idmap))
    for b in blocks_list:
        if b["type"] in ("Start", "Timer"):
            f = fn("start" if b["type"] == "Start" else "tick", b)
            L.append(f"_step({b['id']!r}, {b['type'].lower()!r}, {f})")
    for b in blocks_list:
        if b["type"] == "Button":
            L.append(f"_step({b['id']!r}, 'click', {widget(b)}.invoke)")
    L.append("root.update()")
    labels = [b for b in blocks_list if b["type"] == "Label"]
    L.append("_T['S'] = {k: str(v)[:300] for k, v in S.items()}")
    L.append("_T['labels'] = {" + ", ".join(f"{b['id']!r}: {widget(b)}.cget('text')" for b in labels) + "}")
    L.append("print('" + TEST_MARKER + "' + _tj.dumps(_T), flush=True)")
    L.append("root.destroy()")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Model schema and prompt
# ---------------------------------------------------------------------------


def build_schema():
    ports = sorted({p for b in LIB.values() for p in list(b["in"]) + list(b["out"])})
    variants = []
    for name, b in LIB.items():
        props = {}
        for s in b["settings"]:
            props[s["key"]] = {"enum": s["options"]} if s["options"] else {"type": "string"}
        variants.append({
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "type": {"const": name},
                "settings": {"type": "object", "properties": props, "additionalProperties": False},
            },
            "required": ["id", "type", "settings"],
            "additionalProperties": False,
        })
    wire = {
        "type": "object",
        "properties": {
            "from_block": {"type": "string"}, "from_port": {"enum": ports},
            "to_block": {"type": "string"}, "to_port": {"enum": ports},
        },
        "required": ["from_block", "from_port", "to_block", "to_port"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "blocks": {"type": "array", "items": {"anyOf": variants}, "minItems": 1},
            "wires": {"type": "array", "items": wire},
        },
        "required": ["blocks", "wires"],
        "additionalProperties": False,
    }


def system_prompt():
    lines = ["You design small Tkinter apps as block graphs. Reply with JSON only.", "", "Blocks:"]
    for name, b in LIB.items():
        ins = ", ".join(f"{p}({t})" for p, t in b["in"].items()) or "none"
        outs = ", ".join(f"{p}({t})" for p, t in b["out"].items()) or "none"
        sets = ", ".join(s["key"] for s in b["settings"]) or "none"
        lines.append(f"- {name}: {b['desc']} Inputs: {ins}. Outputs: {outs}. Settings: {sets}.")
    lines += [
        "",
        "Rules:",
        "1. Exactly one Window block.",
        "2. Every Button, Label and Entry has its 'parent' input wired from the Window's 'children' output.",
        "3. A wire goes from an output port to an input port of the same type (ui, flow or text).",
        "4. flow ports trigger actions (Button.click -> HttpGet.run). text ports carry data (HttpGet.body -> Label.text).",
        "5. Give each block a short unique lowercase id such as btn1 or lbl1.",
        "6. Use only the blocks and ports listed above. Every setting value is a string.",
        "7. Blocks with a 'run' input do nothing until a flow wire reaches it. Use Start.start for things that "
        "happen when the app opens, Button.click for user actions and Timer.tick for repeats.",
        "8. To do several steps in order, wire each block's 'done' output into the next block's 'run' input.",
        "9. A text output can feed several inputs. Each text input accepts only one wire.",
        "",
        "Example request: a window with a button that fetches a URL and shows the result",
        "Example answer:",
        json.dumps(DEMO),
        "",
        "Example request: a clock that updates every second",
        "Example answer:",
        json.dumps(CLOCK_DEMO),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model backend: GGUFllama's HTTP API (it owns the model, the VRAM guard and the loading)
# ---------------------------------------------------------------------------

def bridge_request(method, path, cfg=None, payload=None, timeout=600, base=None):
    base = (base or (cfg or {}).get("bridge_url") or DEFAULT_BRIDGE).rstrip("/")
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode("utf-8")).get("error")
        except Exception:
            msg = None
        raise RuntimeError(msg or f"GGUFllama returned HTTP {e.code}.")
    except TimeoutError:
        raise RuntimeError("GGUFllama took too long to reply.")
    except OSError:
        raise RuntimeError(f"Can't reach GGUFllama at {base}. Open GGUFllama, then right-click its "
                           "title bar and choose Start HTTP API.")


def call_local(messages, schema, cfg):
    out = bridge_request("POST", "/api/chat", cfg, {
        "messages": messages, "schema": schema,
        "temperature": float(cfg.get("temperature", 0.2)), "max_tokens": 2048})
    return out["content"]


MAX_ATTEMPTS = 4


def _source_type(graph, bid, port):
    """Type of the block wired into bid.port, or None."""
    types = {b["id"]: b["type"] for b in graph.get("blocks") or []}
    for w in graph.get("wires") or []:
        if w.get("to_block") == bid and w.get("to_port") == port:
            return types.get(w.get("from_block"))
    return None


def _untriggered(graph):
    """Blocks with a 'run' input that no Button/Timer/Start chain ever reaches."""
    blocks = {b["id"]: b for b in graph.get("blocks") or []}
    wires = graph.get("wires") or []
    fired = {i for i, b in blocks.items() if b["type"] in TRIGGERS}
    stack = list(fired)
    while stack:
        cur = stack.pop()
        outs = LIB[blocks[cur]["type"]]["out"]
        for w in wires:
            if w.get("from_block") == cur and outs.get(w.get("from_port")) == "flow":
                t = w.get("to_block")
                if t in blocks and t not in fired:
                    fired.add(t)
                    stack.append(t)
    return [b for b in blocks.values() if "run" in LIB[b["type"]]["in"] and b["id"] not in fired]


def test_graph(graph, timeout=20):
    """Build the app in test mode and run it headless with network/files/popups simulated.

    Returns {ran, ok, failures, notes, labels, stubbed, skipped}. Only 'failures' should be sent
    back to the model; 'notes' are things the simulated data can't prove either way.
    """
    res = {"ran": False, "ok": True, "failures": [], "notes": [], "labels": {},
           "stubbed": [], "skipped": None}

    def fail(msg):
        res["failures"].append(msg)
        res["ok"] = False

    blocks = {b["id"]: b for b in graph.get("blocks") or []}
    for b in _untriggered(graph):
        fail(f"{b['type']} block '{b['id']}' never runs: nothing triggers its 'run' input. Wire a "
             "Button 'click', Timer 'tick' or Start 'start' into it, or another block's 'done'.")

    try:
        code = generate(graph, test=True)
    except Exception as e:
        fail(f"Code generation failed: {e}")
        return res
    try:
        compile(code, "app_test.py", "exec")
    except SyntaxError as e:
        fail(f"The generated code has a syntax error on line {e.lineno}: {e.msg}.")
        return res

    if getattr(sys, "frozen", False):
        py = shutil.which("python") or shutil.which("py")
    else:
        py = sys.executable
    if not py:
        res["skipped"] = "Python wasn't found on PATH, so the app couldn't be test-run."
        return res
    cmd = [py]
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        xvfb = shutil.which("xvfb-run")
        if not xvfb:
            res["skipped"] = "No display is available, so the app couldn't be test-run."
            return res
        cmd = [xvfb, "-a", py]

    tmp = tempfile.mkdtemp(prefix="block_test_")
    path = os.path.join(tmp, "app_test.py")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            r = subprocess.run(cmd + [path], capture_output=True, text=True, timeout=timeout,
                               cwd=tmp, creationflags=flags)
        except subprocess.TimeoutExpired:
            fail(f"The app didn't finish its test run within {timeout} seconds. "
                 "Something in it may be stuck or waiting.")
            return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    res["ran"] = True
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith(TEST_MARKER)), None)
    if line is None:
        both = (r.stderr or "") + "\n" + (r.stdout or "")   # xvfb-run merges stderr into stdout
        last = next((ln for ln in reversed(both.splitlines()) if ln.strip()), "")
        fail("The app crashed before its test finished" + (f": {last.strip()}" if last else "."))
        return res
    try:
        data = json.loads(line[len(TEST_MARKER):])
    except ValueError:
        fail("The test report couldn't be read.")
        return res

    seen = set()
    for e in data.get("errors") or []:
        bid = e.get("block") or ""
        typ = blocks.get(bid, {}).get("type", "block")
        msg = f"{typ} block '{bid}' raised {e.get('error')}" if bid else f"The app raised {e.get('error')}"
        if msg not in seen:
            seen.add(msg)
            fail(msg)
    for k, v in (data.get("S") or {}).items():
        if not str(v).startswith("Error:"):
            continue
        bid = k.split(".", 1)[0]
        typ = blocks.get(bid, {}).get("type", "block")
        upstream = _source_type(graph, bid, "json") if typ == "JsonGet" else None
        if typ == "JsonGet" and upstream in ("HttpGet", "ReadFile"):
            res["notes"].append(f"JsonGet '{bid}' said '{v}', but the test used sample data, "
                                "so this can't be checked without the real response.")
        else:
            fail(f"{typ} block '{bid}' produced: {v}")
    res["labels"] = data.get("labels") or {}
    for lid, text in res["labels"].items():
        if text == "" and _source_type(graph, lid, "text"):
            res["notes"].append(f"Label '{lid}' is still empty after the test.")
    res["stubbed"] = sorted(set(data.get("stubbed") or []))
    res["popups"] = data.get("popups") or []
    return res


def _test_feedback(t):
    lines = ["I built your graph into an app and ran it (network, files and popups were simulated with "
             "sample data), and it had problems:"]
    lines += ["- " + f for f in t["failures"]]
    if t["labels"]:
        lines.append("After firing every trigger, the labels showed: " +
                     ", ".join(f"{k} = {v!r}" for k, v in t["labels"].items()) + ".")
    lines.append("Fix the wiring or settings and return the full corrected graph as JSON.")
    return "\n".join(lines)


def ask_model(prompt, cfg):
    schema = build_schema()
    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": prompt}]
    call = call_local
    do_test = cfg.get("auto_test", True) is not False
    graph, errs, test = {"blocks": [], "wires": []}, ["No graph returned."], None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        text = call(messages, schema, cfg)
        test = None
        try:
            graph = repair(json.loads(text))
            errs = validate(graph)
        except (ValueError, TypeError, AttributeError) as e:
            errs = [f"The output wasn't a valid graph ({e})."]
        if not errs and do_test:
            test = test_graph(graph)
            if test["failures"]:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": _test_feedback(test)})
                if attempt < MAX_ATTEMPTS:
                    continue
                return {"graph": graph, "attempts": attempt, "errors": [], "test": test}
        if not errs:
            return {"graph": graph, "attempts": attempt, "errors": [], "test": test}
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content":
                         "That graph has errors:\n- " + "\n- ".join(errs) +
                         "\nReturn the full corrected graph as JSON."})
    return {"graph": graph, "attempts": MAX_ATTEMPTS, "errors": errs, "test": test}


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.get("/api/library")
def api_library():
    return jsonify({"lib": LIB, "demo": DEMO, "default_bridge": DEFAULT_BRIDGE})


@app.get("/api/status")
def api_status():
    try:
        d = bridge_request("GET", "/api/status", timeout=2, base=request.args.get("bridge") or None)
    except RuntimeError as e:
        return jsonify({"reachable": False, "loaded": False, "reason": str(e)})
    return jsonify({"reachable": True, "loaded": bool(d.get("loaded")),
                    "model": d.get("model"), "reason": d.get("reason")})


@app.post("/api/generate")
def api_generate():
    graph = (request.get_json(silent=True) or {}).get("graph") or {}
    errors = validate(graph)
    try:
        code = generate(graph)
    except Exception as e:
        code = f"# Fix the problems above to generate code.\n# ({e})\n"
    return jsonify({"code": code, "errors": errors})


@app.post("/api/test")
def api_test():
    graph = (request.get_json(silent=True) or {}).get("graph") or {}
    errors = validate(graph)
    if errors:
        return jsonify({"error": "Fix the graph problems first: " + errors[0]}), 400
    try:
        return jsonify(test_graph(graph))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/ask")
def api_ask():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    cfg = data.get("cfg") or {}
    if not prompt:
        return jsonify({"error": "Describe the app you want first."}), 400
    try:
        return jsonify(ask_model(prompt, cfg))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/run")
def api_run():
    code = (request.get_json(silent=True) or {}).get("code") or ""
    path = os.path.join(tempfile.gettempdir(), f"block_app_{int(time.time())}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    if getattr(sys, "frozen", False):
        py = shutil.which("python") or shutil.which("py")
    else:
        py = sys.executable
    if not py:
        return jsonify({"ok": False, "message": "Python wasn't found on PATH, so the app can't run. "
                                                 "Save the .py and run it yourself."})
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen([py, path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, creationflags=flags)
    try:
        out, _ = proc.communicate(timeout=1.5)
        return jsonify({"ok": proc.returncode == 0, "message": "The app exited.", "output": out or ""})
    except subprocess.TimeoutExpired:
        return jsonify({"ok": True, "message": "App is running in its own window.", "output": ""})


def main():
    url = f"http://127.0.0.1:{APP_PORT}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print("Block Editor running at", url)
    app.run(host="127.0.0.1", port=APP_PORT, debug=False, threaded=True)


# ---------------------------------------------------------------------------
# Front end (vanilla HTML/CSS/JS, embedded so PyInstaller bundles one file)
# ---------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Block Editor</title>
<style>
:root{--bg:#14161a;--panel:#1b1e24;--panel2:#23272f;--line:#2d323b;--text:#e6e8ec;--dim:#9aa3b2;
--flow:#e9edf3;--txt:#38bdf8;--ui:#8b93a7;--accent:#38bdf8;--ok:#34d399;--warn:#fbbf24;--bad:#f87171}
*{box-sizing:border-box}
body{margin:0;font:14px/1.45 "Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--text);
height:100vh;display:flex;flex-direction:column}
button,input,select{font:inherit;color:var(--text)}
button{background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:6px 12px;cursor:pointer}
button:hover{border-color:#4a5262}
button.primary{background:var(--accent);border-color:var(--accent);color:#06222e;font-weight:600}
input[type=text],input[type=number],select{background:#12141a;border:1px solid var(--line);border-radius:6px;
padding:7px 10px;width:100%}
input:focus,select:focus,button:focus-visible{outline:2px solid var(--accent);outline-offset:0}
header{display:flex;align-items:center;gap:10px;padding:10px 14px;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0;font-weight:600;margin-right:auto}
.pill{font-size:12px;color:var(--dim);display:flex;align-items:center;gap:6px}
.dot{width:8px;height:8px;border-radius:50%;background:#555}
.dot.on{background:var(--ok)}
#askRow{display:flex;gap:8px;padding:10px 14px;background:var(--panel);border-bottom:1px solid var(--line)}
#status{padding:0 14px;font-size:13px;display:none}
#status.show{display:block;margin:8px 14px 0;padding:8px 12px;border-radius:6px}
#status.ok{background:#12352b;color:var(--ok)}
#status.warn{background:#3b3010;color:var(--warn)}
#status.bad{background:#3d1a1a;color:var(--bad)}
#status.busy{background:var(--panel2);color:var(--dim)}
#status ul{margin:4px 0 0;padding-left:18px}
#app{flex:1;display:grid;grid-template-columns:minmax(0,1fr) 360px;min-height:0;margin-top:8px}
#left{display:flex;flex-direction:column;min-height:0;min-width:0}
#palette,#bar{display:flex;flex-wrap:wrap;gap:6px;padding:6px 14px}
.chip{font-size:12px;padding:3px 10px;border-radius:999px;border:1px solid transparent}
.chip.ui{background:#312e5a;color:#c9c4ff}.chip.io{background:#0f3a35;color:#7ee8d6}.chip.logic{background:#43310d;color:#fcd47a}
#bar button{font-size:12px;padding:4px 10px}
#canvas{flex:1;position:relative;overflow:hidden;min-height:320px;touch-action:none;cursor:grab;
background-image:radial-gradient(#2a2f38 1px,transparent 1px);background-size:24px 24px;border-top:1px solid var(--line)}
#canvas.panning{cursor:grabbing}
#world{position:absolute;left:0;top:0;transform-origin:0 0}
#wires{position:absolute;left:0;top:0;width:1px;height:1px;overflow:visible;pointer-events:none}
.wire{fill:none;stroke-width:2.2;pointer-events:none}
.wire.t-flow{stroke:var(--flow)}.wire.t-text{stroke:var(--txt)}.wire.t-ui{stroke:#64748b;stroke-dasharray:5 4;stroke-width:1.6}
.wire.sel{stroke:var(--warn);stroke-width:3.2}
.hit{fill:none;stroke:transparent;stroke-width:16;pointer-events:stroke;cursor:pointer}
.node{position:absolute;background:var(--panel);border:1px solid var(--line);border-radius:8px;user-select:none}
.node.sel{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.head{height:30px;display:flex;align-items:center;justify-content:space-between;padding:0 10px;border-radius:7px 7px 0 0;cursor:move;font-weight:600;font-size:13px}
.head .nid{font-weight:400;font-size:11px;opacity:.7}
.cat-ui .head{background:#312e5a;color:#d4d0ff}.cat-io .head{background:#0f3a35;color:#a7f0e3}.cat-logic .head{background:#43310d;color:#fde3a5}
.prow{height:26px;position:relative;display:flex;justify-content:space-between;align-items:center;padding:0 14px;font-size:12px;color:var(--text)}
.sub{height:26px;padding:0 10px;font-size:12px;color:var(--dim);display:flex;align-items:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border-top:1px solid var(--line);margin-top:6px}
.port{position:absolute;top:6px;width:14px;height:14px;border-radius:50%;background:var(--panel);border:2px solid var(--c);cursor:crosshair}
.port::before{content:"";position:absolute;inset:-7px}
.port.in{left:-7px}.port.out{right:-7px}
.port.t-flow{--c:var(--flow)}.port.t-text{--c:var(--txt)}.port.t-ui{--c:#7b8499}
.port.out{background:var(--c)}
.port.ok{box-shadow:0 0 0 4px rgba(56,189,248,.35)}
#side{border-left:1px solid var(--line);background:var(--panel);display:flex;flex-direction:column;min-height:0;overflow:auto}
.sec{padding:14px;border-bottom:1px solid var(--line)}
.ptitle{font-weight:600;font-size:15px}.dim{color:var(--dim);font-weight:400;font-size:12px}
.pdesc{color:var(--dim);font-size:13px;margin:4px 0 12px}
.field{display:block;margin-bottom:10px}.field span{display:block;font-size:12px;color:var(--dim);margin-bottom:3px}
.ports{font-size:12px;color:var(--dim);margin:6px 0 12px}
button.danger{color:var(--bad)}
#codeHead{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
#codeHead b{margin-right:auto;font-size:13px}
#codeHead button{font-size:12px;padding:4px 10px}
#codeErrors{display:none;margin:10px 0 0;padding:8px 8px 8px 24px;background:#3b3010;color:var(--warn);border-radius:6px;font-size:12px}
#code{margin:10px 0 0;padding:10px;background:#101216;border:1px solid var(--line);border-radius:6px;font:12px/1.55 Consolas,"Cascadia Mono",monospace;overflow:auto;white-space:pre;max-height:380px}
#runOut{margin:8px 0 0;font-size:12px;color:var(--dim);white-space:pre-wrap;font-family:Consolas,monospace}
dialog{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:18px;width:min(420px,92vw)}
dialog::backdrop{background:rgba(0,0,0,.6)}
dialog h2{margin:0 0 12px;font-size:16px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.chk{display:flex;align-items:center;gap:8px;margin:6px 0 12px;font-size:13px}
.chk input{width:auto}
@media(max-width:900px){body{height:auto;min-height:100vh}#app{grid-template-columns:1fr}#canvas{height:62vh;flex:none}#side{border-left:0;border-top:1px solid var(--line)}}
</style>
</head>
<body>
<header>
  <h1>Block Editor</h1>
  <span class="pill"><span class="dot" id="dot"></span><span id="modelTxt">Checking GGUFllama…</span></span>
  <button id="settingsBtn">Settings</button>
</header>
<div id="askRow">
  <input type="text" id="prompt" placeholder="A window with a button that fetches a URL and shows the result" aria-label="Describe the app">
  <button class="primary" id="askBtn">Ask</button>
</div>
<div id="status" role="status"></div>
<div id="app">
  <div id="left">
    <div id="palette"></div>
    <div id="bar">
      <button id="layoutBtn">Auto-layout</button>
      <button id="fitBtn">Fit</button>
      <button id="zoomIn" aria-label="Zoom in">+</button>
      <button id="zoomOut" aria-label="Zoom out">&minus;</button>
      <button id="saveBtn">Save graph</button>
      <button id="loadBtn">Load graph</button>
      <button id="clearBtn">Clear</button>
      <input type="file" id="fileIn" accept=".json" hidden>
    </div>
    <div id="canvas"><div id="world"><svg id="wires"></svg></div></div>
  </div>
  <div id="side">
    <div class="sec" id="panel"></div>
    <div class="sec">
      <div id="codeHead"><b>Generated app.py</b>
        <button id="copyBtn">Copy</button><button id="savePyBtn">Save .py</button><button id="testBtn" title="Run the app headless with sample data and report problems">Test</button><button class="primary" id="runBtn">Run</button></div>
      <ul id="codeErrors"></ul>
      <pre id="code"></pre>
      <div id="runOut"></div>
    </div>
  </div>
</div>

<dialog id="dlg">
  <h2>Settings</h2>
  <label class="field"><span>GGUFllama API address</span><input type="text" id="cBridge"></label>
  <label class="field"><span>Temperature</span><input type="number" id="cTemp" min="0" max="1" step="0.1"></label>
  <label style="display:flex;gap:8px;align-items:center;font-size:13px;margin:0 0 12px"><input type="checkbox" id="cTest"> Test the app after building and let the model fix problems</label>
  <p style="font-size:12px;color:var(--dim);margin:0 0 12px">Models are loaded in GGUFllama (Models window). Start its HTTP API from the title-bar right-click menu.</p>
  <div style="display:flex;justify-content:flex-end;gap:8px"><button id="dlgCancel">Cancel</button><button class="primary" id="dlgSave">Save settings</button></div>
</dialog>

<script>
const NODE_W=176,HEAD=30,ROW=26,SUB=26;
let LIB={},graph={blocks:[],wires:[]},sel=null,view={x:40,y:30,z:1},drag=null,DEFAULT_BRIDGE='';
const $=s=>document.querySelector(s);
const canvas=$('#canvas'),world=$('#world'),svg=$('#wires');
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const byId=id=>graph.blocks.find(b=>b.id===id);

function dims(b){const t=LIB[b.type];const rows=Math.max(Object.keys(t.in).length,Object.keys(t.out).length);
  return{w:NODE_W,h:HEAD+rows*ROW+(t.settings.length?SUB+6:0)+4};}
function portPos(b,side,name){const t=LIB[b.type];const keys=Object.keys(side==='in'?t.in:t.out);const i=keys.indexOf(name);
  return{x:b.x+(side==='in'?0:NODE_W),y:b.y+HEAD+i*ROW+ROW/2};}
function toWorld(cx,cy){const r=canvas.getBoundingClientRect();return{x:(cx-r.left-view.x)/view.z,y:(cy-r.top-view.y)/view.z};}
function applyView(){world.style.transform=`translate(${view.x}px,${view.y}px) scale(${view.z})`;
  canvas.style.backgroundPosition=`${view.x}px ${view.y}px`;canvas.style.backgroundSize=`${24*view.z}px ${24*view.z}px`;}
function curve(a,b){const dx=Math.max(40,Math.abs(b.x-a.x)/2);return`M${a.x} ${a.y} C${a.x+dx} ${a.y},${b.x-dx} ${b.y},${b.x} ${b.y}`;}

function render(){
  world.querySelectorAll('.node').forEach(n=>n.remove());
  graph.blocks.forEach(b=>{
    const t=LIB[b.type],d=dims(b),ins=Object.entries(t.in),outs=Object.entries(t.out);
    const n=document.createElement('div');n.className=`node cat-${t.cat}`+(sel&&sel.kind==='block'&&sel.id===b.id?' sel':'');
    n.style.cssText=`left:${b.x}px;top:${b.y}px;width:${d.w}px;height:${d.h}px`;n.dataset.id=b.id;
    n.innerHTML=`<div class="head"><span>${esc(b.type)}</span><span class="nid">${esc(b.id)}</span></div>`;
    const rows=Math.max(ins.length,outs.length);
    for(let i=0;i<rows;i++){
      const r=document.createElement('div');r.className='prow';
      const l=document.createElement('span'),rt=document.createElement('span');
      if(ins[i]){l.textContent=ins[i][0];const p=document.createElement('div');p.className=`port in t-${ins[i][1]}`;p.dataset.id=b.id;p.dataset.port=ins[i][0];r.appendChild(p);}
      if(outs[i]){rt.textContent=outs[i][0];const p=document.createElement('div');p.className=`port out t-${outs[i][1]}`;p.dataset.id=b.id;p.dataset.port=outs[i][0];
        p.addEventListener('pointerdown',e=>startWire(e,b,outs[i][0],outs[i][1]));r.appendChild(p);}
      r.prepend(l);r.appendChild(rt);n.appendChild(r);
    }
    if(t.settings.length){const s=t.settings[0];const v=(b.settings[s.key]??s.default)+'';
      const sub=document.createElement('div');sub.className='sub';sub.textContent=`${s.key}: ${v}`;n.appendChild(sub);}
    const head=n.querySelector('.head');
    head.addEventListener('pointerdown',e=>{
      if(e.button>0)return;e.stopPropagation();select({kind:'block',id:b.id});
      const st=toWorld(e.clientX,e.clientY),ox=b.x,oy=b.y;head.setPointerCapture(e.pointerId);
      const mv=ev=>{const p=toWorld(ev.clientX,ev.clientY);b.x=Math.round(ox+p.x-st.x);b.y=Math.round(oy+p.y-st.y);
        n.style.left=b.x+'px';n.style.top=b.y+'px';drawWires();};
      const up=()=>{head.removeEventListener('pointermove',mv);head.removeEventListener('pointerup',up);};
      head.addEventListener('pointermove',mv);head.addEventListener('pointerup',up);});
    n.addEventListener('pointerdown',e=>{e.stopPropagation();if(!e.target.closest('.head'))select({kind:'block',id:b.id});});
    world.appendChild(n);
  });
  drawWires();
}

function drawWires(){
  let h='';
  graph.wires.forEach((w,i)=>{
    const a=byId(w.from_block),b=byId(w.to_block);if(!a||!b)return;
    const t=LIB[a.type].out[w.from_port];const d=curve(portPos(a,'out',w.from_port),portPos(b,'in',w.to_port));
    const s=sel&&sel.kind==='wire'&&sel.i===i;
    h+=`<path class="wire t-${t}${s?' sel':''}" d="${d}"/><path class="hit" data-i="${i}" d="${d}"/>`;
  });
  if(drag&&drag.to)h+=`<path class="wire t-${drag.type}" style="opacity:.6" d="${curve(drag.from,drag.to)}"/>`;
  svg.innerHTML=h;
  svg.querySelectorAll('.hit').forEach(p=>p.addEventListener('pointerdown',e=>{e.stopPropagation();select({kind:'wire',i:+p.dataset.i});}));
}

function select(s){sel=s;world.querySelectorAll('.node').forEach(n=>n.classList.toggle('sel',!!s&&s.kind==='block'&&n.dataset.id===s.id));drawWires();renderPanel();}

function startWire(e,b,name,type){
  e.stopPropagation();e.preventDefault();const port=e.target;port.setPointerCapture(e.pointerId);
  drag={from:portPos(b,'out',name),type,to:null};
  world.querySelectorAll('.port.in').forEach(p=>{const tb=byId(p.dataset.id);
    if(tb&&tb.id!==b.id&&LIB[tb.type].in[p.dataset.port]===type)p.classList.add('ok');});
  const mv=ev=>{drag.to=toWorld(ev.clientX,ev.clientY);drawWires();};
  const up=ev=>{port.removeEventListener('pointermove',mv);port.removeEventListener('pointerup',up);
    const t=document.elementFromPoint(ev.clientX,ev.clientY);const pt=t&&t.closest?t.closest('.port.in'):null;
    if(pt)connect(b.id,name,pt.dataset.id,pt.dataset.port);
    drag=null;world.querySelectorAll('.port.ok').forEach(p=>p.classList.remove('ok'));drawWires();};
  port.addEventListener('pointermove',mv);port.addEventListener('pointerup',up);
}

function connect(fb,fp,tb,tp){
  const a=byId(fb),b=byId(tb);if(!a||!b||fb===tb)return;
  const ft=LIB[a.type].out[fp],tt=LIB[b.type].in[tp];
  if(ft!==tt){setStatus('warn',`Can't wire ${ft} to ${tt}. Ports must be the same type.`);return;}
  if(tt!=='flow')graph.wires=graph.wires.filter(w=>!(w.to_block===tb&&w.to_port===tp));
  if(graph.wires.some(w=>w.from_block===fb&&w.from_port===fp&&w.to_block===tb&&w.to_port===tp))return;
  graph.wires.push({from_block:fb,from_port:fp,to_block:tb,to_port:tp});setStatus();render();scheduleGen();
}

function renderPanel(){
  const p=$('#panel');p.innerHTML='';
  const add=(tag,cls,html)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(html!==undefined)e.innerHTML=html;p.appendChild(e);return e;};
  if(sel&&sel.kind==='block'&&byId(sel.id)){
    const b=byId(sel.id),t=LIB[b.type];
    add('div','ptitle',`${esc(b.type)} <span class="dim">${esc(b.id)}</span>`);add('p','pdesc',esc(t.desc));
    t.settings.forEach(s=>{
      const wrap=add('label','field');const sp=document.createElement('span');sp.textContent=s.label;wrap.appendChild(sp);
      let inp;const cur=b.settings[s.key]??s.default;
      if(s.options){inp=document.createElement('select');s.options.forEach(o=>{const op=document.createElement('option');op.value=o;op.textContent=o;inp.appendChild(op);});}
      else{inp=document.createElement('input');inp.type='text';}
      inp.value=cur;inp.addEventListener('input',()=>{b.settings[s.key]=inp.value;render();scheduleGen();});wrap.appendChild(inp);
    });
    const io=[...Object.entries(t.in).map(([k,v])=>`${k} (${v})`),...Object.entries(t.out).map(([k,v])=>`${k} (${v})`)];
    add('div','ports',`Inputs: ${Object.entries(t.in).map(([k,v])=>k+' ('+v+')').join(', ')||'none'}<br>Outputs: ${Object.entries(t.out).map(([k,v])=>k+' ('+v+')').join(', ')||'none'}`);
    const del=add('button','danger','Delete block');del.addEventListener('click',()=>{
      graph.blocks=graph.blocks.filter(x=>x.id!==b.id);graph.wires=graph.wires.filter(w=>w.from_block!==b.id&&w.to_block!==b.id);
      sel=null;render();renderPanel();scheduleGen();});
  }else if(sel&&sel.kind==='wire'&&graph.wires[sel.i]){
    const w=graph.wires[sel.i];
    add('div','ptitle','Wire');add('p','pdesc',`${esc(w.from_block)}.${esc(w.from_port)} &rarr; ${esc(w.to_block)}.${esc(w.to_port)}`);
    const del=add('button','danger','Delete wire');del.addEventListener('click',()=>{graph.wires.splice(sel.i,1);sel=null;render();renderPanel();scheduleGen();});
  }else{
    add('div','ptitle','Nothing selected');
    add('p','pdesc','Select a block to edit its settings. Drag from a filled dot on the right of a block to a highlighted dot on another block to wire them. Tap a wire to select it.');
    add('div','ports','<span style="color:var(--flow)">&#9679;</span> flow triggers &nbsp; <span style="color:var(--txt)">&#9679;</span> text data &nbsp; <span style="color:#7b8499">&#9679;</span> ui parent');
  }
}

function setStatus(kind,msg,list){
  const s=$('#status');if(!kind){s.className='';s.textContent='';return;}
  s.className='show '+kind;s.textContent=msg;
  if(list&&list.length){const ul=document.createElement('ul');list.forEach(x=>{const li=document.createElement('li');li.textContent=x;ul.appendChild(li);});s.appendChild(ul);}
}

function autoLayout(){
  const B=graph.blocks,depth={};B.forEach(b=>depth[b.id]=0);
  for(let i=0;i<B.length;i++){let ch=false;
    graph.wires.forEach(w=>{if(depth[w.from_block]===undefined||depth[w.to_block]===undefined)return;
      if(depth[w.to_block]<depth[w.from_block]+1){depth[w.to_block]=depth[w.from_block]+1;ch=true;}});
    if(!ch)break;}
  const cols={};B.forEach(b=>{(cols[depth[b.id]]=cols[depth[b.id]]||[]).push(b);});
  Object.keys(cols).forEach(c=>{let y=40;cols[c].forEach(b=>{b.x=40+c*(NODE_W+90);b.y=y;y+=dims(b).h+28;});});
}
function fit(){
  if(!graph.blocks.length){view={x:40,y:30,z:1};applyView();return;}
  let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
  graph.blocks.forEach(b=>{const d=dims(b);x0=Math.min(x0,b.x);y0=Math.min(y0,b.y);x1=Math.max(x1,b.x+d.w);y1=Math.max(y1,b.y+d.h);});
  const cw=canvas.clientWidth,ch=canvas.clientHeight,bw=x1-x0,bh=y1-y0;
  const z=Math.max(.3,Math.min(1,(cw-60)/bw,(ch-60)/bh));
  view.z=z;view.x=(cw-bw*z)/2-x0*z;view.y=Math.max(20,(ch-bh*z)/2)-y0*z;applyView();
}
function zoomAt(cx,cy,f){const r=canvas.getBoundingClientRect(),px=cx-r.left,py=cy-r.top,nz=Math.max(.3,Math.min(1.6,view.z*f));
  view.x=px-(px-view.x)*(nz/view.z);view.y=py-(py-view.y)*(nz/view.z);view.z=nz;applyView();}

canvas.addEventListener('pointerdown',e=>{
  if(e.target.closest('.node')||e.target.closest('.hit'))return;
  select(null);canvas.setPointerCapture(e.pointerId);canvas.classList.add('panning');
  const sx=e.clientX,sy=e.clientY,ox=view.x,oy=view.y;
  const mv=ev=>{view.x=ox+ev.clientX-sx;view.y=oy+ev.clientY-sy;applyView();};
  const up=()=>{canvas.classList.remove('panning');canvas.removeEventListener('pointermove',mv);canvas.removeEventListener('pointerup',up);};
  canvas.addEventListener('pointermove',mv);canvas.addEventListener('pointerup',up);
});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoomAt(e.clientX,e.clientY,e.deltaY<0?1.1:.9);},{passive:false});
document.addEventListener('keydown',e=>{
  if(/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName))return;
  if((e.key==='Delete'||e.key==='Backspace')&&sel){
    if(sel.kind==='block'){const id=sel.id;graph.blocks=graph.blocks.filter(b=>b.id!==id);graph.wires=graph.wires.filter(w=>w.from_block!==id&&w.to_block!==id);}
    else graph.wires.splice(sel.i,1);
    sel=null;render();renderPanel();scheduleGen();}
});

function addBlock(type){
  const base=type.toLowerCase().slice(0,4);let n=1;while(byId(base+n))n++;
  const c=toWorld(canvas.getBoundingClientRect().left+canvas.clientWidth/2,canvas.getBoundingClientRect().top+canvas.clientHeight/2);
  const b={id:base+n,type,settings:{},x:Math.round(c.x-NODE_W/2+(Math.random()*40-20)),y:Math.round(c.y-30+(Math.random()*40-20))};
  LIB[type].settings.forEach(s=>b.settings[s.key]=s.default);graph.blocks.push(b);render();select({kind:'block',id:b.id});scheduleGen();
}

let genT;function scheduleGen(){clearTimeout(genT);genT=setTimeout(gen,250);}
async function gen(){
  try{const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph})});
    const d=await r.json();$('#code').textContent=d.code;const ul=$('#codeErrors');ul.innerHTML='';
    d.errors.forEach(e=>{const li=document.createElement('li');li.textContent=e;ul.appendChild(li);});ul.style.display=d.errors.length?'block':'none';}catch(e){}
}

const CFG_KEY='blockEditorCfg';
function getCfg(){const c=JSON.parse(localStorage.getItem(CFG_KEY)||'{}');
  return Object.assign({bridge_url:DEFAULT_BRIDGE,temperature:0.2,auto_test:true},c);}
function openSettings(){const c=getCfg();$('#cBridge').value=c.bridge_url;$('#cTemp').value=c.temperature;$('#cTest').checked=c.auto_test!==false;$('#dlg').showModal();}
function saveSettings(){localStorage.setItem(CFG_KEY,JSON.stringify({bridge_url:$('#cBridge').value.trim()||DEFAULT_BRIDGE,
  temperature:+$('#cTemp').value,auto_test:$('#cTest').checked}));$('#dlg').close();}

async function refreshModel(){
  try{const d=await(await fetch('/api/status?bridge='+encodeURIComponent(getCfg().bridge_url))).json();
    $('#dot').className='dot'+(d.loaded?' on':'');
    $('#modelTxt').textContent=d.loaded?('GGUFllama: '+d.model):(d.reachable?'GGUFllama: no model loaded':'GGUFllama not reachable');
    $('#modelTxt').title=d.reason||'';}catch(e){}
}

async function ask(){
  const prompt=$('#prompt').value.trim();
  if(!prompt){setStatus('warn','Describe the app you want first.');return;}
  const cfg=getCfg();$('#askBtn').disabled=true;
  setStatus('busy',cfg.auto_test===false?'Asking GGUFllama and drawing the graph…':'Asking GGUFllama, drawing the graph and test-running the app…');
  try{
    const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,cfg})});
    const d=await r.json();if(!r.ok||d.error)throw new Error(d.error||'Request failed');
    graph=d.graph;autoLayout();sel=null;render();renderPanel();fit();scheduleGen();
    const n=`${graph.blocks.length} blocks, ${graph.wires.length} wires.`;
    const tries=d.attempts>1?` (took ${d.attempts} tries)`:'';
    if(d.errors.length)setStatus('warn',`${n} Some problems remain after ${d.attempts} tries. Fix them by hand:`,d.errors);
    else if(d.test)showTest(d.test,n,tries);
    else setStatus('ok',`${n} All ports valid${tries}.`);
  }catch(err){setStatus('bad',err.message);}
  finally{$('#askBtn').disabled=false;refreshModel();}
}

function showTest(t,pre,tries){
  pre=pre?pre+' ':'';tries=tries||'';
  if(t.failures&&t.failures.length)setStatus('warn',`${pre}It still fails its test${tries}. Fix these by hand or ask again:`,t.failures);
  else if(t.ran)setStatus('ok',`${pre}Built, ran headless and tested clean${tries}.${t.notes&&t.notes.length?' Notes:':''}`,t.notes);
  else setStatus('warn',`${pre}All ports valid, but the test was skipped: ${t.skipped||'unknown reason'}`);
}
$('#testBtn').onclick=async()=>{
  setStatus('busy','Running the app headless to test it…');
  try{const r=await fetch('/api/test',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph})});
    const t=await r.json();if(!r.ok||t.error)throw new Error(t.error||'Test failed');
    const sim=(t.stubbed||[]).length?' Simulated: '+t.stubbed.join(', ')+'.':'';
    showTest(t,'');if(sim)$('#runOut').textContent='Network and files were simulated during the test.'+sim;
  }catch(err){setStatus('bad',err.message);}
};
function download(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}

$('#askBtn').onclick=ask;$('#prompt').addEventListener('keydown',e=>{if(e.key==='Enter')ask();});
$('#layoutBtn').onclick=()=>{autoLayout();render();fit();};$('#fitBtn').onclick=fit;
$('#zoomIn').onclick=()=>zoomAt(canvas.getBoundingClientRect().left+canvas.clientWidth/2,canvas.getBoundingClientRect().top+canvas.clientHeight/2,1.2);
$('#zoomOut').onclick=()=>zoomAt(canvas.getBoundingClientRect().left+canvas.clientWidth/2,canvas.getBoundingClientRect().top+canvas.clientHeight/2,.83);
$('#saveBtn').onclick=()=>download('graph.json',JSON.stringify(graph,null,2),'application/json');
$('#loadBtn').onclick=()=>$('#fileIn').click();
$('#fileIn').onchange=e=>{const f=e.target.files[0];if(!f)return;const rd=new FileReader();rd.onload=()=>{
  try{const g=JSON.parse(rd.result);if(!Array.isArray(g.blocks)||!Array.isArray(g.wires))throw 0;
    g.blocks.forEach(b=>{b.settings=b.settings||{};});graph=g;if(graph.blocks.some(b=>b.x===undefined))autoLayout();
    sel=null;render();renderPanel();fit();scheduleGen();setStatus('ok','Graph loaded.');}
  catch(x){setStatus('bad','That file isn\'t a block graph.');}};rd.readAsText(f);e.target.value='';};
$('#clearBtn').onclick=()=>{if(!confirm('Clear the whole graph?'))return;graph={blocks:[],wires:[]};sel=null;render();renderPanel();scheduleGen();setStatus();};
$('#copyBtn').onclick=async()=>{try{await navigator.clipboard.writeText($('#code').textContent);setStatus('ok','Code copied.');}catch(e){setStatus('warn','Copy failed. Select the code and copy it by hand.');}};
$('#savePyBtn').onclick=()=>download('app.py',$('#code').textContent,'text/x-python');
$('#runBtn').onclick=async()=>{const out=$('#runOut');out.textContent='Starting…';
  const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:$('#code').textContent})});
  const d=await r.json();out.textContent=d.message+(d.output?'\n'+d.output:'');};
$('#settingsBtn').onclick=openSettings;$('#dlgSave').onclick=()=>{saveSettings();refreshModel();};$('#dlgCancel').onclick=()=>$('#dlg').close();

(async function init(){
  const d=await(await fetch('/api/library')).json();LIB=d.lib;DEFAULT_BRIDGE=d.default_bridge;
  const pal=$('#palette');Object.keys(LIB).forEach(t=>{const c=document.createElement('button');c.className='chip '+LIB[t].cat;c.textContent='+ '+t;c.title=LIB[t].desc;c.onclick=()=>addBlock(t);pal.appendChild(c);});
  graph=JSON.parse(JSON.stringify(d.demo));autoLayout();render();renderPanel();applyView();requestAnimationFrame(fit);scheduleGen();refreshModel();setInterval(refreshModel,5000);
  window.addEventListener('resize',()=>{});
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
