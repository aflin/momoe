#!/usr/bin/env python3
"""A graded task check for a change that alters the model's output (written for STRATA_ROUTE_PRIOR, docs/ROUTE_PRIOR.md).

140 tasks, each graded automatically:
  code  50  HumanEval (every 3rd problem): the reply's function must pass the problem's unit tests
  math  50  GSM8K test set (every 26th): the reply's "ANSWER:" line must equal the reference number
  copy  20  generated: reverse a list of codes, extract e-mail addresses, change one JSON field, replace a word
  tool  20  generated: an OpenAI tools conversation against a fake shop (prices, stock, orders, exchange rates);
            the final answer must be the right total

    python tools/task_check.py run --config CFG.json --arm NAME [--env K=V ...] --samples 0,1 --out results.jsonl
    python tools/task_check.py report results.jsonl [more.jsonl ...]

The two datasets are not in the repository; put them in data/task_check (or set TASK_CHECK_DATA):
    curl -L https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz | gunzip > data/task_check/HumanEval.jsonl
    curl -L -o data/task_check/gsm8k_test.jsonl \
        https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
The code tasks run the model's code in a subprocess (python -I, a temporary folder, 2 GB of memory, 10 s of CPU):
that is a limit, not a sandbox, so run it on a machine where that is acceptable.

`run` starts serve/server.py with the config (and the arm's environment) once per sample, sends every task with
the config's own sampling (thinking off), grades it, and appends one line per task to --out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import resource
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("TASK_CHECK_ROOT", Path(__file__).resolve().parents[1]))   # the serve/server.py to start
DATA = Path(os.environ.get("TASK_CHECK_DATA", ROOT / "data" / "task_check"))


# ------------------------------------------------------------------------------------------------ tasks
def code_tasks():
    rows = [json.loads(l) for l in (DATA / "HumanEval.jsonl").read_text().splitlines() if l.strip()]
    out = []
    for r in rows[::3][:50]:
        out.append({"id": r["task_id"], "kind": "code",
                    "prompt": "Complete the following Python function. Reply with the complete function (with any imports "
                              "it needs) in one ```python code block.\n\n```python\n" + r["prompt"] + "```",
                    "header": r["prompt"], "test": r["test"], "entry": r["entry_point"]})
    return out


def math_tasks():
    rows = [json.loads(l) for l in (DATA / "gsm8k_test.jsonl").read_text().splitlines() if l.strip()]
    out = []
    for i, r in enumerate(rows[::26][:50]):
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"id": f"gsm8k/{i * 26}", "kind": "math",
                    "prompt": r["question"] + "\n\nSolve it step by step, then give the final answer on its own last line "
                              "as: ANSWER: <number>", "answer": ans})
    return out


WORDS = ("amber basil cedar delta ember fable garnet harbor island jasper kettle lantern meadow nectar orchard pepper "
         "quarry raven saddle timber umber velvet willow yonder zephyr anchor bramble copper drift falcon").split()


def copy_tasks():
    rng = random.Random(1234)
    out = []
    for i in range(5):   # reverse a list of codes
        codes = ["".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8)) for _ in range(30)]
        out.append({"id": f"copy/reverse{i}", "kind": "copy",
                    "prompt": "Here are 30 codes, one per line:\n" + "\n".join(codes) + "\n\nWrite the same 30 codes in reverse "
                              "order, one per line, and nothing else.", "expect": list(reversed(codes)), "check": "lines"})
    for i in range(5):   # extract e-mail addresses
        names = rng.sample(WORDS, 8)
        doms = ["example.com", "mail.test", "corp.example.org", "uni.example.edu"]
        emails = [f"{n}.{rng.choice(WORDS)}{rng.randint(1, 99)}@{rng.choice(doms)}" for n in names]
        filler = []
        for e in emails:
            filler.append(f"{rng.choice(WORDS).capitalize()} {rng.choice(WORDS)} wrote from {e} about the {rng.choice(WORDS)} "
                          f"{rng.choice(WORDS)} on day {rng.randint(1, 28)}.")
            filler.append(f"The {rng.choice(WORDS)} {rng.choice(WORDS)} was {rng.randint(2, 900)} units.")
        out.append({"id": f"copy/emails{i}", "kind": "copy",
                    "prompt": " ".join(filler) + "\n\nList every e-mail address in the text above, in the order they appear, "
                              "as a JSON array of strings. Reply with the JSON array only.", "expect": emails, "check": "json"})
    for i in range(5):   # change one JSON field
        obj = {w: (rng.randint(1, 10000) if rng.random() < 0.5 else rng.choice(WORDS)) for w in rng.sample(WORDS, 15)}
        key = rng.choice(list(obj))
        new = rng.randint(10001, 99999)
        want = dict(obj)
        want[key] = new
        out.append({"id": f"copy/json{i}", "kind": "copy",
                    "prompt": "Here is a JSON object:\n" + json.dumps(obj) + f"\n\nReturn the same object with the value of "
                              f"\"{key}\" changed to {new}, every other field unchanged. Reply with the JSON object only.",
                    "expect": want, "check": "json"})
    for i in range(5):   # replace a word everywhere in a passage
        text = " ".join(rng.choice(WORDS) for _ in range(120))
        old = max(set(text.split()), key=text.split().count)
        new = "PLACEHOLDER" + str(i)
        out.append({"id": f"copy/replace{i}", "kind": "copy",
                    "prompt": "Copy this text exactly, but replace every occurrence of the word \"" + old + "\" with \"" + new +
                              "\". Reply with the text only.\n\n" + text,
                    "expect": " ".join(new if w == old else w for w in text.split()), "check": "words"})
    return out


SHOP_TOOLS = [
    {"type": "function", "function": {"name": "lookup_item", "description": "Price (USD) and stock of an item by SKU.",
                                      "parameters": {"type": "object", "properties": {"sku": {"type": "string"}},
                                                     "required": ["sku"]}}},
    {"type": "function", "function": {"name": "get_order", "description": "The lines (sku, quantity) of an order.",
                                      "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                                                     "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "exchange_rate", "description": "How many units of `to` one USD buys.",
                                      "parameters": {"type": "object", "properties": {"to": {"type": "string"}},
                                                     "required": ["to"]}}},
]
RATES = {"EUR": 0.92, "GBP": 0.79, "JPY": 151.0, "CAD": 1.37}


def tool_tasks():
    rng = random.Random(99)
    out = []
    for i in range(20):
        items = {f"SKU-{rng.randint(1000, 9999)}": {"price": round(rng.uniform(2, 300), 2), "stock": rng.randint(0, 40)}
                 for _ in range(6)}
        skus = list(items)
        lines = [(s, rng.randint(1, 5)) for s in rng.sample(skus, rng.randint(2, 3))]
        cur = rng.choice(list(RATES))
        usd = sum(items[s]["price"] * q for s, q in lines)
        total = round(usd * RATES[cur], 2)
        order_id = f"ORD-{rng.randint(10000, 99999)}"
        env = {"items": items, "orders": {order_id: [{"sku": s, "quantity": q} for s, q in lines]}}
        if i < 10:
            ask = ("I want to buy " + ", ".join(f"{q} of {s}" for s, q in lines) + f". What is the total price in {cur}? "
                   "Use the tools to look up the prices and the exchange rate. Give the final total rounded to 2 decimals on "
                   "its own last line as: TOTAL: <number>")
        else:
            ask = (f"What is the total price of order {order_id} in {cur}? Use the tools to find the order's lines, the "
                   "prices and the exchange rate. Give the final total rounded to 2 decimals on its own last line as: "
                   "TOTAL: <number>")
        out.append({"id": f"tool/{i}", "kind": "tool", "prompt": ask, "env": env, "answer": total})
    return out


def all_tasks():
    return code_tasks() + math_tasks() + copy_tasks() + tool_tasks()


# ------------------------------------------------------------------------------------------------ grading
def last_code_block(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else text


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    os.setsid()


def grade_code(t, text):
    code = last_code_block(text)
    imports = "\n".join(l for l in t["header"].splitlines() if l.startswith(("import ", "from ")))
    prog = imports + "\n" + code + "\n\n" + t["test"] + f"\n\ncheck({t['entry']})\n"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "prog.py"
        p.write_text(prog)
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], cwd=d, capture_output=True, timeout=20,
                               preexec_fn=_limits, env={"PATH": "/usr/bin:/bin"})
            return r.returncode == 0, r.stderr.decode(errors="replace")[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def num(s):
    s = s.replace(",", "").replace("$", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def grade_math(t, text):
    m = re.findall(r"ANSWER:\s*\**\s*([^\n]*)", text)
    if not m:
        return False, "no ANSWER line"
    got = num(m[-1])
    return got is not None and abs(got - float(t["answer"])) < 1e-6, f"got {m[-1][:40]!r} want {t['answer']}"


def strip_fence(text):
    m = re.search(r"```(?:json|text)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def grade_copy(t, text):
    body = strip_fence(text)
    if t["check"] == "lines":
        got = [l.strip() for l in body.splitlines() if l.strip()]
        return got == t["expect"], f"{sum(a == b for a, b in zip(got, t['expect']))}/{len(t['expect'])} lines"
    if t["check"] == "json":
        try:
            return json.loads(body) == t["expect"], "json compared"
        except ValueError:
            return False, "not JSON"
    return body.split() == t["expect"].split(), "words compared"


def grade_tool(t, text, calls):
    m = re.findall(r"TOTAL:\s*\**\s*([^\n]*)", text)
    if not m:
        return False, f"no TOTAL line ({calls} tool calls)"
    got = num(m[-1])
    return got is not None and abs(got - t["answer"]) <= 0.011, f"got {m[-1][:30]!r} want {t['answer']} ({calls} calls)"


# ------------------------------------------------------------------------------------------------ requests
def chat(port, messages, max_tokens, tools=None):
    body = {"messages": messages, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def run_tool(env, name, args):
    if name == "lookup_item":
        it = env["items"].get(str(args.get("sku", "")).strip())
        return it if it else {"error": "unknown sku"}
    if name == "get_order":
        o = env["orders"].get(str(args.get("order_id", "")).strip())
        return {"lines": o} if o else {"error": "unknown order"}
    if name == "exchange_rate":
        r = RATES.get(str(args.get("to", "")).strip().upper())
        return {"usd_to": r} if r else {"error": "unknown currency"}
    return {"error": "unknown tool"}


def do_task(port, t):
    if t["kind"] != "tool":
        r = chat(port, [{"role": "user", "content": t["prompt"]}], 1536)
        text = r["choices"][0]["message"].get("content") or ""
        ok, why = {"code": grade_code, "math": grade_math, "copy": grade_copy}[t["kind"]](t, text)
        return ok, why, text, r.get("usage", {}).get("completion_tokens", 0)
    msgs = [{"role": "user", "content": t["prompt"]}]
    calls = toks = 0
    text = ""
    for _ in range(10):
        r = chat(port, msgs, 768, SHOP_TOOLS)
        toks += r.get("usage", {}).get("completion_tokens", 0)
        m = r["choices"][0]["message"]
        tc = m.get("tool_calls") or []
        text = m.get("content") or ""
        if not tc:
            break
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tc})
        for c in tc:
            calls += 1
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except ValueError:
                args = {}
            msgs.append({"role": "tool", "tool_call_id": c.get("id", ""),
                         "content": json.dumps(run_tool(t["env"], c["function"]["name"], args))})
    ok, why = grade_tool(t, text, calls)
    return ok, why, text, toks


# ------------------------------------------------------------------------------------------------ server
def wait_ready(port, proc, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    chat(port, [{"role": "user", "content": "Say hi."}], 8)
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("server not ready")


def cmd_run(a):
    tasks = all_tasks()
    if a.only:
        tasks = [t for t in tasks if t["kind"] in a.only.split(",")]
    env = dict(os.environ)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    log = Path(a.out).with_suffix(f".{a.arm}.server.log")
    for s in [int(x) for x in a.samples.split(",")]:
        proc = subprocess.Popen([a.python or sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata",
                                 "--config", a.config, "--port", str(a.port)], cwd=str(ROOT), env=env,
                                stdout=open(log, "a"), stderr=subprocess.STDOUT)
        try:
            wait_ready(a.port, proc)
            print(f"{time.strftime('%T')} {a.arm} sample {s}: ready", flush=True)
            n_ok = 0
            for i, t in enumerate(tasks):
                t0 = time.time()
                try:
                    ok, why, text, toks = do_task(a.port, t)
                except Exception as e:  # noqa: BLE001 - a failed request is a failed task, logged
                    ok, why, text, toks = False, f"error: {e}", "", 0
                n_ok += ok
                with open(a.out, "a") as f:
                    f.write(json.dumps({"arm": a.arm, "sample": s, "id": t["id"], "kind": t["kind"], "ok": ok,
                                        "why": why, "tokens": toks, "s": round(time.time() - t0, 2),
                                        "text": text}) + "\n")
                if (i + 1) % 20 == 0:
                    print(f"{time.strftime('%T')} {a.arm} sample {s}: {i + 1}/{len(tasks)} done, {n_ok} passed", flush=True)
            print(f"{time.strftime('%T')} {a.arm} sample {s}: finished {n_ok}/{len(tasks)}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(10)
    return 0


def cmd_report(a):
    rows = []
    for p in a.files:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    arms = sorted({r["arm"] for r in rows})
    kinds = ["code", "math", "copy", "tool"]
    print(f"{'arm':10s} " + " ".join(f"{k:>14s}" for k in kinds + ["all"]) + "   samples")
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        cells = []
        for k in kinds + ["all"]:
            sub = [r for r in rs if k == "all" or r["kind"] == k]
            n, ok = len(sub), sum(r["ok"] for r in sub)
            cells.append(f"{ok:4d}/{n:<4d}{100 * ok / max(n, 1):5.1f}%")
        print(f"{arm:10s} " + " ".join(f"{c:>14s}" for c in cells) + f"   {sorted({r['sample'] for r in rs})}")
    # paired by (task, sample): tasks one arm passes and the other fails
    if len(arms) == 2:
        A, B = arms
        pa = {(r["id"], r["sample"]): r["ok"] for r in rows if r["arm"] == A}
        pb = {(r["id"], r["sample"]): r["ok"] for r in rows if r["arm"] == B}
        both = set(pa) & set(pb)
        a_only = sum(pa[k] and not pb[k] for k in both)
        b_only = sum(pb[k] and not pa[k] for k in both)
        n = a_only + b_only
        # two-sided sign test on the discordant pairs
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(a_only, b_only) + 1)) / 2 ** n) if n else 1.0
        print(f"\npaired (task, sample): {len(both)}; {A} only passes {a_only}, {B} only passes {b_only}; sign test p = {p:.2g}")
        tps = {arm: [r["tokens"] / r["s"] for r in rows if r["arm"] == arm and r["s"] > 0 and r["tokens"] > 50]
               for arm in arms}
        print("tokens/s per task (median): " + ", ".join(f"{arm} {sorted(v)[len(v) // 2]:.1f}" for arm, v in tps.items() if v))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--env", action="append", default=[])
    r.add_argument("--samples", default="0")
    r.add_argument("--out", required=True)
    r.add_argument("--port", type=int, default=8095)
    r.add_argument("--only", default="")
    r.add_argument("--python", default="")
    p = sp.add_parser("report")
    p.add_argument("files", nargs="+")
    sp.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "report":
        return cmd_report(a)
    ts = all_tasks()
    print(len(ts), {k: sum(t["kind"] == k for t in ts) for k in ("code", "math", "copy", "tool")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
