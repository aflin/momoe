#!/usr/bin/env python3
"""MoMoE: more MoEs, built on Strata (https://github.com/Niko1221/Strata).

    momoe setup [setup options]          install a model or change its settings (setup.py --setup), then start it
    momoe start CONFIG [setup options]   start the server from a config, in the background; --host, --api-key,
                                         --draft-vocab, --vram-reserve-mib apply to this start, --save keeps them
    momoe stop                           stop it
    momoe status                         is it running, which model, where
    momoe list                           the configs in config/
    momoe log [server|engine|unsloth|lmstudio|all] [-n N] [--no-follow]
                                         show the last N lines (20) of the logs in logs/ and follow them as they grow;
                                         by default the server's and the running (or last) config's engine log
    momoe run claude|codex|opencode [--config CONFIG] [their arguments]
                                         start that agent, talking to this server (started first if needed)
    momoe run unsloth|lmstudio [--config CONFIG]
                                         open that app, pointed at this server

`momoe run` with no server running starts --config CONFIG, else asks which config when there are several (Enter: the
one started last; without a terminal it takes that one).  --config goes right after the tool's name (or before it):
later arguments are the tool's own.  With a server running it is used; a --config naming another one is refused
(momoe stop first).

Nothing is installed by `momoe run`: a tool that is missing is named with how to install it.  The agents get their
settings for that run only (environment variables and command-line overrides): your own configs stay as they are.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / ".momoe"
# every log is in ROOT / "logs": the engines' (each config's "log"), the server's, the apps' (unsloth.log, ...)
STATE = STATE_DIR / "server.json"
WIN = os.name == "nt"
PY = ROOT / (".venv/Scripts/python.exe" if WIN else ".venv/bin/python")


def say(msg=""):
    print(msg, flush=True)


def die(msg, code=1):
    print(f"momoe: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


# ------------------------------------------------------------------------------------------------ configs and state
def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def configs() -> list:
    """The model configs in config/, the most recently started first: strata-*.json, and *.json named at setup
    (they carry "setup_tag")."""
    move_configs()
    found = []
    for p in config_dir().glob("*.json"):
        if p.name.endswith(".shared-settings.json"):
            continue
        c = read_json(p)
        if isinstance(c, dict) and c.get("exe") and isinstance(c.get("args"), list) and (
                p.name.startswith("strata-") or isinstance(c.get("setup_tag"), str)):
            found.append(p)
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def config_dir() -> Path:
    return ROOT / "config"                           # the model configs (setup writes them there)


def move_configs() -> None:
    """Configs left in the top level (before config/) move there: setup.py's move_configs."""
    if any(ROOT.glob("*.json")):
        sys.path.insert(0, str(ROOT))
        try:
            import setup
            setup.move_configs(ROOT)
        except Exception as e:                       # noqa: BLE001 - listing must not fail over it
            say(f"  could not move the configs into {config_dir()} ({e})")


def find_config(name: str) -> Path:
    """A config by its path, or by its name in config/ (".json" and the "strata-" prefix may be left off)."""
    move_configs()
    d = config_dir()
    for p in (Path(name), d / name, d / (name + ".json"), d / f"strata-{name.lower()}.json"):
        if p.is_file():
            return p.resolve()
    have = ", ".join(p.name for p in configs()) or "none yet (momoe setup)"
    die(f"no config {name!r} in {d} (configs: {have})")


def state():
    s = read_json(STATE)
    return s if isinstance(s, dict) and s.get("pid") else None


def alive(pid) -> bool:
    try:
        import psutil
        return psutil.pid_exists(int(pid)) and psutil.Process(int(pid)).status() != psutil.STATUS_ZOMBIE
    except ImportError:
        if WIN:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
            return str(pid) in out
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False
    except Exception:  # noqa: BLE001 - gone between the two calls
        return False


def api_key(cfg_path) -> str | None:
    """The server's key (the first of "key1,key2"), from its config."""
    c = read_json(cfg_path) or {}
    k = c.get("api_key")
    if isinstance(k, list):
        k = k[0] if k else None
    return str(k).split(",")[0].strip() if k else None


def server_key(s: dict) -> str | None:
    """The running server's key: one given for this start only (momoe start --api-key without --save, kept in its
    state), else its config's."""
    k = s.get("api_key")
    if isinstance(k, list):
        k = k[0] if k else None
    return str(k).split(",")[0].strip() if k else api_key(s.get("config"))


def base_url(s: dict) -> str:
    host = s.get("host") or "127.0.0.1"
    host = "127.0.0.1" if host in ("0.0.0.0", "", "::", "localhost") else host
    return f"http://{'[' + host + ']' if ':' in host else host}:{s['port']}"


def http(method, url, key=None, body=None, timeout=5.0):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def health(s: dict):
    try:
        return http("GET", base_url(s) + "/health", server_key(s), timeout=3)
    except (OSError, ValueError, urllib.error.URLError):
        return None


def running():
    """The momoe server that runs now (its state), or None; a stale state file is removed."""
    s = state()
    if s is None:
        return None
    if not alive(s["pid"]):
        try:
            STATE.unlink()
        except OSError:
            pass
        return None
    return s


# ------------------------------------------------------------------------------------------------ setup / start / stop
def setup_cmd(args) -> list:
    if not PY.exists():
        # the first setup: the starter makes .venv (and installs Python if needed), then runs setup.py
        return (["cmd", "/c", str(ROOT / "START-HERE.bat")] if WIN else [str(ROOT / "setup.sh")]) + args
    return [str(PY), str(ROOT / "setup.py"), *args]


def cmd_setup(args) -> int:
    s = running()
    if s:
        say(f"Stopping {s.get('model') or 'the running model'} first: setup measures the free VRAM.")
        if cmd_stop([]) != 0:
            return 1
    return subprocess.call(setup_cmd(["--setup", "--detach", *args]), cwd=str(ROOT))


def start_config(cfg: Path, extra=(), foreground=False) -> int:
    s = running()
    if s:
        if Path(s.get("config", "")).resolve() == cfg.resolve():
            say(f"{s.get('model') or cfg.name} is already running: {base_url(s)}/v1 (momoe stop ends it)")
            return 0
        die(f"{s.get('model') or Path(s['config']).name} is running ({Path(s['config']).name}): momoe stop first")
    move_log(cfg)
    return subprocess.call(setup_cmd(["--config", str(cfg), *([] if foreground else ["--detach"]), *extra]),
                           cwd=str(ROOT))


def move_log(cfg: Path) -> None:
    """A config from before logs/ keeps its engine log in the MoMoE folder itself: the log moves to logs/ and the
    config says so (once; anything else is left alone)."""
    c = read_json(cfg)
    old = Path(c["log"]) if isinstance(c, dict) and c.get("log") else None
    if old is None or old.parent.resolve() != ROOT.resolve():
        return
    new = ROOT / "logs" / old.name
    try:
        (ROOT / "logs").mkdir(exist_ok=True)
        if old.exists() and not new.exists():
            old.replace(new)
        c["log"] = str(new)
        cfg.write_text(json.dumps(c, indent=1), encoding="utf-8")
    except OSError as e:
        say(f"  the log stays at {old} ({e})")
        return
    say(f"  the engine log is now {new}")


def cmd_start(args) -> int:
    if not args or args[0].startswith("-"):
        have = configs()
        die("which config? momoe start CONFIG" + (f" (here: {', '.join(p.name for p in have)})" if have else
                                                  " (none yet: momoe setup)"))
    fg = "--foreground" in args
    rest = [a for a in args[1:] if a != "--foreground"]
    return start_config(find_config(args[0]), rest, fg)


def cmd_stop(args) -> int:
    s = running()
    if s is None:
        say("No momoe server is running.")
        return 0
    pid = s["pid"]
    say(f"Stopping {s.get('model') or 'the server'} (pid {pid}) ...")
    try:
        http("POST", base_url(s) + "/shutdown", server_key(s), {}, timeout=10)
    except (OSError, ValueError, urllib.error.URLError) as e:
        say(f"  the server did not take the stop request ({e}): ending the process")
        terminate(pid)
    end = time.time() + 90                            # the engine frees its pinned memory (tens of GB) first
    while alive(pid) and time.time() < end:
        time.sleep(0.5)
    if alive(pid):
        say("  still running after 90 s: ending it")
        terminate(pid, hard=True)
        time.sleep(2)
    if alive(pid):
        die(f"could not stop pid {pid}")
    try:
        STATE.unlink()
    except OSError:
        pass
    say("Stopped.")
    return 0


def terminate(pid, hard=False):
    """The server and its children (the engine, the image encoder)."""
    try:
        import psutil
        p = psutil.Process(int(pid))
        procs = p.children(recursive=True) + [p] if hard else [p]
        for q in procs:
            try:
                q.kill() if hard else q.terminate()     # SIGTERM: the server's Ctrl+C path (QUIT to the engine)
            except psutil.Error:
                pass
    except ImportError:
        if WIN:
            subprocess.call(["taskkill", "/PID", str(pid), "/T"] + (["/F"] if hard else []))
        else:
            import signal
            os.kill(int(pid), signal.SIGKILL if hard else signal.SIGTERM)
    except Exception:  # noqa: BLE001 - it ended meanwhile
        pass


def cmd_status(args) -> int:
    s = running()
    if s is None:
        last = state_last()
        last = Path(last).name if last and Path(last).is_file() else (configs()[0].name if configs() else None)
        say("No momoe server is running." + (f" Last config: {last}" if last else ""))
        return 3
    h = health(s)
    up = time.time() - float(s.get("started", time.time()))
    say(f"Model:   {s.get('model')}  ({Path(s['config']).name})")
    say(f"State:   {'ready' if h else 'starting (or not answering)'}, pid {s['pid']}, up {int(up // 60)} min")
    say(f"API:     {base_url(s)}/v1   (Anthropic: {base_url(s)}/v1/messages)")
    if h:
        say(f"Context: {h.get('max_context')} tokens{', images on' if h.get('images') else ''}"
            f"{', API key required' if h.get('api_key') else ''}")
    say(f"Log:     {s.get('log')}")
    return 0


LOG_NAMES = ("server", "engine", "unsloth", "lmstudio")


def log_files(names) -> list:
    """(tag, path) of the logs `momoe log` follows: the server's, the running (else the last started) config's engine
    log, Unsloth Studio's, LM Studio's."""
    out = []
    for n in names:
        if n == "engine":
            s = running()
            cfg = s["config"] if s else state_last()
            c = read_json(cfg) if cfg else None
            if isinstance(c, dict) and c.get("log"):
                out.append(("engine", Path(c["log"])))
        else:
            out.append((n, ROOT / "logs" / f"{n}.log"))
    return out


def tail_lines(path: Path, n: int) -> tuple:
    """The last n lines of path and its size (the end is read, not the whole file)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max(n, 1) * 400))
            data = f.read()
    except OSError:
        return [], 0
    lines = data.decode("utf-8", "replace").splitlines()
    return (lines[-n:] if n > 0 else []), size


def cmd_log(args) -> int:
    n, follow, names = 20, True, []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-n", "--lines") and i + 1 < len(args) and args[i + 1].isdigit():
            n, i = int(args[i + 1]), i + 1
        elif a.startswith("-n") and a[2:].isdigit():
            n = int(a[2:])
        elif a == "--no-follow":
            follow = False
        elif a == "all":
            names += list(LOG_NAMES)
        elif a in LOG_NAMES:
            names.append(a)
        else:
            die(f"log: unknown {a!r} (server, engine, unsloth, lmstudio, all; -n N; --no-follow)")
        i += 1
    files = log_files(list(dict.fromkeys(names or ["server", "engine"])))
    if not files:
        die("no log to show: no config has been started yet (momoe start CONFIG)")
    tag = len(files) > 1
    pos = {}
    for t, path in files:
        lines, pos[path] = tail_lines(path, n)
        say(f"==> {path}{'' if path.exists() else ' (not there yet)'} <==")
        for line in lines:
            say(f"[{t}] {line}" if tag else line)
    if not follow:
        return 0
    say(f"--- following {len(files)} log{'s' if len(files) > 1 else ''} (Ctrl+C ends) ---")
    part = {path: b"" for _, path in files}
    try:
        while True:
            for t, path in files:
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size < pos[path]:            # started again (server.log is rewritten at each start)
                    pos[path], part[path] = 0, b""
                    say(f"==> {path} started again <==")
                if size == pos[path]:
                    continue
                with open(path, "rb") as f:
                    f.seek(pos[path])
                    data = part[path] + f.read(size - pos[path])
                pos[path] = size
                *done, part[path] = data.split(b"\n")
                for line in done:
                    line = line.decode("utf-8", "replace").rstrip("\r")
                    say(f"[{t}] {line}" if tag else line)
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0


def cmd_list(args) -> int:
    s = running()
    have = configs()
    if not have:
        say("No configs yet: momoe setup")
        return 0
    for p in have:
        c = read_json(p) or {}
        on = s is not None and Path(s["config"]).resolve() == p.resolve()
        args = c.get("args", [])
        flag = next((f for f in ("--max-context", "--ctx") if f in args[:-1]), None)   # (--ctx: Momoe's ggml engine)
        ctx = args[args.index(flag) + 1] if flag else "?"
        say(f"{'*' if on else ' '} {p.name:36} {c.get('model_name', ''):40} context {ctx}{'  (running)' if on else ''}")
    return 0


# ------------------------------------------------------------------------------------------------ run <tool>
def pick_config() -> Path:
    """The config `momoe run` starts when none was named: the only one, else asked (Enter: the one started last, or
    the newest); without a terminal, that default."""
    have = configs()
    if not have:
        die("no model is set up yet: momoe setup")
    last = Path(state_last()) if state_last() else None
    default = next((p for p in have if last is not None and p.resolve() == last.resolve()), have[0])
    if len(have) == 1 or not sys.stdin.isatty():
        return default
    have = [default, *(p for p in have if p != default)]
    say("No momoe server is running. Which config?")
    for i, p in enumerate(have, 1):
        c = read_json(p) or {}
        say(f"  {i}) {p.name:36} {c.get('model_name', '')}" + ("   (started last)" if p == default and last else ""))
    while True:
        try:
            typed = input(f"Config [1]: ").strip()
        except EOFError:
            return default
        if not typed:
            return default
        if typed.isdigit() and 1 <= int(typed) <= len(have):
            return have[int(typed) - 1]
        hit = [p for p in have if typed in (p.name, p.stem, p.stem[len("strata-"):])]
        if hit:
            return hit[0]
        say(f"  a number from 1 to {len(have)}, or a config's name")


def ensure_server(cfg: Path | None = None):
    """The running server's state and /health; one is started first when none runs: `cfg` (--config), else
    pick_config.  A running server is used as is; a `cfg` that is not the one running is refused."""
    s = running()
    if s is not None and cfg is not None and Path(s.get("config", "")).resolve() != cfg.resolve():
        die(f"{s.get('model') or Path(s['config']).name} is running ({Path(s['config']).name}), not {cfg.name}: "
            "momoe stop first, or leave out --config to use it")
    if s is None:
        cfg = cfg or pick_config()
        say(f"Starting {cfg.name} first ...")
        if start_config(cfg) != 0:
            die("the server did not start (the lines above)")
        s = running()
        if s is None:
            die("the server did not start")
    end = time.time() + 1800
    h = health(s)
    if h is None:
        say("Waiting for the server to be ready ...")
    while h is None and time.time() < end:
        if not alive(s["pid"]):
            die(f"the server ended: {s.get('log')}")
        time.sleep(1)
        h = health(s)
    if h is None:
        die("the server does not answer")
    return s, h


def state_last():
    """The config started last (setup.py writes it: kept after a stop, for momoe run)."""
    s = read_json(STATE_DIR / "last.json") or {}
    return s.get("config")


def which(name):
    return shutil.which(name)


def missing(tool, how) -> int:
    say(f"{tool} is not installed. To install it:")
    for line in how:
        say("  " + line)
    say(f"then run: momoe run {sys.argv[2] if len(sys.argv) > 2 else ''}".rstrip())
    return 2


def launch(cmd, env) -> int:
    """The agent in this terminal (it replaces momoe on Linux, so Ctrl+C and the terminal are its own)."""
    if WIN:
        return subprocess.call(cmd, env=env)
    os.execvpe(cmd[0], cmd, env)
    return 0


def detached(cmd, log: Path, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS} if WIN \
        else {"start_new_session": True}
    with open(log, "ab") as out:
        return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env, **kw)


INSTALL = {
    "claude": ["curl -fsSL https://claude.ai/install.sh | bash        (Windows: irm https://claude.ai/install.ps1 | iex)",
               "or: npm install -g @anthropic-ai/claude-code"],
    "codex": ["npm install -g @openai/codex", "(https://github.com/openai/codex)"],
    "opencode": ["curl -fsSL https://opencode.ai/install | bash", "or: npm install -g opencode-ai"],
    "unsloth": ["curl -fsSL https://unsloth.ai/install.sh | sh",
                "(Windows and more: https://unsloth.ai/docs/get-started/install)"],
    "lmstudio": ["download LM Studio from https://lmstudio.ai/download and start it once"],
}


def run_claude(s, h, args) -> int:
    exe = which("claude")
    if exe is None:
        return missing("Claude Code", INSTALL["claude"])
    env = dict(os.environ)
    # these would send it to another endpoint, or to Anthropic with the user's own key
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_UNIX_SOCKET", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
              "CLAUDE_CODE_USE_FOUNDRY", "ANTHROPIC_BEDROCK_BASE_URL", "ANTHROPIC_VERTEX_BASE_URL",
              "ANTHROPIC_FOUNDRY_BASE_URL"):
        env.pop(k, None)
    model, ctx = h.get("model") or "momoe", int(h.get("max_context") or 0)
    env.update({
        "ANTHROPIC_BASE_URL": base_url(s),
        "ANTHROPIC_AUTH_TOKEN": server_key(s) or "momoe",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model, "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model, "ANTHROPIC_SMALL_FAST_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",        # a per-request header in the prompt would defeat the prompt cache
    })
    if ctx:
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(ctx)
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(ctx)
    say(f"Claude Code -> {base_url(s)} ({model}, context {ctx})")
    return launch([exe, *args], env)


def run_codex(s, h, args) -> int:
    exe = which("codex")
    if exe is None:
        return missing("Codex", INSTALL["codex"])
    model, ctx = h.get("model") or "momoe", int(h.get("max_context") or 0)
    env = dict(os.environ)
    over = ['model_provider="momoe"', 'model_providers.momoe.name="MoMoE"',
            f'model_providers.momoe.base_url="{base_url(s)}/v1"', 'model_providers.momoe.wire_api="responses"',
            "model_providers.momoe.stream_idle_timeout_ms=600000",   # a first, long prompt can take minutes
            "show_raw_agent_reasoning=true"]                         # the model writes no reasoning summaries
    if ctx:
        over.append(f"model_context_window={ctx}")
    key = server_key(s)
    if key:
        env["MOMOE_API_KEY"] = key
        over.append('model_providers.momoe.env_key="MOMOE_API_KEY"')
    cmd = [exe]
    for o in over:
        cmd += ["-c", o]
    say(f"Codex -> {base_url(s)}/v1 ({model}, context {ctx})")
    return launch([*cmd, "-m", model, *args], env)


def run_opencode(s, h, args) -> int:
    exe = which("opencode")
    if exe is None:
        return missing("OpenCode", INSTALL["opencode"])
    model, ctx = h.get("model") or "momoe", int(h.get("max_context") or 0)
    entry = {"name": model}
    if ctx:
        entry["limit"] = {"context": ctx, "output": min(32768, ctx // 4)}
    cfg = {"$schema": "https://opencode.ai/config.json",
           "provider": {"momoe": {"npm": "@ai-sdk/openai-compatible", "name": "MoMoE",
                                  "options": {"baseURL": f"{base_url(s)}/v1", "apiKey": server_key(s) or "momoe"},
                                  "models": {model: entry}}},
           "model": f"momoe/{model}", "small_model": f"momoe/{model}"}
    env = dict(os.environ)
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(cfg)  # this run only: ~/.config/opencode is not touched
    say(f"OpenCode -> {base_url(s)}/v1 ({model}, context {ctx})")
    return launch([exe, *args], env)


UNSLOTH_DB = Path.home() / ".unsloth" / "studio" / "studio.db"
UNSLOTH_URL = "http://127.0.0.1:8888"
UNSLOTH_ID = "6d6f6d6f65000000"                      # the MoMoE provider's row (16 hex digits, as Studio's own)


def unsloth_url(args) -> str:
    """Where Studio answers with these options of `unsloth studio` (-H/--host, -p/--port); 0.0.0.0 is reached on
    127.0.0.1."""
    host, port = "127.0.0.1", "8888"
    for i, a in enumerate(args):
        k, eq, v = a.partition("=")
        if not eq:
            v = args[i + 1] if i + 1 < len(args) else ""
        if k in ("-H", "--host") and v:
            host = "127.0.0.1" if v in ("0.0.0.0", "::") else v
        elif k in ("-p", "--port") and v:
            port = v
    return f"http://{'[' + host + ']' if ':' in host else host}:{port}"


def unsloth_up(url=UNSLOTH_URL) -> bool:
    try:
        return http("GET", url + "/api/health", timeout=3).get("status") == "healthy"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def unsloth_provider(s, h) -> None:
    """Studio's provider list (its own SQLite table, as its Settings page writes it): one "MoMoE" entry that
    points at this server, added or brought up to date.  The other entries stay as they are."""
    model, url = h.get("model") or "momoe", f"{base_url(s)}/v1"
    now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    con = sqlite3.connect(str(UNSLOTH_DB), timeout=10)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(llm_providers)")}
        if not cols:
            raise sqlite3.OperationalError("no llm_providers table (an older Unsloth Studio?)")
        row = con.execute("SELECT id FROM llm_providers WHERE id = ?", (UNSLOTH_ID,)).fetchone()
        models = json.dumps([model])
        if row:
            con.execute("UPDATE llm_providers SET display_name = 'MoMoE', base_url = ?, models_json = ?, available_models_json = ?, "
                        "is_enabled = 1, updated_at = ? WHERE id = ?", (url, models, models, now, UNSLOTH_ID))
        else:
            con.execute("INSERT INTO llm_providers (id, provider_type, display_name, base_url, models_json, "
                        "available_models_json, created_at, updated_at) VALUES (?, 'llama_cpp', 'MoMoE', ?, ?, ?, "
                        "?, ?)", (UNSLOTH_ID, url, models, models, now, now))
        con.commit()
    finally:
        con.close()


def run_unsloth(s, h, args) -> int:
    exe = which("unsloth")
    if exe is None:
        return missing("Unsloth Studio", INSTALL["unsloth"])
    started, url = False, unsloth_url(args)
    if not unsloth_up(url):
        log = ROOT / "logs" / "unsloth.log"
        say(f"Starting Unsloth Studio in the background (its output: {log}) ...")
        start = log.stat().st_size if log.exists() else 0
        proc = detached([exe, "studio", *args], log)
        started = True
        end = time.time() + 300
        while not unsloth_up(url) and time.time() < end and proc.poll() is None:
            time.sleep(1)
        if not unsloth_up(url):
            if proc.poll() is not None:
                with open(log, "rb") as f:
                    f.seek(start)
                    out = f.read().decode(errors="replace")
                tail = "\n".join(out.strip().splitlines()[-15:])
                hint = ("\n(-h is Studio's --help; its host option is -H, for example: momoe run unsloth -H 0.0.0.0)"
                        if "-h" in args else "")
                die(f"Unsloth Studio stopped (exit {proc.returncode}) before it answered at {url}. The end of its "
                    f"output ({log}):\n{tail}{hint}")
            die(f"Unsloth Studio did not answer at {url} within 5 minutes: {log}")
    try:
        unsloth_provider(s, h)
    except sqlite3.Error as e:
        say(f"Could not add the MoMoE provider to Unsloth Studio ({e}). Add it in Studio's settings: provider "
            f"llama.cpp, base URL {base_url(s)}/v1")
    else:
        say(f"Unsloth Studio {'started' if started else 'is running'}: {url}")
        say(f"  Its provider \"MoMoE\" points at {base_url(s)}/v1 ({h.get('model')}): pick it in the model menu.")
        if not started:
            say("  (reload the Studio page if it was open)")
    if server_key(s):
        say("  This server needs an API key: enter it in the MoMoE provider's settings in Studio.")
    return 0


def lmstudio_paths():
    """LM Studio's app and its `lms` command, where its installers put them."""
    home = Path.home()
    lms = which("lms") or next((str(p) for p in (home / ".lmstudio/bin/lms", home / ".lmstudio/bin/lms.exe")
                                if p.is_file()), None)
    app = which("lm-studio") or which("lmstudio")
    if app is None:
        cands = []
        if WIN:
            cands = [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/LM Studio/LM Studio.exe"]
        elif sys.platform == "darwin":
            cands = [Path("/Applications/LM Studio.app")]
        else:
            for d in (home / "Applications", home / "Downloads", home, Path("/opt")):
                cands += sorted(d.glob("LM-Studio*.AppImage")) + sorted(d.glob("LM_Studio*.AppImage"))
            cands += [Path("/opt/LM Studio/lm-studio"), Path("/opt/LM-Studio/lm-studio"), Path("/usr/bin/lm-studio")]
        app = next((str(p) for p in cands if p.exists()), None)
    return app, lms


LMS_PLUGIN = "nm-z/openai-compat-endpoint"          # the community plugin LM Studio reaches another server through


def lmstudio_home() -> Path:
    return Path.home() / ".lmstudio"


def lmstudio_running() -> bool:
    """LM Studio (its window, or the service `lms` starts) runs now."""
    try:
        import psutil
    except ImportError:
        return False
    names = {"lm-studio", "lm studio", "lm studio.exe", "lm studio helper"}
    for p in psutil.process_iter(["name"]):
        try:
            if (p.info.get("name") or "").lower() in names:
                return True
        except (psutil.Error, OSError):
            pass
    return False


def superjson_load(path: Path) -> dict:
    """LM Studio's SuperJSON files: {"json": ..., "meta": {"values": {key: ["map"]}}}; the maps as [[key, value], ..]."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) and isinstance(d.get("json"), dict) else {}
    except (OSError, ValueError):
        return {}


def lmstudio_plugin_config(base: str, key: str | None) -> bool:
    """The plugin's global settings (LM Studio's .internal/global-plugin-configs.json: {plugins: Map(plugin ->
    {fields: [{key, value}]})}) point at `base`, and carry the server's API key when it has one; other plugins'
    settings are kept.  True when the file was written."""
    path = lmstudio_home() / ".internal" / "global-plugin-configs.json"
    d = superjson_load(path)
    plugins = d.get("json", {}).get("plugins") if d else None
    plugins = [x for x in plugins if isinstance(x, list) and len(x) == 2] if isinstance(plugins, list) else []
    entry = next((x for x in plugins if x[0] == LMS_PLUGIN), None)
    fields = list((entry[1] or {}).get("fields") or []) if entry and isinstance(entry[1], dict) else []
    want = {"overrideBaseUrl": base, **({"openaiApiKey": key} if key else {})}
    have = {f.get("key"): f.get("value") for f in fields if isinstance(f, dict)}
    if all(have.get(k) == v for k, v in want.items()):
        return False
    fields = [f for f in fields if not (isinstance(f, dict) and f.get("key") in want)]
    fields += [{"key": k, "value": v} for k, v in want.items()]
    plugins = [x for x in plugins if x[0] != LMS_PLUGIN] + [[LMS_PLUGIN, {"fields": fields}]]
    meta = (d.get("meta") or {}) if d else {}
    values = dict(meta.get("values") or {})
    values["plugins"] = ["map"]
    out = {**(d or {}), "json": {**((d or {}).get("json") or {}), "plugins": plugins}, "meta": {**meta, "values": values}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".momoe-tmp")
    tmp.write_text(json.dumps(out), encoding="utf-8")
    os.replace(tmp, path)
    return True


def lmstudio_model_list(model: str) -> bool:
    """The plugin's model list (~/.lmstudio/openai-compat-endpoint/models-cache.json, read when it starts) has the
    server's model, so it is in LM Studio's menu without first sending a message.  True when written."""
    path = lmstudio_home() / "openai-compat-endpoint" / "models-cache.json"
    d = read_json(path)
    d = d if isinstance(d, dict) else {}
    models = [m for m in (d.get("models") or []) if isinstance(m, str)]
    if model in models:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": [model, *models], "lastSelected": d.get("lastSelected") or model}, indent=2),
                    encoding="utf-8")
    return True


def lmstudio_prepare(s, h, lms) -> bool:
    """LM Studio set up to reach this server: the plugin's settings and model list written when they are not, then
    the plugin installed with `lms get` when it is not.  The settings come first: LM Studio reads them when it starts,
    and `lms get` starts it (as a service).  False when LM Studio must be restarted to read them."""
    was_running = lmstudio_running()
    wrote = lmstudio_plugin_config(f"{base_url(s)}/v1", server_key(s))
    wrote = lmstudio_model_list(h.get("model") or "") or wrote if h.get("model") else wrote
    if wrote:
        say(f"  LM Studio's {LMS_PLUGIN} plugin set to {base_url(s)}/v1 ({h.get('model')})")
    if not (lmstudio_home() / "extensions" / "plugins" / LMS_PLUGIN / "manifest.json").is_file():
        if lms is None:
            say(f"  The {LMS_PLUGIN} plugin is not installed, and LM Studio's lms command is not found: install it "
                f"from https://lmstudio.ai/{LMS_PLUGIN} (\"Run in LM Studio\")")
        else:
            say(f"  Installing LM Studio's {LMS_PLUGIN} plugin (lms get) ...")
            r = subprocess.run([lms, "get", LMS_PLUGIN, "-y"], stdin=subprocess.DEVNULL, capture_output=True, text=True)
            if r.returncode != 0:
                say(f"  lms get failed ({r.returncode}): {(r.stderr or r.stdout).strip()[-300:]}")
                say(f"  install it from https://lmstudio.ai/{LMS_PLUGIN} (\"Run in LM Studio\")")
    if wrote and was_running:
        say("  LM Studio is running and reads these settings when it starts: quit it (also from the tray icon) and run")
        say("  momoe run lmstudio again.")
        return False
    return True


def has_display() -> bool:
    """A desktop app can open a window from here: Windows and macOS always; Linux with DISPLAY or WAYLAND_DISPLAY
    (not set in an ssh terminal, or a tmux / screen started outside the desktop)."""
    return WIN or sys.platform == "darwin" or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def run_lmstudio(s, h, args) -> int:
    app, lms = lmstudio_paths()
    if app is None:
        return missing("LM Studio", INSTALL["lmstudio"])
    if not lmstudio_prepare(s, h, lms):
        return 1
    log = ROOT / "logs" / "lmstudio.log"
    (ROOT / "logs").mkdir(exist_ok=True)
    seen = log.stat().st_size if log.exists() else 0
    if sys.platform == "darwin":
        detached(["open", "-a", app], log)
    else:
        proc = detached([app, *args], log)
        end = time.time() + 8                          # a start that fails ends within seconds (no display, ...)
        while proc.poll() is None and time.time() < end:
            time.sleep(0.5)
        if proc.poll() is not None:
            with open(log, "rb") as f:
                f.seek(seen)
                new = f.read().decode("utf-8", "replace")
            if "Another instance" not in new:          # an LM Studio already open is brought forward instead
                say(f"LM Studio did not start (exit code {proc.returncode}); the end of {log}:")
                for line in new.strip().splitlines()[-8:]:
                    say("  " + line)
                return 1
    say(f"LM Studio is starting. In a chat, open \"Pick a model\": under Your Generators, {LMS_PLUGIN}")
    say(f"(start it if it says Not Running), then {h.get('model')} in its Model setting.  The plugin is a community")
    say("one, not LM Studio's own.  momoe stop ends this server; closing LM Studio does not.")
    return 0


TOOLS = {"claude": run_claude, "codex": run_codex, "opencode": run_opencode, "unsloth": run_unsloth,
         "lmstudio": run_lmstudio}


def run_config(args: list) -> tuple:
    """`momoe run`'s own --config CONFIG (or --config=CONFIG), before the tool's name or right after it: (the config
    named or None, the arguments without it).  Only one is taken: a --config further on is the tool's own (Codex has
    one)."""
    out = list(args)
    for i in (0, 1):
        a = out[i] if i < len(out) else ""
        if a != "--config" and not a.startswith("--config="):
            continue
        if "=" in a:
            name, n = a.split("=", 1)[1], 1
        elif i + 1 < len(out):
            name, n = out[i + 1], 2
        else:
            die("--config takes a config's name: momoe run TOOL --config CONFIG (momoe list shows them)")
        del out[i:i + n]
        return find_config(name), out
    return None, out


def cmd_run(args) -> int:
    cfg, args = run_config(args)
    if not args or args[0] not in TOOLS:
        die("momoe run " + "|".join(TOOLS) + " [--config CONFIG] [arguments]")
    tool = args[0]
    # a missing tool is named before the model is started for it
    if tool == "lmstudio":
        if lmstudio_paths()[0] is None:
            return missing("LM Studio", INSTALL["lmstudio"])
    elif which(tool) is None:
        return missing({"claude": "Claude Code", "codex": "Codex", "opencode": "OpenCode",
                        "unsloth": "Unsloth Studio"}[tool], INSTALL[tool])
    if tool == "lmstudio" and not has_display():       # checked before a server is started for it
        die("LM Studio is a desktop app, and this terminal has no display (DISPLAY is not set: an ssh terminal, or a "
            "tmux / screen started outside the desktop). Run it from a terminal on the desktop, or set DISPLAY first "
            "(export DISPLAY=:0 for the first desktop)")
    s, h = ensure_server(cfg)
    return TOOLS[tool](s, h, args[1:])


def main(argv) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        say(__doc__.strip())
        return 0
    cmds = {"setup": cmd_setup, "start": cmd_start, "stop": cmd_stop, "status": cmd_status, "list": cmd_list,
            "log": cmd_log, "run": cmd_run}
    if argv[0] not in cmds:
        die(f"unknown command {argv[0]!r}: momoe " + "|".join(cmds) + " (momoe help)")
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(130)
