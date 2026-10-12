#!/usr/bin/env python3
"""Momoe: more MoEs, built on Strata (https://github.com/Niko1221/Strata).

    momoe setup [setup options]          install a model or change its settings (setup.py --setup), then start it
    momoe start CONFIG [setup options]   start the server from a config, in the background
    momoe stop                           stop it
    momoe status                         is it running, which model, where
    momoe list                           the configs in config/
    momoe run claude|codex|opencode [their arguments]
                                         start that agent, talking to this server (started first if needed)
    momoe run unsloth|lmstudio           open that app, pointed at this server

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
        return http("GET", base_url(s) + "/health", api_key(s.get("config")), timeout=3)
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
    return subprocess.call(setup_cmd(["--config", str(cfg), *([] if foreground else ["--detach"]), *extra]),
                           cwd=str(ROOT))


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
        http("POST", base_url(s) + "/shutdown", api_key(s.get("config")), {}, timeout=10)
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
        say("No momoe server is running." + (f" Last config: {configs()[0].name}" if configs() else ""))
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


def cmd_list(args) -> int:
    s = running()
    have = configs()
    if not have:
        say("No configs yet: momoe setup")
        return 0
    for p in have:
        c = read_json(p) or {}
        on = s is not None and Path(s["config"]).resolve() == p.resolve()
        ctx = c["args"][c["args"].index("--max-context") + 1] if "--max-context" in c.get("args", [])[:-1] else "?"
        say(f"{'*' if on else ' '} {p.name:36} {c.get('model_name', ''):40} context {ctx}{'  (running)' if on else ''}")
    return 0


# ------------------------------------------------------------------------------------------------ run <tool>
def ensure_server():
    """The running server's state and /health; one is started first (the last config started) when none runs."""
    s = running()
    if s is None:
        last = Path(state_last()) if state_last() else None
        cfg = last if last is not None and last.is_file() else (configs() or [None])[0]
        if cfg is None:
            die("no model is set up yet: momoe setup")
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
    STATE_DIR.mkdir(exist_ok=True)
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
        "ANTHROPIC_AUTH_TOKEN": api_key(s["config"]) or "momoe",
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
    over = ['model_provider="momoe"', 'model_providers.momoe.name="Momoe"',
            f'model_providers.momoe.base_url="{base_url(s)}/v1"', 'model_providers.momoe.wire_api="responses"',
            "model_providers.momoe.stream_idle_timeout_ms=600000",   # a first, long prompt can take minutes
            "show_raw_agent_reasoning=true"]                         # the model writes no reasoning summaries
    if ctx:
        over.append(f"model_context_window={ctx}")
    key = api_key(s["config"])
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
           "provider": {"momoe": {"npm": "@ai-sdk/openai-compatible", "name": "Momoe",
                                  "options": {"baseURL": f"{base_url(s)}/v1", "apiKey": api_key(s["config"]) or "momoe"},
                                  "models": {model: entry}}},
           "model": f"momoe/{model}", "small_model": f"momoe/{model}"}
    env = dict(os.environ)
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(cfg)  # this run only: ~/.config/opencode is not touched
    say(f"OpenCode -> {base_url(s)}/v1 ({model}, context {ctx})")
    return launch([exe, *args], env)


UNSLOTH_DB = Path.home() / ".unsloth" / "studio" / "studio.db"
UNSLOTH_URL = "http://127.0.0.1:8888"
UNSLOTH_ID = "6d6f6d6f65000000"                      # the Momoe provider's row (16 hex digits, as Studio's own)


def unsloth_up() -> bool:
    try:
        return http("GET", UNSLOTH_URL + "/api/health", timeout=3).get("status") == "healthy"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def unsloth_provider(s, h) -> None:
    """Studio's provider list (its own SQLite table, as its Settings page writes it): one "Momoe" entry that
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
            con.execute("UPDATE llm_providers SET base_url = ?, models_json = ?, available_models_json = ?, "
                        "is_enabled = 1, updated_at = ? WHERE id = ?", (url, models, models, now, UNSLOTH_ID))
        else:
            con.execute("INSERT INTO llm_providers (id, provider_type, display_name, base_url, models_json, "
                        "available_models_json, created_at, updated_at) VALUES (?, 'llama_cpp', 'Momoe', ?, ?, ?, "
                        "?, ?)", (UNSLOTH_ID, url, models, models, now, now))
        con.commit()
    finally:
        con.close()


def run_unsloth(s, h, args) -> int:
    exe = which("unsloth")
    if exe is None:
        return missing("Unsloth Studio", INSTALL["unsloth"])
    started = False
    if not unsloth_up():
        log = STATE_DIR / "unsloth.log"
        say(f"Starting Unsloth Studio in the background (its output: {log}) ...")
        detached([exe, "studio", *args], log)
        started = True
        end = time.time() + 300
        while not unsloth_up() and time.time() < end:
            time.sleep(1)
        if not unsloth_up():
            die(f"Unsloth Studio did not answer at {UNSLOTH_URL} within 5 minutes: {log}")
    try:
        unsloth_provider(s, h)
    except sqlite3.Error as e:
        say(f"Could not add the Momoe provider to Unsloth Studio ({e}). Add it in Studio's settings: provider "
            f"llama.cpp, base URL {base_url(s)}/v1")
    else:
        say(f"Unsloth Studio {'started' if started else 'is running'}: {UNSLOTH_URL}")
        say(f"  Its provider \"Momoe\" points at {base_url(s)}/v1 ({h.get('model')}): pick it in the model menu.")
        if not started:
            say("  (reload the Studio page if it was open)")
    if api_key(s["config"]):
        say("  This server needs an API key: enter it in the Momoe provider's settings in Studio.")
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
            cands += [Path("/opt/LM Studio/lm-studio"), Path("/usr/bin/lm-studio")]
        app = next((str(p) for p in cands if p.exists()), None)
    return app, lms


def run_lmstudio(s, h, args) -> int:
    app, lms = lmstudio_paths()
    if app is None:
        return missing("LM Studio", INSTALL["lmstudio"])
    log = STATE_DIR / "lmstudio.log"
    if sys.platform == "darwin":
        detached(["open", "-a", app], log)
    else:
        detached([app, *args], log)
    say("LM Studio is starting. It runs its own models, so it reaches this server through a plugin:")
    say("  1. In LM Studio, install the plugin openai-compat-endpoint:")
    say("       https://lmstudio.ai/nm-z/openai-compat-endpoint   (\"Run in LM Studio\")")
    say(f"  2. In the plugin's settings, set the base URL to {base_url(s)}/v1"
        + (" and the API key to the server's key" if api_key(s["config"]) else ""))
    say(f"  3. Pick {h.get('model')} in the model menu.")
    say("  (The plugin is a community one, not LM Studio's own.)")
    return 0


TOOLS = {"claude": run_claude, "codex": run_codex, "opencode": run_opencode, "unsloth": run_unsloth,
         "lmstudio": run_lmstudio}


def cmd_run(args) -> int:
    if not args or args[0] not in TOOLS:
        die("momoe run " + "|".join(TOOLS) + " [arguments]")
    tool = args[0]
    # a missing tool is named before the model is started for it
    if tool == "lmstudio":
        if lmstudio_paths()[0] is None:
            return missing("LM Studio", INSTALL["lmstudio"])
    elif which(tool) is None:
        return missing({"claude": "Claude Code", "codex": "Codex", "opencode": "OpenCode",
                        "unsloth": "Unsloth Studio"}[tool], INSTALL[tool])
    s, h = ensure_server()
    return TOOLS[tool](s, h, args[1:])


def main(argv) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        say(__doc__.strip())
        return 0
    cmds = {"setup": cmd_setup, "start": cmd_start, "stop": cmd_stop, "status": cmd_status, "list": cmd_list,
            "run": cmd_run}
    if argv[0] not in cmds:
        die(f"unknown command {argv[0]!r}: momoe " + "|".join(cmds) + " (momoe help)")
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(130)
