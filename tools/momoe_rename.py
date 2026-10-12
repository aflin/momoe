"""tools/momoe_rename.py - Momoe's visible name: the engine's log prefixes ("strata serve:" -> "momoe serve:", "[strata]"
-> "[momoe]", "strata: ..." -> "momoe: ...") and "Strata" in the messages people read (setup, the server, the tools, the
engine's errors).  The C++ namespaces, include paths, STRATA_* settings, the GGUF keys (strata.ple.*), the shared
settings folder, Strata-data, upstream URLs, docstrings and docs keep Strata's names.

Forward only, and running it again changes nothing.  Merging a Strata release: revert the commit this script made, merge,
run it again, commit.

    python tools/momoe_rename.py          # rewrite the files
    python tools/momoe_rename.py --check  # list what it would change, exit 1 if anything
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# the engine's log prefixes (src/, include/): "strata <part>:" at the start of a line it prints
ENGINE_PARTS = ("generate", "serve", "mtp", "verify", "batch", "trace", "prefill", "hc", "pipeline", "dbg", "rss", "pf-step",
                "aux cpus", "aux")
PREFIX = re.compile(r"(?:(?<=\\n)|(?<=\\t)|\b)strata (" + "|".join(re.escape(p) for p in sorted(ENGINE_PARTS, key=len, reverse=True)) + r"):")
BRACKET = re.compile(r"\[strata\]")
BARE = re.compile(r'(["\'])strata: ')                     # "strata: warning: ..." (the engine's own warnings)
# "Strata" the program, in a message; not Strata-data, Strata-main, Strata*, a URL's /Strata, a host name, or Strata's
# own repo
WORD = re.compile(r"(?<![/\w-])Strata(?![-\w*/]|\.\w)")

# strings about Strata's downloadable engine releases keep the name: they are Strata's
RELEASES = re.compile(r"ready-made|GitHub|from Strata \d|Strata (?:AMD )?engine\"|this Strata version")
CPP = (".cpp", ".cu", ".hpp", ".cuh", ".inl", ".h")
CPP_DIRS = ("src", "include", "tests", "tools/vision", "bench/micro")
PY_FILES = ("setup.py", "momoe.py", "chat.py")
PY_DIRS = ("serve", "tools", "tests")
CPP_STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')


def cpp(text: str) -> str:
    text = PREFIX.sub(lambda m: "momoe " + m.group(1) + ":", text)
    text = BRACKET.sub("[momoe]", text)
    text = BARE.sub(lambda m: m.group(1) + "momoe: ", text)
    # "Strata" inside string literals only (comments keep the project's name)
    out = []
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("//"):
            out.append(line)
            continue
        out.append(CPP_STRING.sub(lambda m: WORD.sub("Momoe", m.group(0)), line))
    return "".join(out)


def docstring_starts(source: str) -> set:
    """(row, col) of the string tokens that are expression statements: docstrings and bare strings."""
    starts = set()
    prev = tokenize.NEWLINE
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.STRING and prev in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT,
                                                    tokenize.ENCODING):
            starts.add(tok.start)
        if tok.type not in (tokenize.COMMENT, tokenize.NL):
            prev = tok.type
    return starts


def py(text: str) -> str:
    text = PREFIX.sub(lambda m: "momoe " + m.group(1) + ":", text)
    text = BRACKET.sub("[momoe]", text)
    text = BARE.sub(lambda m: m.group(1) + "momoe: ", text)
    try:
        skip = docstring_starts(text)
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return text
    lines = text.splitlines(keepends=True)
    edits = []                                         # (row, col0, col1, new) on one line, or whole multi-line strings
    for tok in toks:
        if tok.type != tokenize.STRING or tok.start in skip or "Strata" not in tok.string:
            continue
        if tok.string.strip("\"'rRbBfFuU") in ("Strata", "Niko1221/Strata"):
            continue                                   # a path or a repo name, not a message
        if RELEASES.search(tok.string):
            continue                                   # Strata's own releases (Momoe publishes none)
        edits.append((tok.start, tok.end, WORD.sub("Momoe", tok.string)))
    for (r0, c0), (r1, c1), new in reversed(edits):   # from the end, so earlier positions stay valid
        if r0 == r1:
            ln = lines[r0 - 1]
            lines[r0 - 1] = ln[:c0] + new + ln[c1:]
        else:
            head, tail = lines[r0 - 1][:c0], lines[r1 - 1][c1:]
            lines[r0 - 1:r1] = [head + new + tail]
    return "".join(lines)


def files():
    for d in CPP_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.suffix in CPP and p.is_file():
                yield p, cpp
    for f in PY_FILES:
        if (ROOT / f).is_file():
            yield ROOT / f, py
    for d in PY_DIRS:
        for p in sorted((ROOT / d).rglob("*.py")):
            if p.name != "momoe_rename.py":
                yield p, py


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only list the files it would change")
    a = ap.parse_args()
    changed = []
    for path, fix in files():
        old = path.read_text(encoding="utf-8", errors="surrogateescape")   # a stray non-UTF-8 byte stays as it was
        new = fix(old)
        if new != old:
            changed.append(path.relative_to(ROOT))
            if not a.check:
                path.write_text(new, encoding="utf-8", errors="surrogateescape")
    for p in changed:
        print(("would change " if a.check else "changed ") + str(p))
    print(f"{len(changed)} file(s)")
    return 1 if a.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
