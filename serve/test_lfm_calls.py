"""LFM2's tool-call format (LfmCallRewriter in OutputParser(lfm=True)): a Python list of calls between
<|tool_call_start|> and <|tool_call_end|>; values as Python or JSON literals, parallel calls, prose that names the tag,
a list that is not calls, an output that ends before the end tag; streamed in pieces or whole, the same events.  And
the chat template's {% generation %} blocks and {{ bos_token }}."""
import json
import random
import tempfile
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, OutputParser, lfm_calls, lfm_forced

TOOLS = [{"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                                              "limit": {"type": "integer"}}}},
         {"name": "write", "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                                          "content": {"type": "string"}}}},
         {"name": "search", "parameters": {"type": "object", "properties": {"q": {"type": "string"},
                                                                           "opts": {"type": "object"}}}}]
S, E = "<|tool_call_start|>", "<|tool_call_end|>"


def run(chunks):
    p = OutputParser(thinking=True, tools=TOOLS, stream_tools=True, lfm=True)
    evs = []
    for c in chunks:
        evs += p.feed(c)
    evs += p.finish()
    out = []
    for e in evs:
        if e.kind not in ("content", "reasoning", "tool_call"):
            continue
        v = e.text if e.kind != "tool_call" else (e.call.name, e.call.arguments)
        if out and e.kind != "tool_call" and out[-1][0] == e.kind:
            out[-1] = (e.kind, out[-1][1] + v)
        else:
            out.append((e.kind, v))
    return out


class LfmCalls(unittest.TestCase):
    CASES = [
        ("thinking</think>Let me look." + S + "[read_file(path='setup.py', limit=40)]" + E,
         [("reasoning", "thinking"), ("content", "Let me look."), ("tool_call", ("read_file", {"path": "setup.py", "limit": 40}))]),
        ("x</think>" + S + "[write(path=\"doc.md\", content='Use <|tool_call_end|> rarely.\\nok')]" + E,
         None),                                          # (the end tag inside a value: see test_end_tag_in_value)
        ("x</think>Plain answer naming " + S + " in text, no call.",
         [("reasoning", "x"), ("content", "Plain answer naming " + S + " in text, no call.")]),
        ("x</think>" + S + "[read_file(path='a.py'), read_file(path='b.py')]" + E,
         [("reasoning", "x"), ("tool_call", ("read_file", {"path": "a.py"})), ("tool_call", ("read_file", {"path": "b.py"}))]),
        ("x</think>" + S + "[search(q='x', opts={\"deep\": true, \"n\": null, \"k\": -2})]" + E,
         [("reasoning", "x"), ("tool_call", ("search", {"q": "x", "opts": {"deep": True, "n": None, "k": -2}}))]),
        ("x</think>" + S + "[read_file(path='t.py')]",
         [("reasoning", "x"), ("tool_call", ("read_file", {"path": "t.py"}))]),
        ("x</think>" + S + "[1, 2, 3]" + E + " done",
         [("reasoning", "x"), ("content", S + "[1, 2, 3]" + E + " done")]),
    ]

    def test_whole_and_streamed(self):
        rnd = random.Random(3)
        for text, want in self.CASES:
            if want is None:
                continue
            self.assertEqual(run([text]), want, text)
            self.assertEqual(run(list(text)), want, text)
            for _ in range(10):
                cuts = sorted(rnd.sample(range(1, len(text)), min(30, len(text) - 1)))
                self.assertEqual(run([text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]), want, text)

    def test_end_tag_in_value(self):
        # the model writes the end tag as a token after the list: a value that holds its text ends the list early,
        # and what was read is not a list of calls, so it stays text (as written)
        text = self.CASES[1][0]
        got = run([text])
        self.assertEqual(got[0], ("reasoning", "x"))
        self.assertEqual("".join(v for k, v in got[1:] if k == "content"), text[len("x</think>"):])

    def test_calls(self):
        self.assertIsNone(lfm_calls("read_file('a.py')"))           # positional: not the format
        self.assertIsNone(lfm_calls("[f(x=open('a'))]"))            # not a literal
        self.assertIn("<function=mcp.fs.read>", lfm_calls("[mcp.fs.read(path='a')]"))
        self.assertIn("<parameter=n>\n3\n</parameter>", lfm_calls("f(n=3)"))   # one call without the list

    def test_forced_opening(self):
        self.assertEqual(lfm_forced("<tool_call>\n<function="), S + "[")
        self.assertEqual(lfm_forced("<tool_call>\n<function=read_file>\n"), S + "[read_file(")
        self.assertIsNone(lfm_forced(None))


class Opening(unittest.TestCase):
    """LFM2.5-8B-A1B's template ends its prompt at "<|im_start|>assistant\n" (the 24B's: "...\n<think>"): the reply
    opens its thinking itself, or answers at once (after a tool's result, measured)."""
    def run_opening(self, chunks):
        p = OutputParser(thinking=True, tools=TOOLS, stream_tools=True, lfm=True, opened=False)
        evs = []
        for c in chunks:
            evs += p.feed(c)
        evs += p.finish()
        out = []
        for e in evs:
            if e.kind in ("content", "reasoning") and e.text:
                if out and out[-1][0] == e.kind:
                    out[-1] = (e.kind, out[-1][1] + e.text)
                else:
                    out.append((e.kind, e.text))
            elif e.kind == "tool_call":
                out.append(("tool_call", (e.call.name, e.call.arguments)))
        return out

    def test_own_think(self):
        text = "<think>Let me check.</think>" + S + "[read_file(path='a.py')]" + E
        want = [("reasoning", "Let me check."), ("tool_call", ("read_file", {"path": "a.py"}))]
        self.assertEqual(self.run_opening([text]), want)
        self.assertEqual(self.run_opening(list(text)), want)

    def test_answer_at_once(self):
        text = "The current weather in Paris is 14\u00b0C with light rain."
        self.assertEqual(self.run_opening([text]), [("content", text)])
        self.assertEqual(self.run_opening(list(text)), [("content", text)])

    def test_whitespace_before_think_and_a_lone_part(self):
        self.assertEqual(self.run_opening(["\n", "<th", "ink>x</think>y"]), [("reasoning", "x"), ("content", "y")])
        self.assertEqual(self.run_opening(["<thi"]), [("content", "<thi")])
        self.assertEqual(self.run_opening(["  \n"]), [])

    def test_template_check(self):
        from serve.server import template_opens_think
        mk = lambda src, out: type("T", (), {"source": src, "render": lambda self, *a, **k: out})()
        self.assertTrue(template_opens_think(mk("x <think> y", "<|im_start|>assistant\n<think>")))
        self.assertFalse(template_opens_think(mk("x <think> y", "<|im_start|>assistant\n")))
        self.assertTrue(template_opens_think(mk("no tag", "<|im_start|>assistant\n")))     # (says nothing of it)


class PromptOpens(unittest.TestCase):
    """Where a reply starts is read from the prompt itself: GLM-5.3-Flash's template ends every prompt with <think>
    (thinking cannot be turned off), so a request without thinking still gets its reasoning apart (measured: it went
    into the answer text)."""
    def svc(self, src, tail):
        from serve.server import Service
        s = Service.__new__(Service)
        s.template = type("T", (), {"source": src})()
        s.tok = type("K", (), {"decode": lambda self, ids: tail})()
        return s

    def test_prompt_ending_in_think(self):
        self.assertTrue(self.svc("x <think> y", "<|assistant|><think>").prompt_opens_think([1, 2, 3]))
        self.assertTrue(self.svc("x <think> y", "assistant\n<think>\n").prompt_opens_think([1]))
        self.assertFalse(self.svc("x <think> y", "<|im_start|>assistant\n").prompt_opens_think([1]))
        self.assertFalse(self.svc("no tag", "<think>").prompt_opens_think([1]))   # (a template that never writes it)
        self.assertFalse(self.svc("x <think> y", "<think>").prompt_opens_think([]))


class LfmTemplate(unittest.TestCase):
    SRC = ("{{- bos_token -}}{%- for m in messages -%}<|im_start|>{{ m.role }}\n"
           "{%- if m.role == 'assistant' -%}{%- generation -%}{{ m.content }}{%- endgeneration -%}"
           "{%- else -%}{{ m.content }}{%- endif -%}<|im_end|>\n{%- endfor -%}")

    def make(self, pack):
        d = Path(tempfile.mkdtemp())
        (d / "chat_template.jinja").write_text(self.SRC, encoding="utf-8")
        if pack is not None:
            (d / "tokenizer.json").write_text(json.dumps(pack), encoding="utf-8")
        return ChatTemplate(d / "chat_template.jinja")

    def test_generation_and_bos(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        self.assertEqual(self.make({"bos_token": "<|startoftext|>"}).render(msgs, add_generation_prompt=False),
                         "<|startoftext|><|im_start|>userhi<|im_end|><|im_start|>assistantyo<|im_end|>")   # (its "-"s)
        # a pack without one (every pack before this): nothing is written, as before
        self.assertTrue(self.make({}).render(msgs, add_generation_prompt=False).startswith("<|im_start|>user"))
        self.assertTrue(self.make(None).render(msgs, add_generation_prompt=False).startswith("<|im_start|>user"))


if __name__ == "__main__":
    unittest.main()
