"""GLM-5's tool-call format (GlmCallRewriter in OutputParser(glm=True)): calls, values that hold the tags, prose that
names the tag, parallel calls, an output that ends on a value; streamed in pieces or whole, the same events."""
import random
import unittest

from serve.frontend import OutputParser, glm_forced

TOOLS = [{"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                                              "limit": {"type": "integer"}}}},
         {"name": "write", "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                                          "content": {"type": "string"}}}}]


def run(chunks):
    p = OutputParser(thinking=True, tools=TOOLS, stream_tools=True, glm=True)
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


class GlmCalls(unittest.TestCase):
    CASES = [
        ("thinking</think>Let me look.\n<tool_call>read_file\n<arg_key>path</arg_key>\n<arg_value>setup.py</arg_value>\n"
         "<arg_key>limit</arg_key>\n<arg_value>40</arg_value>\n</tool_call>",
         [("reasoning", "thinking"), ("content", "Let me look."), ("tool_call", ("read_file", {"path": "setup.py", "limit": 40}))]),
        ("x</think><tool_call>write<arg_key>path</arg_key><arg_value>doc.md</arg_value><arg_key>content</arg_key>"
         "<arg_value>Use <arg_value>v</arg_value> and </arg_value> tags.\nok</arg_value></tool_call>",
         [("reasoning", "x"), ("tool_call", ("write", {"path": "doc.md", "content": "Use <arg_value>v</arg_value> and </arg_value> tags.\nok"}))]),
        ("x</think>Plain answer with a <tool_call> mention in text, no call.",
         [("reasoning", "x"), ("content", "Plain answer with a <tool_call> mention in text, no call.")]),
        ("x</think><tool_call>read_file<arg_key>path</arg_key><arg_value>a.py</arg_value></tool_call>\n"
         "<tool_call>read_file<arg_key>path</arg_key><arg_value>b.py</arg_value></tool_call>",
         [("reasoning", "x"), ("tool_call", ("read_file", {"path": "a.py"})), ("tool_call", ("read_file", {"path": "b.py"}))]),
        ("x</think><tool_call>write<arg_key>path</arg_key><arg_value>t</arg_value><arg_key>content</arg_key>"
         "<arg_value>ends here</arg_value>",
         [("reasoning", "x"), ("tool_call", ("write", {"path": "t", "content": "ends here"}))]),
    ]

    def test_whole_and_streamed(self):
        rnd = random.Random(3)
        for text, want in self.CASES:
            self.assertEqual(run([text]), want)
            self.assertEqual(run(list(text)), want)
            for _ in range(10):
                cuts = sorted(rnd.sample(range(1, len(text)), min(30, len(text) - 1)))
                self.assertEqual(run([text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]), want)

    def test_forced_opening(self):
        self.assertEqual(glm_forced("<tool_call>\n<function="), "<tool_call>")
        self.assertEqual(glm_forced("<tool_call>\n<function=write>\n"), "<tool_call>write")
        self.assertIsNone(glm_forced(None))


if __name__ == "__main__":
    unittest.main()
