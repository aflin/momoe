"""serve/test_gemma_calls.py - Gemma 4's thinking and tool calls through OutputParser (GemmaRewriter): its markers
rewritten into the parser's own as they arrive, whole or a few characters a delta."""
import unittest

from serve.frontend import OutputParser, gemma_call, gemma_forced

TOOLS = [{"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                                            "limit": {"type": "integer"}}}}]


def run(text, step, thinking=True):
    p = OutputParser(thinking=thinking, tools=TOOLS, gemma=True, opened=False)
    ev = []
    for i in range(0, len(text), step):
        ev += p.feed(text[i:i + step])
    ev += p.finish("stop")
    out = []
    for e in ev:
        if e.kind == "tool_call":
            out.append(("tool_call", e.call.name, e.call.arguments))
        elif e.kind in ("reasoning", "content"):
            if out and out[-1][0] == e.kind:
                out[-1] = (e.kind, out[-1][1] + e.text)
            else:
                out.append((e.kind, e.text))
    return out


class GemmaCalls(unittest.TestCase):
    CASES = [
        ("<|channel>thought\nThinking here.<channel|>The answer.",
         [("reasoning", "Thinking here."), ("content", "The answer.")]),
        ('Let me look.<|tool_call>call:read_file{path:<|"|>setup.py<|"|>,limit:40}<tool_call|>',
         [("content", "Let me look."), ("tool_call", "read_file", {"path": "setup.py", "limit": 40})]),
        ('<|channel>thought\nhmm<channel|><|tool_call>call:read_file{<|"|>path<|"|>:<|"|>a, b}.py<|"|>}<tool_call|>',
         [("reasoning", "hmm"), ("tool_call", "read_file", {"path": "a, b}.py"})]),
        ("Plain text with <|tool_call> mention but no end", [("content", "Plain text with <|tool_call> mention but no end")]),
        # the reply stops at <|tool_response> (a stop token): the call without its end tag
        ('<|tool_call>call:read_file{path:<|"|>x.py<|"|>}', [("tool_call", "read_file", {"path": "x.py"})]),
    ]

    def test_cases_whole_and_split(self):
        for text, want in self.CASES:
            for step in (len(text), 1, 3, 17):
                self.assertEqual(run(text, step), want, (text, step))

    def test_values(self):
        got = gemma_call('call:f{a:<|"|>s<|"|>,b:[1,2.5,true,null],c:{d:<|"|>e<|"|>},n:-3}')
        self.assertIn("<function=f>", got)
        self.assertIn('[1, 2.5, true, null]', got)
        self.assertIn('{"d": "e"}', got)
        self.assertIn("\n-3\n", got)
        self.assertIsNone(gemma_call("not a call"))
        self.assertIsNone(gemma_call('call:f{a:<|"|>unclosed}'))

    def test_forced(self):
        self.assertEqual(gemma_forced("<tool_call>\n<function="), "<|tool_call>call:")
        self.assertEqual(gemma_forced("<tool_call>\n<function=read_file>\n"), "<|tool_call>call:read_file{")


if __name__ == "__main__":
    unittest.main()
