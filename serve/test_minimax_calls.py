"""MiniMax M3's reply format (its chat template's): thinking between <mm:think> and </mm:think> (a reply that does not
think starts with </mm:think> alone), and tool calls as namespaced XML - read by MinimaxRewriter into OutputParser's
form, whatever the deltas' sizes.

    python -m unittest serve.test_minimax_calls
"""
import unittest

from serve.frontend import CALL_START, FUNC_START, MM_NS, OutputParser, minimax_calls, minimax_forced

N = MM_NS
TOOLS = [{"name": "get_weather", "parameters": {"type": "object", "properties": {
             "city": {"type": "string"}, "days": {"type": "integer"}, "opts": {"type": "object"}}}},
         {"name": "Write", "parameters": {"type": "object", "properties": {
             "file_path": {"type": "string"}, "content": {"type": "string"}}}}]


def run(text, step, reason="stop"):
    p = OutputParser(thinking=True, tools=TOOLS, stream_tools=False, recover=True, opened=False, minimax=True)
    ev = []
    for i in range(0, len(text), step):
        ev += p.feed(text[i:i + step])
    ev += p.finish(reason)
    out = []
    for e in ev:
        if e.kind in ("reasoning", "content"):
            if out and out[-1][0] == e.kind:
                out[-1] = (e.kind, out[-1][1] + e.text)
            else:
                out.append((e.kind, e.text))
        elif e.kind == "tool_call":
            out.append(("tool_call", e.call.name, e.call.arguments))
    return out


def call(name, args):
    return N + '<invoke name="' + name + '">' + "".join(N + "<" + k + ">" + v + N + "</" + k + ">" for k, v in args) + N + "</invoke>\n"


class Thinking(unittest.TestCase):
    def test_a_thought_then_the_answer(self):
        for step in (1, 3, 1000):
            self.assertEqual(run("<mm:think>Check the map.</mm:think>Canberra.", step),
                             [("reasoning", "Check the map."), ("content", "Canberra.")])

    def test_no_thinking_prefix(self):
        for step in (1, 2, 1000):
            self.assertEqual(run("</mm:think>Canberra is the capital.", step), [("content", "Canberra is the capital.")])


class Calls(unittest.TestCase):
    def test_typed_and_nested_arguments(self):
        nested = N + "<units>C" + N + "</units>" + N + "<list>" + N + "<item>1" + N + "</item>" + N + "<item>x" + N + "</item>" + N + "</list>"
        text = ("<mm:think>Need weather.</mm:think>" + N + "<tool_call>\n"
                + call("get_weather", [("city", "Paris"), ("days", "3"), ("opts", nested)]) + N + "</tool_call>")
        for step in (1, 3, 1000):
            self.assertEqual(run(text, step), [("reasoning", "Need weather."),
                                               ("tool_call", "get_weather", {"city": "Paris", "days": 3,
                                                                             "opts": {"units": "C", "list": [1, "x"]}})])

    def test_a_value_with_its_own_tags_stays_whole(self):
        text = "</mm:think>" + N + "<tool_call>\n" + call("Write", [("file_path", "a.html"), ("content", "<div>x < y</div>\n")]) + N + "</tool_call>"
        for step in (1, 4, 1000):
            self.assertEqual(run(text, step), [("tool_call", "Write", {"file_path": "a.html", "content": "<div>x < y</div>\n"})])

    def test_two_calls_in_one_block(self):
        text = ("</mm:think>" + N + "<tool_call>\n" + call("get_weather", [("city", "Paris")])
                + call("get_weather", [("city", "Oslo")]) + N + "</tool_call>")
        self.assertEqual([e[2]["city"] for e in run(text, 5) if e[0] == "tool_call"], ["Paris", "Oslo"])

    def test_not_calls_stay_text(self):
        self.assertIsNone(minimax_calls("no tags here"))
        self.assertIsNone(minimax_calls(N + "<city>Paris" + N + "</city>"))           # (no invoke)
        self.assertIsNone(minimax_calls(N + '<invoke name="x">' + N + "<a>1"))        # (unclosed)

    def test_a_forced_call_opens_in_its_form(self):
        self.assertEqual(minimax_forced(CALL_START + "\n" + FUNC_START + "get_weather>\n"),
                         "</mm:think>" + N + "<tool_call>\n" + N + '<invoke name="get_weather">')


if __name__ == "__main__":
    unittest.main()
