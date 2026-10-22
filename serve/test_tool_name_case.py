"""serve/test_tool_name_case.py - a call naming a declared tool in another case (Flint wrote `bash` for Claude Code's
`Bash`) is that tool's call, with the declared spelling; whole or a character a delta, in Qwen's form and through
GLM's rewriter (Flint's template); a declared name is never changed, nor one that matches two declared names."""
import unittest

from serve.frontend import OutputParser

CALL = "<tool_call>\n<function={}>\n<parameter=command>\npwd && ls -la\n</parameter>\n</function>\n</tool_call>"


def calls(text, names, step=None, glm=False):
    tools = [{"name": n, "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}} for n in names]
    p = OutputParser(thinking=False, tools=tools, glm=glm, opened=False)
    ev = []
    step = step or len(text) or 1
    for i in range(0, len(text), step):
        ev += p.feed(text[i:i + step])
    ev += p.finish("stop")
    return ([(e.call.name, e.call.arguments) for e in ev if e.kind == "tool_call"],
            "".join(e.text for e in ev if e.kind == "content"))


class NameCase(unittest.TestCase):
    def test_other_case_is_the_declared_tool(self):
        for glm in (False, True):
            for step in (None, 1, 5):
                got, text = calls(CALL.format("bash") * 2, ["Bash", "Read"], step, glm)
                self.assertEqual(got, [("Bash", {"command": "pwd && ls -la"})] * 2, (glm, step))
                self.assertEqual(text.strip(), "")

    def test_declared_names_unchanged(self):
        got, _ = calls(CALL.format("bash"), ["bash", "Bash"])
        self.assertEqual(got, [("bash", {"command": "pwd && ls -la"})])

    def test_two_matches_stay_text(self):
        got, text = calls(CALL.format("BASH"), ["bash", "Bash"])
        self.assertEqual(got, [])
        self.assertIn("<function=BASH>", text)

    def test_prose_untouched(self):
        for t in ("Use <function=bash to call it.", "a <function=\nbash> b", "x <function=bash> y"):
            got, text = calls(t, ["Bash"])
            self.assertEqual(text, t.replace("<function=bash>", "<function=Bash>") if "<function=bash>" in t else t)


if __name__ == "__main__":
    unittest.main()
