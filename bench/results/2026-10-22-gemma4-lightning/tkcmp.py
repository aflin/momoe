"""tkcmp.py GGUF PORT: strata_tokenizer's ids against llama.cpp's /tokenize (parse_special off and on)."""
import json, sys, urllib.request
sys.path.insert(0, "/usr3/local/src/momoe/tools")
import strata_tokenizer as ST
tk = ST.Tokenizer.from_gguf(sys.argv[1])
port = sys.argv[2]
def ll(text, special):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", json.dumps({"content": text, "add_special": False, "parse_special": special}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["tokens"]
corpus = ["", "Hello, world!", "  leading and trailing  ", "a\n\n\nb", "\n", "\n\n\n\n\n\n\n", " \n \n",
          "def f(x):\n\treturn x  # comment\n", "你好，世界", "مرحبا",
          "\U0001f600\U0001f680\U0001f1fa\U0001f1f8", "é́ combining", "\x00\x01\x7f control",
          " non-breaking space", "1234567890 3.14159", "MixedCASE_and-dashes", "\r\n\r\n", "  ",
          "x" * 3000, "The quick brown fox jumps over the lazy dog.   Multiple   spaces.",
          "<|turn>user\nHi there<turn|>\n<|turn>model\n", "<bos>text<eos>", "<|channel>thought\n<channel|>answer",
          "日本語のテキストと한국어 텍스트", "Ελληνικά και русский текст", "tab\tseparated\tvalues\n",
          "```python\nimport os\nprint(os.getcwd())\n```", "URL: https://example.com/a?b=c&d=e#f", "﻿BOM", "\U00020000\U0002a6d6 rare CJK",
          "​‌‍ zero width", "  ", "▁ already escaped ▁▁"]
bad = 0
for special in (False, True):
    for s in corpus:
        a, b = tk.encode(s, parse_special=special), ll(s, special)
        if a != b:
            bad += 1
            print("DIFF special=%s %r\n  strata %s\n  llama  %s" % (special, s[:50], a[:20], b[:20]))
print("%d texts x 2: %d differ" % (len(corpus), bad))
