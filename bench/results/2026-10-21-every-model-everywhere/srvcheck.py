"""srvcheck.py [PORT] - a running momoe server: a chat (speed), a tool call (whole and streamed), a tool result."""
import json, sys, time, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8080"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def post(body, stream=False):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=1800)
    if not stream:
        return json.load(r)
    calls, text = {}, ""
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])["choices"][0].get("delta", {})
        text += d.get("content") or ""
        for c in d.get("tool_calls") or []:
            e = calls.setdefault(c.get("index", 0), {"name": "", "arguments": ""})
            f = c.get("function", {})
            e["name"] += f.get("name") or ""
            e["arguments"] += f.get("arguments") or ""
    return calls, text


t = time.time()
j = post({"model": "x", "max_tokens": 400, "messages": [{"role": "user", "content":
          "Write a Python function that merges two sorted lists, and explain it briefly."}]})
dt = time.time() - t
u = j["usage"]
print(f"chat: {u['prompt_tokens']} prompt + {u['completion_tokens']} tokens in {dt:.1f} s = "
      f"{u['completion_tokens'] / dt:.1f} tok/s overall; finish {j['choices'][0]['finish_reason']}")
m = j["choices"][0]["message"]
print("  text:", repr((m.get("content") or "")[-160:]))

ask = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
j = post({"model": "x", "max_tokens": 1500, "messages": ask, "tools": TOOLS})
m = j["choices"][0]["message"]
tc = m.get("tool_calls") or []
print("tool call (whole):", [(c["function"]["name"], c["function"]["arguments"]) for c in tc] or "NONE",
      j["choices"][0]["finish_reason"])
calls, _ = post({"model": "x", "max_tokens": 1500, "messages": ask, "tools": TOOLS, "stream": True}, stream=True)
print("tool call (streamed):", [(c["name"], c["arguments"]) for c in calls.values()] or "NONE")
if tc:
    msgs = ask + [{"role": "assistant", "content": m.get("content") or "", "tool_calls": tc},
                  {"role": "tool", "tool_call_id": tc[0]["id"], "content": json.dumps({"temp_c": 14, "sky": "rain"})}]
    j = post({"model": "x", "max_tokens": 1500, "messages": msgs, "tools": TOOLS})
    print("after the result:", repr((j["choices"][0]["message"].get("content") or "")[-200:]))
