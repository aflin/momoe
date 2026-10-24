"""tools/confidence_gate.py - the confidence-gate experiment (design/confidence-gate.md): does a low-bit model's top-two gap say when it disagrees with a higher-bit one?

  confidence_gate.py ref   PORT OUT.json          the reference model writes greedy answers: ids, its own top-2 gaps
  confidence_gate.py probe PORT REF.json OUT.json  another model, fed the reference answers token by token (teacher forcing):
                                        its top word and its top-2 gap at each position
llama-server with the model loaded; temperature 0; thinking off (Qwen3.6's template written by hand).
"""
import json, math, sys, time, urllib.request

PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list, with a short docstring.",
    "Explain in plain words how a hash table handles collisions.",
    "A train leaves at 9:40 and arrives at 13:05. How long is the trip? Show the steps.",
    "You are a coding agent. The test test_parse_date fails with ValueError: unconverted data remains: Z. "
    "Say what you would check first and the fix you would try.",
    "Write a short scene in which a lighthouse keeper finds a message in a bottle.",
    "List five differences between TCP and UDP, one line each.",
    "Write a bash one-liner that counts the lines of every .py file under the current folder and prints the total.",
    "What is the derivative of x^3 * ln(x)? Explain each step.",
]
N_NEW = 200

def post(port, path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))

def prompt_ids(port, q):
    text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return post(port, "/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]

def top2(entry):
    """(top id, gap in logits) from a completion_probabilities entry (top_logprobs: id, logprob)"""
    tl = entry["top_logprobs"]
    return tl[0]["id"], (tl[0]["logprob"] - tl[1]["logprob"]) if len(tl) > 1 else 99.0

def ref(port, out):
    res = []
    for q in PROMPTS:
        ids = prompt_ids(port, q)
        r = post(port, "/completion", {"prompt": ids, "n_predict": N_NEW, "temperature": 0, "n_probs": 2,
                                       "cache_prompt": True, "post_sampling_probs": False})
        probs = r["completion_probabilities"]
        toks = [p["id"] for p in probs]
        gaps = [top2(p)[1] for p in probs]
        res.append({"q": q, "prompt": ids, "ref": toks, "ref_gap": gaps, "text": r["content"]})
        print(f"ref: {len(toks)} tokens: {r['content'][:70]!r}", flush=True)
    json.dump(res, open(out, "w"))

def probe(port, refp, out):
    res = []
    t0 = time.time()
    for item in json.load(open(refp)):
        tops, gaps = [], []
        for i in range(len(item["ref"])):
            r = post(port, "/completion", {"prompt": item["prompt"] + item["ref"][:i], "n_predict": 1, "temperature": 0,
                                           "n_probs": 2, "cache_prompt": True, "post_sampling_probs": False})
            t, g = top2(r["completion_probabilities"][0])
            tops.append(t); gaps.append(g)
        agree = sum(a == b for a, b in zip(tops, item["ref"])) / len(tops)
        res.append({**item, "top": tops, "gap": gaps})
        print(f"probe: {agree:.1%} agree on {len(tops)} ({time.time() - t0:.0f} s)", flush=True)
    json.dump(res, open(out, "w"))

if __name__ == "__main__":
    {"ref": lambda: ref(sys.argv[2], sys.argv[3]), "probe": lambda: probe(sys.argv[2], sys.argv[3], sys.argv[4])}[sys.argv[1]]()
