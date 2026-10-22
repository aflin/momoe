"""same.py REF.tok ENGINE.out: how many of the engine's greedy tokens equal llama.cpp's, from the start."""
import sys

r = open(sys.argv[1]).read().split()
lines = [l for l in open(sys.argv[2]) if l.startswith("tokens:")]
m = lines[0].split()[1:] if lines else []
n = min(len(r), len(m))
print("  greedy vs llama.cpp:", next((i for i, (a, b) in enumerate(zip(r, m)) if a != b), n), "of", n)
