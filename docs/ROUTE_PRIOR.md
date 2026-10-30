# Residency-biased routing (STRATA_ROUTE_PRIOR): off in MoMoE

Strata 0.2.7 turned on, by default, a routing change for NVIDIA cards of 13 GiB or less: the router prefers experts
that are already on the card. MoMoE took Strata 0.2.7's engine but left this off, and Strata 0.2.8 turned it off
again too (#4402: repetition loops and weaker results in long agent sessions). This page is the measurement behind
MoMoE's choice: on a 12 GB card it makes Qwen3.8-Flash-Next write 41-78% faster, and it makes the answers clearly worse,
tool use most of all.

## What it does

Each token, a mixture-of-experts layer's router scores all 512 experts and runs the 10 best. With
`STRATA_ROUTE_PRIOR=λ` (`src/kernels/cuda/route_prior.cu`), the 10 are picked again after a bonus of λ is added to the
score of every expert already in the card's expert cache. A resident expert that scored a little lower then takes the
place of a non-resident one that scored higher. The expert it pushed out is never computed, and the one it kept is
weighted with its own router weight. Fewer experts have to come from RAM, which is where the speed comes from; the
model runs a different set of experts from the one it chose, which is where the quality goes.

This is not what Strata's adaptive expert tier or MoMoE's `--hybrid` do. Those compute every expert the router chose
(on the card, by the CPU, or read over PCIe) and change only where it is computed and what the cache holds next; the
answer stays the router's, apart from rounding.

Strata 0.2.7's default: λ = 1.5 on a discrete NVIDIA card of 13 GiB or less, off elsewhere. Strata checked it on an
RTX 3060 12 GB with 80 tasks: no copy or code errors; agent tasks passed 4.5 of 10 on average over 4 runs at 1.5
against 5.6 of 10 over 7 runs with it off, which they put within the run-to-run spread. Strata 0.2.8 (2026-10-29)
took the default back after reports of repetition loops in long agent sessions; with nothing set it routes as 0.2.6.

MoMoE: off on every card, as Strata 0.2.8. `STRATA_ROUTE_PRIOR=<λ>` turns it on (1.5 is Strata's value;
`src/core/verify.cpp`, `g_route_prior`). It only acts on Flash-Next's shape (512 experts, 10 a token).

## Measured

Qwen3.8-Flash-Next IQ3_XXS, the `merge-strata-0.2.7` engine, 2026-10-28. The PC: an RTX 4070 Ti 12 GB (sm_89), 63 GB
RAM, Linux, CUDA 13.0. The engine's own expert cache there holds 2,897 of the model's 24,576 experts. The raw results
are in [bench/results/2026-10-28-route-prior](../bench/results/2026-10-28-route-prior/).

### Speed

`tools/ab_engine.py` (the server's timings, greedy, 3 rounds, medians), the config setup writes (MTP drafting,
int8 K/V, 64K context):

| Request | Off | λ = 1.0 | λ = 1.5 |
| --- | ---: | ---: | ---: |
| chat 1, writing | 69.4 tok/s | 106.0 (+53%) | 123.6 (+78%) |
| chat 2 | 63.8 | 84.8 (+33%) | 89.7 (+41%) |
| chat 3 | 62.1 | 89.2 (+44%) | 93.6 (+51%) |
| after a 6,345-token prompt | 58.0 | 88.4 (+52%) | 84.0 (+45%) |

Reading the long prompt itself was unchanged (1,571-1,604 tok/s). Across the task check below, whose requests are
short and include tool round trips, the 4070 Ti wrote 63.0 tok/s off and 94.8 at λ = 1.5 (medians per task).

### Next-token predictions

`tools/quant_quality.py` (the model reads 4,995 tokens of code, docs, English, other languages and an agent transcript
through the decode path, and every next-token distribution is compared with the run with the prior off; the same
engine, expert cache and settings otherwise, with no adaptation, so the runs are deterministic: a repeat run gives
KL 0):

| λ | Same top token | KL mean | Perplexity (off: 4.138) | Agent transcript perplexity (off: 4.815) |
| --- | ---: | ---: | ---: | ---: |
| 1.0 | 87.2% | 0.144 | 4.362 (+5%) | 5.698 (+18%) |
| 1.5 | 82.5% | 0.251 | 4.630 (+12%) | 6.001 (+25%) |
| 2.0 | 74.9% | 0.455 | 5.397 (+30%) | 6.370 (+32%) |

With the cache at 2,897 experts, the 4070 Ti's own. With 1,500 the numbers are close (λ = 1.5: KL 0.212, 83.5%).
For scale:
- IQ3_XXS against Q2_0, a whole quant level apart: KL 0.094 ([MAC.md](MAC.md)).
- The same engine with the prior off, 1,500 against 2,897 cached experts (the experts on the card are computed by
  the GPU's kernels, the others by the CPU's, which round differently): KL 0.025.

At Strata's λ = 1.5 the model's first choice of next token changes about once every six tokens, and the loss is more
than going down a quant level.

### Answers to graded tasks

`tools/task_check.py`: 140 tasks, each graded by a program. Code: 50 HumanEval problems (every third), the reply's
function must pass the problem's tests. Math: 50 GSM8K test problems (every 26th), the last `ANSWER:` line must be the
right number. Copy: 20 generated tasks (write 30 codes in reverse order, list the e-mail addresses in a text, change
one field of a JSON object, replace a word everywhere in a passage), checked exactly. Tool use: 20 generated tasks
through the OpenAI tools API against a made-up shop (an item's price, an order's lines, an exchange rate), the final
total must be right.

Each task 4 times per setting, with the config's sampling (temperature 1.0, top-p 0.95, top-k 20), thinking off. Two
of the four on the 4070 Ti, two on a DGX Spark running the same engine (built for sm_121) with its expert cache held
at 2,897 experts so it keeps what the 12 GB card keeps. The order of the two settings alternated.

| | Code | Math | Copy | Tool use | All |
| --- | ---: | ---: | ---: | ---: | ---: |
| Off | 191/200 (95.5%) | 191/200 (95.5%) | 78/80 (97.5%) | 76/80 (95.0%) | 536/560 (95.7%) |
| λ = 1.5 | 187/200 (93.5%) | 171/200 (85.5%) | 72/80 (90.0%) | 42/80 (52.5%) | 472/560 (84.3%) |

Paired by task and sample: 72 pass only with the prior off, 8 only with it on (a sign test gives p = 5×10⁻¹⁴). Per
machine: the 4070 Ti 95.0% against 80.4%, the Spark 96.4% against 88.2%.

How the tasks failed with the prior on (against off, over the 560):
- tool use: no final answer 34 times (4 off), a wrong total 4 (0 off). The model called tools that do not exist,
  written as text (`load_skill`, `calculate_expression`), stopped in the middle of a sentence ("EUR = 1,981.30 × 0.92
  = 1,822.796 ≈ 1"), or kept calling tools;
- math: a wrong number 18 (7 off), no answer line 11 (2 off);
- copy: a JSON object with more changed than the one field 6 (1 off), a reversed list that came back wrong 2 (1 off);
- code: 13 failed tests (9 off), the smallest difference, which may be why a code check did not show it.

Not measured: thinking on; other models (it only acts on Flash-Next's shape); λ = 1.0 in the task check; long agent
sessions.

### What this means

The speed is real, and so is the cost: the answers change wherever the router's choices would have needed an expert
that is not on the card, and in tool use, the work MoMoE is mostly for, they fail almost half the time. MoMoE keeps it
off. If you want the speed for plain chat and can check the answers, `STRATA_ROUTE_PRIOR=1.0` loses less than 1.5
(KL 0.14 against 0.25), but that setting has no task check here.

## Strata's other defaults measured here

Each against the same engine with the setting off, on the 4070 Ti PC above unless said:

| Setting | Strata's default | Speed | Next-token KL | Graded tasks | MoMoE |
| --- | --- | --- | ---: | --- | --- |
| Residency-biased routing (`STRATA_ROUTE_PRIOR=1.5`) | 0.2.7: on, cards of 13 GiB or less; 0.2.8: off | decode +41-78% | 0.251 | 84.3% against 95.7% | **off** |
| Route tail skip (`STRATA_ROUTE_TAIL_SKIP=7`) | on, CUDA | decode +27-35% | 0.064 | 95.6% against 91.9% | kept |
| DeepSeek's five experts (`--experts 5`) | on, 16 GiB of cards or less | decode +11-31% | 0.030 | not run | kept |
| Int8 prompt attention (`STRATA_PROMPT_ATTN_IMMA`) | on, sm_80 and newer | prompt +0.4% on sm_89 | 0.014 (prompt path) | not run | **off on sm_89** |

For scale: two quants a level apart (IQ3_XXS against Q2_0) differ by KL 0.094.


**The int8 prompt attention on sm_89.** Strata 0.2.7 also makes `STRATA_PROMPT_ATTN_IMMA` a default on every NVIDIA card
from sm_80 (measured on sm_120 and sm_86: +1.1-1.8% prompt). It reads the prompt's int8 K/V on the int8 tensor cores
with the other operand in three 8-bit parts (24 bits), so it is as precise as the kernel it replaces; only the order of
the additions differs. On the 4070 Ti it was +0.4% (1,589 against 1,583 tok/s, within noise), and it changes answers:
the prompt path's predictions differ from the old kernel by KL 0.014 (20 prompts, every top token the same), the size
of a change of summation order alone (the old kernel's own other order, `STRATA_PROMPT_ATTN_V1=1`: 0.010). Not
faster, so MoMoE leaves sm_89 on the old kernel (`strata::prompt_attn_imma_default`); `STRATA_PROMPT_ATTN_IMMA=1` turns
it on. Other cards keep Strata's default.

**Kept: DeepSeek's five experts.** Strata's DeepSeek engine runs 5 of the model's 6 experts a token on cards of 16 GiB
or less, on a GPU that shares the PC's memory (Strix Halo), and on two or more RDNA4 cards that hold half the experts
(`src/dsv41/ds4_experts_rule.hpp`; `--experts 6` for all six). Unlike the route prior it keeps the model's own choice:
the router's top 5, their weights renormalised, so only the weakest of the six is dropped. Strata measured it on
4 x R9700 (KL against the reference 0.110 with 5, 0.108 with 6) and Strix Halo (decode +11-13%), not on a 12 GB card.

Measured here: IST-DAS 2.00 bpw, `strata-dsv41` from this merge, the 4070 Ti PC above (15,025 of the experts on the
CPU, in the model's files), 2026-10-28.

| | 5 experts | 6 experts |
| --- | ---: | ---: |
| Decode, three chats (`tools/ab_engine.py`, 3 rounds, medians) | 3.7 / 5.1 / 4.2 tok/s | 3.2 / 4.2 / 3.2 (5: +16% / +21% / +31%) |
| Decode after a 3,170-token prompt | 4.2 | 3.8 (+11%) |
| Reading the 4,995-token text set below | 910 s | 1,125 s |

The long prompt's own read could not be compared: the engine's saved prompt cache answered it after the first run.

Next-token predictions (the engine's `--score`, the same text set as above in DeepSeek's tokens, compared with
`tools/quant_quality.py`; a second run with 6 experts gave KL 0):

| Text | Same top token | KL | Perplexity, 5 against 6 |
| --- | ---: | ---: | ---: |
| code | 93.8% | 0.032 | 4.452 / 4.498 |
| docs | 90.3% | 0.039 | 23.874 / 23.217 (+2.8%) |
| English | 89.5% | 0.032 | 5.108 / 5.251 |
| other languages | 94.5% | 0.029 | 8.119 / 8.110 |
| agent transcript | 96.5% | 0.016 | 2.818 / 2.851 |
| all | 92.9% | 0.030 | 6.589 / 6.616 |

KL 0.030 is an eighth of the route prior's 0.25 at λ = 1.5 and about a third of a quant level (0.094); the text is not
harder for it to predict (perplexity -0.4% over the set, the agent transcript least changed). MoMoE keeps Strata's
default: on a 12 GB card it writes 11-31% faster for a change the size of a rounding change. Not measured: graded tasks
(DeepSeek writes 3-5 tok/s here: the task check above would take days), long prompts, the 2.25 and 1.75 bpw sets.

**Kept: the route tail skip.** Strata's engine skips, in a verify window, an expert that is not on the card when every
token of the window ranks it 7th or lower of its 10 (`STRATA_ROUTE_TAIL_SKIP=7`, on by default on CUDA since Strata
0.1.42, `src/core/expert_source.cpp`; `0` turns it off): it is neither copied in nor computed, and the other experts
keep their weights. Like DeepSeek's five experts it drops the router's weakest choices rather than choosing others.
Setup turns it off itself for Unsloth's Flash-Next files when every expert is in RAM (Strata 0.1.44: a missed expert
is cheap to fetch there, and the skip still changes the answers); everywhere else it stays on.

Measured: Qwen3.8-Flash-Next IQ3_XXS, the 4070 Ti PC above, its own 2,897-expert cache, 2026-10-29.

| | Tail skip on (7) | Off |
| --- | ---: | ---: |
| Decode, three chats (`tools/ab_engine.py`, 3 rounds, medians) | 68.7 / 63.9 / 62.7 tok/s | 50.8 / 49.7 / 49.4 |
| Decode after a 6,345-token prompt | 63.3 | 49.2 (on: +27-35%) |
| Copy tasks (`tools/task_check.py`, 20 tasks x 4) | 77/80 (96.2%) | 72/80 (90.0%) |
| Tool use tasks (20 x 4) | 76/80 (95.0%) | 75/80 (93.8%) |

The prompt's own read did not change (1,564-1,576 tok/s). Paired by task, 11 pass only with the skip on and 5 only
with it off (a sign test gives p = 0.21: no difference shown, and none against it). Next-token predictions against the
skip off (the same text set as above): the same top token 91.5%, KL 0.064 (code 0.027, docs 0.074, English 0.041,
other languages 0.021, the agent transcript 0.154), perplexity 4.112 against 4.138 (the agent transcript 4.619
against 4.815). So it changes the wording about once every twelve tokens without making the text harder for the model
to predict, unlike the route prior (perplexity +12%, the agent transcript +25%). MoMoE keeps Strata's default. The
graded task check of the route prior above ran with the tail skip on, as Strata's default: both of its arms had it.
Not measured: math and code tasks with the skip off, other quants, other cards.
