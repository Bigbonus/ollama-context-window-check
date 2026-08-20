### A model that advertises 40,960 tokens was being served 4,096

I've been running knowledge distillation experiments on consumer hardware: one
RTX 5080 (16GB VRAM), 32GB of system RAM, an 850W supply, and Qwen3 at 1.7B,
4B, 8B, and 14B, all QLoRA 4-bit.

Two things I had written down as results were properties of my measurement
setup. Both were silent. I'm writing them up because the first one is not
something I configured, and the second is a bug I'd expect to find in a lot of
hand-written eval code.

#### 1. The context window nobody set

What I found when I actually checked:

```
qwen3.context_length = 40960        model's own declared context length
Modelfile parameters                no num_ctx
                                    (repeat_penalty / stop / temperature /
                                     top_k / top_p only)
OLLAMA_CONTEXT_LENGTH               not set
                                    (only OLLAMA_MODELS and OLLAMA_HOST)
measured boundary                   prompt_eval_count 4,046-4,050 passes,
                                    breaks above
```

The model declares 40,960. It was being served roughly 4,096. Nothing in the
model file and nothing in my environment asked for that.

The server states the number itself — but only when it refuses:

```
request (7851 tokens) exceeds the available context size (4096 tokens),
try increasing it
```

`"type": "exceed_context_size_error"`. That is the 4B rejecting a 44,426-character
prompt. Given the same prompt, the 14B says nothing at all. It returns 200.

**Success is the failure mode.** Same request, same prompt, three models:

| Model | HTTP | What came back |
|---|---|---|
| 4B | 400 | Rejected |
| 9B\* | 400 | Rejected |
| 14B | **200** | `prompt_eval_count` 2,050, wrong answer |

(Why 2,050 rather than 4,096, I don't know. I'm reporting the number I measured.
It is reproducible: the check script below, on an unrelated prompt, lands on
2,050 as well.)

\* Nemotron Nano 9B v2 — a different model family, used here only as a second
inference target. The 4B and 14B are Qwen3.

The 14B returns HTTP 200. A successful response, correctly formed, containing a
confident wrong answer. Nothing in the response says most of the prompt was
discarded. If you test with the small models you conclude something is broken;
if you test with the large one you conclude the model is weak. I did the latter,
for weeks.

**The truncation takes the front.** The tail of the prompt survives; the head
does not. Your system prompt, formatting rules, few-shot examples, persona
definition, and the earlier turns of a conversation are exactly what gets
discarded first. The model answers from a prompt that contains the question but
not the instructions.

If your notes say the model "forgets its instructions in long sessions" or
"drifts out of character," check this before believing it.

#### 2. You cannot answer this from the documentation

I looked. Three of Ollama's own documentation pages give three different
answers:

- the [FAQ](https://docs.ollama.com/faq#how-can-i-specify-the-context-window-size)
  says the default is 4096
- the [Modelfile reference](https://docs.ollama.com/modelfile#valid-parameters-and-values)
  says `num_ctx` defaults to 2048
- the [context length page](https://docs.ollama.com/context-length) says the
  default is selected from available VRAM: 4k below 24 GiB, 32k from 24 to 48
  GiB, 256k at 48 GiB and above

Three pages, three defaults. This is why "I read the docs" doesn't protect you
here, and why people who have carefully configured everything else still have
this sitting in their setup.

<sub>Checked against `ollama/ollama` `main` on 2026-08-20. Commit-pinned copies,
in case these get reconciled:
[faq.mdx](https://github.com/ollama/ollama/blob/948f69330acf96a2310f1b53fdfc211731a386d8/docs/faq.mdx) ·
[modelfile.mdx](https://github.com/ollama/ollama/blob/6a261db7d87b13d76c5197cec636a0a3951afb36/docs/modelfile.mdx) ·
[context-length.mdx](https://github.com/ollama/ollama/blob/f0078ae4766d0d570e196158f20dde309bd96124/docs/context-length.mdx)</sub>

The VRAM tiering, if that's the behaviour on your build, also explains why this
rarely comes up in discussion. My card is 16,303 MiB, which is 15.92 GiB —
under the 24 GiB threshold, so the 4k tier. Anyone on a 24GB or 48GB card never
sees it. The people most likely to hit this are the ones least likely to be in
the conversation about it.

My own numbers are consistent with that reading:

```
card             16,303 MiB = 15.92 GiB   -> below 24 GiB -> 4k tier
measured boundary  prompt_eval_count 4,046-4,050 passes  -> consistent with 4,096
model declares     qwen3.context_length = 40,960          -> served ~1/10
```

I'm reporting this as consistency, not as proof of mechanism. I haven't read the
source.

The practical conclusion is the same either way: don't take the number from a
page, read it off the running server.

#### Two checks that take one line each

**`ollama ps`** prints the applied context on builds that carry the column:

```
NAME                    ID              SIZE      PROCESSOR    CONTEXT    UNTIL
qwen3-4b-custom:q4km    69bf8b840ba0    5.1 GB    100% GPU     16384      4 minutes from now
```

That 16384 is a value I set explicitly. With nothing specified you should
expect to see 4096 there, and that alone tells you where you stand. (The
context length page above uses this same command as its own verification step.)

**`prompt_eval_count`** in the API response works on every build. It's the
number of prompt tokens actually evaluated — not the number you sent. If you
sent 10,000 tokens and it comes back near 4,096, you've found it.

#### Repro

Same prompt twice, changing only whether `options.num_ctx` is present.

```bash
curl -s http://127.0.0.1:11434/api/chat -d '{
  "model":"qwen3:14b",
  "stream":false,
  "think":false,
  "messages":[
    {"role":"system","content":"<6000+ characters of text, with a needle on the first line>"},
    {"role":"user","content":"What is the needle value?"}
  ],
  "options":{"num_predict":256,"temperature":0}
}' | jq '.prompt_eval_count, .done_reason, .message.content'
```

Add `"num_ctx":16384` to `options` and it answers correctly.

Put the needle on the **first** line, not the last. A needle at the end survives
truncation, so the test passes while the bug is still there.

## Check your own setup

```bash
./check_context.sh qwen3:14b        # bash + curl + jq
python check_context.py qwen3:14b   # standard library only, no jq
```

Both send the same needle prompt twice — once with the server default, once with
an explicit `num_ctx` — and compare `prompt_eval_count`. The Python version is
there because `jq` isn't present by default on Windows, which is where this was
found.

Output on a server carrying the small default, from the 14B — the silent case:

```
--- run 1: server default (no num_ctx)
    HTTP status      : 200
    prompt_eval_count: 2050
    answer           : ACCESS
    needle found     : NO

--- run 2: explicit num_ctx=16384
    HTTP status      : 200
    prompt_eval_count: 7855
    answer           : XJ7Q-4419-KTM
    needle found     : yes

===
TRUNCATION DETECTED.
  default  : 2050 prompt tokens evaluated
  explicit : 7855 prompt tokens evaluated
```

Run 1 returned 200 and an answer. The answer is the word `ACCESS`, picked out of
a log line near the end — the front of the prompt, where the code was, is gone.
The 4B given the same prompt returns 400 and tells you why.

#### Fixes, in increasing order of permanence

```bash
# per request
"options": {"num_ctx": 16384}

# server-wide
OLLAMA_CONTEXT_LENGTH=16384 ollama serve

# baked into a derived model
cat > Modelfile << 'EOF'
FROM qwen3:14b
PARAMETER num_ctx 16384
EOF
ollama create qwen3-14b-16k -f Modelfile
```

`PARAMETER num_ctx` in a Modelfile takes precedence over the environment
variable — check with `ollama show --modelfile <model>`. If Ollama runs as a
service, the variable has to be set where the service sees it, not in your
shell.

One more routing trap: requests through the OpenAI-compatible endpoint
(`/v1/chat/completions`) often don't carry `num_ctx` through at all. If you go
through an agent framework, a RAG pipeline, or a chat frontend, assume it sets
its own value until `prompt_eval_count` proves otherwise.

#### Why you can't just set it to 40,960

A larger window means a larger KV cache, and on a 16GB card that competes
directly with the weights. Peak VRAM during QLoRA 4-bit training, from 135
recorded runs:

| Model | LoRA config | Peak allocated | Peak reserved |
|---|---|---|---|
| 1.7B | r13 / a26 | 3.29-3.30 GB | 3.72 GB |
| 4B | r12-19 | 4.19-5.17 GB | 5.08-5.26 GB |
| 8B | r15-29 | 9.61-10.04 GB | 9.68-10.28 GB |
| 14B | r32 / a64 | 15.06-15.17 GB | **16.30-16.71 GB** |

The card has 15.92 GiB usable. The 14B runs reserved more than that. They ran,
but against the ceiling the whole time — which is also why I couldn't keep a 14B
loaded for inference alongside anything else. Together they needed 16.43 GB.

So the fix isn't free. On this hardware, raising the context window and running
a 14B are in direct competition, and that tradeoff is invisible until you know
the window was never what you thought it was.

For reference on the other axis: identical 8-question, 4,800-token set, 40.4
tok/s with an unmerged 4-bit adapter versus 84.6 tok/s with merged bf16 weights
at batch 8 — 2.08x. Over 520 questions, 2.14 hours versus 1.02.

#### 3. My scoring script was counting empty as wrong

This one was entirely mine.

I had a confabulation rate of 62% recorded. The real number was 40%.

The difference was empty responses. The model had exhausted its generation
budget on reasoning tokens and returned nothing, and my scoring script — finding
no correct answer in an empty string — filed it as a confabulation. Those are
not the same failure. One is a model asserting something false. The other is a
model that never got to answer.

At `num_predict=64`, every response came back empty. All of them.

The same bug explained something else I'd been carrying: apparent
non-determinism at temperature 0. The reasoning-token budget was where the
variance was entering. With thinking disabled, 24 out of 24 runs matched
exactly.

I'd guess this exact bug is sitting in a lot of hand-written eval harnesses,
because the natural way to score is "does the expected answer appear in the
output," and an empty string fails that check the same way a wrong answer does.

I discarded and re-collected more than 17,000 rows after finding these two.

#### What I'm not claiming

Verified on ollama 0.32.9, Windows, GGUF, via `POST /api/chat`. Not tested on
other versions, other platforms, llama.cpp directly, or vLLM. I'm not claiming
a universal default.

I don't use MMLU, GSM8K, HumanEval, or JGLUE. My metrics are my own and my
harness is hand-written — which is how I produced a 62% that was really 40%.
None of this is a claim about whether distillation works or about relative model
quality.

What I think does generalize: if you're evaluating local models on your own
hardware with your own scripts, some fraction of what you've written down is
about your setup rather than the model. The two above cost me weeks, and each
would have been caught by one check.

#### Checks I run now, before recording anything

- `ollama ps` and `prompt_eval_count` on every run
- Needle at the front of the prompt, never the end
- Empty and truncated outputs in their own bucket, never folded into a
  correctness class
- A determinism check — same input, same seed, N runs — before trusting any
  measured difference
- Base model name, start/end timestamps, elapsed seconds, GPU name and library
  versions in every training record (learned by having 65 of 135 runs I can no
  longer attribute)

---

## Data and further reading

- Experiment data: https://doi.org/10.5281/zenodo.21983630
- I wrote up the full set of experiments, including the failures not covered here, as a book: [Raising Your Own AI on a Home PC](https://www.amazon.com/dp/B0HDPMNMTQ)
