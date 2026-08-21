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
$ ollama show qwen3:14b
    context length      40960          the model's own declared length

$ ollama show --modelfile qwen3:14b | grep PARAMETER
PARAMETER repeat_penalty 1            no num_ctx anywhere
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
PARAMETER temperature 0.6
PARAMETER top_k 20
PARAMETER top_p 0.95

OLLAMA_CONTEXT_LENGTH                 not set — checked in all three Windows
                                      scopes; only OLLAMA_MODELS and
                                      OLLAMA_HOST are set
```

Then ask the running server what it actually gave the model. Load it with
nothing specified and read the `CONTEXT` column:

```
$ ollama ps
NAME         ID              SIZE      PROCESSOR    CONTEXT    UNTIL
qwen3:14b    bdbd181c33f2    9.6 GB    100% GPU     4096       4 minutes from now
```

A stock model, pulled as-is, declaring 40,960, served 4,096. Nothing in the
model file and nothing in my environment asked for it.

The server states the number itself — but only when it refuses:

```
request (7851 tokens) exceeds the available context size (4096 tokens),
try increasing it
```

`"type": "exceed_context_size_error"`. That is one model refusing a 44 KB prompt
and telling you exactly what is wrong. Another model, same server, same request,
says nothing at all and returns 200.

**Success is the failure mode.** Same server, same prompt, no `num_ctx` on any
of them. None of the four sets `num_ctx` in its Modelfile:

| Model | Template | Declares | HTTP | What came back |
|---|---|---|---|---|
| `qwen3:4b` | Go (no Jinja in the build) | 262,144 | **200** | `prompt_eval_count` 2,050, wrong answer |
| `qwen3:14b` | Go (no Jinja in the build) | 40,960 | **200** | `prompt_eval_count` 2,050, wrong answer |
| a 4B I built | Jinja (carried by the GGUF) | 40,960 | 400 | Rejected, naming 4,096 |
| a 9B I built | Jinja (carried by the GGUF) | 1,048,576 | 400 | Rejected, naming 4,096 |

**It is not model size.** That was my first reading and it was wrong: a stock 4B
and a stock 14B behave identically here, and two models I built locally reject
instead. The declared context length doesn't predict it either — the model
advertising 1,048,576 is one of the ones that refuses at 4,096. Nor is it
`ollama create`: a derivative built from a Modelfile containing nothing but
`FROM qwen3:14b` truncates exactly like its parent.

**It is the chat template.** A maintainer asked what architecture the
self-built models were, which sent me looking at what else differed. The 4B I
built reports the same architecture, parameter count, embedding length and
quantization as stock `qwen3:4b`. What differs is which template path the
server takes. A maintainer later corrected my description of this: **the Jinja
template is the default, and the Go template is used only when there is no
Jinja template, or when `OLLAMA_GO_TEMPLATE=1` is set.** So the split is not
"registry ships Go" — it is that the GGUFs I imported *carry* a Jinja template
and the registry builds do not, which is why the latter fall back to Go. On
this machine `OLLAMA_GO_TEMPLATE` is unset, and `ollama show --template`
reports Jinja for both self-built models and Go for both registry models.

Swapping only the template, on the same weights:

```
self-built 4B, Jinja template from the GGUF   HTTP 400, exceed_context_size_error
same weights + qwen3:4b's Go template         HTTP 200, prompt_eval_count 2050
stock qwen3:4b, Go template                   HTTP 200, prompt_eval_count 2050
```

The Modelfile for the middle row is two lines: `FROM` the self-built model, and
`TEMPLATE` set to the output of `ollama show --template qwen3:4b`. The
unmodified model still returned 400 in the same session, so it isn't drift.

Which of the two is preferable is a matter of taste — the 400 is at least
honest — but **whether you get told depends on where your GGUF came from**, and
nothing surfaces that. I'm reporting the measurement; I haven't read the source
and don't know if the asymmetry is intended.

(The window is 4,096 — `ollama ps` says so, and the models that refuse say so.
So why does truncation leave 2,050, about half of it? **This has an answer, and
it isn't mine.** See below.)

The 200 is the dangerous case. A successful response, correctly formed,
containing a confident wrong answer, with nothing in it to say most of the
prompt was discarded. If you happen to test with a model that refuses, you go
looking for a configuration problem and you find one. If you test with a model
that truncates, you conclude the model is weak. I did the latter, for weeks.

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

My card is consistent with the 4k tier:

```
card             16,303 MiB = 15.92 GiB   -> below 24 GiB -> 4k tier
server's refusal "available context size (4096 tokens)"  -> 4,096, stated
model declares   qwen3.context_length = 40,960           -> a fraction served
```

Consistency, not proof of mechanism — I haven't read the source. And it accounts
for only one of the two numbers: the window is 4,096, but the models that
truncate rather than refuse evaluate 2,050. The VRAM tier explains the 4,096.

The practical conclusion is the same either way: don't take the number from a
page, read it off the running server.

#### 2,050 is `num_ctx / 2 + 2`, and someone had already found it

I filed the above upstream and had the answer within the hour, which is the
argument for filing rather than blogging. It is
[ollama/ollama#17427](https://github.com/ollama/ollama/issues/17427), reported
in July on entirely different hardware:

> the effective usable **prompt** token window is consistently and exactly
> **half** the configured `num_ctx` (plus a small +2 constant)

`4096 / 2 + 2 = 2050`. Exact. In that thread rick-github explains why: when the
prompt exceeds the context buffer, the tokens are reduced to half the buffer
rather than trimmed to fit it, and the server logs it as
`limit=2050 keep=4`. spenceclark measured the same formula at several window
sizes on a different model, and independently confirmed with a canary that the
system prompt is what disappears.

Two things follow. **Truncation is a cliff, not a gradient** — a prompt one
token over a 4,096 window doesn't lose one token, it loses about 2,047. And
**my "unexplained" number was already documented**; I hadn't found it because I
didn't know what to search for. That is the ordinary case, and it is why the
first thing to do with a measurement you can't explain is to publish it
somewhere the people who can explain it will see it.

The other half of the report — why two models refuse with a 400 instead of
truncating — was settled the same way, by a maintainer asking one question I
hadn't thought to ask myself. It's the template, above. Both halves of this
writeup were answered by other people within a few hours of publishing it, and
neither would have been if I'd kept measuring on my own.

**One correction went the other way.** I reported that the pruning description
in #17427 did not fit what I saw: my request is a two-message list — a ~7.8k
token system message and a short user message — and under the described stage 1
the system message should have been dropped, leaving `prompt_eval_count` in the
tens. Observed was 2,050, with the answer quoting the tail of the system
message. rick-github then corrected the description:

> All of the messages between S and U3 are dropped and then S and U3 are
> concatenated to form the final prompt. It's that prompt that is subsequently
> truncated to 2050 tokens.

So the system message is not dropped; it is concatenated with the last user
message and the whole thing is then cut. That matches what the outputs showed.
Two rounds of measurement, and the description of the mechanism moved.

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
number of prompt tokens actually evaluated — not the number you sent. Send
10,000 tokens; if a few thousand come back, you've found it. Don't expect it to
equal the window: mine reports 4,096 in `ollama ps` and evaluates 2,050.

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

On Windows, don't inline a prompt this size into the command as above: the
command line caps out around 32 KB and you'll get a malformed request back
rather than a truncated one. Put the body in a file and use `--data-binary @body.json`,
or just run one of the scripts below, which do that already.

## Check your own setup

```bash
python check_context.py qwen3:14b     # standard library only, no dependencies
bash check_context.sh qwen3:14b       # same test, needs curl and jq
```

Both send the same needle prompt twice — once with the server default, once with
an explicit `num_ctx` — and compare `prompt_eval_count`. The Python one is listed
first because it needs nothing installed; `jq` isn't present by default on
Windows, which is where this was found. Invoke the shell one as `bash
check_context.sh` rather than `./check_context.sh` if the execute bit didn't
survive the trip.

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

## Postscript: the repro script had the same bug

The first version of `check_context.sh` passed the haystack through argv.
On Windows that hits the ~32KB command line limit, `jq` fails, and curl sends
an empty body. Ollama returns 400 "missing request body" — and my script,
seeing a 400, reported "REJECTED (prompt exceeded the served window)."

A request that was never sent, reported as a context-window rejection. The
instrument fabricated the phenomenon it was built to detect. It now checks
that the error body actually mentions the context size before making that
claim, and reports REQUEST FAILED otherwise.

---

## Upstream

Reported as [ollama/ollama#17889](https://github.com/ollama/ollama/issues/17889),
alongside the existing reports of silent truncation going back to 2024 (#4967,
#14259, #14262).

The 2,050 was answered there within the hour by a pointer to
[#17427](https://github.com/ollama/ollama/issues/17427), which had the formula
already: `num_ctx / 2 + 2`. Credit to rick-github and spenceclark. The 400
versus 200 split turned out to be the chat template; see above. Credit to
rick-github for the question that found it.

## Data and further reading

- Experiment data for this writeup: https://doi.org/10.5281/zenodo.21983630
- A log of what broke across the wider program — including the bug in the first
  version of this repository's own script — kept in a fixed schema with a field
  that cannot be reconstructed afterwards, *why it looked true*:
  [llm-retraction-log](https://github.com/Bigbonus/llm-retraction-log)
  ([10.5281/zenodo.22048923](https://doi.org/10.5281/zenodo.22048923))

### The books

Both come out of the machine described at the top of this page.

**Raising Your Own AI on a Home PC** — the six months that produced these
experiments. It opens with the morning an AI told me *"Actually, that isn't
distillation."* I had spent half a year collecting 133 draft-and-correction
pairs as experience points for a student model whose weights had been updated
exactly zero times. The book keeps the wiring mistakes, the scoring mistakes
and the failed predictions, including the run where the machine score improved
while a blinded human comparison gave the trained side 0 wins, 9 losses and
11 ties.
[Kindle $9.99](https://www.amazon.com/dp/B0HCT93JX3) ·
[Paperback $12.99](https://www.amazon.com/dp/B0HDPMNMTQ) ·
data at [10.5281/zenodo.21730423](https://doi.org/10.5281/zenodo.21730423)

**Applied AI Distillation: Make AI Yours** — five practical paths for adapting
a model to your own use, starting from a no-training baseline, with runnable
fixtures and known-answer tests.
[Kindle $9.99](https://www.amazon.com/dp/B0HFJNDJV6) ·
[Paperback $79.90](https://www.amazon.com/dp/B0HFKFZ5Q2)

**The script above needs none of this.** It is one file, MIT, and it answers
one question about your own server.
