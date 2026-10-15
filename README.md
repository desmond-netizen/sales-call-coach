# Sales Call Coach

Point it at a folder of sales call transcripts. It finds the pains and objections your prospects raise, ranks them across calls, grades how your reps handle each objection, and prints a coaching card for any single call.

Every finding is backed by a line the prospect actually said. If the model cannot quote it, it does not count.

## Why the verification step matters

LLMs are good at spotting a price objection and bad at admitting when there was none. Ask "did the prospect object to price?" across 200 calls and you get confident yeses that never happened, or quotes that were really the rep talking.

So the model is never trusted on its own. For every flag it must return verbatim prospect lines as evidence. The code then checks each quote against the transcript: normalized text, fuzzy matched, and only against lines spoken by the PROSPECT. A flag counts only when a quote matches (0.85 or higher by default) and the model's confidence clears the threshold (0.6 by default). Everything else is dropped and counted, so the report tells you how many flags held up.

That is the difference between a report you can show your team and a report you have to double check.

## Quick start

Needs Python 3.9 or newer. No other dependencies for the mock and Claude Code backends.

```bash
git clone <this repo>
cd sales-call-coach

# 1. See it work offline on the fictional sample calls (free, no model)
python3 sales_coach.py --mock all examples
open reports/report.md
python3 sales_coach.py --mock coach examples/2026-03-11_brightside-dental.vtt

# 2. Make it yours
python3 sales_coach.py init          # copies coach.config.json here; edit reps, offer, objections
```

Then pick a backend.

**Claude Code (no API key).** Uses your Claude subscription through headless Claude Code. Install Claude Code and log in once (`claude`, then `/login`).

```bash
python3 sales_coach.py --backend claude-code all ~/calls
```

**Anthropic API.** Uses the Anthropic Python SDK with structured output and a cached system prompt.

```bash
pip install anthropic                # SDK 1.x needs Python 3.10+
export ANTHROPIC_API_KEY=...
python3 sales_coach.py all ~/calls   # backend "auto" picks anthropic when the key is set
```

The default model is `claude-sonnet-5-5`. Change it with `--model`, the `SALES_COACH_MODEL` env var, or `model.anthropic_model` in the config.

## Transcript formats

Put one call per file in a folder. Subfolders are read too.

| Format | What it reads |
|---|---|
| `.txt`, `.md` | `Name: text` lines. Timestamps are fine: `[00:01:02] Name: text`, `00:01:02 Name: text`, `Name (00:01:02): text`. Otter style (`Name  0:05` on its own line, text below) works. Lines without a speaker join the previous line. |
| `.vtt` | WebVTT. Speaker from `<v Name>` (Teams) or a `Name:` prefix (Zoom). |
| `.srt` | SubRip. Speaker from a `Name:` prefix. |

In `.vtt` and `.srt`, a cue with no speaker is given to the previous speaker. Consecutive lines by the same speaker are merged.

**Who is the rep.** Speakers listed in `reps` (or passed with `--rep "Name"`) are REP. A first name matches a full name in either direction ("Jordan" matches "Jordan Blake"). If speaker labels include emails, set `team_domain` and anyone at that domain is a REP. Everyone else is PROSPECT. Diarized labels like "Speaker 1" work if you list the right one as a rep.

**Metadata.** The call date comes from a `YYYY-MM-DD` in the filename (else the file date). The title is the rest of the filename.

**Wins (optional).** Add a `won.txt` to the folder with one transcript filename per line for calls that became customers. The report then adds "won when raised" columns and a section on what the won calls did differently.

## Config

`coach.config.json` holds everything specific to your business:

| Key | What it is |
|---|---|
| `reps`, `team_domain` | who counts as your side |
| `offer` | one line on what you sell; goes into the prompt |
| `pains` | label id to `name` and `description` |
| `objections` | label id to `name`, `description` and `play` (the coaching line shown when it is handled badly) |
| `handling_levels` | the 0 to 4 rubric for objection handling |
| `discovery_levels`, `next_step_levels` | 0 to 3 rubrics for rep behaviour |
| `thresholds` | `confidence`, `quote_match`, `min_quote_words`, and the cutoffs used for OK / FIX |
| `model` | `backend` (`auto`, `anthropic`, `claude-code`, `mock`), model ids, `effort`, `fallbacks`, `max_retries` |
| `chunk_chars`, `workers`, `cache`, `reports_dir` | plumbing |

The defaults are written for a B2B service sold on a retainer (agencies, consultants, software setup). Rewrite the labels in your own words. Better descriptions beat more labels.

Config lookup order: `--config path`, then `./coach.config.json`, then the copy next to the script.

## Commands

```bash
python3 sales_coach.py init                  # copy the default config into the current folder
python3 sales_coach.py analyze <folder>      # model pass on every new or changed transcript
python3 sales_coach.py report [folder]       # ranked markdown report (no model calls)
python3 sales_coach.py coach <file>          # one-call feedback card (--out card.md to save it)
python3 sales_coach.py all <folder>          # analyze + report
```

Global flags: `--config`, `--backend`, `--mock`, `--model`, `--rep "Name"` (repeatable). `report` takes `--top N` (quotes shown for the top N labels) and `--out path` (default `reports/report.md`).

Results are cached in `coach_cache.sqlite`, keyed by a hash of the file plus the prompt, schema, model and reps. Rerunning only pays for new or changed transcripts. Changing a threshold needs no model calls: just rerun `report`. Changing labels or reps re-analyzes.

## What you get

From the mock run on the three sample calls (`python3 sales_coach.py --mock all examples`):

```
**Verification:** the model flagged 26 pains and objections. 23 were backed by a prospect
quote found in the transcript. 3 were dropped because the quote was not there (or was the rep talking).

**Price too high** (2 calls)
- "Three grand a month is steep for us." (2026-03-04_summit-ridge-roofing.txt, 2026-03-04)
- "Man, that's expensive. That's more than I pay my nephew by a lot." (2026-03-18_harbor-line-hvac.srt, 2026-03-18)
- Best handled: 2026-03-04_summit-ridge-roofing.txt by Jordan Blake (3/3)
- Play: Do not defend the number. Go back to the cost of the problem they gave you ...

| Rep | Calls | Discovery (0 to 3) | Got a number on the pain | Tied offer to their words | Asked for the sale | Firm next step | ...
| Jordan Blake | 2 | 2.5 | 100% | 100% | 100% | 100% | ...
| Priya Shah | 1 | 0.0 | 0% | 0% | 0% | 0% | ...

## Coaching priorities
1. **Need to think about it, send me info**: 2 calls, handled 1.0/3 on average. Ask what specifically ...
```

The report sections: pains ranked by share of calls, objections ranked with average handling, best quotes per label, verified lines outside your taxonomy, rep scorecard, won vs lost behaviours (with `won.txt`), coaching priorities (frequency times how badly it is handled), and method and usage.

The coach card shows OK / FIX on discovery depth, pain quantified, offer tied to their words, asked for the sale, next step firmness and rep talk share, then the verified pains and objections with the play for any that were handled badly, and plain coach notes.

The mock backend answers from keyword rules and invents one fake quote per call on purpose, so you can watch the verifier drop it. Its numbers are not real analysis.

## Cost

One model call per transcript part. A part is about 22,000 characters, which is roughly 30 minutes of conversation, so most calls take 1 or 2 requests. The system prompt (taxonomy and rubrics, about 2k tokens) is cached, so after the first request you pay full price only for the transcript and the answer.

Rough estimate with `claude-sonnet-5-5` at medium effort: a few cents per part, so about 3 to 10 US cents per call. Check your real numbers: the tool prints token usage after every run and the report lists the running total. With the Claude Code backend the calls count against your subscription usage instead.

## Limits

- The taxonomy is fixed up front. Lines outside it are listed, not counted. Read that section and add labels when something repeats.
- Speaker labels must be right. If the transcript tool merged two people into one speaker, the role check is wrong for that call.
- Fuzzy matching tolerates small transcription and punctuation differences. A heavy paraphrase fails, which is the point, but a very short generic quote ("yeah, that's too much") can match more than one moment. `min_quote_words` guards the worst of it.
- Handling grades and rep behaviour scores are model judgments. They are not verified by quotes. Treat them as a ranking signal, not a verdict on one call.
- Small samples swing hard. Shares from 10 calls are anecdotes.
- Won vs lost comparisons are correlation. Warmer prospects make it easier to ask for the sale.

## Tests

```bash
python3 -m unittest
```

Offline. Covers the three transcript formats, rep detection, quote verification and the full mock pipeline.

## Files

- `sales_coach.py`: CLI, parsing, chunking, verification, report and coach card.
- `backends.py`: `anthropic`, `claude-code` and `mock` backends.
- `coach.config.json`: default taxonomy, rubrics, thresholds and model settings.
- `examples/`: three short fictional calls (a roofing company, a dental practice, an HVAC company) plus a sample `won.txt`.
- `SKILL.md`: instructions so Claude Code can set this up and run it for you.
