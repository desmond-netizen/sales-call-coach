---
name: sales-call-coach
description: Mine sales call transcripts for prospect pain points and objections, verified with verbatim prospect quotes, rank them across calls, grade how reps handle objections, and write a coaching card for one call. Use when the user has call transcripts (.txt, .md, .vtt, .srt) and asks what prospects struggle with, what objections come up most, how their reps are doing, or for feedback on a specific sales call.
---

# Sales Call Coach

`sales_coach.py` in this folder does the work. You set up the config, run it, and explain the results.

## When to use

- "What objections do we hear most?" or "What pains do prospects bring up?" across many calls.
- "How is each rep doing on discovery, asking for the sale, next steps?"
- "Give me feedback on this call" for one transcript.
- Not for live calls, audio files (transcribe first), or a single quote lookup (just read the file).

## 1. Set up the config from the user's business

Run `python3 sales_coach.py init` in the user's working folder (or edit `coach.config.json` there if it exists). Then ask the user, briefly and in one message:

1. **Who are the reps?** Their names exactly as they appear as speaker labels in the transcripts. Open one transcript and show the speaker names you found so they can point at theirs. If labels include emails, ask for the company email domain (`team_domain`).
2. **What do you sell?** One line, including the price shape (setup fee, retainer, one-off). This goes into `offer`.
3. **What objections do you hear?** Their top 5 to 10 in their own words, and what a great answer sounds like for each.
4. **What problems do prospects come in with?** Optional; the defaults cover most B2B services.

Then edit the config:

- Put names in `reps`, the domain in `team_domain`, the line in `offer`.
- Rewrite `objections` to match what they said. Each entry needs `name`, `description` (what counts, in plain words) and `play` (the coaching line). Keep ids short and snake_case. Delete labels that do not apply to them. Never invent their prices, offers or case studies in a `play`; use their words or keep it generic ("your entry offer", "a capped test budget").
- Adjust `pains` the same way.
- Leave `handling_levels`, rubrics and thresholds alone unless asked.

## 2. Run it

Always do a free dry run first to check parsing and rep detection:

```bash
python3 sales_coach.py --mock all <folder>
```

If a file fails with "no REP speaker found", the rep names do not match the speaker labels. Fix `reps` or pass `--rep "Name"`.

Then the real run. Pick the backend:

- `--backend claude-code`: uses the user's Claude login, no API key. Needs `claude` on PATH and logged in. If it says "not logged in", tell the user to run `claude` in a terminal and `/login` once.
- `--backend anthropic` (or `auto` with `ANTHROPIC_API_KEY` set): Anthropic API, default model `claude-sonnet-5-5`. This spends money. Tell the user the number of transcripts and the rough cost (about 3 to 10 US cents per call) and get a yes before running a large folder.

```bash
python3 sales_coach.py --backend claude-code all <folder>
python3 sales_coach.py --backend claude-code coach <file> --out reports/<name>-card.md
```

Reruns are cached by file hash, so only new or changed transcripts cost anything. `report` alone makes no model calls; use it after changing thresholds.

If the user has a list of calls that closed, write their filenames into `<folder>/won.txt` (one per line) before running `report`.

## 3. Read the report back to the user

The report is `reports/report.md`. Lead with what matters, in this order:

1. **Verification line.** How many flags the model raised and how many held up with a real prospect quote. If many were dropped, say so: the labels may be vague, or speaker labels may be wrong.
2. **Top 3 objections** by share of calls, with one real quote each and the average handling score (0 ignored, 3 resolved).
3. **Coaching priorities.** Objections that come up often and are handled badly. These are the training topics.
4. **Rep scorecard.** Call out the biggest gap per rep (no number on the pain, no ask, no dated next step, talking over 60%).
5. **Outside the taxonomy.** If the same kind of line shows up more than twice, suggest a new label.

For a single call, summarize the FIX rows of the coach card and the one or two plays that would have changed the call.

Be honest about limits: handling and rep scores are model judgments (not quote verified), and fewer than about 20 calls is a small sample. Never quote a line that is not in the report or the transcript.
