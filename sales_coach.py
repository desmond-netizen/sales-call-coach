#!/usr/bin/env python3
"""Sales Call Coach: mine sales call transcripts for prospect pains and objections, rank them,
grade how reps handle objections, and print a coaching card per call.

The model only answers structured questions. Code owns the rest: parsing, chunking, quote
verification, counting, ranking and the coaching text. A flag counts only when its quote is
found in what the PROSPECT actually said and its confidence clears the threshold.

Commands
  init                 copy the default coach.config.json into the current folder
  analyze <folder>     run the model on every transcript in the folder (cached by file hash)
  report [folder]      write the ranked markdown report from the cache
  coach <file>         one-call feedback card
  all <folder>         analyze + report

Backends: anthropic (ANTHROPIC_API_KEY), claude-code (your Claude login), mock (offline).
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import backends

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_VERSION = 1
EXTS = (".txt", ".md", ".vtt", ".srt")


# ---------------------------------------------------------------- config
def load_config(path):
    for p in (path, "coach.config.json", os.path.join(HERE, "coach.config.json")):
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["_path"] = os.path.abspath(p)
            return cfg
    sys.exit("no coach.config.json found")


# ---------------------------------------------------------------- parsing
TS = r"[\[(]?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?[\])]?"
SPEAKER = re.compile(r"^(?P<name>[^:\[\]()]{1,60}?)\s*(?:" + TS + r")?\s*:\s+(?P<text>.+)$")
OTTER = re.compile(r"^(?P<name>[^:]{1,60}?)\s{2,}" + TS + r"$")


def speaker_line(line):
    """'Name: text' -> (name, text). Rejects sentence-like names."""
    m = SPEAKER.match(line)
    if not m:
        return None
    name = m.group("name").strip().strip("*").strip()
    if not name or len(name.split()) > 4 or name[0].isdigit():
        return None
    return name, m.group("text").strip()


def parse_text(raw):
    rows = []
    for line in raw.splitlines():
        line = line.strip().replace("**", "")
        if not line or line.startswith("#") or line.startswith("---"):
            continue
        line = re.sub(r"^[-*>]\s+", "", line)
        m = OTTER.match(line)
        if m:
            rows.append([m.group("name").strip(), ""])
            continue
        line = re.sub(r"^" + TS + r"\s*[-|]?\s*", "", line)
        if not line or re.fullmatch(TS, line):
            continue
        sp = speaker_line(line)
        if sp:
            rows.append(list(sp))
        elif rows:
            rows[-1][1] = (rows[-1][1] + " " + line).strip()
    return rows


def parse_cues(raw):
    """WebVTT and SRT. Speaker from <v Name> or a 'Name:' prefix; otherwise the previous speaker."""
    rows, last = [], "Unknown"
    for block in re.split(r"\n\s*\n", raw.replace("\r", "")):
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if not lines or lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        while lines and "-->" not in lines[0]:
            lines = lines[1:]  # cue id or SRT index
        text_lines = lines[1:]
        for t in text_lines:
            v = re.match(r"^<v(?:\.[^ >]*)?\s+([^>]+)>(.*)$", t)
            if v:
                name, t = v.group(1).strip(), v.group(2)
            else:
                sp = speaker_line(re.sub(r"<[^>]+>", "", t))
                name, t = sp if sp else (last, t)
            t = re.sub(r"<[^>]+>", "", t).strip()
            if t:
                rows.append([name, t])
                last = name
    return rows


def parse_file(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    rows = parse_cues(raw) if path.lower().endswith((".vtt", ".srt")) else parse_text(raw)
    return [(n, t) for n, t in rows if t.strip()]


class Roles:
    def __init__(self, cfg, extra=()):
        self.reps = {r.strip().lower() for r in list(cfg.get("reps", [])) + list(extra) if r.strip()}
        self.domain = (cfg.get("team_domain") or "").lower().lstrip("@")

    def role(self, name):
        n = (name or "").strip().lower()
        if n in ("", "unknown", "unknown speaker"):
            return "UNKNOWN"
        if self.domain and re.search(r"@" + re.escape(self.domain) + r"\b", n):
            return "REP"
        n = re.sub(r"\s*<[^>]*>", "", n).strip()
        if n in self.reps:
            return "REP"
        first = n.split()[0]
        for r in self.reps:
            if (len(n.split()) == 1 or len(r.split()) == 1) and r.split()[0] == first:
                return "REP"
        return "PROSPECT"


def utterances(rows, roles):
    """Merge consecutive lines by the same speaker."""
    out = []
    for name, text in rows:
        if out and out[-1]["name"] == name:
            out[-1]["text"] += " " + text
        else:
            out.append({"name": name, "text": text})
    for u in out:
        u["role"] = roles.role(u["name"])
        u["words"] = len(u["text"].split())
    return out


def chunk(utts, size):
    chunks, cur, n = [], [], 0
    for u in utts:
        line = "%s %s: %s" % (u["role"], u["name"], u["text"])
        if cur and n + len(line) > size:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(u)
        n += len(line) + 1
    if cur:
        chunks.append(cur)
    return chunks


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def call_meta(path):
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    m = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
    date = m.group(1) if m else datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")
    title = re.sub(r"[_-]+", " ", re.sub(r"\d{4}-\d{2}-\d{2}", "", stem)).strip() or stem
    if title.islower():
        title = title.title()
    return base, title, date


def list_transcripts(folder):
    if os.path.isfile(folder):
        return [folder]
    out = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(EXTS) and f.lower() not in ("readme.md", "won.txt"):
                out.append(os.path.join(root, f))
    return sorted(out)


# ---------------------------------------------------------------- prompt and schema
def build_system(cfg):
    L = ["You review B2B sales call transcripts. The seller sells %s." % cfg.get("offer", "a B2B service"),
         "",
         "Speakers are tagged REP (the seller's team), PROSPECT (the potential customer) or UNKNOWN.",
         "You get one excerpt of a call at a time. Judge only that excerpt.",
         "",
         "Rules",
         "- A pain or objection is present only if a PROSPECT says it or clearly confirms it. A REP suggesting it does not count. Something about another business does not count.",
         "- evidence: copy 1 to 3 lines exactly as the PROSPECT said them in the excerpt. Verbatim: no paraphrase, no fixed grammar, no merged lines, no ellipses. 6 to 40 words each. Never quote a REP. Empty when not present.",
         "- confidence: your probability from 0 to 1 that the label truly applies in this excerpt.",
         "- handling: 0 when the objection is not raised; otherwise 1 to 4 using the handling levels.",
         "- unlisted: prospect pains or objections that fit none of the labels, each with one verbatim prospect line.",
         "- rep: judge the REP side of this excerpt. next_step is judged only when the excerpt is the final part of the call; otherwise set it to 0.",
         "",
         "Pain points (problems the prospect has in their own business)"]
    for k, v in cfg["pains"].items():
        L.append("- %s: %s" % (k, v["description"]))
    L += ["", "Objections (hesitations or reasons not to buy, or not to buy now)"]
    for k, v in cfg["objections"].items():
        L.append("- %s: %s" % (k, v["description"]))
    L += ["", "Handling levels"]
    L += ["%d. %s" % (i, t) for i, t in enumerate(cfg["handling_levels"])]
    L += ["", "Rep behaviour",
          "- discovery: how thoroughly the REP diagnoses the prospect's business."]
    L += ["  %d. %s" % (i, t) for i, t in enumerate(cfg["discovery_levels"])]
    L += ["- pain_quantified: the REP gets the PROSPECT to put a number on the problem or goal (lost revenue, cost per lead, deal value, customers needed, revenue target).",
          "- offer_tied_to_pain: the REP explicitly connects the offer to a problem the PROSPECT described in their own words.",
          "- asked_for_sale: the REP directly asks the PROSPECT to buy, sign, pay, or commit to a package or start date.",
          "- next_step: how firm the next step is when the call ends."]
    L += ["  %d. %s" % (i, t) for i, t in enumerate(cfg["next_step_levels"])]
    return "\n".join(L)


def build_schema(cfg):
    def item(handling):
        p = {"evidence": {"type": "array", "items": {"type": "string"}},
             "present": {"type": "boolean"},
             "confidence": {"type": "number"}}
        if handling:
            p["handling"] = {"type": "integer", "enum": list(range(len(cfg["handling_levels"])))}
        return {"type": "object", "properties": p, "required": list(p), "additionalProperties": False}

    def group(tax, handling):
        return {"type": "object", "properties": {k: item(handling) for k in tax},
                "required": list(tax), "additionalProperties": False}

    rep = {"discovery": {"type": "integer", "enum": list(range(len(cfg["discovery_levels"])))},
           "pain_quantified": {"type": "boolean"},
           "offer_tied_to_pain": {"type": "boolean"},
           "asked_for_sale": {"type": "boolean"},
           "next_step": {"type": "integer", "enum": list(range(len(cfg["next_step_levels"])))}}
    unl = {"kind": {"type": "string", "enum": ["pain", "objection"]},
           "summary": {"type": "string"},
           "evidence": {"type": "string"}}
    props = {
        "pains": group(cfg["pains"], False),
        "objections": group(cfg["objections"], True),
        "unlisted": {"type": "array", "items": {"type": "object", "properties": unl, "required": list(unl), "additionalProperties": False}},
        "rep": {"type": "object", "properties": rep, "required": list(rep), "additionalProperties": False},
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def analysis_key(cfg, backend, system, schema):
    blob = json.dumps({"v": PROMPT_VERSION, "system": system, "schema": schema, "backend": backend.name,
                       "model": backend.model, "chunk_chars": cfg["chunk_chars"],
                       "reps": sorted(cfg.get("reps", [])), "domain": cfg.get("team_domain", "")}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- cache
def open_cache(cfg):
    db = sqlite3.connect(cfg.get("cache") or "coach_cache.sqlite", check_same_thread=False)
    db.executescript("""
    CREATE TABLE IF NOT EXISTS calls (file_hash TEXT, akey TEXT, path TEXT, title TEXT, date TEXT,
        utts TEXT, answers TEXT, created TEXT, PRIMARY KEY (file_hash, akey));
    CREATE TABLE IF NOT EXISTS runs (ts TEXT, cmd TEXT, backend TEXT, model TEXT, requests INTEGER,
        input_tokens INTEGER, output_tokens INTEGER, cache_read INTEGER, cache_write INTEGER, cost_usd REAL);
    """)
    return db


def log_run(db, cmd, be):
    u = be.usage
    if u.requests:
        db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)", (datetime.now().isoformat(timespec="seconds"), cmd,
                   be.name, be.model, u.requests, u.input_tokens, u.output_tokens, u.cache_read, u.cache_write, u.cost_usd))
        db.commit()


# ---------------------------------------------------------------- analyze
def analyze_utterances(utts, title, cfg, be, system, schema):
    parts = chunk(utts, cfg["chunk_chars"])
    answers = []
    for i, part in enumerate(parts):
        final = i == len(parts) - 1
        body = "\n".join("%s %s: %s" % (u["role"], u["name"], u["text"]) for u in part)
        user = "Call: %s\nExcerpt: part %d of %d%s\n\n<transcript>\n%s\n</transcript>" % (
            title, i + 1, len(parts), " (final part)" if final else "", body)
        ctx = {"lines": [(u["role"], u["name"], u["text"]) for u in part], "final": final,
               "pains": list(cfg["pains"]), "objections": list(cfg["objections"])}
        answers.append(be.ask(system, user, schema, ctx))
    return answers


def analyze_files(paths, cfg, be, db, roles, quiet=False):
    """Analyze every path not already cached. Returns {path: (file_hash, akey)}."""
    system, schema = build_system(cfg), build_schema(cfg)
    akey = analysis_key(cfg, be, system, schema)
    done = {r[0] for r in db.execute("SELECT file_hash FROM calls WHERE akey=?", (akey,))}
    todo, keys = [], {}
    for p in paths:
        h = file_hash(p)
        keys[p] = (h, akey)
        if h not in done:
            todo.append((p, h))
    if not quiet:
        print("%d transcripts, %d cached, %d to analyze with %s (%s)" % (len(paths), len(paths) - len(todo), len(todo), be.name, be.model))

    def work(p, h):
        base, title, date = call_meta(p)
        utts = utterances(parse_file(p), roles)
        if not any(u["role"] == "PROSPECT" for u in utts):
            raise ValueError("no PROSPECT speaker found")
        if not any(u["role"] == "REP" for u in utts):
            raise ValueError("no REP speaker found; add your reps to the config or pass --rep")
        ans = analyze_utterances(utts, title, cfg, be, system, schema)
        return p, h, base, title, date, utts, ans

    def save(res):
        p, h, base, title, date, utts, ans = res
        db.execute("INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?,?)", (h, akey, os.path.abspath(p), title, date,
                   json.dumps(utts), json.dumps(ans), datetime.now().isoformat(timespec="seconds")))
        db.commit()
        if not quiet:
            print("  analyzed %s (%d part%s)" % (base, len(ans), "" if len(ans) == 1 else "s"))

    failed = []
    if todo and be.warm_cache:
        p, h = todo.pop(0)
        try:
            save(work(p, h))
        except (backends.BackendError, ValueError) as e:
            failed.append(p)
            print("  fail %s: %s" % (os.path.basename(p), e))
    with ThreadPoolExecutor(max(1, int(cfg.get("workers", 4)))) as ex:
        futs = {ex.submit(work, p, h): p for p, h in todo}
        for f in as_completed(futs):
            try:
                save(f.result())
            except backends.BackendError as e:
                failed.append(futs[f])
                print("  fail %s: %s" % (os.path.basename(futs[f]), e))
                if "not logged in" in str(e).lower() or "api key" in str(e).lower():
                    for g in futs:
                        g.cancel()
            except Exception as e:  # one bad file should not stop the batch
                failed.append(futs[f])
                print("  fail %s: %s" % (os.path.basename(futs[f]), e))
    if not quiet and be.usage.requests:
        print("usage", be.usage.as_dict())
    return keys, failed


# ---------------------------------------------------------------- verification
def norm(s):
    s = (s or "").lower().replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def match_one(q, texts):
    """Best fuzzy match of normalized quote q inside any of the normalized texts (0..1)."""
    qw = q.split()
    n = len(qw)
    qset = set(qw)
    best = 0.0
    for t in texts:
        if q in t:
            return 1.0
        tw = t.split()
        for size in (n - 1, n, n + 1):
            if size < 1 or size > len(tw):
                continue
            for i in range(0, len(tw) - size + 1):
                win = tw[i:i + size]
                if len(qset.intersection(win)) < 0.6 * n:
                    continue
                sm = difflib.SequenceMatcher(None, q, " ".join(win), autojunk=False)
                if sm.quick_ratio() <= best:
                    continue
                best = max(best, sm.ratio())
    return best


def quote_score(quote, prospect_texts, min_words):
    """How well a quote matches what the PROSPECT said. Ellipsis parts must each match."""
    parts = [norm(p) for p in re.split("\\.\\.\\.|\u2026", quote or "")]
    parts = [p for p in parts if p]
    if not parts or sum(len(p.split()) for p in parts) < min_words:
        return 0.0
    return min(match_one(p, prospect_texts) for p in parts)


def score_call(row, cfg):
    """Turn cached chunk answers into one call record, keeping only verified flags."""
    th = cfg["thresholds"]
    utts = json.loads(row["utts"])
    answers = json.loads(row["answers"])
    pros = [norm(u["text"]) for u in utts if u["role"] == "PROSPECT"]
    c = {"path": row["path"], "file": os.path.basename(row["path"]), "title": row["title"], "date": row["date"],
         "pain": {}, "obj": {}, "handle": {}, "rejected": [], "flagged": 0, "unlisted": []}
    for kind, key in (("pain", "pains"), ("obj", "objections")):
        tax = cfg[key]
        for i, a in enumerate(answers):
            for label, it in (a.get(key) or {}).items():
                if label not in tax or not it.get("present") or (it.get("confidence") or 0) < th["confidence"]:
                    continue
                c["flagged"] += 1
                good = None
                for q in it.get("evidence") or []:
                    s = quote_score(q, pros, th["min_quote_words"])
                    if s >= th["quote_match"] and (good is None or s > good[1]):
                        good = (q, s)
                if not good:
                    c["rejected"].append((kind, label, (it.get("evidence") or [""])[0]))
                    continue
                prev = c[kind].get(label)
                if prev is None or it["confidence"] > prev["confidence"]:
                    c[kind][label] = {"quote": good[0], "match": good[1], "confidence": it["confidence"], "chunk": i}
                    if kind == "obj":
                        h = it.get("handling") or 0
                        c["handle"][label] = (h - 1) if h >= 1 else None
    # a label verified in one part but rejected in another still counts once
    c["rejected"] = [r for r in c["rejected"] if r[1] not in c[r[0]]]
    for a in answers:
        for u in a.get("unlisted") or []:
            if quote_score(u.get("evidence"), pros, th["min_quote_words"]) >= th["quote_match"]:
                c["unlisted"].append(u)
    reps = [a.get("rep") or {} for a in answers]
    c["discovery"] = max([r.get("discovery", 0) for r in reps] or [0])
    for k in ("pain_quantified", "offer_tied_to_pain", "asked_for_sale"):
        c[k] = any(r.get(k) for r in reps)
    c["next_step"] = reps[-1].get("next_step", 0) if reps else 0
    words = defaultdict(int)
    for u in utts:
        words[(u["role"], u["name"])] += u["words"]
    rep_words = {n: w for (r, n), w in words.items() if r == "REP"}
    c["rep"] = max(rep_words, key=rep_words.get) if rep_words else "Unknown"
    rw = sum(rep_words.values())
    pw = sum(w for (r, _), w in words.items() if r == "PROSPECT")
    c["rep_share"] = rw / max(1, rw + pw)
    return c


# ---------------------------------------------------------------- report helpers
def pct(a, b):
    return "%d%%" % round(100.0 * a / b) if b else "-"


def calls(n):
    return "%d call%s" % (n, "" if n == 1 else "s")


def trim(text, n):
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "..."


def name_of(cfg, kind, label):
    tax = cfg["pains"] if kind == "pain" else cfg["objections"]
    return tax.get(label, {}).get("name") or label.replace("_", " ").capitalize()


def load_rows(db, akey=None, hashes=None):
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT * FROM calls ORDER BY created").fetchall()
    db.row_factory = None
    if hashes is not None:
        rows = [r for r in rows if (r["file_hash"], r["akey"]) in hashes]
    elif akey:
        rows = [r for r in rows if r["akey"] == akey]
    latest = {}
    for r in rows:
        latest[r["path"]] = r  # newest analysis per file wins
    return list(latest.values())


def read_won(folder):
    p = os.path.join(folder, "won.txt") if folder and os.path.isdir(folder) else None
    if not p or not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return {l.strip() for l in f if l.strip() and not l.startswith("#")}


# ---------------------------------------------------------------- report
def cmd_report(args, cfg, db, be, keys=None):
    th = cfg["thresholds"]
    akey = analysis_key(cfg, be, build_system(cfg), build_schema(cfg))
    if keys is None and args.folder:
        keys = {p: (file_hash(p), akey) for p in list_transcripts(args.folder)}
    rows = load_rows(db, akey=akey, hashes=set(keys.values()) if keys else None)
    if not rows:
        sys.exit("no analyzed calls in the cache; run analyze first")
    C = [score_call(r, cfg) for r in rows]
    N = len(C)
    won = read_won(args.folder) if getattr(args, "folder", None) else None
    for c in C:
        c["won"] = (c["file"] in won) if won is not None else None
    dates = sorted(c["date"] for c in C)
    flagged = sum(c["flagged"] for c in C)
    verified = sum(len(c["pain"]) + len(c["obj"]) for c in C)
    rejected = sum(len(c["rejected"]) for c in C)

    L = ["# Sales calls: pains and objections ranked\n"]
    L.append("**Scope:** %d calls, %s to %s. Every count is a call where the prospect said it and the quote was found in the prospect's own words in the transcript.\n" % (N, dates[0], dates[-1]))
    L.append("**Verification:** the model flagged %d pains and objections. %d were backed by a prospect quote found in the transcript. %d were dropped because the quote was not there (or was the rep talking).\n" % (flagged, verified, rejected))
    if won is not None:
        L.append("**Won calls (won.txt):** %d of %d.\n" % (sum(1 for c in C if c["won"]), N))

    def ranking(kind, key, title):
        L.append("## %s\n" % title)
        rows_ = []
        for label in cfg[key]:
            hit = [c for c in C if label in c[kind]]
            rows_.append((label, hit))
        rows_.sort(key=lambda r: -len(r[1]))
        head = "| # | %s | Calls | Share of calls |" % ("Pain point" if kind == "pain" else "Objection")
        sep = "|---|---|---|---|"
        if kind == "obj":
            head += " Avg handling (0 to 3) |"
            sep += "---|"
        if won is not None:
            head += " Won when raised |"
            sep += "---|"
        L.append(head + "\n" + sep)
        for i, (label, hit) in enumerate(rows_, 1):
            line = "| %d | %s | %d | %s |" % (i, name_of(cfg, kind, label), len(hit), pct(len(hit), N))
            if kind == "obj":
                hs = [c["handle"][label] for c in hit if c["handle"].get(label) is not None]
                line += " %s |" % ("%.1f" % (sum(hs) / len(hs)) if hs else "-")
            if won is not None:
                line += " %s |" % pct(sum(1 for c in hit if c["won"]), len(hit))
            L.append(line)
        L.append("\n### In their words\n")
        for label, hit in rows_[:args.top]:
            if not hit:
                continue
            qs = sorted(((c[kind][label]["confidence"] * c[kind][label]["match"], c) for c in hit), key=lambda q: -q[0])
            L.append("**%s** (%s)" % (name_of(cfg, kind, label), calls(len(hit))))
            seen = set()
            for _, c in qs:
                q = c[kind][label]["quote"]
                if q in seen:
                    continue
                seen.add(q)
                L.append("- \"%s\" (%s, %s)" % (trim(q, 300), c["file"], c["date"]))
                if len(seen) == 3:
                    break
            if kind == "obj":
                best = sorted((c for c in hit if c["handle"].get(label) is not None), key=lambda c: -c["handle"][label])[:1]
                if best:
                    L.append("- Best handled: %s by %s (%d/3)" % (best[0]["file"], best[0]["rep"], best[0]["handle"][label]))
                L.append("- Play: " + cfg["objections"][label].get("play", ""))
            L.append("")
        return rows_

    ranking("pain", "pains", "Biggest pain points (what prospects walk in with)")
    obj_rows = ranking("obj", "objections", "Most common objections")

    unl = [(c, u) for c in C for u in c["unlisted"]]
    if unl:
        L.append("## Outside the taxonomy\n")
        L.append("Verified prospect lines that fit none of your labels. If one keeps coming up, add it to the config.\n")
        for c, u in unl[:15]:
            L.append("- %s: %s. \"%s\" (%s)" % (u.get("kind"), trim(u.get("summary"), 120), trim(u.get("evidence"), 200), c["file"]))
        L.append("")

    L.append("## Rep scorecard\n")
    head = "| Rep | Calls | Discovery (0 to 3) | Got a number on the pain | Tied offer to their words | Asked for the sale | Firm next step | Objection handling (0 to 3) | Rep talk share |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    if won is not None:
        head += " Won |"
        sep += "---|"
    L.append(head + "\n" + sep)
    by_rep = defaultdict(list)
    for c in C:
        by_rep[c["rep"]].append(c)
    for rep, cs in sorted(by_rep.items(), key=lambda x: -len(x[1])):
        hs = [c["handle"][l] for c in cs for l in c["obj"] if c["handle"].get(l) is not None]
        n = len(cs)
        line = "| %s | %d | %.1f | %s | %s | %s | %s | %s | %s |" % (
            rep, n, sum(c["discovery"] for c in cs) / n,
            pct(sum(c["pain_quantified"] for c in cs), n), pct(sum(c["offer_tied_to_pain"] for c in cs), n),
            pct(sum(c["asked_for_sale"] for c in cs), n), pct(sum(c["next_step"] >= th["firm_next_step"] for c in cs), n),
            "%.1f" % (sum(hs) / len(hs)) if hs else "-", pct(sum(c["rep_share"] for c in cs), n))
        if won is not None:
            line += " %s |" % pct(sum(1 for c in cs if c["won"]), n)
        L.append(line)
    L.append("\nRep is whoever on your side talked most on the call. Firm next step means a dated follow-up or a commitment on the call.\n")

    if won is not None:
        L.append("## What the won calls did differently\n")
        L.append("| On the call | Won when it happened | Won when it did not |\n|---|---|---|")
        tests = [("Rep asked for the sale", lambda c: c["asked_for_sale"]),
                 ("Prospect put a number on the problem", lambda c: c["pain_quantified"]),
                 ("Offer tied back to their own words", lambda c: c["offer_tied_to_pain"]),
                 ("Deep discovery", lambda c: c["discovery"] >= th["deep_discovery"]),
                 ("Firm next step", lambda c: c["next_step"] >= th["firm_next_step"])]
        for name, f in tests:
            yes = [c for c in C if f(c)]
            no = [c for c in C if not f(c)]
            L.append("| %s | %s (%s) | %s (%s) |" % (name, pct(sum(c["won"] for c in yes), len(yes)), calls(len(yes)),
                                                pct(sum(c["won"] for c in no), len(no)), calls(len(no))))
        L.append("\nCorrelation, not proof: warmer prospects also make it easier to ask and to book.\n")

    L.append("## Coaching priorities\n")
    L.append("Ranked by how often the objection comes up times how badly it is handled.\n")
    pri = []
    for label, hit in obj_rows:
        hs = [c["handle"][label] for c in hit if c["handle"].get(label) is not None]
        avg = sum(hs) / len(hs) if hs else 3
        if len(hit) >= th["min_calls_for_priority"] and avg < 2.5:
            pri.append(((3 - avg) * len(hit), label, len(hit), avg))
    pri.sort(key=lambda p: -p[0])
    if not pri:
        L.append("Not enough repeated objections yet (needs %d or more calls per objection)." % th["min_calls_for_priority"])
    for i, (_, label, n, avg) in enumerate(pri[:5], 1):
        L.append("%d. **%s**: %s, handled %.1f/3 on average. %s" % (i, name_of(cfg, "obj", label), calls(n), avg, cfg["objections"][label].get("play", "")))
    L.append("")

    runs = db.execute("SELECT backend, model, sum(requests), sum(input_tokens), sum(output_tokens), sum(cache_read), sum(cost_usd) FROM runs GROUP BY 1, 2").fetchall()
    L.append("## Method and limits\n")
    L.append("- Each transcript is split into parts of about %d characters with speakers tagged REP or PROSPECT. One structured model call per part returns every label with present, confidence, verbatim evidence and (for objections) a handling level, plus rep behaviour." % cfg["chunk_chars"])
    L.append("- A flag counts only when confidence is %.2f or higher and one of its quotes matches the prospect's own words at %.2f or higher (normalized fuzzy match, checked in code)." % (th["confidence"], th["quote_match"]))
    L.append("- Handling grade is the model's 1 to 4 level minus 1: 0 ignored or defensive, 3 resolved and the prospect moved forward.")
    L.append("- The taxonomy lives in %s. Anything outside it is only listed, not counted." % os.path.basename(cfg["_path"]))
    for b, m, rq, it, ot, cr, cost in runs:
        L.append("- Usage so far with %s (%s): %s requests, %s input tokens, %s output tokens, %s cache reads%s." % (
            b, m, format(rq, ","), format(it, ","), format(ot, ","), format(cr, ","), ", reported cost US$%.2f" % cost if cost else ""))
    out = args.out or os.path.join(cfg.get("reports_dir") or "reports", "report.md")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s (%d calls, %d verified flags, %d dropped)" % (out, N, verified, rejected))


# ---------------------------------------------------------------- coach
def cmd_coach(args, cfg, db, be, roles):
    th = cfg["thresholds"]
    keys, failed = analyze_files([args.file], cfg, be, db, roles, quiet=True)
    log_run(db, "coach", be)
    if failed:
        sys.exit(1)
    rows = load_rows(db, hashes={keys[args.file]})
    c = score_call(rows[0], cfg)
    others = [score_call(r, cfg) for r in load_rows(db, akey=keys[args.file][1]) if r["path"] != c["path"]]

    def flag(ok):
        return "OK" if ok else "FIX"

    L = ["# Call feedback: %s (%s)" % (c["title"], c["date"]), "", "Rep: **%s**" % c["rep"], "",
         "| Check | Result | |", "|---|---|---|",
         "| Discovery depth | %d / 3 | %s |" % (c["discovery"], flag(c["discovery"] >= th["deep_discovery"])),
         "| Prospect put a number on the problem | %s | %s |" % ("yes" if c["pain_quantified"] else "no", flag(c["pain_quantified"])),
         "| Offer tied to their own words | %s | %s |" % ("yes" if c["offer_tied_to_pain"] else "no", flag(c["offer_tied_to_pain"])),
         "| Asked for the sale | %s | %s |" % ("yes" if c["asked_for_sale"] else "no", flag(c["asked_for_sale"])),
         "| Next step firmness | %d / 3 | %s |" % (c["next_step"], flag(c["next_step"] >= th["firm_next_step"])),
         "| Rep talk share | %d%% | %s |" % (round(c["rep_share"] * 100), flag(c["rep_share"] <= th["max_rep_talk_share"])),
         "", "## Pains the prospect named"]
    pains = sorted(c["pain"], key=lambda l: -c["pain"][l]["confidence"])
    if not pains:
        L.append("None verified.")
    for l in pains:
        L.append("- **%s**: \"%s\"" % (name_of(cfg, "pain", l), trim(c["pain"][l]["quote"], 220)))
    L.append("\n## Objections")
    objs = sorted(c["obj"], key=lambda l: -c["obj"][l]["confidence"])
    if not objs:
        L.append("None verified.")
    for l in objs:
        h = c["handle"].get(l)
        L.append("- **%s** (handled %s/3): \"%s\"" % (name_of(cfg, "obj", l), h if h is not None else "?", trim(c["obj"][l]["quote"], 220)))
        if h is None or h < 2:
            L.append("  - Play: " + cfg["objections"][l].get("play", ""))
            best = sorted((o for o in others if (o["handle"].get(l) or 0) >= 2), key=lambda o: -o["handle"][l])[:1]
            if best:
                L.append("  - Watch: %s, %s handled it %d/3" % (best[0]["file"], best[0]["rep"], best[0]["handle"][l]))
    if c["rejected"]:
        L.append("\n_%d flag%s dropped: the model's quote was not found in what the prospect said._" % (len(c["rejected"]), "" if len(c["rejected"]) == 1 else "s"))
    L.append("\n## Coach notes")
    notes = []
    if c["discovery"] < th["deep_discovery"]:
        notes.append("Discovery stayed shallow. Before pitching, get what they tried, what it cost, why now, and what happens if nothing changes.")
    if not c["pain_quantified"]:
        notes.append("No number on the pain. Ask how many leads or customers a month now, what a customer is worth, how many more they need. Price objections are easier against a number they said.")
    if not c["offer_tied_to_pain"]:
        notes.append("The pitch was not tied back to their words. Repeat their problem back before each part of the offer.")
    if not c["asked_for_sale"]:
        notes.append("You did not ask for the sale. Ask directly for the package and a start date.")
    if c["next_step"] < th["firm_next_step"]:
        notes.append("Call ended without a dated next step. Book the follow-up on the call, with every decision maker on it.")
    if c["rep_share"] > th["max_rep_talk_share"]:
        notes.append("You talked %d%% of the time. Aim under 50%% on first calls: ask, then wait." % round(c["rep_share"] * 100))
    if not notes:
        notes.append("Clean call on the fundamentals. Keep the structure.")
    L += ["- " + n for n in notes]
    out = "\n".join(L) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print("wrote", args.out)
    print(out)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="path to coach.config.json (default: ./coach.config.json, then the bundled one)")
    p.add_argument("--backend", choices=["auto", "anthropic", "claude-code", "mock"])
    p.add_argument("--mock", action="store_true", help="same as --backend mock (offline, free)")
    p.add_argument("--model", help="model id for the chosen backend")
    p.add_argument("--rep", action="append", default=[], help="extra rep speaker name (repeatable)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    s = sub.add_parser("analyze")
    s.add_argument("folder")
    s = sub.add_parser("report")
    s.add_argument("folder", nargs="?")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--out")
    s = sub.add_parser("coach")
    s.add_argument("file")
    s.add_argument("--out")
    s = sub.add_parser("all")
    s.add_argument("folder")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--out")
    a = p.parse_args()

    if a.cmd == "init":
        if os.path.exists("coach.config.json"):
            sys.exit("coach.config.json already exists here")
        shutil.copy(os.path.join(HERE, "coach.config.json"), "coach.config.json")
        print("wrote coach.config.json: set reps, offer, pains and objections for your business")
        return

    cfg = load_config(a.config)
    mcfg = dict(cfg.get("model") or {})
    if a.model:
        mcfg["anthropic_model"] = mcfg["claude_code_model"] = a.model
    name = "mock" if a.mock else (a.backend or os.environ.get("SALES_COACH_BACKEND") or mcfg.get("backend") or "auto")
    try:
        be = backends.make(name, mcfg)
    except backends.BackendError as e:
        sys.exit(str(e))
    roles = Roles(cfg, a.rep)
    if a.rep:
        cfg["reps"] = list(cfg.get("reps", [])) + a.rep
    db = open_cache(cfg)
    if a.cmd in ("analyze", "all"):
        paths = list_transcripts(a.folder)
        if not paths:
            sys.exit("no .txt, .md, .vtt or .srt transcripts in %s" % a.folder)
        keys, _ = analyze_files(paths, cfg, be, db, roles)
        log_run(db, a.cmd, be)
        if a.cmd == "all":
            cmd_report(a, cfg, db, be, keys)
    elif a.cmd == "report":
        cmd_report(a, cfg, db, be)
    elif a.cmd == "coach":
        cmd_coach(a, cfg, db, be, roles)


if __name__ == "__main__":
    main()
