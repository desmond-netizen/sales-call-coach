"""Model backends. Each one takes (system, user, schema, ctx) and returns the parsed JSON answer.

  anthropic    Anthropic Python SDK. Structured output, cached system prompt. Needs ANTHROPIC_API_KEY.
  claude-code  headless Claude Code (`claude -p`). Uses your Claude login, no API key.
  mock         deterministic fake answers from keyword rules. Offline, free, for tests and demos.

Code never trusts these answers on their own: sales_coach.py checks every quote against the transcript.
"""
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time

FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1"}


class BackendError(RuntimeError):
    pass


class Usage:
    def __init__(self):
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read = 0
        self.cache_write = 0
        self.cost_usd = 0.0

    def as_dict(self):
        return dict(requests=self.requests, input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                    cache_read=self.cache_read, cache_write=self.cache_write, cost_usd=round(self.cost_usd, 4))


# ---------------------------------------------------------------- anthropic
class AnthropicBackend:
    name = "anthropic"
    warm_cache = True  # run one request alone first so the rest read the cached system prompt

    def __init__(self, mcfg):
        try:
            import anthropic
        except ImportError:
            raise BackendError("pip install anthropic (needs Python 3.10+), or use --backend claude-code")
        self.sdk = anthropic
        self.model = os.environ.get("SALES_COACH_MODEL") or mcfg.get("anthropic_model") or "claude-sonnet-5-5"
        self.effort = mcfg.get("effort") or None
        self.fallbacks = bool(mcfg.get("fallbacks")) and self.model in FALLBACK_MODELS
        self.client = anthropic.Anthropic(max_retries=int(mcfg.get("max_retries", 4)))
        self.usage = Usage()

    def ask(self, system, user, schema, ctx):
        output_config = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort:
            output_config["effort"] = self.effort
        kw = dict(
            model=self.model,
            max_tokens=16000,
            # the taxonomy and rubric never change between requests, so cache them
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
        )
        try:
            if self.fallbacks:
                # on a safety decline the API retries on a fallback model inside the same call
                r = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
            else:
                r = self.client.messages.create(**kw)
        except self.sdk.AuthenticationError:
            raise BackendError("Anthropic rejected the API key (check ANTHROPIC_API_KEY)")
        except self.sdk.NotFoundError:
            raise BackendError("model %r not found; set model.anthropic_model in the config" % self.model)
        except self.sdk.BadRequestError as e:
            raise BackendError("bad request: %s" % e.message)
        u = r.usage
        self.usage.requests += 1
        self.usage.input_tokens += u.input_tokens or 0
        self.usage.output_tokens += u.output_tokens or 0
        self.usage.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        self.usage.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0
        if r.stop_reason == "refusal":
            raise BackendError("model declined this excerpt (refusal)")
        if r.stop_reason == "max_tokens":
            raise BackendError("answer cut off at max_tokens; lower chunk_chars")
        text = next((b.text for b in r.content if b.type == "text"), "")
        return json.loads(text)


# ---------------------------------------------------------------- claude code
class ClaudeCodeBackend:
    name = "claude-code"
    warm_cache = False

    def __init__(self, mcfg):
        self.bin = shutil.which("claude")
        if not self.bin:
            raise BackendError("`claude` not found on PATH; install Claude Code or use --backend anthropic")
        self.model = os.environ.get("SALES_COACH_MODEL") or mcfg.get("claude_code_model") or ""
        self.effort = mcfg.get("effort") or ""
        self.tries = int(mcfg.get("max_retries", 4)) + 1
        self.usage = Usage()
        self.cwd = tempfile.mkdtemp(prefix="sales-coach-")  # empty dir: no project files leak into the prompt

    def ask(self, system, user, schema, ctx):
        cmd = [self.bin, "-p", "--output-format", "json", "--json-schema", json.dumps(schema),
               "--system-prompt", system, "--tools", "", "--no-session-persistence"]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        env = dict(os.environ)
        env.pop("CLAUDE_CODE_SIMPLE", None)  # bare mode ignores the subscription login
        last = None
        for i in range(self.tries):
            try:
                p = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=900, cwd=self.cwd, env=env)
            except subprocess.TimeoutExpired:
                last = "timed out"
                continue
            try:
                out = json.loads(p.stdout)
            except ValueError:
                last = (p.stderr or p.stdout or "no output").strip()[:500]
                time.sleep(min(30, 2 ** i + random.random()))
                continue
            if isinstance(out, list):  # some versions print a list of events
                out = next((e for e in reversed(out) if e.get("type") == "result"), {})
            u = out.get("usage") or {}
            self.usage.requests += 1
            self.usage.input_tokens += u.get("input_tokens", 0) or 0
            self.usage.output_tokens += u.get("output_tokens", 0) or 0
            self.usage.cache_read += u.get("cache_read_input_tokens", 0) or 0
            self.usage.cache_write += u.get("cache_creation_input_tokens", 0) or 0
            self.usage.cost_usd += out.get("total_cost_usd", 0) or 0
            if out.get("is_error"):
                msg = str(out.get("result") or out.get("subtype") or "error")
                if "login" in msg.lower() or "logged in" in msg.lower():
                    raise BackendError("Claude Code is not logged in. Run `claude` once in a terminal and use /login, then retry. (%s)" % msg)
                last = msg
                time.sleep(min(30, 2 ** i + random.random()))
                continue
            data = out.get("structured_output")
            if data is None:
                data = parse_json_text(out.get("result", ""))
            if data is not None:
                return data
            last = "no JSON in result"
        raise BackendError("claude -p failed: %s" % last)


def parse_json_text(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------- mock
# Keyword rules for the default labels. Unknown labels fall back to the words in the label id.
MOCK_HINTS = {
    "not_enough_leads": ["referral", "word of mouth", "phone isn't ringing", "slow", "dried up", "not enough"],
    "low_lead_quality": ["tire kicker", "unqualified", "junk", "shared lead", "price shopping"],
    "leads_not_converting": ["no-show", "no show", "don't show", "never call back", "follow up", "follow-up"],
    "burned_by_vendors": ["last agency", "previous agency", "got burned", "burned", "took our money", "last company"],
    "marketing_not_profitable": ["wasted", "cost per lead", "not working", "boosted posts", "lost money"],
    "lead_sources_too_expensive": ["per lead", "directory", "lead service", "too much per"],
    "owner_time_bottleneck": ["no time", "don't have time", "wear all the hats", "stretched", "on the roof all day"],
    "no_system_in_place": ["website is", "no website", "no system", "no process", "outdated"],
    "slow_response": ["miss calls", "missed calls", "voicemail", "after hours", "front desk"],
    "inconsistent_revenue": ["feast or famine", "seasonal", "slow months", "up and down", "winter"],
    "growth_plateau": ["second location", "expand", "grow", "stuck", "plateau"],
    "tech_overwhelm": ["overwhelm", "confusing", "not a tech", "don't understand"],
    "low_trust_or_visibility": ["reviews", "nobody knows", "stand out", "visibility"],
    "cash_flow_strain": ["cash flow", "tight right now", "debt", "loan"],
    "price_too_high": ["expensive", "too much", "price is", "pricey", "steep", "more than i expected"],
    "no_budget_now": ["no budget", "can't afford", "don't have the money", "budget"],
    "extra_costs_on_top": ["on top of", "ad spend", "plus the", "extra cost"],
    "need_partner_approval": ["my wife", "my husband", "my partner", "business partner", "run it by", "talk to my"],
    "need_to_think": ["think about it", "send me", "sleep on it", "get back to you", "mull it over"],
    "burned_before_trust": ["how is this different", "heard that before", "burned before", "trust"],
    "will_it_work_for_me": ["case study", "in my area", "my industry", "work for us", "other practices", "proof"],
    "wants_guarantee": ["guarantee", "pay per lead", "refund", "risk free"],
    "contract_length": ["contract", "month to month", "locked in", "twelve months"],
    "bad_timing": ["busy season", "next quarter", "after the holidays", "right now isn't", "timing"],
    "cannot_handle_more_work": ["can't handle", "capacity", "crews are booked", "short staffed", "no one to answer"],
    "doing_it_themselves": ["my nephew", "in house", "in-house", "we do it ourselves", "another company", "other quote"],
    "automation_skepticism": ["robot", "bot", "ai", "automated", "impersonal"],
    "compliance_or_risk": ["hipaa", "compliance", "legal", "license", "privacy"],
    "channel_skepticism": ["facebook", "instagram", "social media", "google ads"],
    "time_commitment_required": ["my time", "how much time", "hours a week", "on my plate"],
}
INVENTED = "Honestly we already tried this exact thing twice and it never worked for us at all."


class MockBackend:
    """Deterministic fake answers. It also invents one quote per excerpt that is not in the
    transcript, so you can watch the verifier drop it in the report."""
    name = "mock"
    warm_cache = False

    def __init__(self, mcfg):
        self.model = "mock"
        self.usage = Usage()

    def ask(self, system, user, schema, ctx):
        self.usage.requests += 1
        lines = ctx["lines"]  # [(role, name, text)] for this excerpt
        seed = int(hashlib.sha256(user.encode()).hexdigest()[:8], 16)
        pros = [(i, t) for i, (r, _, t) in enumerate(lines) if r == "PROSPECT"]
        reps = [(i, t) for i, (r, _, t) in enumerate(lines) if r == "REP"]

        def find(label):
            keys = MOCK_HINTS.get(label) or [w for w in label.split("_") if len(w) > 3]
            for i, t in pros:
                low = t.lower()
                for k in keys:
                    if re.search(r"\b%s\b" % re.escape(k), low):
                        return i, sentence_with(t, k)
            return None, None

        out = {"pains": {}, "objections": {}, "unlisted": [], "rep": {}}
        missing = []
        for kind in ("pains", "objections"):
            for j, label in enumerate(ctx[kind]):
                i, quote = find(label)
                item = {"evidence": [], "present": False, "confidence": round(0.05 + (seed >> j) % 10 / 100, 2)}
                if quote:
                    item = {"evidence": [quote], "present": True, "confidence": round(0.7 + (seed >> j) % 25 / 100, 2)}
                else:
                    missing.append((kind, label))
                if kind == "objections":
                    item["handling"] = handling_guess(lines, i) if quote else 0
                out[kind][label] = item
        if missing:  # the deliberate hallucination
            kind, label = missing[seed % len(missing)]
            out[kind][label].update(evidence=[INVENTED], present=True, confidence=0.81)
            if kind == "objections":
                out[kind][label]["handling"] = 2
        rep_text = " ".join(t for _, t in reps).lower()
        pro_text = " ".join(t for _, t in pros).lower()
        tail = " ".join(t for _, _, t in lines[-6:]).lower()
        next_step = 0
        if ctx["final"]:
            if re.search(r"\b(let's do it|sign|send the agreement|card|get started)\b", tail):
                next_step = 3
            elif re.search(r"\b(monday|tuesday|wednesday|thursday|friday|tomorrow|next week|\d{1,2}(:\d{2})? ?(am|pm))\b", tail):
                next_step = 2
            elif re.search(r"\b(think|send|touch base|get back)\b", tail):
                next_step = 1
        out["rep"] = {
            "discovery": min(3, sum(t.count("?") for _, t in reps) // 4),
            "pain_quantified": bool(re.search(r"\d|\b(hundred|thousand|percent)\b", pro_text)),
            "offer_tied_to_pain": bool(re.search(r"you (said|mentioned|told me)", rep_text)),
            "asked_for_sale": bool(re.search(r"(get you started|move forward|ready to start|sign you up|start on)", rep_text)),
            "next_step": next_step,
        }
        return out


def sentence_with(text, key):
    parts = re.split(r"(?<=[.!?])\s+", text)
    for s in parts:
        if key in s.lower() and len(s.split()) >= 4:
            return s
    return text if len(text.split()) <= 60 else " ".join(text.split()[:60])


def handling_guess(lines, i):
    """Next two REP turns after the objection: a question back, substance, numbers and
    the prospect's own words each score a point."""
    reply = " ".join(t for r, _, t in lines[i + 1:i + 5] if r == "REP")
    if not reply:
        return 1
    score = 1 + ("?" in reply) + (len(reply.split()) > 25)
    score += bool(re.search(r"\d|\b(hundred|thousand)\b", reply)) + bool(re.search(r"you (said|told me|mentioned)", reply.lower()))
    return min(4, score)


def make(name, mcfg):
    if name == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif shutil.which("claude"):
            name = "claude-code"
        else:
            raise BackendError("no backend: set ANTHROPIC_API_KEY, install Claude Code, or pass --mock")
    return {"anthropic": AnthropicBackend, "claude-code": ClaudeCodeBackend, "mock": MockBackend}[name](mcfg)
