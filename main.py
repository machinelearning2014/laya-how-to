"""A comprehensive tour of the laya API.

Everything here runs against the already-cached `english` checkpoint. Nothing downloads
the multilingual or typed-decisions models; where those matter, `route()` shows the
decision they would be chosen for, and it needs no weights at all.

    .venv/bin/python main.py

Every printed value in the comments below is real output from this file.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import torch

import laya
from laya import BaseHook, Router, apply_confidence_gate
from laya.calibrate import records_from_labeled

# The shipped english checkpoint carries a `choice:11+` temperature outside the accepted
# range and emits a RuntimeWarning on load. Section 05 explains it; it affects no argmax
# here, so it is silenced to keep the tour readable.
warnings.filterwarnings("ignore")

STATE = (
    "Hi, we were billed twice for March. Please refund the duplicate today or we will "
    "cancel our plan."
)

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle `message`?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is `message`?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does `message` threaten to cancel or leave?"},
    "sentiment": {"type": "score", "instructions": "How unhappy is the user in `message`?",
                  "criteria": ["neutral", "annoyed", "angry", "furious"]},
    "refund_requested": {"type": "noul", "instructions": "Does `message` ask for money back?",
                         "criteria": {"true": "asks for a refund or a charge reversed",
                                      "false": "no refund requested"}},
    # `labels` only changes the text shown to the model. The returned value is still P(true).
    "security_related": {"type": "noul",
                         "instructions": "Does `message` report fraud, account compromise or a data breach?",
                         "labels": {"true": "security incident", "false": "ordinary support"}},
}

# Checkpoints fetched from this repo's GitHub Releases with fetch_model.py, if any.
# `Router(models=...)` replaces the whole table, so start from the built-ins and
# override only the names found locally -- otherwise routing to a checkpoint you
# have not fetched would fail with "unknown model".
LOCAL_MODELS = os.path.expanduser("~/.laya/models")


def model_sources() -> dict:
    models = dict(laya.DEFAULT_MODELS)
    for name in models:
        local = os.path.join(LOCAL_MODELS, name)
        if os.path.isfile(os.path.join(local, "model.safetensors")):
            models[name] = local
    return models


router = Router(models=model_sources())


def section(n: int, title: str) -> None:
    print(f"\n{'=' * 74}\n{n:02d}. {title}\n{'=' * 74}")


# ---------------------------------------------------------------------------
# 01. Routing: pick a checkpoint without loading one
# ---------------------------------------------------------------------------
def show_routing() -> None:
    section(1, "Routing (loads no weights)")

    for text in [STATE,
                 "Bonjour, merci de m'aider avec ma facture.",
                 "Mein Konto wurde zweimal belastet, bitte erstatten Sie den Betrag."]:
        d = router.route(text)
        print(f"  {d['model']:<13} {d['reason']}")

    # route_batch answers many routing decisions in one pass.
    decisions = router.route_batch([{"state": t, "questions": {}} for t in
                                    [STATE, "Bonjour, merci de m'aider avec ma facture."]])
    print("  batch ->", [d["model"] for d in decisions])

    print("\n  Precedence: model > task > detected workflow > lang > lang_guess > detection > default")
    print("  Aliases:    en/laya/default | multi/ml | typed/decisions")

    # Where each checkpoint is loaded from: a copy fetched by fetch_model.py, or the Hub.
    sources = {name: ("local" if spec != laya.DEFAULT_MODELS[name] else "hub")
               for name, spec in model_sources().items()}
    print("  Sources:   ", ", ".join(f"{n}={s}" for n, s in sources.items()))
    if "local" not in sources.values():
        print("             (fetch one with: python fetch_model.py english)")


# ---------------------------------------------------------------------------
# 02. The three question types
# ---------------------------------------------------------------------------
def show_question_types() -> None:
    section(2, "Question types: choice, score, noul")

    result = router.predict(STATE, QUESTIONS)
    a = result["answers"]

    print("  choice ->", a["department"]["choice"], "of", list(QUESTIONS["department"]["criteria"]))
    print("  score  ->", a["urgency"]["score"], "on a 0..2 scale (an expected level, not an index)")
    print("  noul   ->", a["churn_risk"]["noul"], "= P(true)")

    print("\n  A score returns a distribution over levels, with a legend:")
    print("   ", json.dumps(a["sentiment"]["probabilities"]), "-> expected", a["sentiment"]["score"])
    print("   ", json.dumps(a["sentiment"]["legend"]))

    print("\n  criteria dicts add descriptions; labels relabel the two noul options:")
    print("    refund_requested ->", a["refund_requested"]["noul"], "(criteria true/false)")

    # Labels are not cosmetic: they change the text the model reads, so the answer can move.
    unlabelled = router.predict(STATE, {"q": {"type": "noul",
                                              "instructions": QUESTIONS["security_related"]["instructions"]}})
    print("\n  the same question, with and without custom labels:")
    print("    without labels ->", unlabelled["answers"]["q"]["noul"])
    print("    with labels    ->", a["security_related"]["noul"], "(security incident / ordinary support)")
    print("    The probability is always P(true); relabelling only changes what the model reads.")


# ---------------------------------------------------------------------------
# 03. Reading the payload
# ---------------------------------------------------------------------------
def show_payload() -> None:
    section(3, "Reading the payload")

    result = router.predict(STATE, QUESTIONS)
    print("  top-level keys ->", list(result.keys()))

    d = result["answers"]["department"]
    print("\n  confidence vs answer_confidence (choice):")
    print("    answer_confidence (max p) ->", d["answer_confidence"], "<- calibrated, gate on this")
    print("    confidence (entropy)      ->", d["confidence"], "<- drifts with option count")

    print("\n  usage reports whether the model saw the whole state:")
    print("   ", json.dumps(result["usage"]))

    print("\n  GATE_STATES ->", laya.GATE_STATES)


# ---------------------------------------------------------------------------
# 04. Batch and long documents
# ---------------------------------------------------------------------------
def show_batch_and_long() -> None:
    section(4, "Batch, truncation and windowed scanning")

    out = router.predict_batch([
        {"state": "I was charged twice, please refund me.", "questions": QUESTIONS},
        {"state": "The app crashes every time I open settings.", "questions": QUESTIONS,
         "max_len": 256},
    ], batch_size=2, sort_by_length=True)
    print("  predict_batch (input order preserved):")
    for r in out:
        print(f"    {r['routing']['model']:<9} {r['answers']['department']['choice']:<10} "
              f"{r['answers']['department']['answer_confidence']}")

    # A state longer than max_len is silently trimmed from the right; usage tells you.
    long_state = "This is a long support thread with many previous messages. " * 60
    truncated = router.predict(long_state, {"x": QUESTIONS["churn_risk"]})
    print("\n  long state ->", {k: v for k, v in truncated["usage"].items()
                                if k in ("state_tokens", "state_tokens_dropped", "truncated")})

    # predict_long windows the state instead of dropping it, then aggregates per type.
    windowed = router.predict_long(long_state, {"x": QUESTIONS["churn_risk"]})
    print("  predict_long ->", windowed["answers"]["x"]["noul"],
          "over", windowed["usage"]["windows"], "windows")
    print("  (noul takes the max P(true) across windows; choice/score take the best window)")


# ---------------------------------------------------------------------------
# 05. Confidence, abstention and the gate
# ---------------------------------------------------------------------------
def show_abstention() -> None:
    section(5, "Abstention: min_confidence, per-bucket maps, the gate helpers")

    gated = router.predict(STATE, QUESTIONS, min_confidence=0.9)
    for qid in ("department", "sentiment"):
        a = gated["answers"][qid]
        print(f"  {qid:<11} {a['answer_confidence']:<7} {a['abstention']:<10} "
              f"threshold {a['abstention_threshold']}"
              f"{'  low_confidence' if a.get('low_confidence') else ''}")

    print("\n  One threshold does not transfer across option counts. A per-bucket map instead:")
    print("  Buckets are named <type>:<option-count band>, e.g. choice:3-5, noul:2, score:6-10.")
    payload = router.predict(STATE, QUESTIONS)
    apply_confidence_gate([payload], {"choice:3-5": 0.95, "score:3-5": 0.5,
                                      "noul:2": 0.6, "default": 0.99})
    for qid, a in payload["answers"].items():
        print(f"    {qid:<18} threshold {a['abstention_threshold']:<5} -> {a['abstention']}")

    print("\n  With no threshold, none of these fields are written at all.")

    print("\n  The clamp behind the load-time warning:")
    print("    fitters accept [0.1, 10], but the agent clamps to [0.5, 5.0] on load,")
    print("    so the shipped choice:11+ value of 0.1006 runs as 0.5 and stays uncalibrated.")


# ---------------------------------------------------------------------------
# 06. Typed output with decide()
# ---------------------------------------------------------------------------
def show_decide() -> None:
    section(6, "Typed output: decide() over a JSON schema")

    schema = {"type": "object", "properties": {
        "department": {"type": "string", "enum": ["billing", "technical", "other"]},
        "urgency": {"type": "integer", "minimum": 0, "maximum": 2},
        "needs_human": {"type": "boolean"}}}

    print("  decide      ->", router.decide(STATE, schema=schema))
    print("  decide_batch ->", router.decide_batch(
        [STATE, "The app crashes on launch."], schema=schema))

    details = router.decide(STATE, schema=schema, return_details=True)
    print("\n  return_details gives a DecisionResult:")
    print("    values             ->", details.values)
    print("    answer_confidence  ->", {k: round(v, 4) for k, v in details.answer_confidence.items()})

    abstained = router.decide(STATE, schema=schema, min_confidence=0.99)
    print("\n  with min_confidence, a field below threshold comes back None, not a guess:")
    print("   ", abstained)

    print("\n  Watch the urgency value above: it reads 0 on a message that says refund today")
    print("  or we cancel. A schema-derived score question has level texts \"0\", \"1\", \"2\"")
    print("  and no descriptions, so the model infers the scale from the property name alone.")
    print("  When a scale needs meaning, write the question explicitly instead of via schema.")

    print("\n  A pydantic model works the same way but needs the extra:")
    print("    pip install 'laya[structured]'   # then: decide(state, schema=TicketModel)")
    print("    (pydantic is not installed in this venv, so only the JSON-schema path runs)")


# ---------------------------------------------------------------------------
# 07. Presets and text helpers
# ---------------------------------------------------------------------------
def show_presets_and_helpers() -> None:
    section(7, "Presets, the backtick convention, and text helpers")

    presets = {"triage": laya.triage_questions, "email": laya.email_questions,
               "guard": laya.guard_questions, "moderation": laya.moderation_questions,
               "router": laya.router_questions}
    for name, factory in presets.items():
        qs = factory()
        print(f"  {name:<11} reads state field {laya.presets.state_field(qs):<9} "
              f"({len(qs)} questions: {', '.join(qs)})")

    print("\n  state_field() is how the CLI and MCP server know where to put your text.")
    print("  Name the field in backticks in every instruction or it resolves to None.")

    print("\n  This file's questions read ->", laya.presets.state_field(QUESTIONS))

    email = laya.email_state("Double charge", "Please refund.\n\nOn Mon, Bob wrote:\n> old thread",
                             sender="c@example.com")
    print("\n  email_state ->", email)
    print("  cleaned     ->", repr(laya.clean_email_body(email["body"])))

    d = laya.detect_language("Bonjour, merci de m'aider avec ma facture.")
    print("\n  detect_language ->", d["language"], "| is_english", d["is_english"],
          "| script", d["script"])
    print("  detect_script('hello') ->", laya.detect_script("hello"),
          "| is_english('hello') ->", laya.is_english("hello"))
    print("  Short text carries too little signal: detect_language('Bonjour, merci') ->",
          laya.detect_language("Bonjour, merci")["language"])


# ---------------------------------------------------------------------------
# 08. Hooks: logging and caching
# ---------------------------------------------------------------------------
def show_hooks() -> None:
    section(8, "Hooks: on_route, on_predict_start/end, and ctx.skip for caching")

    def key(ctx):
        return json.dumps([ctx.states[0], ctx.questions], sort_keys=True, default=str)

    class LogAndCache(BaseHook):
        def __init__(self):
            self.cache, self.events = {}, []

        def on_route(self, ctx):
            self.events.append(("route", ctx.decision["model"]))

        def on_predict_start(self, ctx):
            hit = self.cache.get(key(ctx))
            if hit is not None:
                ctx.skip([hit])              # short-circuit: no forward pass runs
                self.events.append(("cache", "hit"))

        def on_predict_end(self, ctx):
            self.cache.setdefault(key(ctx), ctx.results[0])
            self.events.append(("end_ms", round(ctx.elapsed_ms, 1)))

    hook = LogAndCache()
    hooked = Router(hooks=[hook], hooks_raise=False, hooks_timeout=1.0)
    qs = {"urgent": {"type": "noul", "instructions": "Is `message` urgent?"}}

    hooked.predict(STATE, qs)   # cold: routes, runs, caches
    hooked.predict(STATE, qs)   # warm: skipped
    for event in hook.events:
        print("  ", event)
    print("\n  The second call returns from cache, so its elapsed_ms is a fraction of the first.")
    print("  Available events: on_route, on_load, on_evict, on_predict_start, on_predict_end, on_error")
    print("  hooks_raise=False downgrades a broken hook to a warning; hooks_timeout bounds each call.")


# ---------------------------------------------------------------------------
# 09. Shortlisting many-label questions
# ---------------------------------------------------------------------------
def show_shortlist() -> None:
    section(9, "Shortlisting a question with more labels than the head budget fits")

    labels = ["billing", "refunds", "technical", "outage", "security", "account",
              "shipping", "cancellation", "upgrade", "pricing", "integrations", "other"]
    many = {"dept": {"type": "choice", "instructions": "Which department handles `request`?",
                     "criteria": {label: label for label in labels}}}

    agent = router.load("english")
    shortlisted = laya.predict_shortlist(
        agent, "I was charged twice and want a refund", many,
        embed_fn=laya.embed_fn_from_agent(agent), k=4)

    info = shortlisted["shortlist"]["dept"]
    print(f"  ranked {info['n']} labels down to k={info['k']}:")
    for label, score in zip(info["labels"], info["scores"]):
        print(f"    {label:<14} {score:.4f}")
    print("  predicted ->", shortlisted["answers"]["dept"]["choice"])

    full = router.predict("I was charged twice and want a refund", many)
    print("\n  The same question over all 12 labels ->", full["answers"]["dept"]["choice"])

    print("\n  That difference is the point. Ranking is by embedding similarity, so a label the")
    print("  embedder scores poorly is removed before the model ever sees it: here the shortlist")
    print("  dropped both 'billing' and 'refunds' from a message about a double charge.")
    print("  Shortlisting buys headroom for many labels and costs accuracy -- measure the two")
    print("  against each other on your own data before adopting it.")


# ---------------------------------------------------------------------------
# 10. Calibration on your own labels
# ---------------------------------------------------------------------------
def show_calibration() -> None:
    section(10, "Calibration: records, temperatures, saving and loading")

    agent = router.load("english")
    question = {"dept": QUESTIONS["department"]}
    refund = [1.0, 0.0, 0.0]
    technical = [0.0, 1.0, 0.0]
    pairs = [("I was charged twice", question, {"dept": refund}),
             ("The app crashes on launch", question, {"dept": technical})] * 6

    records = records_from_labeled(agent, pairs)
    print(f"  records_from_labeled -> {len(records)} records; one per (state, question) pair")
    print("  each record is (qtype, logits, target, k) -- what the fitters consume")

    fitted = laya.fit_temperatures(records)
    print("\n  fit_temperatures ->")
    print("    temperature        ->", fitted["temperature"], "(one per type: choice, score, noul)")
    print("    temperature_by_options ->", fitted["temperature_by_options"],
          "  <- empty: buckets need 2000+ examples each")

    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "calibration.json")
    agent.save_calibration(path)
    print("\n  save_calibration writes temperatures + checkpoint identity, no weights ->",
          os.path.basename(path))
    print("  load_calibration(path) restores them.")

    print("\n  Related fitters: fit_one_temperature | fit_temperature_map | fit_abstention_thresholds")
    print("  | fit_binning_map + apply_binning_map (for a bucket temperature cannot fix)")


# ---------------------------------------------------------------------------
# 11. The training primitives
# ---------------------------------------------------------------------------
def show_training_primitives() -> None:
    section(11, "Training primitives (no model needed)")

    q = torch.tensor([[[0.80, 0.15, 0.05]]])       # reported distribution
    target = torch.tensor([[[1.0, 0.0, 0.0]]])     # teacher distribution
    qtype = torch.tensor([0])                      # 0 = choice
    mask = torch.tensor([[1, 1, 1]])

    print("  proper_reward ->", round(float(laya.proper_reward(q, target, qtype, mask)), 4))
    print("    = log score + 0.5 * spherical - RPS (the RPS term applies to score questions only)")
    print("    The fine-tuning scripts pass w_sph=0.75, not the library default of 0.5.")

    batch = {"target": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
             "ep_group": torch.tensor([0, 0]),
             "ep_step": torch.tensor([0, 1])}
    print("\n  td_lambda_targets (lam=1.0) ->",
          laya.td_lambda_targets(torch.tensor([0.2, 0.9]), batch, lam=1.0).tolist())
    print("    Bootstraps targets across a multi-turn trajectory. Unused by the")
    print("    typed-decisions fine-tune, whose data is single-turn.")

    print("\n  Scoring helpers: ece_score, answer_confidence, confidence_from_probs")
    p = np.array([0.90, 0.80, 0.60])
    correct = np.array([1, 1, 0])
    print("    ece_score ->", round(laya.ece_score(p, correct), 4))
    print("    answer_confidence([0.9,0.06,0.04], 3) ->",
          round(laya.answer_confidence(np.array([0.9, 0.06, 0.04]), 3), 4))


# ---------------------------------------------------------------------------
# 12. What needs a separate process
# ---------------------------------------------------------------------------
def show_out_of_process() -> None:
    section(12, "Features that need a separate process (shown, not run)")

    commands = [
        ("HTTP API", "laya-serve", "POST /v1/systemone, /v1/systemone/batch, GET /health"),
        ("MCP server", "laya-mcp-server", "laya_predict, laya_route, laya_decide, laya_preset, ..."),
        ("CLI", 'laya "text" --preset triage --predict', "--batch, --questions, --json, --min-confidence"),
        ("Evaluation", "laya-evals run dataset.jsonl --min-accuracy 0.9", "ECE, Brier, AURC, per-slice gates"),
        ("Fine-tuning", "torchrun --nproc_per_node=2 train_ddp.py", "the package ships no trainer"),
    ]
    for name, command, note in commands:
        print(f"  {name:<13} {command}")
        print(f"  {'':<13} {note}")

    print("\n  Integrations are import-and-wire rather than call-and-print:")
    print("    from laya import LayaRouter, LayaGuardrail, LayaTriage, LayaEvaluator, LayaDecision")
    print("    LangChain / LangGraph, LlamaIndex selectors, CrewAI routing.")


def main() -> None:
    show_routing()
    show_question_types()
    show_payload()
    show_batch_and_long()
    show_abstention()
    show_decide()
    show_presets_and_helpers()
    show_hooks()
    show_shortlist()
    show_calibration()
    show_training_primitives()
    show_out_of_process()


if __name__ == "__main__":
    main()
