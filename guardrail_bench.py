#!/usr/bin/env python3
"""Measure whether laya's confidence is trustworthy enough to automate on.

The whole value proposition rests on one claim: that `answer_confidence` is calibrated
enough to threshold, so you can act on the answers it is sure about and abstain on the
rest. That claim is easy to assert and easy to falsify, so this measures it on labelled
data and prints the numbers.

It reports three things a deployment decision actually turns on:

  1. **Accuracy** on your labelled set, per question.
  2. **Calibration** -- ECE and Brier, and whether fitting a temperature improves them.
  3. **The selective curve** -- as you raise the confidence threshold, what fraction of
     inputs still get an answer, and how accurate is it on those. This is the trade you
     are actually making; accuracy alone hides it.

    python guardrail_bench.py                          # the bundled guardrail set
    python guardrail_bench.py --data mine.jsonl        # your own labelled rows
    python guardrail_bench.py --questions q.json       # your own questions
    python guardrail_bench.py --target-error 0.05      # operating point to solve for

Row format is one JSON object per line: {"text": "...", "injection": 0|1}. Extra keys are
ignored, so a note or a source field is fine. `--preset` picks which question set to
evaluate; every yes/no question in it is scored against the same binary label.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import laya
from laya import Router
from laya.calibrate import records_from_labeled
from laya.common import TEMP_MAX, TEMP_MIN

from local_models import is_fully_local, missing, model_sources

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "guardrail_eval.jsonl")


def load_rows(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise SystemExit(
            f"no such data file: {path}\n"
            f"  the bundled set is {DEFAULT_DATA}\n"
            f"  bring your own with: --data rows.jsonl"
        )
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{n} is not valid JSON: {e}") from None
            if "text" not in row or "injection" not in row:
                raise SystemExit(f"{path}:{n} needs both 'text' and 'injection'")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path} has no rows")
    return rows


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def ece_at(records, temperature: float) -> float:
    """ECE over the records with every question's logits scaled by `temperature`."""
    confs, correct = [], []
    for _qtype, z, target, k in records:
        p = softmax(np.asarray(z[:k], dtype=float) / temperature)
        confs.append(float(p.max()))
        correct.append(float(int(p.argmax()) == int(np.argmax(target[:k]))))
    return laya.ece_score(np.array(confs), np.array(correct))


def selective_curve(conf: np.ndarray, correct: np.ndarray, thresholds) -> list[tuple]:
    rows = []
    for t in thresholds:
        covered = conf >= t
        n = int(covered.sum())
        if n == 0:
            rows.append((t, 0.0, float("nan"), float("nan"), 0))
            continue
        acc = float(correct[covered].mean())
        rows.append((t, float(covered.mean()), acc, 1.0 - acc, n))
    return rows


def bar(fraction: float, width: int = 24) -> str:
    if fraction != fraction:  # NaN
        return " " * width
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "#" * filled + "." * (width - filled)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=DEFAULT_DATA, help="labelled JSONL rows")
    ap.add_argument("--preset", default="guard", help="preset question set to evaluate")
    ap.add_argument("--questions", help="JSON file of your own questions instead of a preset")
    ap.add_argument("--label", default="injection", help="the 0/1 field in each row")
    ap.add_argument("--target-error", type=float, default=0.10,
                    help="error rate to solve an operating point for (default 0.10)")
    ap.add_argument("--only", help="comma-separated question ids to score (default: all yes/no ones)")
    ap.add_argument("--show-errors", type=int, default=8,
                    help="how many misclassifications to list (default 8)")
    args = ap.parse_args()

    rows = load_rows(args.data)
    ys = np.array([int(r[args.label]) for r in rows])
    texts = [r["text"] for r in rows]
    positives = int(ys.sum())

    if args.questions:
        if not os.path.isfile(args.questions):
            raise SystemExit(f"no such questions file: {args.questions}")
        with open(args.questions) as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise SystemExit(f"{args.questions} must contain a JSON object of questions")
        # Accept both shapes the CLI takes: a bare id -> definition mapping, or the
        # {"state_key": ..., "questions": {...}} wrapper.
        inner = loaded.get("questions")
        questions = inner if isinstance(inner, dict) else loaded
        source = args.questions
    else:
        questions = getattr(laya, f"{args.preset}_questions")()
        source = f"preset: {args.preset}"

    noul = {qid: spec for qid, spec in questions.items() if spec.get("type") == "noul"}
    if args.only:
        wanted = [q.strip() for q in args.only.split(",") if q.strip()]
        unknown = [q for q in wanted if q not in noul]
        if unknown:
            raise SystemExit(f"--only names {unknown}, not yes/no questions in {source}")
        noul = {q: noul[q] for q in wanted}
    if not noul:
        raise SystemExit(f"{source} has no yes/no questions to score against a 0/1 label")

    sources = model_sources()
    where = "local mirror" if is_fully_local(sources) else f"hub (missing {', '.join(missing(sources))})"
    print(f"\n  data      {args.data}")
    print(f"  rows      {len(rows)}  ({positives} positive, {len(rows) - positives} negative)")
    print(f"  questions {source}  ->  {', '.join(noul)}")
    print(f"  weights   {where}")

    router = Router(models=sources)
    results = router.predict_batch([{"state": t, "questions": noul} for t in texts])

    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    primary = None

    print(f"\n{'=' * 74}\n  Per question\n{'=' * 74}")
    print(f"  {'question':<20} {'acc':>7} {'ECE':>7} {'Brier':>7}   correct@100% / covered")
    for qid in noul:
        p = np.array([r["answers"][qid]["noul"] for r in results])
        pred = (p >= 0.5).astype(int)
        correct = (pred == ys).astype(float)
        conf = np.maximum(p, 1.0 - p)
        acc = float(correct.mean())
        ece = laya.ece_score(conf, correct)
        brier = float(((p - ys) ** 2).mean())
        print(f"  {qid:<20} {acc:>6.1%} {ece:>7.3f} {brier:>7.3f}   {bar(acc)}")
        if primary is None or acc > primary[1]:
            primary = (qid, acc, p, pred, correct, conf)

    qid, acc, p, pred, correct, conf = primary
    print(f"\n  best question: {qid}  ({acc:.1%} at full coverage, "
          f"{len(rows) - int(correct.sum())} of {len(rows)} wrong)")

    print(f"\n{'=' * 74}\n  Selective curve -- {qid}\n{'=' * 74}")
    print(f"  {'threshold':>9} {'coverage':>9} {'accuracy':>9} {'error':>7}   {'':<24} n")
    curve = selective_curve(conf, correct, thresholds)
    for t, cov, a, err, n in curve:
        print(f"  {t:>9.2f} {cov:>8.1%} {a:>8.1%} {err:>7.1%}   {bar(cov):<24} {n}")

    met = [(t, cov, a, err, n) for t, cov, a, err, n in curve if err == err and err <= args.target_error]
    print()
    if met:
        t, cov, a, err, n = met[0]
        print(f"  lowest threshold meeting a {args.target_error:.0%} error target: {t:.2f}"
              f"  ->  answers {cov:.0%} of inputs at {a:.1%} accuracy ({n} items)")
    else:
        print(f"  no threshold in {thresholds[0]:.2f}-{thresholds[-1]:.2f} meets a "
              f"{args.target_error:.0%} error target on this set.")

    # ---- where the errors are: a single accuracy number hides the interesting split ----
    hard = np.array([str(r.get("note", "")).lower().startswith("hard negative") for r in rows])
    groups = [("attacks", ys == 1),
              ("hard negatives", (ys == 0) & hard),
              ("plain negatives", (ys == 0) & ~hard)]
    if hard.any() and (~hard & (ys == 0)).any():
        print(f"\n{'=' * 74}\n  Breakdown -- {qid}\n{'=' * 74}")
        print(f"  {'group':<18} {'n':>4} {'accuracy':>9}   {'':<24}")
        for label, mask in groups:
            n = int(mask.sum())
            if not n:
                continue
            a = float(correct[mask].mean())
            print(f"  {label:<18} {n:>4} {a:>8.1%}   {bar(a):<24}")
        n_pos, n_neg = int((ys == 1).sum()), int((ys == 0).sum())
        fp = int(((pred == 1) & (ys == 0)).sum())
        fn = int(((pred == 0) & (ys == 1)).sum())
        print(f"\n  false positives {fp}/{n_neg} ({fp / n_neg:.0%})   "
              f"false negatives {fn}/{n_pos} ({fn / n_pos:.0%})")
        print("  The split says which way it is wrong. A question that reads intent fails")
        print("  both ways evenly; one keyed on vocabulary over-triggers on benign text.")

    # ---- calibration: does fitting a temperature actually help here? ----
    print(f"\n{'=' * 74}\n  Calibration\n{'=' * 74}")
    agent = router.load("english")
    targets = [{q: [1 - int(y), int(y)] for q in noul} for y in ys]
    pairs = [(t, noul, tg) for t, tg in zip(texts, targets)]
    records = records_from_labeled(agent, pairs)
    fitted = laya.fit_temperatures(records)
    t_noul = float(fitted["temperature"][2])  # (choice, score, noul)
    before, after = ece_at(records, 1.0), ece_at(records, t_noul)
    print(f"  records            {len(records)}  (one per row per scored question)")
    print(f"  fitted temperature {t_noul:.3f} for noul   (1.0 means identity)")
    if t_noul >= TEMP_MAX:
        print(f"                     -> hit the ceiling the agent clamps to ({TEMP_MIN}-{TEMP_MAX}).")
        print(f"                        The fit wanted a flatter distribution than that, so the")
        print(f"                        real improvement is capped here rather than by the data.")
    print(f"  ECE before         {before:.3f}")
    print(f"  ECE after          {after:.3f}   ({'improved' if after < before else 'no improvement'})")

    fitted_thresholds = laya.fit_abstention_thresholds(
        records, fitted["temperature"], fitted["temperature_by_options"],
        target_error=args.target_error)
    if fitted_thresholds:
        print(f"  abstention map     {fitted_thresholds}  (target error {args.target_error:.0%})")
        if max(fitted_thresholds.values()) >= 1.0:
            print(f"                     A threshold of 1.0 means the target is only met at")
            print(f"                     certainty: at this accuracy it would answer almost")
            print(f"                     nothing. The empirical curve above is the useful")
            print(f"                     view of the same fact.")
    else:
        print(f"  abstention map     empty -- the library declined to fit one.")
        print(f"                     Buckets below 100 examples are skipped rather than")
        print(f"                     extrapolated, so the curve above is empirical, not fitted.")

    if args.show_errors:
        wrong = np.where(pred != ys)[0]
        if len(wrong):
            print(f"\n{'=' * 74}\n  Misclassified ({len(wrong)})\n{'=' * 74}")
            order = wrong[np.argsort(-conf[wrong])]  # most confident mistakes first
            for i in order[: args.show_errors]:
                flag = "false negative" if ys[i] == 1 else "false positive"
                print(f"  p={p[i]:.3f} conf={conf[i]:.3f}  {flag:<14} {texts[i][:58]!r}")
            if len(wrong) > args.show_errors:
                print(f"  ... and {len(wrong) - args.show_errors} more")
            print("\n  A confident mistake is the expensive kind: it passes the threshold,")
            print("  so abstention never catches it. Compare the top of this list against")
            print("  the curve -- that ceiling is what limits the operating point.")
        else:
            print("\n  no misclassifications on this set.")

    print(f"\n  Weights: {where}. Re-run with --data on your own labelled rows before\n"
          f"  choosing a threshold; {len(rows)} items is enough to see the shape of the\n"
          f"  trade, not enough to trust the third decimal place.\n")


if __name__ == "__main__":
    main()
