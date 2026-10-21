"""Experiment 1C: Are the prompt difficulty labels real?

The routing experiments (`exp_routing.py`) assign each prompt a hand-written
difficulty label of easy, medium or hard. Those labels are the ground truth
that the static-feature routing result was measured against. They were never
validated.

This experiment replaces the hand labels with three measured difficulty
signals and re-tests the static prefill features against them:

1. 1 - similarity      Small model's greedy output against the large model's.
2. nll(large|small)    How surprised the small model is by what the large
                       model actually produced, teacher-forced. Independent
                       of string matching.
3. top1 disagreement   Fraction of the large model's tokens the small model
                       would not have picked.

Signals 2 and 3 need one extra small-model forward pass per prompt and do not
depend on the decoding path, so they act as a second opinion on signal 1.

Requires a completed `prompt_routing` run for the large-model outputs.
"""

from __future__ import annotations

import glob
import json
import math
from statistics import mean

import torch
import torch.nn.functional as F

from src.experiments.exp_routing import ROUTING_PROMPTS
from src.experiments.runner import ExperimentRunner

STATIC_FEATURES = [
    "first_token_entropy", "first_token_top_prob", "first_token_top5_mass",
    "first_token_top10_mass", "avg_attention_entropy", "avg_attention_sparsity",
    "max_sink_mass", "unique_token_ratio", "avg_gen_entropy", "avg_gen_top_prob",
    "min_gen_top_prob", "tokens_above_90pct", "tokens_above_95pct",
    "output_repetition", "routing_score", "input_token_count",
    "prefill_time_ms", "output_len",
]
LABEL_NUM = {"easy": 0, "medium": 1, "hard": 2}


def _pearson(a, b):
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da * db else 0.0


def _spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    return _pearson(rank(a), rank(b))


def _p_value(r, n):
    """Two-sided p for a correlation. Normal approximation, adequate as a screen."""
    if abs(r) >= 1 or n <= 2:
        return 0.0
    t = abs(r) * math.sqrt((n - 2) / (1 - r * r))
    return 2 * (1 - 0.5 * (1 + math.erf(t / math.sqrt(2))))


@torch.no_grad()
def _score_large_output_under_small(trials):
    """NLL and top-1 agreement of the large model's output under the small model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager",
    ).eval()

    out = {}
    for t in trials:
        pname = t["config"]["prompt"]
        large_text = t.get("metadata", {}).get("large_text", "")
        if not large_text.strip():
            out[pname] = None  # large model produced nothing to score
            continue
        p_ids = tok(ROUTING_PROMPTS[pname], return_tensors="pt").input_ids.to("cuda")
        c_ids = tok(large_text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        ids = torch.cat([p_ids, c_ids], dim=1)
        logits = model(input_ids=ids).logits.float()
        cut = p_ids.shape[1]
        tgt = ids[0, cut:]
        step_logits = logits[0, cut - 1:-1]
        lp = F.log_softmax(step_logits, dim=-1)
        out[pname] = {
            "nll": -lp.gather(1, tgt.unsqueeze(1)).squeeze(1).mean().item(),
            "top1_agreement": (step_logits.argmax(-1) == tgt).float().mean().item(),
            "n_scored": int(tgt.numel()),
        }
        print(f"  {pname:<18} nll={out[pname]['nll']:.3f}  top1={out[pname]['top1_agreement']:.3f}")

    del model
    torch.cuda.empty_cache()
    return out


def run_difficulty_validation():
    static_files = sorted(glob.glob("reports/experiments/prompt_routing_2026042*.json"))
    routing_files = sorted(glob.glob("reports/experiments/prompt_routing_2026091*.json"))
    if not static_files or not routing_files:
        raise FileNotFoundError(
            "Needs the April static-feature run and a corrected routing run. "
            "Run `python -m src.experiments.exp_routing` first."
        )
    static = {t["config"]["prompt"]: {**t["metrics"], "label": t["config"]["expected_difficulty"]}
              for t in json.load(open(static_files[-1]))["trials"]}
    routing_trials = json.load(open(routing_files[-1]))["trials"]
    routing = {t["config"]["prompt"]: t["metrics"] for t in routing_trials}

    print("Scoring large-model outputs under the small model...")
    scored = _score_large_output_under_small(routing_trials)

    prompts = [p for p in static if scored.get(p) and p in routing]
    dropped = sorted(set(static) - set(prompts))
    n = len(prompts)
    print(f"\n{n} prompts usable, dropped {dropped} (no large-model output to score)")

    measures = {
        "hand_label": [LABEL_NUM[static[p]["label"]] for p in prompts],
        "inv_similarity": [-routing[p]["small_quality"] for p in prompts],
        "nll_large_under_small": [scored[p]["nll"] for p in prompts],
        "top1_disagreement": [1 - scored[p]["top1_agreement"] for p in prompts],
    }

    runner = ExperimentRunner(
        name="difficulty_validation",
        description=(
            "Validates the hand-assigned prompt difficulty labels against three measured "
            "difficulty signals, then re-tests the April static prefill features and the "
            "trajectory features against measured difficulty instead of the labels."
        ),
    )

    # Agreement between the candidate difficulty definitions.
    names = list(measures)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = _spearman(measures[a], measures[b])
            with runner.trial({"comparison": "measure_agreement", "a": a, "b": b}) as trial:
                trial.record("spearman", rho)
                trial.record("p_value", _p_value(rho, n))
                trial.record("n", float(n))

    # Static and trajectory features against every difficulty definition.
    bonferroni = 0.05 / len(STATIC_FEATURES)
    for feat in STATIC_FEATURES:
        v = [static[p][feat] for p in prompts]
        for target in ["hand_label", "nll_large_under_small", "inv_similarity"]:
            rho = _spearman(v, measures[target])
            pv = _p_value(rho, n)
            with runner.trial({"comparison": "static_feature", "feature": feat, "target": target}) as trial:
                trial.record("spearman", rho)
                trial.record("p_value", pv)
                trial.record("survives_bonferroni", float(pv < bonferroni))
                trial.record("n", float(n))

    for feat in ["avg_k_prob", "avg_k_entropy"]:
        v = [routing[p][feat] for p in prompts]
        for target in ["hand_label", "nll_large_under_small", "inv_similarity"]:
            rho = _spearman(v, measures[target])
            pv = _p_value(rho, n)
            with runner.trial({"comparison": "trajectory_feature", "feature": feat, "target": target}) as trial:
                trial.record("spearman", rho)
                trial.record("p_value", pv)
                trial.record("survives_bonferroni", float(pv < bonferroni))
                trial.record("n", float(n))

    runner.save()
    runner.to_csv()

    # Readable summary
    print("\n" + "=" * 70)
    print("DIFFICULTY LABEL VALIDATION")
    print("=" * 70)
    print("\nAgreement between difficulty definitions (Spearman):")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"  {a:<24} vs {b:<24} {_spearman(measures[a], measures[b]):+.3f}")

    print(f"\nStatic prefill features, hits at Bonferroni p<{bonferroni:.4f}:")
    for target in ["hand_label", "nll_large_under_small", "inv_similarity"]:
        hits = []
        for feat in STATIC_FEATURES:
            rho = _spearman([static[p][feat] for p in prompts], measures[target])
            if _p_value(rho, n) < bonferroni:
                hits.append((feat, rho))
        print(f"  vs {target:<24} {len(hits)} hit(s): " +
              (", ".join(f"{f} ({r:+.2f})" for f, r in hits) if hits else "none"))

    print("\nTrajectory features (K=3):")
    for feat in ["avg_k_prob", "avg_k_entropy"]:
        v = [routing[p][feat] for p in prompts]
        parts = []
        for target in ["nll_large_under_small", "inv_similarity"]:
            rho = _spearman(v, measures[target])
            parts.append(f"{target}: rho={rho:+.3f} p={_p_value(rho, n):.4f}")
        print(f"  {feat:<16} " + " | ".join(parts))

    print("\nMean small-model similarity by difficulty definition:")
    q = {p: routing[p]["small_quality"] for p in prompts}
    for lab in ["easy", "medium", "hard"]:
        v = [q[p] for p in prompts if static[p]["label"] == lab]
        if v:
            print(f"  hand label {lab:<8} n={len(v)}  {mean(v):.3f}")
    srt = sorted(prompts, key=lambda p: scored[p]["nll"])
    third = max(1, len(srt) // 3)
    for label, grp in [("low nll", srt[:third]), ("mid", srt[third:2 * third]), ("high nll", srt[2 * third:])]:
        print(f"  measured {label:<9} n={len(grp)}  {mean(q[p] for p in grp):.3f}")

    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_difficulty_validation()
