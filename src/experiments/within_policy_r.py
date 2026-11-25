"""Within-policy correlation between retained attention mass and replayed NLL.

For each policy (h2o, window, hybrid) and each model, correlate the median
retention over windows with the median NLL increase over windows across the six
budgets (Pearson r over 6 points), with a 95% bootstrap interval that resamples
windows. Retention is the prompt-query mass ("pq") of each policy's true keep set,
as stored by exp_hindsight_retention.py, so hybrid is scored on sink + recent +
top-mass fill rather than on a sink-plus-recent curve.

No model is loaded; this reads committed result files only.

Usage: venv/bin/python -m src.experiments.within_policy_r
"""
from __future__ import annotations

import glob
import json
import random
from statistics import median

import numpy as np

from src.experiments.exp_hindsight_retention import BUDGETS, POLICIES, load_nll

D = "reports/experiments"
MODELS = {
    "Qwen2.5-0.5B": (f"{D}/hindsight_retention_qwen2.5-0.5b_20261030_062616.json",
                     [f"{D}/e2e_kv_logprob_qwen2.5_0.5b_20260916_213111.json",
                      f"{D}/e2e_kv_logprob_qwen2.5_0.5b_long1024_w*.json"]),
    "Qwen2.5-7B": (f"{D}/hindsight_retention_qwen2.5-7b_20261030_064400.json",
                   [f"{D}/e2e_kv_logprob_qwen2.5_7b_long1024_w*.json"]),
    "Mistral-7B-Instruct-v0.2": (f"{D}/hindsight_retention_mistral-7b-instruct-v0.2_20261029_145424.json",
                                 [f"{D}/e2e_kv_logprob_mistral_7b_instruct_v0.2_long1024_w*.json"]),
}


def within_r(ret, nll, ws, policy):
    x = [median(ret[w][f"{policy}|{b}"]["pq"] for w in ws) for b in BUDGETS]
    y = [median(nll[w][(policy, b)] for w in ws) for b in BUDGETS]
    return float(np.corrcoef(x, y)[0, 1])


def main(B: int = 2000, seed: int = 0) -> None:
    rng = random.Random(seed)
    print("Within-policy Pearson r, retention (pq, true keep set) vs NLL increase, over 6 budgets;")
    print("medians over windows; 95% bootstrap CI resamples windows.\n")
    for label, (ret_path, globs) in MODELS.items():
        ret = {int(k): v for k, v in json.load(open(ret_path))["retention"].items()}
        files = [f for g in globs for f in sorted(glob.glob(g))]
        nll, _ = load_nll(files, n_windows=20)
        ws = sorted(w for w in ret if w in nll and all((p, b) in nll[w] for p in POLICIES for b in BUDGETS))
        cells = []
        for p in POLICIES:
            boots = sorted(within_r(ret, nll, [rng.choice(ws) for _ in ws], p) for _ in range(B))
            boots = [b for b in boots if b == b]
            lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
            cells.append(f"{p} {within_r(ret, nll, ws, p):+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"{label:26s} {len(ws)} windows   " + "   ".join(cells))


if __name__ == "__main__":
    main()
