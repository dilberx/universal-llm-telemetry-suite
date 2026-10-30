"""Does hindsight attention mass predict replay quality when prompt-query mass does not?

The proxy audit scores a keep set by the attention mass it retains, where mass is
summed over the prompt's own queries. H2O keeps the keys with the most of that
mass, so it tops the proxy by construction. The replay then asks something else:
how well do the next tokens decode when the cache holds only that set. Those
tokens' queries are not in the prompt-query mass.

This script measures the retention of each keep set under four masses, all taken
from one full-cache teacher-forced pass over the prompt plus the logged target
tokens:

  pq     the audit's proxy: layer, head and prompt-query summed mass
  norm   the same mass divided by the number of queries that can see each key
         (control: removes the causal-visibility advantage of early keys)
  hs     hindsight: mass the continuation queries (the decode steps) put on each
         prompt key, summed over layers and heads
  hsA/B  hindsight from the first and second half of the decode steps
         (split-half noise floor: how stable the hindsight signal is)

Keep sets are those the replay builds at its first prune (budget = int(n * frac),
four sink tokens), by calling the replay's own `_keep_indices`. Later prunes add
the new tokens and shift the budget by at most a few slots; this is not modelled.
Retention for each measure is correlated with the replay's measured NLL increase
(Kendall tau-b) over the 18 policy-budget cells (h2o, window, hybrid; six budgets),
as the median over windows, with a bootstrap over windows.

Run: python -m src.experiments.exp_hindsight_retention
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import time
from statistics import median, mean

import torch
import torch.nn.functional as F

from src.experiments.exp_e2e_kv_cache import _keep_indices, _select_prompts

BUDGETS = (0.1, 0.2, 0.3, 0.5, 0.75, 0.9)
POLICIES = ("h2o", "window", "hybrid")
MEASURES = ("pq", "norm", "hs", "hsA", "hsB")


def kendall_tau_b(x: list[float], y: list[float]) -> float:
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    denom = ((conc + disc + tx) * (conc + disc + ty)) ** 0.5
    return (conc - disc) / denom if denom else float("nan")


def load_nll(files: list[str], n_windows: int) -> dict[int, dict[tuple[str, float], float]]:
    """window -> (policy, budget) -> nll_delta_vs_full, plus window -> target tokens."""
    nll: dict[int, dict] = {}
    targets: dict[int, list[int]] = {}
    for path in files:
        for t in json.load(open(path))["trials"]:
            c = t["config"]
            name = c.get("prompt", "")
            if "_w" not in name:
                continue
            w = int(name.rsplit("_w", 1)[1])
            if w >= n_windows:
                continue
            if c.get("policy") == "full":
                targets[w] = t["metadata"]["target_tokens"]
            elif c.get("policy") in POLICIES:
                nll.setdefault(w, {})[(c["policy"], c["budget_frac"])] = t["metrics"]["nll_delta_vs_full"]
    return nll, targets


@torch.no_grad()
def window_masses(model, tokenizer, prompt: str, targets: list[int]) -> dict[str, torch.Tensor]:
    """pq, norm, hs, hsA, hsB mass vectors over the prompt keys of one window."""
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    n = int(inputs["input_ids"].shape[1])
    out = model(**inputs, use_cache=True, output_attentions=True)
    pq = sum(a.detach().float().sum(dim=(0, 1, 2)) for a in out.attentions)
    cache = out.past_key_values
    del out

    rows = []
    for i, tok in enumerate(targets[:-1]):                 # the replay feeds all but the last target
        pos = n + i
        o = model(input_ids=torch.tensor([[tok]], device=device), past_key_values=cache,
                  use_cache=True, output_attentions=True,
                  position_ids=torch.tensor([[pos]], device=device),
                  cache_position=torch.tensor([pos], device=device))
        cache = o.past_key_values
        rows.append(sum(a.detach().float()[0, :, 0, :n].sum(dim=0) for a in o.attentions))
        del o
    rows = torch.stack(rows)                                # [steps, n]
    half = rows.shape[0] // 2
    visible = (n - torch.arange(n, device=pq.device)).float()   # queries that can see key j
    return {"pq": pq, "norm": pq / visible, "hs": rows.sum(0),
            "hsA": rows[:half].sum(0), "hsB": rows[half:].sum(0), "n": n}


def retention_table(masses: dict, device) -> dict[tuple[str, float], dict[str, float]]:
    n = masses["n"]
    positions = torch.arange(n, device=device)
    table = {}
    for pol in POLICIES:
        for frac in BUDGETS:
            budget = min(max(4, int(n * frac)), n)
            keep = _keep_indices(pol, masses["pq"], budget, positions, sink_tokens=4)
            table[(pol, frac)] = {m: float(masses[m][keep].sum() / masses[m].sum()) for m in MEASURES}
    # control: H2O on the causal-normalised mass, read out under hindsight
    for frac in BUDGETS:
        budget = min(max(4, int(n * frac)), n)
        keep = torch.topk(masses["norm"], k=budget).indices
        table[("h2o_norm", frac)] = {m: float(masses[m][keep].sum() / masses[m].sum()) for m in MEASURES}
    return table


def analyse(ret: dict[int, dict], nll: dict[int, dict], windows: list[int], B: int = 2000, seed: int = 0):
    rng = random.Random(seed)
    cells = [(p, b) for p in POLICIES for b in BUDGETS]

    def cell_median(ws, getter, cell):
        return median(getter(w, cell) for w in ws)

    def tau_for(ws, m):
        xs = [cell_median(ws, lambda w, c: ret[w][c][m], c) for c in cells]
        ys = [cell_median(ws, lambda w, c: nll[w][c], c) for c in cells]
        return kendall_tau_b(xs, ys)

    def agree(ws, m):
        hit = 0
        for b in BUDGETS:
            best_nll = min(POLICIES, key=lambda p: cell_median(ws, lambda w, c: nll[w][c], (p, b)))
            best_ret = max(POLICIES, key=lambda p: cell_median(ws, lambda w, c: ret[w][c][m], (p, b)))
            hit += best_nll == best_ret
        return hit

    res = {}
    for m in MEASURES:
        boots = sorted(tau_for([rng.choice(windows) for _ in windows], m) for _ in range(B))
        boots = [x for x in boots if x == x]
        res[m] = {"tau": tau_for(windows, m), "ci": (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]),
                  "per_window_tau": [tau_for([w], m) for w in windows], "best_agree": agree(windows, m)}
    # reliability of the hindsight signal: A against B retention over the cells
    xa = [cell_median(windows, lambda w, c: ret[w][c]["hsA"], c) for c in cells]
    xb = [cell_median(windows, lambda w, c: ret[w][c]["hsB"], c) for c in cells]
    res["split_half_tau"] = kendall_tau_b(xa, xb)
    return res


def run(model_name="Qwen/Qwen2.5-0.5B", logprob_glob=None, n_windows=5, offload=False, out_dir="reports/experiments"):
    logprob_glob = logprob_glob or "reports/experiments/e2e_kv_logprob_qwen2.5_0.5b_2026091*.json"
    files = []
    for pattern in logprob_glob.split(","):                 # several globs may be given, comma-separated
        found = []
        for f in sorted(glob.glob(pattern)):
            d = json.load(open(f))
            pt = [t["metrics"].get("prompt_tokens", 0) for t in d["trials"]]
            if pt and median(pt) >= 1000:
                found.append(f)
        files += found[-1:] if "0.5b_2026091" in pattern else found
    nll, targets = load_nll(files, n_windows)
    windows = sorted(set(nll) & set(targets))
    print(f"replay results: {len(files)} file(s), windows {windows}", flush=True)

    if offload:
        from transformers import AutoTokenizer
        from src.kv_profiler.loading import limit_gpu_memory, load_for_profiling
        limit_gpu_memory()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = load_for_profiling(model_name, dtype=torch.bfloat16, headroom_gib=3.5)
    else:
        from src.experiments.exp_real_model import load_model
        model, tokenizer = load_model(model_name)
    prompts = _select_prompts("long", tokenizer, (1024,), long_windows=n_windows)

    ret, t0 = {}, time.time()
    for w in windows:
        masses = window_masses(model, tokenizer, prompts[f"wikitext_1024tok_w{w}"], targets[w])
        ret[w] = retention_table(masses, model.device)
        print(f"  window {w}: {masses['n']} tokens, {len(targets[w]) - 1} decode steps, {time.time() - t0:.0f}s", flush=True)

    res = analyse(ret, nll, windows)
    print(f"\nKendall tau-b between retention and NLL increase over 18 policy-budget cells "
          f"({len(windows)} windows; negative = retention predicts quality):")
    print(f"{'measure':<7}{'tau':>8}   95% CI over windows        agree(best policy, of 6)   per-window tau")
    for m in MEASURES:
        r = res[m]
        print(f"{m:<7}{r['tau']:>+8.3f}   [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
              f"{r['best_agree']:>20}/6      {[round(x, 2) for x in r['per_window_tau']]}")
    print(f"split-half tau between hsA and hsB retention (noise floor of the hindsight signal): {res['split_half_tau']:+.3f}")

    print("\nMedian retention by policy under pq and hs (rows: policy, columns: budget):")
    for m in ("pq", "hs"):
        print(f"  measure {m}")
        for pol in POLICIES + ("h2o_norm",):
            print(f"    {pol:<9}" + "".join(f"{median(ret[w][(pol, b)][m] for w in windows):>8.3f}" for b in BUDGETS))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"{out_dir}/hindsight_retention_{model_name.rstrip('/').split('/')[-1].lower()}_{stamp}.json"
    json.dump({"model": model_name, "windows": windows, "result": res,
               "retention": {str(w): {f"{p}|{b}": v for (p, b), v in ret[w].items()} for w in windows}},
              open(path, "w"), indent=1, default=lambda o: list(o))
    print(f"\nsaved {path}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--logprob-glob", default=None)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--offload", action="store_true")
    a = ap.parse_args()
    run(a.model, a.logprob_glob, a.windows, a.offload)
