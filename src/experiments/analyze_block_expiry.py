"""Aggregate the block-expiry runs into the tables the write-up needs.

Reads the newest `block_expiry_<tag>_n<N>_*.json` per context length, averages
over windows, and prints:

  1. concentration: how much of the context holds the mass,
  2. dead blocks: pooled across layers versus judged per layer,
  3. the RFC's literal threshold under each reading of its units,
  4. the cost, in KL and NLL, of freeing a given share of blocks, for the mass
     signal against two rules that ignore attention, interpolated at the 30 and
     60 percent footprint reductions the RFC projects,
  5. the dormant-then-needed test: what happens to a passage that goes quiet and
     is then copied back.

Run: python -m src.experiments.analyze_block_expiry [--tag q05b]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from statistics import mean

TARGETS = (0.30, 0.60)
ARMS = ("mass", "random_nosink", "oldest")


def interp_at(points: list[tuple[float, float]], target: float) -> float | None:
    """Linear interpolation of y at x = target over (x, y) points.

    Returns None when the points do not bracket the target, because
    extrapolating a cost curve would invent a number.
    """
    pts = sorted(p for p in points if p[0] == p[0])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= target <= x1 and x1 > x0:
            return y0 + (y1 - y0) * (target - x0) / (x1 - x0)
    for x, y in pts:
        if abs(x - target) < 1e-12:
            return y
    return None


def load_runs(tag: str, pattern_dir: str = "reports/experiments") -> dict[int, dict]:
    """Every window of every length, merged per length.

    Each window is its own result file, `block_expiry_<tag>_n<N>_w<W>_<stamp>.json`,
    so that a run interrupted part-way keeps what it finished. When a window was
    run more than once the newest file wins.
    """
    newest: dict[tuple[int, int], str] = {}
    for path in sorted(glob.glob(f"{pattern_dir}/block_expiry_{tag}_n*_w*_*.json")):
        m = re.search(r"_n(\d+)_w(\d+)_\d{8}_\d{6}\.json$", path)
        if m:
            newest[(int(m.group(1)), int(m.group(2)))] = path
    merged: dict[int, dict] = {}
    for (n, w), path in sorted(newest.items()):
        d = json.load(open(path))
        if not d["trials"]:
            continue
        if n not in merged:
            merged[n] = {"trials": [], "files": []}
        merged[n]["trials"].extend(d["trials"])
        merged[n]["files"].append(path)
    return dict(sorted(merged.items()))


def _part(run: dict, which: str) -> list[dict]:
    return [t for t in run["trials"] if t["config"]["part"] == which]


def summarise(runs: dict[int, dict]) -> dict:
    out = {"concentration": {}, "dead": {}, "cost": {}, "interp": {}, "dormant": {}}
    for n, run in runs.items():
        a = _part(run, "A")
        out["concentration"][n] = {
            "windows": len(a),
            "blocks": mean(t["metrics"]["n_blocks_end"] for t in a),
            "b90": mean(t["metrics"]["blocks_for_90pct"] for t in a),
            "b99": mean(t["metrics"]["blocks_for_99pct"] for t in a),
        }
        dead = defaultdict(lambda: defaultdict(list))
        for t in a:
            for name, v in t["metadata"]["dead_by_theta"].items():
                for key in ("pooled", "all_layers", "layer_median", "layer_max", "theta_rel_uniform"):
                    dead[name][key].append(v[key])
        out["dead"][n] = {nm: {k: mean(v) for k, v in kv.items()} for nm, kv in dead.items()}

        cost = defaultdict(lambda: defaultdict(list))
        for t in _part(run, "B"):
            c, m = t["config"], t["metrics"]
            if c["theta_name"] == "none":
                continue
            for key in ("expired_frac_end", "kl_mean", "nll_delta", "top1_agree", "acc_delta"):
                cost[(c["theta_name"], c["arm"])][key].append(m[key])
        out["cost"][n] = {k: {kk: mean(vv) for kk, vv in kv.items()} | {"n": len(kv["kl_mean"])}
                          for k, kv in cost.items()}

        for arm in ARMS:
            keyed = [(nm, v) for (nm, a_), v in out["cost"][n].items()
                     if a_ == arm and not nm.startswith("literal")]
            out["interp"][(n, arm)] = {
                tgt: (interp_at([(v["expired_frac_end"], v["kl_mean"]) for _, v in keyed], tgt),
                      interp_at([(v["expired_frac_end"], v["nll_delta"]) for _, v in keyed], tgt))
                for tgt in TARGETS}

        dorm = defaultdict(lambda: defaultdict(list))
        full_acc, full_nll = [], []
        for t in _part(run, "C"):
            c, m = t["config"], t["metrics"]
            for key in ("passage_blocks_expired", "copy_acc", "copy_acc_delta", "copy_nll_delta",
                        "all_blocks_expired_at_copy_start"):
                dorm[(c["theta_name"], c["arm"])][key].append(m[key])
            full_acc.append(m["copy_acc_full"])
            full_nll.append(m["copy_nll_full"])
        out["dormant"][n] = {
            "rows": {k: {kk: mean(vv) for kk, vv in kv.items()} | {"n": len(kv["copy_acc"])}
                     for k, kv in dorm.items()},
            "full_acc": mean(full_acc) if full_acc else None,
            "full_nll": mean(full_nll) if full_nll else None,
        }
    return out


def print_report(s: dict) -> None:
    ns = list(s["concentration"])
    print("\n1. CONCENTRATION (pooled over layers, last read)")
    print(f"{'N':>7}{'windows':>9}{'blocks':>8}{'90% of mass in':>16}{'99% of mass in':>16}")
    for n in ns:
        c = s["concentration"][n]
        print(f"{n:>7}{c['windows']:>9}{c['blocks']:>8.0f}{c['b90']:>15.1%}{c['b99']:>16.1%}")

    print("\n2. DEAD BLOCKS by threshold relative to the uniform share (1/blocks)")
    print("   pooled = decided on layer-mean mass; all = dead in every layer; "
          "layer med/max = judged per layer")
    names = [k for k in next(iter(s["dead"].values())) if k.startswith("rel_")]
    for nm in names:
        print(f"\n   theta = {nm.replace('rel_', '')} x uniform")
        print(f"   {'N':>7}{'pooled':>9}{'all':>8}{'layer med':>11}{'layer max':>11}")
        for n in ns:
            v = s["dead"][n][nm]
            print(f"   {n:>7}{v['pooled']:>9.1%}{v['all_layers']:>8.1%}"
                  f"{v['layer_median']:>11.1%}{v['layer_max']:>11.1%}")

    print("\n3. THE RFC'S THRESHOLD, theta = 1e-3, under each reading of its units")
    print(f"   {'N':>7}  {'reading':<28}{'x uniform':>11}{'dead (pooled)':>15}{'expired':>9}{'KL':>9}{'dNLL':>8}")
    for n in ns:
        for nm in ("literal_sum_heads_layers", "literal_sum_heads", "literal_as_normalised"):
            d = s["dead"][n][nm]
            c = s["cost"][n].get((nm, "mass"))
            tail = (f"{c['expired_frac_end']:>9.1%}{c['kl_mean']:>9.4f}{c['nll_delta']:>8.3f}"
                    if c else f"{'0.0%':>9}{'-':>9}{'-':>8}")
            print(f"   {n:>7}  {nm:<28}{d['theta_rel_uniform']:>11.3g}{d['pooled']:>15.1%}{tail}")

    print("\n4. COST OF FREEING A SHARE OF BLOCKS on ordinary text (interpolated, KL)")
    print("   mass = the RFC's signal; rand = random, sink protected; oldest = sink + recent window")
    print(f"   {'N':>7}{'freed':>7} | {'mass':>9}{'rand':>9}{'oldest':>9} | {'dNLL mass':>10}{'rand':>8}{'oldest':>8}")
    f = lambda v, w=9, p=4: f"{v:>{w}.{p}f}" if v is not None else f"{'n/a':>{w}}"
    for n in ns:
        for tgt in TARGETS:
            kl = [s["interp"][(n, a)][tgt][0] for a in ARMS]
            nl = [s["interp"][(n, a)][tgt][1] for a in ARMS]
            print(f"   {n:>7}{tgt:>7.0%} | {f(kl[0])}{f(kl[1])}{f(kl[2])} | {f(nl[0], 10, 3)}{f(nl[1], 8, 3)}{f(nl[2], 8, 3)}")

    print("\n5. DORMANT THEN NEEDED: a passage goes quiet for 160 tokens, then is copied back")
    for n in ns:
        d = s["dormant"][n]
        if not d["rows"]:
            continue
        print(f"\n   N={n}   full cache: copy accuracy {d['full_acc']:.2f}, copy NLL {d['full_nll']:.3f}")
        print(f"   {'theta':<24}{'arm':<9}{'all freed':>10}{'passage freed':>15}{'copy acc':>10}{'dNLL':>8}{'n':>4}")
        for (nm, arm), v in d["rows"].items():
            print(f"   {nm:<24}{arm:<9}{v['all_blocks_expired_at_copy_start']:>10.0%}"
                  f"{v['passage_blocks_expired']:>15.0%}{v['copy_acc']:>10.2f}{v['copy_nll_delta']:>8.2f}{v['n']:>4}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="q05b")
    args = ap.parse_args()
    runs = load_runs(args.tag)
    if not runs:
        raise SystemExit(f"no block_expiry_{args.tag}_n*_*.json results with trials found")
    print_report(summarise(runs))
