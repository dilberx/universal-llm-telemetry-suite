#!/usr/bin/env python3
"""Pre-flight resource check for resource-heavy runs.

Reports what is currently consuming GPU and RAM, attributes each consumer to a
project by its working directory, and exits non-zero when something outside
this repository is holding significant resources.

It never stops anything. Deciding whether another project's job should be
interrupted is the user's call, so this prints what is running, who it belongs
to, and what it costs, and leaves the decision alone.

Usage:
    python scripts/preflight_resources.py            # check and report
    python scripts/preflight_resources.py --quiet    # only report problems

Exit codes: 0 clear, 1 foreign consumers found, 2 could not read GPU state.

Notes on this environment, learned the hard way:
  * `nvidia-smi --query-compute-apps` returns no rows under WSL, so VRAM
    cannot be attributed to a PID. Total free VRAM plus a list of candidate
    processes is the best available substitute.
  * `pgrep -f <name>` matches the checking process's own command line. Every
    scan here excludes self, the parent shell and shell wrappers, because a
    false "still running" reading is worse than no reading.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# A process is worth mentioning above this resident size.
RSS_GB_THRESHOLD = 0.5
# Free VRAM below this makes a 7B-class run unlikely to fit on the GPU.
LOW_VRAM_GIB = 4.0

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gpu_state() -> tuple[float, float] | None:
    """(used_gib, free_gib), or None if the GPU cannot be queried."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        if out.returncode != 0:
            return None
        used, free = (float(x.strip()) / 1024 for x in out.stdout.split(",")[:2])
        return used, free
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _cmdline(pid: str) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\x00", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def _rss_gb(pid: str) -> float:
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 ** 2
    except (OSError, ValueError):
        pass
    return 0.0


def heavy_processes() -> list[dict]:
    """Python-ish processes above the RSS threshold, attributed to a project.

    Excludes this process, its parents and shell wrappers, so the check cannot
    report itself as a resource hog.
    """
    mine = {str(os.getpid()), str(os.getppid())}
    found = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or pid in mine:
            continue
        cmd = _cmdline(pid)
        if not cmd or cmd.startswith(("/bin/bash", "/bin/sh", "bash ", "sh ")):
            continue
        if "preflight_resources" in cmd:
            continue
        if not any(k in cmd for k in ("python", "torch", "vllm", "accelerate")):
            continue
        rss = _rss_gb(pid)
        if rss < RSS_GB_THRESHOLD:
            continue
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            cwd = ""
        found.append({
            "pid": pid, "rss_gb": rss, "cmd": cmd, "cwd": cwd,
            "project": os.path.basename(cwd) if cwd else "unknown",
            "is_ours": bool(cwd) and os.path.abspath(cwd).startswith(REPO),
        })
    return sorted(found, key=lambda p: -p["rss_gb"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true", help="print only if something is competing")
    args = ap.parse_args()

    gpu = gpu_state()
    procs = heavy_processes()
    foreign = [p for p in procs if not p["is_ours"]]
    low_vram = gpu is not None and gpu[1] < LOW_VRAM_GIB

    if args.quiet and not foreign and not low_vram:
        return 0

    print("=" * 68)
    print("PRE-FLIGHT RESOURCE CHECK")
    print("=" * 68)
    if gpu is None:
        print("GPU: could not query nvidia-smi")
    else:
        used, free = gpu
        flag = "  <-- tight for a 7B-class run" if low_vram else ""
        print(f"GPU: {used:.1f} GiB used, {free:.1f} GiB free{flag}")
        print("     (WSL does not attribute VRAM to PIDs; the list below is by "
              "resident memory)")

    if not procs:
        print("\nNo heavy processes found.")
    else:
        print(f"\nProcesses above {RSS_GB_THRESHOLD} GiB resident:")
        for p in procs:
            tag = "this repo" if p["is_ours"] else f"OTHER: {p['project']}"
            print(f"  [{tag}] pid {p['pid']}  {p['rss_gb']:.1f} GiB")
            print(f"      {p['cmd'][:96]}")
            if p["cwd"]:
                print(f"      cwd: {p['cwd']}")

    if foreign or low_vram:
        print("\n" + "-" * 68)
        print("DECISION NEEDED")
        print("-" * 68)
        if foreign:
            print("These belong to other work and are competing for this machine:")
            for p in foreign:
                print(f"  pid {p['pid']}  {p['rss_gb']:.1f} GiB  {p['project']}")
            print("\nNothing has been stopped. Options:")
            print("  * wait for it to finish")
            print("  * let it run and accept slower CPU-offloaded execution")
            print(f"  * stop it yourself:  kill {' '.join(p['pid'] for p in foreign)}")
        if low_vram and not foreign:
            print(f"Free VRAM is {gpu[1]:.1f} GiB with no competing process "
                  "identified. Something outside this shell may hold it.")
        return 1

    print("\nClear to run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
