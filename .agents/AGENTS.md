# Agent Instructions

## Git Workflow

- Only stage changes (`git add`). Never commit.
- After staging, generate a commit message for user review. User will commit manually.
- Create git branches using the format `dilberx/<feature>`, for example `dilberx/add-benchmark-config`.

## MR Creation Advice

- Always add the linked issue when creating an MR.
- Always add the default reviewers when creating an MR.

## Content Generation Rules

- No marketing fluff.
- No em dashes.
- No overuse of adverbs or adjectives.
- No rhymes.
- No meaningless filler statements.
- Write plain, direct, human-sounding text.

## Before Resource-Heavy Runs

Run the pre-flight check before anything that loads a model, profiles, or runs
an experiment:

```bash
python scripts/preflight_resources.py
```

It reports GPU and resident memory, attributes each heavy process to a project
by its working directory, and exits non-zero when something outside this repo
is competing. Exit 1 means report it and stop.

Rules:

- **Never stop another project's process.** This machine is shared with
  `llm-finetune-pipeline` and others. Identify what is running, say what it is
  and what it costs, and let the user decide. Killing someone else's job to
  free a GPU is not a call to make unilaterally.
- **Report before proceeding.** Name the process, the project it belongs to,
  and the resource it holds. Then offer the options: wait, proceed with CPU
  offload and accept it being slower, or have the user stop it.
- **Do not trust `pgrep -f <name>` for liveness.** It matches the checking
  shell's own command line, which produced two false "still running" readings
  in one session. Use a PID file with `kill -0`, or exclude self explicitly.
- **`nvidia-smi --query-compute-apps` returns no rows under WSL**, so VRAM
  cannot be attributed to a PID here. Use total free VRAM plus the resident
  memory listing.
- **Decide GPU placement from free VRAM at load time**, never hardcode a
  `device_map`. Free VRAM on this machine has been observed swinging by 5 GiB
  as other jobs move through phases, so a placement that fits when a script is
  written can OOM when it runs. `src/kv_profiler/loading.py` does this, with a
  retry that shifts more to CPU on OOM.
- **Cap CUDA memory before any long-context run.** Under WSL the driver does
  not fail when PyTorch reserves more than the card holds, it spills the overflow
  into system RAM and pages it across PCIe on every step. Measured here: a 0.5B
  model at 32K tokens had a real peak of 3.8 GiB but a reserved pool of 10.6 GiB,
  prefill took 350 s, and decode ran at 146 ms/step. With
  `limit_gpu_memory()` from `src/kv_profiler/loading.py`, prefill took 22 s and
  decode 54 ms/step. Warning signs: sustained PCIe traffic in the GB/s range
  (`nvidia-smi dmon -s t`) from a model that fits in VRAM, or reserved memory
  above total VRAM in `torch.cuda.memory_reserved()`.
- **Print progress inside any step that can run longer than a few minutes.** A
  window that is silent for 20 minutes cannot be told apart from a stuck one,
  and an ETA extrapolated from a shorter length was wrong by more than 2x here.

## Private Notes (do not commit)

Private working notes live outside the repo:

`/home/dilbersha/dev/llm-inference-benchmark-private-notes`

Contents:

- `claims_ledger.md` - every public claim with its evidence file and required caveat. Check a claim here before writing it into README or reports.
- `novelty_acquisition_plan.md` - novelty positioning, six-layer coverage gaps, acquisition readiness plan.
- `content_agent_handoff_20260426.md` - repo state, evidence numbers, content editing rules.
- `novelty_expansion_20260916.md` - literature scan of 2026 work, which of our directions are now prior art, and the open indirect research transfers. Read before making any novelty claim in public text.
- `two_track_plan_20260916.md` - acquisition and publication track plan, asset split, venue choice, 90-day sequence, and the IP gate. Supersedes the 30-day plan in the acquisition plan.
- resume `.tex` files.

Rules:

- Read these for context. Never copy acquisition strategy, novelty positioning, or private claims analysis into the public repo.
- Public text must stay conservative and match the claims ledger, including the caveat column.
- Proxy metrics (attention-mass retention) must be labeled as proxy metrics, not end-to-end quality.
- If you need the notes reachable from inside the repo, symlink the folder as `private-notes`. That name is gitignored.
