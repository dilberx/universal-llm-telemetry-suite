"""
Unified Experiment Runner & Data Collector.

Every experiment in the inference optimization suite runs through this
framework. It captures ALL metrics, hardware context, and raw outputs
in a structured JSON format for reproducibility and cross-experiment analysis.

Usage:
    runner = ExperimentRunner("token_confidence_threshold")
    with runner.trial(config={"threshold": 0.95, "model": "Qwen-0.5B"}) as trial:
        # ... run experiment ...
        trial.record("throughput_tok_s", 142.5)
        trial.record("quality_score", 0.98)
        trial.record("memory_mb", 3200)
    runner.save()  # writes to reports/experiments/
"""

from __future__ import annotations

import json
import os
import platform
import time
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from contextlib import contextmanager

import torch


@dataclass
class HardwareSnapshot:
    """Captures hardware context for reproducibility."""
    gpu_name: str = "N/A"
    gpu_count: int = 0
    gpu_vram_mb: float = 0
    gpu_driver: str = "N/A"
    cuda_version: str = "N/A"
    cpu_name: str = ""
    cpu_cores: int = 0
    ram_gb: float = 0
    python_version: str = ""
    torch_version: str = ""
    os_info: str = ""

    @classmethod
    def capture(cls) -> "HardwareSnapshot":
        snap = cls()
        snap.python_version = platform.python_version()
        snap.torch_version = torch.__version__
        snap.os_info = f"{platform.system()} {platform.release()}"
        snap.cpu_name = platform.processor() or "unknown"
        snap.cpu_cores = os.cpu_count() or 0

        try:
            import psutil
            snap.ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        except ImportError:
            snap.ram_gb = 0

        if torch.cuda.is_available():
            snap.gpu_count = torch.cuda.device_count()
            snap.gpu_name = torch.cuda.get_device_name(0)
            snap.gpu_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            snap.cuda_version = torch.version.cuda or "N/A"
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                snap.gpu_driver = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(snap.gpu_driver, bytes):
                    snap.gpu_driver = snap.gpu_driver.decode()
            except Exception:
                snap.gpu_driver = "unknown"

        return snap


@dataclass
class TrialResult:
    """Single trial (one configuration) within an experiment."""
    trial_id: str
    config: dict[str, Any]
    config_hash: str
    metrics: dict[str, float]
    metadata: dict[str, Any]
    start_time: str
    end_time: str
    duration_seconds: float
    status: str = "completed"  # completed, failed, skipped
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentReport:
    """Complete experiment with all trials and context."""
    experiment_name: str
    experiment_description: str
    hardware: HardwareSnapshot
    trials: list[TrialResult]
    timestamp: str
    total_duration_seconds: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment_name,
            "description": self.experiment_description,
            "hardware": asdict(self.hardware),
            "timestamp": self.timestamp,
            "total_duration_seconds": self.total_duration_seconds,
            "seed": self.seed,
            "num_trials": len(self.trials),
            "trials": [t.to_dict() for t in self.trials],
        }

    def summary(self) -> str:
        """Quick summary of results."""
        lines = [
            f"\n{'='*70}",
            f"EXPERIMENT: {self.experiment_name}",
            f"{'='*70}",
            f"Hardware: {self.hardware.gpu_name or 'CPU'} | "
            f"Trials: {len(self.trials)} | "
            f"Duration: {self.total_duration_seconds:.1f}s",
            "",
        ]

        if self.trials:
            # Collect all metric names
            all_metrics = set()
            for t in self.trials:
                all_metrics.update(t.metrics.keys())

            # Show best/worst for each metric
            for metric in sorted(all_metrics):
                values = [t.metrics[metric] for t in self.trials if metric in t.metrics]
                if values:
                    lines.append(
                        f"  {metric}: min={min(values):.4f}  max={max(values):.4f}  "
                        f"mean={sum(values)/len(values):.4f}  ({len(values)} trials)"
                    )

        return "\n".join(lines)


class TrialContext:
    """Context manager for a single trial — collects metrics."""

    def __init__(self, trial_id: str, config: dict[str, Any]):
        self.trial_id = trial_id
        self.config = config
        self.config_hash = hashlib.md5(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        self.metrics: dict[str, float] = {}
        self.metadata: dict[str, Any] = {}
        self.start_time = ""
        self.end_time = ""
        self.duration = 0.0
        self.status = "running"
        self.error = None
        self._start_perf = 0.0

    def record(self, name: str, value: float) -> None:
        """Record a metric value."""
        self.metrics[name] = value

    def record_meta(self, name: str, value: Any) -> None:
        """Record metadata (non-numeric context)."""
        self.metadata[name] = value

    def to_result(self) -> TrialResult:
        return TrialResult(
            trial_id=self.trial_id,
            config=self.config,
            config_hash=self.config_hash,
            metrics=self.metrics,
            metadata=self.metadata,
            start_time=self.start_time,
            end_time=self.end_time,
            duration_seconds=self.duration,
            status=self.status,
            error=self.error,
        )


class ExperimentRunner:
    """
    Runs experiments and collects all data points.

    Usage:
        runner = ExperimentRunner(
            name="token_confidence",
            description="Measuring when argmax is safe to use"
        )
        for threshold in [0.9, 0.95, 0.99]:
            with runner.trial({"threshold": threshold}) as t:
                result = run_experiment(threshold)
                t.record("skip_rate", result.skip_rate)
                t.record("quality_delta", result.quality_delta)

        runner.save()
        print(runner.report.summary())
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        output_dir: str = "./reports/experiments",
        seed: int = 42,
    ):
        self.name = name
        self.description = description
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.trials: list[TrialResult] = []
        self.hardware = HardwareSnapshot.capture()
        self._start_time = time.time()
        self._trial_counter = 0

        # Set global seed
        torch.manual_seed(seed)

    @contextmanager
    def trial(self, config: dict[str, Any]):
        """Context manager for a single trial."""
        self._trial_counter += 1
        trial_id = f"{self.name}_t{self._trial_counter:04d}"

        ctx = TrialContext(trial_id, config)
        ctx.start_time = datetime.now().isoformat()
        ctx._start_perf = time.perf_counter()

        try:
            yield ctx
            ctx.status = "completed"
        except Exception as e:
            ctx.status = "failed"
            ctx.error = str(e)
            print(f"  ⚠️ Trial {trial_id} failed: {e}")
        finally:
            ctx.end_time = datetime.now().isoformat()
            ctx.duration = time.perf_counter() - ctx._start_perf
            self.trials.append(ctx.to_result())

            # Print progress
            status_icon = "✓" if ctx.status == "completed" else "✗"
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in ctx.metrics.items())
            config_str = ", ".join(f"{k}={v}" for k, v in config.items())
            print(f"  [{status_icon}] {trial_id} | {config_str} | {metrics_str}")

    @property
    def report(self) -> ExperimentReport:
        """Build the experiment report."""
        return ExperimentReport(
            experiment_name=self.name,
            experiment_description=self.description,
            hardware=self.hardware,
            trials=self.trials,
            timestamp=datetime.now().isoformat(),
            total_duration_seconds=time.time() - self._start_time,
            seed=self.seed,
        )

    def save(self, filename: str | None = None) -> Path:
        """Save experiment results to JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{ts}.json"
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(self.report.to_dict(), f, indent=2, default=str)
        print(f"\n📊 Results saved to {path}")
        return path

    def to_csv(self, filename: str | None = None) -> Path:
        """Export trials as CSV for easy analysis."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{ts}.csv"
        path = self.output_dir / filename

        # Collect all config keys and metric keys
        config_keys = set()
        metric_keys = set()
        for t in self.trials:
            config_keys.update(t.config.keys())
            metric_keys.update(t.metrics.keys())

        config_keys = sorted(config_keys)
        metric_keys = sorted(metric_keys)

        with open(path, "w") as f:
            headers = ["trial_id", "status", "duration_s"] + \
                      [f"cfg_{k}" for k in config_keys] + \
                      [f"metric_{k}" for k in metric_keys]
            f.write(",".join(headers) + "\n")

            for t in self.trials:
                row = [t.trial_id, t.status, f"{t.duration_seconds:.4f}"]
                row += [str(t.config.get(k, "")) for k in config_keys]
                row += [str(t.metrics.get(k, "")) for k in metric_keys]
                f.write(",".join(row) + "\n")

        print(f"📄 CSV exported to {path}")
        return path
