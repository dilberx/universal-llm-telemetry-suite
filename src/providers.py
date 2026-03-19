"""
providers.py — Pluggable TelemetryProvider architecture.
"""

from __future__ import annotations
import csv
import os
import platform
import plistlib
import re
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Any
from datetime import datetime, timezone

class TelemetryProvider(ABC):
    @abstractmethod
    def get_hardware_info(self) -> dict:
        pass

    @abstractmethod
    def get_cli_flags(self) -> list[str]:
        """Returns platform-specific flags to ensure machine-readable output."""
        pass

    @abstractmethod
    def start(self, model_name: str, thermal_log_csv: str) -> None:
        pass

    @abstractmethod
    def stop(self) -> dict:
        pass

    def set_target_pid(self, pid: int) -> None:
        """Optional: link a specific process PID for telemetry tracking."""
        pass

# ---------------------------------------------------------------------------
# NvidiaProvider (CUDA/WSL)
# ---------------------------------------------------------------------------
class NvidiaProvider(TelemetryProvider):
    def __init__(self, gpu_index: int = 0):
        import pynvml
        self._pynvml = pynvml
        self._gpu_index = gpu_index
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._vram_list, self._power_list, self._temp_list, self._clock_list = [], [], [], []

    def get_cli_flags(self) -> list[str]:
        # Standard flags work fine on Linux/WSL
        return ["--no-conversation", "--single-turn", "--log-disable"]

    def get_hardware_info(self) -> dict:
        pynvml = self._pynvml
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                "gpu_name": gpu_name,
                "total_vram_gb": round(mem_info.total / (1024 ** 3), 2),
                "base_clock_mhz": 0.0, # Optional: fetch via nvml if needed
                "memory_type": "discrete",
                "isolation_level": "gpu_isolated",
            }
        except Exception:
            return {"gpu_name": "Nvidia GPU", "total_vram_gb": 0.0, "memory_type": "discrete", "isolation_level": "gpu_isolated", "base_clock_mhz": 0.0}

    def start(self, model_name: str, thermal_log_csv: str) -> None:
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll, args=(model_name, thermal_log_csv), daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        if self._stop_event: self._stop_event.set()
        if self._thread: self._thread.join()
        return {
            "max_vram_mb": round(max(self._vram_list, default=0.0), 2),
            "avg_power_watts": round(_safe_mean(self._power_list), 2),
            "avg_temp_c": round(_safe_mean(self._temp_list), 2),
            "avg_clock_mhz": round(_safe_mean(self._clock_list), 2),
        }

    
    def _poll(self, model_name: str, thermal_log_csv: str) -> None:
        # We add 'tasks' to the samplers to get better CPU/GPU breakdown
        cmd = ["sudo", "powermetrics", "--samplers", "cpu_power,gpu_power,thermal", "-i", "500", "-f", "text"]
        self._pm_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        
        buffer = ""
        with open(thermal_log_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            for line in self._pm_proc.stdout:
                if self._stop_event.is_set(): break
                buffer += line
                
                if "*****" in line:
                    p_watt = self._parse_power(buffer)
                    t_c = self._parse_temp(buffer)
                    m_mb = _available_memory_mb()
                    
                    if p_watt > 0: self._power_list.append(p_watt)
                    if t_c > 0: self._temp_list.append(t_c)
                    self._mem_peak_mb = max(self._mem_peak_mb, m_mb)
                    
                    writer.writerow([time.time(), model_name, m_mb, p_watt, t_c, 0.0])
                    f.flush()
                    buffer = "" # Reset buffer for next sample

    def _parse_power(self, sample: str) -> float:
        """Sum all power components if 'Package Power' isn't explicitly found."""
        # 1. Primary: Package Power (Total SoC)
        pkg = re.search(r"Package Power:\s*([\d.]+)\s*mW", sample)
        if pkg: return float(pkg.group(1)) / 1000.0
        
        # 2. Fallback: Sum components
        cpu = re.search(r"CPU Power:\s*([\d.]+)\s*mW", sample)
        gpu = re.search(r"GPU Power:\s*([\d.]+)\s*mW", sample)
        ane = re.search(r"ANE Power:\s*([\d.]+)\s*mW", sample)
        
        total_mw = 0.0
        if cpu: total_mw += float(cpu.group(1))
        if gpu: total_mw += float(gpu.group(1))
        if ane: total_mw += float(ane.group(1))
        
        return total_mw / 1000.0

# ---------------------------------------------------------------------------
# AppleSiliconProvider (Metal/macOS)
# ---------------------------------------------------------------------------
class AppleSiliconProvider(TelemetryProvider):
    # Flexible regex: match any "<Label> Power: <N> mW" line
    _RE_POWER_VAL = re.compile(r"([\w\s]+) Power:\s*([\d.]+)\s*mW")
    # SMC die temperature: "CPU die temperature: 45.67 C"
    _RE_TEMP_SMC  = re.compile(r"CPU die temperature:\s*([\d.]+)\s*C", re.I)
    # Legacy fallback
    _RE_TEMP_LEGACY = re.compile(r"die-temp\s*=\s*([\d.]+)")

    def __init__(self):
        self._stop_event, self._thread, self._pm_proc = None, None, None
        self._power_list, self._temp_list = [], []
        self._mem_start_mb, self._mem_peak_mb = 0.0, 0.0
        self._target_pid: Optional[int] = None

    def set_target_pid(self, pid: int) -> None:
        self._target_pid = pid

    def get_cli_flags(self) -> list[str]:
        return ["--no-conversation", "--single-turn", "--simple-io", "--no-display-prompt", "--log-disable"]

    def get_hardware_info(self) -> dict:
        return {
            "gpu_name": _apple_chip_name(),
            "total_vram_gb": _apple_total_ram_gb(),
            "base_clock_mhz": 0.0,
            "memory_type": "unified",
            "isolation_level": "system_level",
        }

    def start(self, model_name: str, thermal_log_csv: str) -> None:
        self._power_list, self._temp_list = [], []
        self._target_pid = None # Reset for new run
        self._mem_start_mb = _available_memory_mb()
        self._mem_peak_mb = 0.0 
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll, args=(model_name, thermal_log_csv), daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        if self._stop_event: self._stop_event.set()
        if self._pm_proc: self._pm_proc.terminate()
        if self._thread: self._thread.join(timeout=5)
        
        # If we tracked a specific PID, use its peak RSS. Otherwise use system delta.
        if self._target_pid and self._mem_peak_mb > 0:
            mem_used_mb = self._mem_peak_mb
        else:
            mem_used_mb = max(0.0, self._mem_peak_mb - self._mem_start_mb) if self._mem_peak_mb > 0 else 0.0

        return {
            "max_vram_mb":     round(mem_used_mb, 2),
            "avg_power_watts": round(_safe_mean(self._power_list), 2),
            "avg_temp_c":      round(_safe_mean(self._temp_list),  2),
            "avg_clock_mhz":   0.0,
        }

    def _poll(self, model_name: str, thermal_log_csv: str) -> None:
        # Robust PIPE approach with NSUnbufferedIO to force immediate flushing
        env = os.environ.copy()
        env["NSUnbufferedIO"] = "YES"
        cmd = ["sudo", "powermetrics",
               "--samplers", "cpu_power,gpu_power,thermal",
               "-i", "500", "-f", "plist"]

        self._pm_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env
        )

        with open(thermal_log_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            buffer = b""

            for chunk in iter(lambda: self._pm_proc.stdout.read(4096), b""):
                if self._stop_event and self._stop_event.is_set():
                    break
                buffer += chunk
                while b"\0" in buffer:
                    plist_bytes, buffer = buffer.split(b"\0", 1)
                    if not plist_bytes.strip():
                        continue

                    try:
                        p = plistlib.loads(plist_bytes)
                        p_watt = p.get('processor', {}).get('combined_power', 0.0) / 1000.0
                        t_c = 0.0  # Thermal pressure doesn't export die temp natively in plist
                        
                        m_mb = _available_memory_mb(self._target_pid)
                        self._mem_peak_mb = max(self._mem_peak_mb, m_mb)

                        if p_watt > 0: self._power_list.append(p_watt)
                        if t_c > 0:    self._temp_list.append(t_c)

                        writer.writerow([time.time(), model_name, m_mb, p_watt, t_c, 0.0])
                        f.flush()
                    except Exception:
                        pass

class NullProvider(TelemetryProvider):
    def get_hardware_info(self) -> dict: return {"gpu_name": "CPU", "total_vram_gb": 0.0, "memory_type": "none", "isolation_level": "none", "base_clock_mhz": 0.0}
    def get_cli_flags(self) -> list[str]: return ["--no-conversation", "--single-turn"]
    def start(self, model_name: str, thermal_log_csv: str) -> None: pass
    def stop(self) -> dict: return {"max_vram_mb": 0.0, "avg_power_watts": 0.0, "avg_temp_c": 0.0, "avg_clock_mhz": 0.0}

def detect_provider(gpu_index: int = 0) -> TelemetryProvider:
    if sys.platform == "darwin" and platform.machine() == "arm64": return AppleSiliconProvider()
    try:
        import pynvml
        pynvml.nvmlInit()
        return NvidiaProvider(gpu_index)
    except: return NullProvider()

def _safe_mean(v): return sum(v)/len(v) if v else 0.0
def _apple_chip_name():
    res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
    return res.stdout.strip() or "Apple Silicon"
def _apple_total_ram_gb():
    res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
    return round(int(res.stdout.strip()) / (1024**3), 2)
def _available_memory_mb(pid: Optional[int] = None):
    """Return memory in MB: process RSS (with children) if PID given, else system used."""
    import psutil
    if pid:
        try:
            proc = psutil.Process(pid)
            total_rss = proc.memory_info().rss
            # Include child processes (llama-cli may fork workers)
            for child in proc.children(recursive=True):
                try:
                    total_rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return round(total_rss / (1024**2), 2)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return round(psutil.virtual_memory().used / (1024**2), 2)