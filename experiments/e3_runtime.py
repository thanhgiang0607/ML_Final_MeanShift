from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn import __version__ as sklearn_version
from sklearn.datasets import make_blobs
import scipy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = [
    "custom_brute",
    "custom_kdtree",
    "custom_kdtree_bin",
    "sklearn",
    "sklearn_bin",
]
METHOD_LABELS = {
    "custom_brute": "Custom brute",
    "custom_kdtree": "Custom KD-Tree",
    "custom_kdtree_bin": "Custom KD-Tree + bin seeding",
    "sklearn": "sklearn (bin seeding off)",
    "sklearn_bin": "sklearn (bin seeding on)",
}
RAW_FIELDS = [
    "experiment", "n", "h", "alpha", "method", "measurement", "repeat", "random_state",
    "elapsed_s", "peak_memory_mb", "n_seeds", "n_clusters",
    "neighbor_queries", "mean_neighbors_per_query", "mean_iterations",
    "status", "error",
]
SUMMARY_FIELDS = [
    "experiment", "n", "h", "alpha", "method", "successful_runs",
    "median_time_s", "q1_time_s", "q3_time_s", "iqr_time_s",
    "median_peak_memory_mb", "q1_peak_memory_mb", "q3_peak_memory_mb",
    "iqr_peak_memory_mb", "speedup_vs_custom_brute",
    "median_n_seeds", "median_n_clusters", "mean_neighbors_per_query",
    "mean_iterations", "status",
]


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy config: {config_path}. "
            "Hãy đặt configs/e3_runtime.yaml trong repository hoặc dùng --config."
        )
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _read_cpu_name() -> str:
    candidates = [platform.processor(), os.environ.get("PROCESSOR_IDENTIFIER", "")]
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        candidates.insert(0, line.split(":", 1)[1].strip())
                        break
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                candidates.insert(0, result.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return next((x for x in candidates if x and x.strip()), platform.machine())


def _read_ram_gb() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    if hasattr(os, "sysconf"):
        try:
            return (
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                / (1024 ** 3)
            )
        except (ValueError, OSError, AttributeError):
            pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / (1024 ** 3)
        except Exception:
            pass
    return None


def write_environment(out_dir: Path) -> dict[str, Any]:
    ram_gb = _read_ram_gb()
    env = {
        "cpu": _read_cpu_name(),
        "logical_cpu_count": os.cpu_count(),
        "ram_gb": round(ram_gb, 3) if ram_gb is not None else None,
        "operating_system": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn_version,
        "script": "experiments/e3_runtime.py",
        "measurement_protocol": (
            "runtime: perf_counter around fit with no tracemalloc/wrappers; "
            "memory: separate tracemalloc fit with no wrappers; "
            "diagnostics: separate fit with neighbor/iteration wrappers"
        ),
        "memory_measurement": "tracemalloc peak Python-traced memory; not total native/process RSS",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "E3 — Environment",
        f"CPU: {env['cpu']}",
        f"Logical CPUs: {env['logical_cpu_count']}",
        f"RAM total (GB): {env['ram_gb'] if env['ram_gb'] is not None else 'unavailable'}",
        f"OS: {env['operating_system']}",
        f"Python: {env['python']}",
        f"NumPy: {env['numpy']}",
        f"SciPy: {env['scipy']}",
        f"scikit-learn: {env['scikit_learn']}",
        "Timing: separate fit runs without tracemalloc and without measurement wrappers.",
        "Memory: one separate fit run with tracemalloc; no measurement wrappers.",
        "Diagnostics: separate fit run with neighbor/iteration wrappers; its time is not reported as runtime.",
        "Memory metric: Python-traced peak; native allocations may not be fully captured.",
    ]
    (out_dir / "env.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env


def _estimate_seed_count(X: np.ndarray, h: float, method: str) -> int:
    uses_bin = method in {"custom_kdtree_bin", "sklearn_bin"}
    if not uses_bin:
        return int(len(X))
    try:
        if method.startswith("custom"):
            from meanshift.seeding import get_bin_seeds
        else:
            sk_ms = importlib.import_module("sklearn.cluster._mean_shift")
            get_bin_seeds = sk_ms.get_bin_seeds
        seeds = get_bin_seeds(X, bin_size=h, min_bin_freq=1)
        return int(len(seeds))
    except Exception:
        return -1


def _mean_neighbor_stats(stats: dict[str, Any]) -> tuple[int, float]:
    count = int(stats.get("neighbor_queries", 0))
    if count <= 0:
        return 0, float("nan")
    return count, float(stats.get("neighbor_count_sum", 0)) / count


def _record_sklearn_neighbor_rows(stats: dict[str, Any], rows: Any) -> None:
    try:
        for row in rows:
            try:
                size = len(row)
            except TypeError:
                size = 1
            stats["neighbor_count_sum"] += int(size)
            stats["neighbor_queries"] += 1
    except TypeError:
        try:
            size = len(rows)
            stats["neighbor_count_sum"] += int(size)
            stats["neighbor_queries"] += 1
        except TypeError:
            pass


def _record_custom_neighbor_result(stats: dict[str, Any], result: Any) -> None:
    try:
        if isinstance(result, tuple) and len(result) == 2:
            rows = [result]
        elif isinstance(result, list):
            rows = result
        else:
            return

        for row in rows:
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                continue
            indices = row[0]
            try:
                neighbor_count = len(indices)
            except TypeError:
                neighbor_count = 1
            stats["neighbor_count_sum"] += int(neighbor_count)
            stats["neighbor_queries"] += 1
    except Exception:
        pass


def _make_estimator(method: str, h: float, cfg: dict[str, Any]):
    max_iter = int(cfg["benchmark"].get("max_iter", 300))
    tol = float(cfg["benchmark"].get("tol", 1e-3))
    if method.startswith("custom"):
        from meanshift import MeanShiftCustom
        if method == "custom_brute":
            backend, bin_seeding = "brute", False
        elif method == "custom_kdtree":
            backend, bin_seeding = "kdtree", False
        elif method == "custom_kdtree_bin":
            backend, bin_seeding = "kdtree", True
        else:
            raise ValueError(f"Unknown custom method: {method}")
        return MeanShiftCustom(
            bandwidth=float(h),
            kernel="flat",
            neighbor_search=backend,
            bin_seeding=bin_seeding,
            max_iter=max_iter,
            tol=tol,
            cluster_all=True,
        )
    if method in {"sklearn", "sklearn_bin"}:
        from sklearn.cluster import MeanShift
        return MeanShift(
            bandwidth=float(h),
            bin_seeding=(method == "sklearn_bin"),
            n_jobs=1,
            max_iter=max_iter,
            cluster_all=True,
        )
    raise ValueError(f"Unknown method: {method}")


def _fit_worker(
    conn,
    X: np.ndarray,
    method: str,
    h: float,
    cfg: dict[str, Any],
    measurement: str,
) -> None:
    import tracemalloc

    try:
        if measurement not in {"time", "memory", "diagnostics"}:
            raise ValueError(f"measurement không hợp lệ: {measurement}")

        with warnings.catch_warnings():
            warnings.simplefilter("default")
            estimator = _make_estimator(method, h, cfg)
            n_seeds = _estimate_seed_count(X, h, method)
            stats: dict[str, Any] = {
                "neighbor_queries": 0,
                "neighbor_count_sum": 0,
                "iterations": [],
            }
            restore_actions = []
            if measurement == "diagnostics" and method.startswith("custom"):
                try:
                    neighbor_module = importlib.import_module("meanshift.neighbors")
                    for cls_name in ("BruteForceSearch", "KDTreeSearch"):
                        cls = getattr(neighbor_module, cls_name, None)
                        if cls is None or not hasattr(cls, "query_radius"):
                            continue
                        original = cls.query_radius

                        def custom_wrapper(self, *args, _original=original, **kwargs):
                            result = _original(self, *args, **kwargs)
                            _record_custom_neighbor_result(stats, result)
                            return result

                        cls.query_radius = custom_wrapper
                        restore_actions.append((cls, "query_radius", original))
                except Exception:
                    pass
            elif measurement == "diagnostics" and method in {"sklearn", "sklearn_bin"}:
                try:
                    sk_ms = importlib.import_module("sklearn.cluster._mean_shift")
                    nn_cls = sk_ms.NearestNeighbors
                    original_radius = nn_cls.radius_neighbors

                    def sklearn_radius_wrapper(self, *args, _original=original_radius, **kwargs):
                        result = _original(self, *args, **kwargs)
                        return_distance = kwargs.get("return_distance", True)
                        if len(args) >= 3:
                            return_distance = args[2]
                        indices = result[1] if return_distance and isinstance(result, tuple) else result
                        _record_sklearn_neighbor_rows(stats, indices)
                        return result

                    nn_cls.radius_neighbors = sklearn_radius_wrapper
                    restore_actions.append((nn_cls, "radius_neighbors", original_radius))

                    original_seed_fit = getattr(sk_ms, "_mean_shift_single_seed", None)
                    if original_seed_fit is not None:
                        def sklearn_seed_wrapper(*args, _original=original_seed_fit, **kwargs):
                            result = _original(*args, **kwargs)
                            if isinstance(result, tuple) and len(result) >= 3:
                                stats["iterations"].append(float(result[2]))
                            return result
                        sk_ms._mean_shift_single_seed = sklearn_seed_wrapper
                        restore_actions.append((sk_ms, "_mean_shift_single_seed", original_seed_fit))
                except Exception:
                    pass

            conn.send({"event": "started"})
            elapsed = float("nan")
            peak_bytes = float("nan")
            if measurement == "memory":
                tracemalloc.start()
            start = time.perf_counter() if measurement == "time" else None
            try:
                estimator.fit(X)
            finally:
                if measurement == "time" and start is not None:
                    elapsed = time.perf_counter() - start
                if measurement == "memory":
                    try:
                        _current, peak_bytes = tracemalloc.get_traced_memory()
                    finally:
                        tracemalloc.stop()
                for obj, attr, original in reversed(restore_actions):
                    setattr(obj, attr, original)

            n_clusters = -1
            centers = getattr(estimator, "cluster_centers_", None)
            if centers is not None:
                try:
                    n_clusters = int(len(centers))
                except TypeError:
                    pass
            if n_clusters < 0:
                labels = getattr(estimator, "labels_", None)
                if labels is not None:
                    n_clusters = int(len(np.unique(labels[labels >= 0])))

            mean_iterations = float("nan")
            per_seed_iterations = getattr(estimator, "seed_n_iter_", None)
            if per_seed_iterations is not None:
                try:
                    arr = np.asarray(per_seed_iterations, dtype=float).ravel()
                    if arr.size:
                        mean_iterations = float(np.mean(arr))
                except (TypeError, ValueError):
                    pass
            if not math.isfinite(mean_iterations) and stats["iterations"]:
                mean_iterations = float(np.mean(stats["iterations"]))

            n_queries, mean_neighbors = _mean_neighbor_stats(stats)
            if measurement != "diagnostics":
                n_queries, mean_neighbors = 0, float("nan")
            if measurement != "memory":
                peak_bytes = float("nan")
            if measurement != "time":
                elapsed = float("nan")

            conn.send({
                "event": "result",
                "data": {
                    "status": "ok",
                    "elapsed_s": float(elapsed),
                    "peak_memory_mb": float(peak_bytes / (1024 ** 2)) if math.isfinite(float(peak_bytes)) else float("nan"),
                    "n_seeds": int(n_seeds),
                    "n_clusters": int(n_clusters),
                    "neighbor_queries": int(n_queries),
                    "mean_neighbors_per_query": float(mean_neighbors),
                    "mean_iterations": float(mean_iterations),
                    "error": "",
                },
            })
    except BaseException as exc:
        try:
            conn.send({
                "event": "result",
                "data": {
                    "status": "error",
                    "elapsed_s": float("nan"),
                    "peak_memory_mb": float("nan"),
                    "n_seeds": -1,
                    "n_clusters": -1,
                    "neighbor_queries": 0,
                    "mean_neighbors_per_query": float("nan"),
                    "mean_iterations": float("nan"),
                    "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
                },
            })
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def fit_with_timeout(
    X: np.ndarray,
    method: str,
    h: float,
    cfg: dict[str, Any],
    timeout_seconds: float,
    measurement: str = "time",
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_fit_worker,
        args=(child_conn, X, method, float(h), cfg, measurement),
    )
    proc.start()
    child_conn.close()
    fit_started_at: float | None = None
    result = None
    timed_out = False
    pipe_error: str | None = None

    try:
        while proc.is_alive():
            try:
                ready = parent_conn.poll(0.1)
            except (EOFError, OSError, ValueError) as exc:
                pipe_error = f"Pipe bị đóng khi chờ worker: {type(exc).__name__}: {exc}"
                break

            if ready:
                try:
                    message = parent_conn.recv()
                except (EOFError, OSError, ValueError) as exc:
                    pipe_error = f"Không đọc được dữ liệu từ worker: {type(exc).__name__}: {exc}"
                    break
                if message:
                    if message.get("event") == "started":
                        fit_started_at = time.monotonic()
                    elif message.get("event") == "result":
                        result = message.get("data")

            if fit_started_at is not None and time.monotonic() - fit_started_at > timeout_seconds:
                timed_out = True
                proc.terminate()
                proc.join(timeout=3)
                if proc.is_alive() and hasattr(proc, "kill"):
                    proc.kill()
                    proc.join(timeout=3)
                break

        proc.join(timeout=1)
        while True:
            try:
                ready = parent_conn.poll(0)
            except (EOFError, OSError, ValueError) as exc:
                if pipe_error is None:
                    pipe_error = f"Pipe đóng khi đọc kết quả cuối: {type(exc).__name__}: {exc}"
                break
            if not ready:
                break
            try:
                message = parent_conn.recv()
            except (EOFError, OSError, ValueError) as exc:
                if pipe_error is None:
                    pipe_error = f"Không đọc được kết quả cuối: {type(exc).__name__}: {exc}"
                break
            if message and message.get("event") == "result":
                result = message.get("data")
    finally:
        try:
            parent_conn.close()
        except OSError:
            pass
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive() and hasattr(proc, "kill"):
                proc.kill()
                proc.join(timeout=2)

    if timed_out:
        return {
            "status": "timeout",
            "elapsed_s": float(timeout_seconds) if measurement == "time" else float("nan"),
            "peak_memory_mb": float("nan"),
            "n_seeds": -1,
            "n_clusters": -1,
            "neighbor_queries": 0,
            "mean_neighbors_per_query": float("nan"),
            "mean_iterations": float("nan"),
            "error": f"quá thời gian (> {timeout_seconds:g} giây) ở lượt đo {measurement}",
        }
    if result is None:
        details = f"Worker kết thúc mà không trả kết quả (exitcode={proc.exitcode})."
        if pipe_error:
            details += f" Chi tiết pipe: {pipe_error}"
        return {
            "status": "error",
            "elapsed_s": float("nan"),
            "peak_memory_mb": float("nan"),
            "n_seeds": -1,
            "n_clusters": -1,
            "neighbor_queries": 0,
            "mean_neighbors_per_query": float("nan"),
            "mean_iterations": float("nan"),
            "error": details,
        }
    return result

def _to_csv_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return value if math.isfinite(float(value)) else ""
    return value


def _append_row(writer: csv.DictWriter, file_obj, row: dict[str, Any]) -> None:
    writer.writerow({k: _to_csv_value(row.get(k, "")) for k in RAW_FIELDS})
    file_obj.flush()


def _base_row(
    experiment: str,
    n: int,
    h: float,
    alpha: float | None,
    method: str,
    measurement: str,
    repeat: int,
    random_state: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "experiment": experiment,
        "n": int(n),
        "h": float(h),
        "alpha": "" if alpha is None else float(alpha),
        "method": method,
        "measurement": measurement,
        "repeat": int(repeat),
        "random_state": int(random_state),
    }
    row.update(result)
    return row


def _skipped_result(status: str, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "elapsed_s": float("nan"),
        "peak_memory_mb": float("nan"),
        "n_seeds": -1,
        "n_clusters": -1,
        "neighbor_queries": 0,
        "mean_neighbors_per_query": float("nan"),
        "mean_iterations": float("nan"),
        "error": reason,
    }


def _append_result_row(writer, file_obj, all_rows, experiment, n, h, alpha,
                       method, measurement, repeat, random_state, result):
    row = _base_row(experiment, n, h, alpha, method, measurement, repeat, random_state, result)
    _append_row(writer, file_obj, row)
    all_rows.append(row)
    return row


def _append_skipped_config(writer, file_obj, all_rows, experiment, n, h, alpha,
                           method, random_state, repeats, reason):
    for repeat in range(1, repeats + 1):
        _append_result_row(
            writer, file_obj, all_rows, experiment, n, h, alpha, method,
            "time", repeat, random_state,
            _skipped_result("skipped_after_timeout", reason),
        )
    for measurement in ("memory", "diagnostics"):
        _append_result_row(
            writer, file_obj, all_rows, experiment, n, h, alpha, method,
            measurement, 0, random_state,
            _skipped_result("skipped_after_timeout", reason),
        )


def _run_one_config(writer, file_obj, all_rows, X, cfg, experiment, n, h, alpha,
                    method, random_state, repeats, timeout):
    for repeat in range(1, repeats + 1):
        result = fit_with_timeout(X, method, h, cfg, timeout, measurement="time")
        _append_result_row(
            writer, file_obj, all_rows, experiment, n, h, alpha, method,
            "time", repeat, random_state, result,
        )
        print(
            f"[E3:{experiment}:time] method={method:18s} n={n:6d} "
            f"alpha={alpha if alpha is not None else '-'} repeat={repeat}/{repeats} "
            f"status={result['status']} time={result['elapsed_s'] if math.isfinite(float(result['elapsed_s'])) else 'NA'}",
            flush=True,
        )
        if result["status"] == "timeout":
            for remaining in range(repeat + 1, repeats + 1):
                _append_result_row(
                    writer, file_obj, all_rows, experiment, n, h, alpha, method,
                    "time", remaining, random_state,
                    _skipped_result("skipped_after_timeout", "bỏ lượt thời gian còn lại sau timeout"),
                )
            for measurement in ("memory", "diagnostics"):
                _append_result_row(
                    writer, file_obj, all_rows, experiment, n, h, alpha, method,
                    measurement, 0, random_state,
                    _skipped_result("skipped_after_timeout", "bỏ phép đo riêng sau timeout ở phép đo thời gian"),
                )
            return True

    mem_result = fit_with_timeout(X, method, h, cfg, timeout, measurement="memory")
    _append_result_row(
        writer, file_obj, all_rows, experiment, n, h, alpha, method,
        "memory", 0, random_state, mem_result,
    )
    print(
        f"[E3:{experiment}:memory] method={method:18s} n={n:6d} "
        f"alpha={alpha if alpha is not None else '-'} status={mem_result['status']} "
        f"peak_MB={mem_result['peak_memory_mb'] if math.isfinite(float(mem_result['peak_memory_mb'])) else 'NA'}",
        flush=True,
    )
    if mem_result["status"] == "timeout":
        _append_result_row(
            writer, file_obj, all_rows, experiment, n, h, alpha, method,
            "diagnostics", 0, random_state,
            _skipped_result("skipped_after_timeout", "bỏ phép đo chẩn đoán sau timeout đo bộ nhớ"),
        )
        return False
    diag_result = fit_with_timeout(X, method, h, cfg, timeout, measurement="diagnostics")
    _append_result_row(
        writer, file_obj, all_rows, experiment, n, h, alpha, method,
        "diagnostics", 0, random_state, diag_result,
    )
    print(
        f"[E3:{experiment}:diagnostics] method={method:18s} n={n:6d} "
        f"alpha={alpha if alpha is not None else '-'} status={diag_result['status']} "
        f"mean_neighbors={diag_result['mean_neighbors_per_query'] if math.isfinite(float(diag_result['mean_neighbors_per_query'])) else 'NA'}",
        flush=True,
    )
    return False


def _run_sweep_n(writer: csv.DictWriter, f, cfg: dict[str, Any], h0: float,
                 n_values: list[int], repeats: int, timeout: float) -> list[dict[str, Any]]:
    ds = cfg["dataset"]
    all_rows: list[dict[str, Any]] = []
    random_state = int(ds.get("random_state", 0))
    for method in METHODS:
        stop_larger_n = False
        for n in n_values:
            if stop_larger_n:
                _append_skipped_config(
                    writer, f, all_rows, "n", n, h0, None, method,
                    random_state, repeats,
                    "bỏ n lớn hơn sau timeout của biến thể này",
                )
                continue
            X, _ = make_blobs(
                n_samples=int(n), centers=int(ds.get("centers", 5)),
                cluster_std=float(ds.get("cluster_std", 1.0)),
                random_state=random_state, n_features=int(ds.get("dimensions", 2)),
            )
            timed_out = _run_one_config(
                writer, f, all_rows, X, cfg, "n", int(n), h0, None,
                method, random_state, repeats, timeout,
            )
            if timed_out:
                stop_larger_n = True
    return all_rows


def _run_sweep_h(writer: csv.DictWriter, f, cfg: dict[str, Any], h0: float,
                 n: int, alphas: list[float], repeats: int, timeout: float) -> list[dict[str, Any]]:
    ds = cfg["dataset"]
    random_state = int(ds.get("random_state", 0))
    X, _ = make_blobs(
        n_samples=int(n), centers=int(ds.get("centers", 5)),
        cluster_std=float(ds.get("cluster_std", 1.0)),
        random_state=random_state, n_features=int(ds.get("dimensions", 2)),
    )
    all_rows: list[dict[str, Any]] = []
    for method in METHODS:
        stop_larger_alpha = False
        for alpha in alphas:
            h = float(alpha) * float(h0)
            if stop_larger_alpha:
                _append_skipped_config(
                    writer, f, all_rows, "h", n, h, alpha, method,
                    random_state, repeats,
                    "bỏ alpha lớn hơn sau timeout của biến thể này",
                )
                continue
            timed_out = _run_one_config(
                writer, f, all_rows, X, cfg, "h", int(n), h, float(alpha),
                method, random_state, repeats, timeout,
            )
            if timed_out:
                stop_larger_alpha = True
    return all_rows

def _numeric(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    vals = []
    for row in rows:
        try:
            v = float(row.get(field, ""))
            if math.isfinite(v):
                vals.append(v)
        except (TypeError, ValueError):
            continue
    return np.asarray(vals, dtype=float)


def summarize(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_fields = ("experiment", "n", "h", "alpha", "method")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = tuple(row.get(k, "") for k in key_fields)
        groups.setdefault(key, []).append(row)

    def quantiles(values: np.ndarray) -> tuple[float, float, float, float]:
        if values.size == 0:
            return (float("nan"),) * 4
        median = float(np.percentile(values, 50))
        if values.size == 1:
            return median, float("nan"), float("nan"), float("nan")
        q1, q3 = np.percentile(values, [25, 75])
        return median, float(q1), float(q3), float(q3 - q1)

    summary: list[dict[str, Any]] = []
    for key, group in groups.items():
        timing_rows = [r for r in group if r.get("measurement") == "time" and r.get("status") == "ok"]
        memory_rows = [r for r in group if r.get("measurement") == "memory" and r.get("status") == "ok"]
        diagnostic_rows = [r for r in group if r.get("measurement") == "diagnostics" and r.get("status") == "ok"]

        time_vals = _numeric(timing_rows, "elapsed_s")
        mem_vals = _numeric(memory_rows, "peak_memory_mb")
        seed_vals = _numeric(timing_rows, "n_seeds")
        cluster_vals = _numeric(timing_rows, "n_clusters")
        neighbor_vals = _numeric(diagnostic_rows, "mean_neighbors_per_query")
        iter_vals = _numeric(diagnostic_rows, "mean_iterations")

        med_t, q1_t, q3_t, iqr_t = quantiles(time_vals)
        med_m, q1_m, q3_m, iqr_m = quantiles(mem_vals)
        time_rows_any = [r for r in group if r.get("measurement") == "time"]
        status = "ok" if time_vals.size else next(
            (r.get("status", "unknown") for r in time_rows_any),
            next((r.get("status", "unknown") for r in group), "unknown"),
        )
        summary.append({
            **dict(zip(key_fields, key)),
            "successful_runs": int(time_vals.size),
            "median_time_s": med_t,
            "q1_time_s": q1_t,
            "q3_time_s": q3_t,
            "iqr_time_s": iqr_t,
            "median_peak_memory_mb": med_m,
            "q1_peak_memory_mb": q1_m,
            "q3_peak_memory_mb": q3_m,
            "iqr_peak_memory_mb": iqr_m,
            "speedup_vs_custom_brute": float("nan"),
            "median_n_seeds": float(np.median(seed_vals)) if seed_vals.size else float("nan"),
            "median_n_clusters": float(np.median(cluster_vals)) if cluster_vals.size else float("nan"),
            "mean_neighbors_per_query": float(np.mean(neighbor_vals)) if neighbor_vals.size else float("nan"),
            "mean_iterations": float(np.mean(iter_vals)) if iter_vals.size else float("nan"),
            "status": status,
        })

    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in summary:
        k = (row["experiment"], row["n"], row["h"], row["alpha"])
        by_key.setdefault(k, {})[row["method"]] = row
    for group in by_key.values():
        baseline = group.get("custom_brute")
        if not baseline or baseline.get("status") != "ok":
            continue
        base_time = baseline.get("median_time_s")
        if not isinstance(base_time, (int, float)) or not math.isfinite(float(base_time)) or base_time <= 0:
            continue
        for row in group.values():
            t = row.get("median_time_s")
            if isinstance(t, (int, float)) and math.isfinite(float(t)) and float(t) > 0:
                row["speedup_vs_custom_brute"] = float(base_time) / float(t)
    return sorted(summary, key=lambda r: (r["experiment"], str(r["method"]), float(r["n"]), float(r["h"])))

def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _to_csv_value(row.get(k, "")) for k in SUMMARY_FIELDS})


def _summary_value(rows: list[dict[str, Any]], experiment: str, method: str,
                   n: int | None = None, alpha: float | None = None) -> dict[str, Any] | None:
    candidates = [r for r in rows if r["experiment"] == experiment and r["method"] == method]
    if n is not None:
        candidates = [r for r in candidates if int(float(r["n"])) == int(n)]
    if alpha is not None:
        candidates = [r for r in candidates if math.isclose(float(r["alpha"]), alpha, rel_tol=1e-9, abs_tol=1e-12)]
    candidates = [r for r in candidates if r["status"] == "ok" and isinstance(r.get("median_time_s"), (int, float)) and math.isfinite(float(r["median_time_s"]))]
    return candidates[0] if candidates else None


def create_plots(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    drew = False
    for method in METHODS:
        rows = [r for r in summary_rows if r["experiment"] == "n" and r["method"] == method and r["status"] == "ok"]
        rows = sorted(rows, key=lambda r: int(float(r["n"])))
        xs = [int(float(r["n"])) for r in rows if _valid_positive(r.get("median_time_s"))]
        ys = [float(r["median_time_s"]) for r in rows if _valid_positive(r.get("median_time_s"))]
        if xs and ys:
            ax.loglog(xs, ys, marker="o", label=METHOD_LABELS[method])
            drew = True
    ax.set_xlabel("Số điểm dữ liệu n")
    ax.set_ylabel("Thời gian fit trung vị (giây)")
    ax.set_title("E3 — Thời gian chạy theo kích thước dữ liệu (log–log)")
    ax.grid(True, which="both", alpha=0.25)
    if drew:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "e3_time_vs_n.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    drew = False
    for method in METHODS:
        rows = [r for r in summary_rows if r["experiment"] == "n" and r["method"] == method and r["status"] == "ok"]
        rows = sorted(rows, key=lambda r: int(float(r["n"])))
        pairs = [(int(float(r["n"])), float(r["median_peak_memory_mb"])) for r in rows
                 if _valid_positive(r.get("median_peak_memory_mb"))]
        if pairs:
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=METHOD_LABELS[method])
            drew = True
    ax.set_xscale("log")
    ax.set_xlabel("Số điểm dữ liệu n (thang log)")
    ax.set_ylabel("Bộ nhớ đỉnh tracemalloc (MB)")
    ax.set_title("E3 — Bộ nhớ đỉnh theo kích thước dữ liệu")
    ax.grid(True, which="both", alpha=0.25)
    if drew:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "e3_memory_vs_n.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    drew = False
    for method in METHODS:
        rows = [r for r in summary_rows if r["experiment"] == "h" and r["method"] == method and r["status"] == "ok"]
        rows = sorted(rows, key=lambda r: float(r["h"]))
        pairs = [(float(r["h"]), float(r["median_time_s"])) for r in rows
                 if _valid_positive(r.get("h")) and _valid_positive(r.get("median_time_s"))]
        if pairs:
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=METHOD_LABELS[method])
            drew = True
    ax.set_xlabel("Bandwidth h")
    ax.set_ylabel("Thời gian fit trung vị (giây)")
    ax.set_title("E3 — Thời gian chạy theo bandwidth h")
    ax.grid(True, alpha=0.25)
    if drew:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "e3_time_vs_h.png", dpi=150)
    plt.close(fig)


def _valid_positive(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def write_report(report_path: Path, cfg: dict[str, Any], h0: float, env: dict[str, Any],
                 summary_rows: list[dict[str, Any]], repeats: int, timeout: float) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    n_values = cfg["n_sweep"]["values"]
    h_sweep = cfg["h_sweep"]
    alphas = h_sweep["alphas"]

    def fmt(value: Any, digits: int = 4) -> str:
        try:
            number = float(value)
            return f"{number:.{digits}g}" if math.isfinite(number) else "—"
        except (TypeError, ValueError):
            return "—"

    def fmt_int(value: Any) -> str:
        try:
            number = float(value)
            return f"{number:.0f}" if math.isfinite(number) else "—"
        except (TypeError, ValueError):
            return "—"

    lines = []
    for n in n_values:
        for method in METHODS:
            r = _summary_value(summary_rows, "n", method, n=int(n))
            if r:
                lines.append(
                    f"| {int(n)} | {METHOD_LABELS[method]} | {fmt(r.get('median_time_s'))} | {fmt(r.get('iqr_time_s'))} | "
                    f"{fmt(r.get('median_peak_memory_mb'))} | {fmt_int(r.get('median_n_clusters'))} | "
                    f"{fmt_int(r.get('median_n_seeds'))} | {fmt(r.get('mean_neighbors_per_query'))} | {fmt(r.get('mean_iterations'))} |"
                )
            else:
                lines.append(f"| {int(n)} | {METHOD_LABELS[method]} | — | — | — | — | — | — | — |")

    lines.extend([])
    for alpha in alphas:
        h = float(alpha) * h0
        for method in METHODS:
            r = _summary_value(summary_rows, "h", method, n=int(h_sweep["n"]), alpha=float(alpha))
            if r:
                lines.append(
                    f"| {float(alpha):g} | {h:.5g} | {METHOD_LABELS[method]} | {fmt(r.get('median_time_s'))} | {fmt(r.get('iqr_time_s'))} | "
                    f"{fmt(r.get('median_peak_memory_mb'))} | {fmt_int(r.get('median_n_clusters'))} | "
                    f"{fmt_int(r.get('median_n_seeds'))} | {fmt(r.get('mean_neighbors_per_query'))} | {fmt(r.get('mean_iterations'))} |"
                )
            else:
                lines.append(f"| {float(alpha):g} | {h:.5g} | {METHOD_LABELS[method]} | — | — | — | — | — | — | — |")

    lines.extend(["", "## 4. Trả lời định lượng các câu hỏi RQ3"])

    def numeric(row: dict[str, Any] | None, key: str) -> float | None:
        if not row:
            return None
        try:
            value = float(row.get(key, float("nan")))
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    def pct_change(start_value: float | None, end_value: float | None) -> float | None:
        if start_value is None or end_value is None or abs(start_value) < 1e-15:
            return None
        return (end_value - start_value) / abs(start_value) * 100.0

    def pct_text(value: float | None) -> str:
        return "không tính được" if value is None else f"{value:+.1f}%"

    def direction(values: list[float]) -> str:
        values = [v for v in values if math.isfinite(v)]
        if len(values) < 2:
            return "chưa đủ điểm hợp lệ để xác định xu hướng"
        diffs = [b - a for a, b in zip(values, values[1:])]
        eps = 1e-12 * max(1.0, max(abs(v) for v in values))
        if all(d >= -eps for d in diffs) and any(d > eps for d in diffs):
            return "tăng hoặc giữ nguyên qua các mức đã đo"
        if all(d <= eps for d in diffs) and any(d < -eps for d in diffs):
            return "giảm hoặc giữ nguyên qua các mức đã đo"
        if all(abs(d) <= eps for d in diffs):
            return "gần như không đổi qua các mức đã đo"
        return "biến động không đơn điệu qua các mức đã đo"

    def get_row(experiment: str, method: str, *, n: int | None = None,
                alpha: float | None = None) -> dict[str, Any] | None:
        return _summary_value(summary_rows, experiment, method, n=n, alpha=alpha)

    lines.extend([])
    comparable_n: list[tuple[int, float, float, dict[str, Any], dict[str, Any]]] = []
    for n in sorted({int(float(v)) for v in n_values}):
        brute = get_row("n", "custom_brute", n=n)
        kd = get_row("n", "custom_kdtree", n=n)
        brute_t, kd_t = numeric(brute, "median_time_s"), numeric(kd, "median_time_s")
        if brute_t is None or kd_t is None or brute_t <= 0 or kd_t <= 0:
            lines.append(f"| {n} | — | — | — | {fmt(numeric(kd, 'mean_neighbors_per_query'))} |")
            continue
        ratio = brute_t / kd_t
        comparable_n.append((n, brute_t, kd_t, brute or {}, kd or {}))
        lines.append(
            f"| {n} | {brute_t:.5g} | {kd_t:.5g} | {ratio:.3f} | {fmt(numeric(kd, 'mean_neighbors_per_query'))} |"
        )

    wins = [entry for entry in comparable_n if entry[2] < entry[1]]
    if wins:
        first_n = wins[0][0]
        later = [entry for entry in comparable_n if entry[0] >= first_n]
        remains_faster = all(kd_t < brute_t for _, brute_t, kd_t, _, _ in later)
        if remains_faster:
            threshold_note = "và KD-Tree vẫn nhanh hơn ở tất cả mức n lớn hơn có số liệu hợp lệ"
        else:
            reversed_ns = [n for n, brute_t, kd_t, _, _ in later if kd_t >= brute_t]
            threshold_note = (
                + ", ".join(f"{n:,}" for n in reversed_ns)
            )
        lines.append(
            f"\n**Kết quả:** trong các mức n được so sánh hợp lệ, KD-Tree lần đầu nhanh hơn brute tại **n = {first_n:,}**; {threshold_note}. Đây là kết quả quan sát trên máy và bộ dữ liệu này, không phải ngưỡng phổ quát."
        )
    elif len(comparable_n) >= 2:
        largest = max(comparable_n, key=lambda entry: entry[0])
        n_max, brute_t, kd_t, _, kd_row = largest
        kd_neighbors = numeric(kd_row, "mean_neighbors_per_query")
        lines.append(
            f"\n**Kết quả:** chưa ghi nhận KD-Tree nhanh hơn brute ở bất kỳ mức n nào có đủ số liệu so sánh ({len(comparable_n)} mức hợp lệ). Tại n lớn nhất có thể so sánh (n={n_max:,}), brute mất {brute_t:.5g} s, KD-Tree mất {kd_t:.5g} s; tỷ số brute/KD-Tree = {brute_t / kd_t:.3f}×."
        )
        possible_reasons = []
        if kd_neighbors is not None and kd_neighbors > 0.25 * n_max:
            possible_reasons.append(
                f"số láng giềng trung bình của KD-Tree ({kd_neighbors:.3g}) đã chiếm hơn 25% n tại mức n này, nên việc truy vấn bán kính có thể phải xử lý nhiều điểm"
            )
        if kd_t >= brute_t:
            possible_reasons.append(
                "chi phí xây dựng/truy vấn cây có thể chưa được bù đắp bởi lượng tính toán tiết kiệm được ở bộ dữ liệu và quy mô đã thử"
            )
        if possible_reasons:
            lines.append("Các cơ chế có thể giải thích kết quả, cần xem như giả thuyết chứ chưa phải quan hệ nhân quả đã được chứng minh: " + "; ".join(possible_reasons) + ".")
        else:
            lines.append("Không nên suy diễn một nguyên nhân duy nhất từ benchmark hiện có; cần kiểm tra thêm số láng giềng, cấu hình và các lượt đo nếu muốn giải thích nguyên nhân.")
    else:
        lines.append("\n**Kết quả:** chưa đủ ít nhất hai mức n có thời gian hợp lệ cho cả brute và KD-Tree để xác định điểm giao hoặc đánh giá khả năng mở rộng.")

    slopes = []
    for method in METHODS:
        rows = [r for r in summary_rows if r.get("experiment") == "n" and r.get("method") == method
                and r.get("status") == "ok" and _valid_positive(r.get("n"))
                and _valid_positive(r.get("median_time_s"))]
        if len(rows) >= 2:
            rows.sort(key=lambda row: float(row["n"]))
            slope = float(np.polyfit(
                np.log10([float(r["n"]) for r in rows]),
                np.log10([float(r["median_time_s"]) for r in rows]), 1
            )[0])
            slopes.append((method, slope, len(rows)))
    lines.extend(["", "**Độ dốc log–log thực nghiệm** (hệ số góc của log10(thời gian median) theo log10(n), chỉ tính tại các điểm đo hợp lệ):"])
    if slopes:
        for method, slope, count in slopes:
            lines.append(f"- {METHOD_LABELS[method]}: độ dốc = **{slope:.3f}**, dựa trên {count} mức n.")
        lines.append("Độ dốc là mô tả thực nghiệm trong khoảng n đã đo; không nên đồng nhất trực tiếp với độ phức tạp tiệm cận lý thuyết.")
    else:
        lines.append("Chưa đủ ít nhất hai mức n hợp lệ cho một phương pháp để ước lượng độ dốc.")

    lines.extend([
        "",
        "### 4.2. Khi h tăng, số láng giềng và thời gian thay đổi thế nào?",
        "Bảng sau so sánh mức alpha nhỏ nhất và lớn nhất có số liệu hợp lệ của từng phương pháp. Phần trăm thay đổi được tính từ điểm đầu đến điểm cuối; phần nhận xét về xu hướng dùng tất cả điểm alpha có dữ liệu.",
        "",
        "| Phương pháp | alpha đầu → cuối | Láng giềng TB/truy vấn đầu → cuối | Thay đổi láng giềng | Thời gian median đầu → cuối (s) | Thay đổi thời gian | Xu hướng thời gian qua các alpha |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    h_trend_data: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    for method in METHODS:
        entries = []
        for alpha in alphas:
            row = get_row("h", method, n=int(h_sweep["n"]), alpha=float(alpha))
            if row is not None:
                entries.append((float(alpha), row))
        entries.sort(key=lambda item: item[0])
        h_trend_data[method] = entries
        valid_time = [(a, r, numeric(r, "median_time_s")) for a, r in entries
                      if numeric(r, "median_time_s") is not None]
        valid_neighbors = [(a, r, numeric(r, "mean_neighbors_per_query")) for a, r in entries
                           if numeric(r, "mean_neighbors_per_query") is not None]
        if not valid_time:
            lines.append(f"| {METHOD_LABELS[method]} | — | — | — | — | — | Không đủ thời gian hợp lệ |")
            continue
        first_a, first_r, first_t = valid_time[0]
        last_a, last_r, last_t = valid_time[-1]
        time_values = [float(t) for _, _, t in valid_time if t is not None]
        time_trend = direction(time_values)
        if valid_neighbors:
            n_first_a, _, n_first = valid_neighbors[0]
            n_last_a, _, n_last = valid_neighbors[-1]
            neighbor_text = f"{fmt(n_first)} → {fmt(n_last)}"
            neighbor_change = pct_text(pct_change(n_first, n_last))
        else:
            neighbor_text = "không thu được"
            neighbor_change = "không tính được"
        lines.append(
            f"| {METHOD_LABELS[method]} | {first_a:g} → {last_a:g} | {neighbor_text} | {neighbor_change} | "
            f"{first_t:.5g} → {last_t:.5g} | {pct_text(pct_change(first_t, last_t))} | {time_trend} |"
        )

    kd_entries = h_trend_data.get("custom_kdtree", [])
    kd_neighbors = [(a, numeric(r, "mean_neighbors_per_query")) for a, r in kd_entries
                    if numeric(r, "mean_neighbors_per_query") is not None]
    kd_times = [(a, numeric(r, "median_time_s")) for a, r in kd_entries
                if numeric(r, "median_time_s") is not None]
    lines.append("")
    if len(kd_neighbors) >= 2:
        neighbor_values = [float(v) for _, v in kd_neighbors]
        lines.append(
            f"**Diễn giải KD-Tree:** số láng giềng trung bình {direction(neighbor_values)} theo các alpha đã đo; từ alpha={kd_neighbors[0][0]:g} đến alpha={kd_neighbors[-1][0]:g}, giá trị đổi từ {neighbor_values[0]:.4g} sang {neighbor_values[-1]:.4g} ({pct_text(pct_change(neighbor_values[0], neighbor_values[-1]) )})."
        )
    else:
        lines.append("**Diễn giải KD-Tree:** không đủ ít nhất hai giá trị `mean_neighbors_per_query` hợp lệ để kết luận xu hướng láng giềng khi h tăng. Hãy kiểm tra `e3_runs.csv` xem lượt diagnostics có ghi được bộ đếm hay không.")
    if len(kd_times) >= 2:
        time_values = [float(v) for _, v in kd_times]
        lines.append(
            f"Thời gian median của KD-Tree {direction(time_values)} theo alpha; từ alpha={kd_times[0][0]:g} đến alpha={kd_times[-1][0]:g}, thời gian đổi từ {time_values[0]:.5g} s sang {time_values[-1]:.5g} s ({pct_text(pct_change(time_values[0], time_values[-1]))})."
        )
    else:
        lines.append("Chưa đủ dữ liệu thời gian của KD-Tree qua nhiều mức alpha để kết luận xu hướng.")
    lines.append("Số láng giềng lớn hơn có thể làm giảm lợi thế của KD-Tree vì truy vấn cần trả về nhiều điểm hơn; chỉ kết luận cơ chế này phù hợp với kết quả nếu số láng giềng thực đo và thời gian cùng ủng hộ nhận xét, không suy ra quan hệ nhân quả chỉ từ lý thuyết.")

    lines.extend([])
    bin_rows: list[dict[str, float]] = []
    for n in sorted({int(float(v)) for v in n_values}):
        kd = get_row("n", "custom_kdtree", n=n)
        kdb = get_row("n", "custom_kdtree_bin", n=n)
        t0, t1 = numeric(kd, "median_time_s"), numeric(kdb, "median_time_s")
        s0, s1 = numeric(kd, "median_n_seeds"), numeric(kdb, "median_n_seeds")
        if t0 is None or t1 is None or t0 <= 0 or t1 <= 0:
            c0, c1 = numeric(kd, "median_n_clusters"), numeric(kdb, "median_n_clusters")
            lines.append(f"| {n} | {fmt_int(s0)} | {fmt_int(s1)} | — | — | {fmt_int(c0)} | {fmt_int(c1)} | {fmt(t0)} | {fmt(t1)} | — | — | — |")
            continue
        seed_saved = (s0 - s1) if s0 is not None and s1 is not None else None
        seed_pct = pct_change(s0, s1) if s0 is not None and s0 > 0 and s1 is not None else None
        seed_pct_saved = -seed_pct if seed_pct is not None else None
        seconds_saved = t0 - t1
        time_pct_saved = (seconds_saved / t0 * 100.0) if t0 > 0 else None
        speedup = t0 / t1
        c0, c1 = numeric(kd, "median_n_clusters"), numeric(kdb, "median_n_clusters")
        lines.append(
            f"| {n} | {fmt_int(s0)} | {fmt_int(s1)} | {fmt_int(seed_saved)} | {pct_text(seed_pct_saved)} | "
            f"{fmt_int(c0)} | {fmt_int(c1)} | {t0:.5g} | {t1:.5g} | {seconds_saved:+.5g} | {pct_text(time_pct_saved)} | {speedup:.3f} |"
        )
        bin_rows.append({
            "n": float(n), "time_without": t0, "time_with": t1, "seconds_saved": seconds_saved,
            "time_pct_saved": time_pct_saved if time_pct_saved is not None else float("nan"),
            "seed_without": s0 if s0 is not None else float("nan"),
            "seed_with": s1 if s1 is not None else float("nan"),
            "seed_saved": seed_saved if seed_saved is not None else float("nan"),
            "seed_pct_saved": seed_pct_saved if seed_pct_saved is not None else float("nan"),
            "speedup": speedup,
        })

    if bin_rows:
        seed_rows = [r for r in bin_rows if math.isfinite(r["seed_pct_saved"])]
        time_rows = [r for r in bin_rows if math.isfinite(r["time_pct_saved"])]
        if seed_rows:
            best_seed = max(seed_rows, key=lambda r: r["seed_pct_saved"])
            if best_seed["seed_pct_saved"] >= 0:
                seed_sentence = "Mức giảm seed lớn nhất theo tỷ lệ"
            else:
                seed_sentence = "Không ghi nhận giảm seed ở các mức n có thể so sánh; mức thay đổi ít bất lợi nhất theo tỷ lệ"
            lines.append(
                f"\n{seed_sentence} là {best_seed['seed_pct_saved']:+.1f}% tại n={int(best_seed['n']):,} "
                f"(từ {best_seed['seed_without']:.0f} xuống {best_seed['seed_with']:.0f} seed)."
            )
        if time_rows:
            best_time = max(time_rows, key=lambda r: r["time_pct_saved"])
            if best_time["time_pct_saved"] >= 0:
                time_sentence = "Mức tiết kiệm thời gian lớn nhất theo tỷ lệ"
            else:
                time_sentence = "Bin seeding không tiết kiệm thời gian ở các mức n so sánh được; trường hợp ít chậm hơn theo tỷ lệ"
            lines.append(
                f"{time_sentence} là {best_time['time_pct_saved']:+.1f}% tại n={int(best_time['n']):,} "
                f"({best_time['seconds_saved']:+.5g} s; speedup={best_time['speedup']:.3f}×). Số âm nghĩa là bin seeding chậm hơn KD-Tree không bin ở cấu hình đó."
            )
        lines.append("Cần đọc đồng thời cột số cụm: giảm seed/tăng tốc không tự động có nghĩa là kết quả phân cụm giữ nguyên. Bảng E3 ghi số cụm để đối chiếu; đánh giá sai khác chất lượng phân cụm chi tiết thuộc các thực nghiệm phù hợp như E1/E6.")
    else:
        lines.append("Chưa đủ các cặp thời gian hợp lệ của KD-Tree có/không bin seeding để định lượng mức giảm. Xem trạng thái timeout/error trong `e3_runs.csv`.")
    lines.extend([])
    report_path.write_text("\n".join(lines), encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser(description="E3 Mean-Shift runtime/memory benchmark")
    parser.add_argument("--config", default="configs/e3_runtime.yaml", help="Đường dẫn YAML tương đối repository root hoặc đường dẫn tuyệt đối")
    parser.add_argument("--repeats", type=int, default=None, help="Số lượt đo/cấu hình; mặc định theo YAML (3 theo phân công C)")
    parser.add_argument("--timeout-seconds", type=float, default=None, help="Timeout cho mỗi lần fit; mặc định theo YAML (600 giây)")
    parser.add_argument("--quick", action="store_true", help="Chạy kiểm tra rút gọn, không dùng làm kết quả báo cáo cuối")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = load_config(config_path)

    out_dir = Path(cfg.get("outputs", {}).get("directory", "results/e3"))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(cfg.get("outputs", {}).get("report", "report/THUC_NGHIEM_E3.md"))
    if not report_path.is_absolute():
        report_path = ROOT / report_path

    repeats_cfg = int(cfg.get("benchmark", {}).get("repeats", 3))
    repeats = int(args.repeats) if args.repeats is not None else repeats_cfg
    timeout = float(args.timeout_seconds if args.timeout_seconds is not None
                    else cfg.get("benchmark", {}).get("timeout_seconds", 600))
    if repeats < 1:
        parser.error("--repeats phải >= 1")
    if timeout <= 0:
        parser.error("--timeout-seconds phải > 0")

    n_values = [int(n) for n in cfg["n_sweep"]["values"]]
    h_n = int(cfg["h_sweep"]["n"])
    alphas = [float(a) for a in cfg["h_sweep"]["alphas"]]
    if args.quick:
        n_values = n_values[:2]
        h_n = min(h_n, max(n_values))
        alphas = alphas[:2]
        if args.repeats is None:
            repeats = 1
        timeout = min(timeout, 60.0)

    from meanshift import estimate_bandwidth
    ds = cfg["dataset"]
    reference_n = int(ds.get("reference_n", 2000))
    X_ref, _ = make_blobs(
        n_samples=reference_n,
        centers=int(ds.get("centers", 5)),
        cluster_std=float(ds.get("cluster_std", 1.0)),
        random_state=int(ds.get("random_state", 0)),
        n_features=int(ds.get("dimensions", 2)),
    )
    h0 = float(estimate_bandwidth(
        X_ref,
        quantile=float(ds.get("bandwidth_quantile", 0.1)),
        n_samples=ds.get("bandwidth_n_samples", None),
        random_state=int(ds.get("random_state", 0)),
    ))
    if not math.isfinite(h0) or h0 <= 0:
        raise RuntimeError(f"estimate_bandwidth trả về h0 không hợp lệ: {h0}")

    print(f"Repository root: {ROOT}")
    print(f"Config: {config_path}")
    print(f"h0={h0:.8g}; n sweep={n_values}; h sweep n={h_n}, alpha={alphas}")
    print(f"Repeats={repeats}; timeout={timeout:g}s; output={out_dir}")
    env = write_environment(out_dir)

    raw_path = out_dir / "e3_runs.csv"
    all_rows: list[dict[str, Any]] = []
    with raw_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        f.flush()
        all_rows.extend(_run_sweep_n(writer, f, cfg, h0, n_values, repeats, timeout))
        all_rows.extend(_run_sweep_h(writer, f, cfg, h0, h_n, alphas, repeats, timeout))

    summary_rows = summarize(all_rows)
    write_summary(out_dir / "e3_summary.csv", summary_rows)
    create_plots(summary_rows, out_dir)

    metadata = {
        "task": "E3 — runtime and memory",
        "config": cfg,
        "config_file": config_path.relative_to(ROOT).as_posix() if config_path.is_relative_to(ROOT) else str(config_path),
        "h0": h0,
        "repeats": repeats,
        "timeout_seconds": timeout,
        "quick_mode": bool(args.quick),
        "methods": METHODS,
        "method_labels": METHOD_LABELS,
        "environment": env,
        "raw_csv": raw_path.relative_to(ROOT).as_posix() if raw_path.is_relative_to(ROOT) else str(raw_path),
        "summary_csv": (out_dir / "e3_summary.csv").relative_to(ROOT).as_posix() if out_dir.is_relative_to(ROOT) else str(out_dir / "e3_summary.csv"),
        "measurement_protocol": {
            "time": "repeated fit runs with perf_counter; tracemalloc off; no instrumentation wrappers",
            "memory": "one separate fit run with tracemalloc; no instrumentation wrappers",
            "diagnostics": "one separate fit run with wrappers for neighbor and iteration statistics; its elapsed time is not used",
        },
        "note": "tracemalloc peak is Python-traced memory, not complete process RSS. Bandwidth quantile defaults to 0.1; inspect median_n_clusters in the report.",
    }
    (out_dir / "e3_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, cfg, h0, env, summary_rows, repeats, timeout)

    successful_time_runs = sum(
        1 for r in all_rows
        if r.get("measurement") == "time" and r.get("status") == "ok"
    )
    successful_memory_runs = sum(
        1 for r in all_rows
        if r.get("measurement") == "memory" and r.get("status") == "ok"
    )
    successful_diagnostic_runs = sum(
        1 for r in all_rows
        if r.get("measurement") == "diagnostics" and r.get("status") == "ok"
    )
    timeouts = sum(1 for r in all_rows if r.get("status") == "timeout")
    errors = sum(1 for r in all_rows if r.get("status") == "error")
    print("\nE3 hoàn tất.")
    print(f"- Lượt đo thời gian thành công: {successful_time_runs}")
    print(f"- Lượt đo bộ nhớ thành công: {successful_memory_runs}")
    print(f"- Lượt chẩn đoán thành công: {successful_diagnostic_runs}")
    print(f"- Timeout: {timeouts}; lỗi: {errors}")
    print(f"- Raw: {raw_path}")
    print(f"- Summary: {out_dir / 'e3_summary.csv'}")
    print(f"- Hình: {out_dir / 'e3_time_vs_n.png'}, {out_dir / 'e3_memory_vs_n.png'}, {out_dir / 'e3_time_vs_h.png'}")
    print(f"- Environment: {out_dir / 'env.txt'}")
    print(f"- Report: {report_path}")
    return 0 if successful_time_runs > 0 else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
