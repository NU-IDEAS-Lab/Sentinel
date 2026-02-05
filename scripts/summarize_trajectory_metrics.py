#!/usr/bin/env python3
"""Summarize trajectory metrics per model.

This script scans trajectory rollout files under `logs/trajectories` and reports,
for each model directory, how many trajectories completed successfully and how
many contain no error messages.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Set, Tuple


@dataclass
class TrajectoryMetrics:
    total: int = 0
    success: int = 0
    valid: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "total_trajs": self.total,
            "success_trajs": self.success,
            "valid_trajs": self.valid,
        }


def iter_traj_files(root: Path) -> Iterator[Tuple[str, Path]]:
    """Yield (model_key, path) pairs for every r0_*.json under root."""
    for path in root.rglob("*.json"):
        if path.name.startswith("ctl"):
            continue
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            # Should not happen, but guard against it.
            rel = path.name
        parts = rel.parts if isinstance(rel, Path) else tuple(rel.split("/"))
        if len(parts) >= 2:
            model_key = "/".join(parts[:2])
        else:
            model_key = parts[0]
        yield model_key, path


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_success(payload: Dict) -> bool:
    return payload.get("success", False)


def check_valid(payload: Dict) -> bool:
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, list):
        return False
    for step in trajectory:
        if not isinstance(step, dict):
            return False
        stepSuccess = step.get("success", False)
        if stepSuccess is False:
            return False
        # message = step.get("errorMessage", "")
    return True


def summarize(root: Path) -> Tuple[Dict[str, TrajectoryMetrics], Set[Path]]:
    metrics: Dict[str, TrajectoryMetrics] = {}
    json_dirs: Set[Path] = set()
    for model_key, path in iter_traj_files(root):
        stats = metrics.setdefault(model_key, TrajectoryMetrics())
        stats.total += 1
        json_dirs.add(path.parent)
        try:
            payload = load_json(path)
        except Exception:
            # Treat unreadable files as neither success nor valid, but they still count toward total.
            continue
        if check_success(payload):
            stats.success += 1
        if check_valid(payload):
            stats.valid += 1
    return metrics, json_dirs


def find_log_only_trials(root: Path, json_dirs: Set[Path]) -> Iterable[Path]:
    """
    Return the subdirectories that contain only log (.txt) files but no traj JSON.
    """
    log_only: Set[Path] = set()
    for log_file in root.rglob("*.txt"):
        if not log_file.is_file():
            continue
        parent = log_file.parent
        if parent not in json_dirs:
            try:
                rel_path = log_file.parent.relative_to(root)
            except ValueError:
                rel_path = log_file.parent
            log_only.add(rel_path)
    return sorted(log_only)


def print_summary(metrics: Dict[str, TrajectoryMetrics]) -> None:
    if not metrics:
        print("No trajectory files found.")
        return

    header = f"{'Model':40}  {'Total':>8}  {'Success':>8}  {'Valid':>8}"
    print(header)
    print("-" * len(header))
    for model_key in sorted(metrics):
        stats = metrics[model_key]
        print(f"{model_key:40}  {stats.total:8d}  {stats.success:8d}  {stats.valid:8d}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize trajectory metrics per model")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("logs/trajectories"),
        help="Root directory containing model trajectory logs (default: logs/trajectories)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to dump the summary as JSON",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    metrics, json_dirs = summarize(root)
    print_summary(metrics)

    log_only_trials = find_log_only_trials(root, json_dirs)
    if log_only_trials:
        print("\nTrials with logs but missing traj JSON:")
        for trial_dir in log_only_trials:
            print(f"  - {trial_dir}")

    if args.json:
        payload = {key: stats.as_dict() for key, stats in sorted(metrics.items())}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nSummary written to {args.json}")


if __name__ == "__main__":
    main()
