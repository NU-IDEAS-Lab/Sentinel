#!/usr/bin/env python3
"""
Rerun eval_llm_astar.py for trials whose compare log directory is missing JSON outputs.

Example:
    python scripts/rerun_missing_trials.py \
        --base logs/trajectories/openai/gpt-5-llm/atomic \
        --compare logs/trajectories/openai/gpt-5-llm-20260127-1345/atomic \
        --data-root data/json_2.1.0/atomic \
        --model openai/gpt-5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Set, List


def collect_trial_jsons(root: Path) -> Dict[str, Set[str]]:
    trials: Dict[str, Set[str]] = {}
    if not root.exists():
        return trials
    for task_dir in root.iterdir():
        if not task_dir.is_dir():
            continue
        for trial_dir in task_dir.glob("trial_*"):
            if not trial_dir.is_dir():
                continue
            rel = str(trial_dir.relative_to(root))
            json_files = {p.name for p in trial_dir.glob("*.json") if p.is_file()}
            trials[rel] = json_files
    return trials


def find_missing_trials(base_trials: Dict[str, Set[str]], compare_trials: Dict[str, Set[str]]) -> List[str]:
    missing: List[str] = []
    
    for trial in sorted(base_trials.keys()):
        compare_jsons = compare_trials.get(trial)
        if len(compare_jsons) < 5:
            missing.append(trial)
    return missing


def run_eval(traj_file: Path, model: str, extra_args: List[str], dry_run: bool = False, vlm: bool = False) -> int:
    cmd = [
        sys.executable,
        "models/eval/eval_llm_astar.py",
        "--debug",
        "--traj_file",
        str(traj_file),
        "--model",
        model,
    ]
    if vlm:
        cmd = [
        sys.executable,
        "models/eval/eval_vlm_step.py",
        "--debug",
        "--traj_file",
        str(traj_file),
        "--model",
        model,
    ]
    cmd.extend(extra_args)
    if dry_run:
        print("[DRY RUN]", " ".join(cmd))
        return 0
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        print(f"  ↳ command failed with exit code {completed.returncode}")
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/json_2.1.0/atomic"), help="Base log root (with reference traj_data.json copies).")
    parser.add_argument("--compare", type=Path, required=True, help="Compare log root to inspect.")
    parser.add_argument("--data-root", type=Path, default=Path("data/json_2.1.0/atomic"),
                        help="Root of the original data trajectories (e.g., data/json_2.1.0/atomic).")
    parser.add_argument("--model", type=str, default="openai/gpt-5", help="Model name passed to eval_llm_astar.py.")
    parser.add_argument("--extra-arg", action="append", default=[],
                        help="Additional arguments to forward to eval_llm_astar.py (repeatable).")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands without executing.")
    parser.add_argument("--vlm", action="store_true", help="Use VLM settings.")
    args = parser.parse_args()

    base_root = args.base.resolve()
    compare_root = args.compare.resolve()
    data_root = args.data_root.resolve()

    if not base_root.exists():
        raise SystemExit(f"Base directory not found: {base_root}")
    if not compare_root.exists():
        raise SystemExit(f"Compare directory not found: {compare_root}")
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    base_trials = collect_trial_jsons(base_root)
    compare_trials = collect_trial_jsons(compare_root)
    missing_trials = find_missing_trials(base_trials, compare_trials)

    if not missing_trials:
        print("No missing trials detected; nothing to rerun.")
        return

    print(f"Found {len(missing_trials)} trials missing JSON outputs in compare.")
    failures = 0
    for trial in missing_trials:
        traj_file = data_root / trial / "traj_data.json"
        if not traj_file.exists():
            print(f"✗ traj_data.json not found for {trial}: {traj_file}")
            failures += 1
            continue
        ret = run_eval(traj_file, args.model, args.extra_arg, dry_run=args.dry_run, vlm=args.vlm)
        if ret != 0:
            failures += 1

    if failures == 0:
        print("All reruns completed successfully.")
    else:
        print(f"Completed reruns with {failures} failures.")


if __name__ == "__main__":
    main()
