#!/usr/bin/env python3
"""Run eval_llm_astar on candle-related trajectories with optional parallelism."""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List


def _iter_task_dirs(root: Path, keyword: str) -> Iterable[Path]:
    keyword = keyword.lower()
    for child in sorted(root.iterdir()):
        if child.is_dir() and keyword in child.name.lower():
            yield child


def _iter_traj_files(task_dir: Path) -> Iterable[Path]:
    for trial_dir in sorted(task_dir.glob('trial_*')):
        traj_file = trial_dir / 'traj_data.json'
        if traj_file.is_file():
            yield traj_file


def _build_command(
    eval_script: Path,
    traj_file: Path,
    args: argparse.Namespace,
    repo_root: Path,
) -> List[str]:
    try:
        traj_arg = traj_file.relative_to(repo_root)
    except ValueError:
        traj_arg = traj_file

    cmd = [sys.executable, str(eval_script), '--traj_file', str(traj_arg)]
    if args.max_steps is not None:
        cmd += ['--max_steps', str(args.max_steps)]
    if args.max_fails is not None:
        cmd += ['--max_fails', str(args.max_fails)]
    if args.smooth_nav:
        cmd.append('--smooth_nav')
    if args.debug:
        cmd.append('--debug')
    if args.model is not None:
        cmd += ['--model', args.model]
    if args.max_tokens is not None:
        cmd += ['--max_tokens', str(args.max_tokens)]
    if args.temperature is not None:
        cmd += ['--temperature', str(args.temperature)]
    if args.top_p is not None:
        cmd += ['--top_p', str(args.top_p)]
    if args.frequency_penalty is not None:
        cmd += ['--frequency_penalty', str(args.frequency_penalty)]
    if args.presence_penalty is not None:
        cmd += ['--presence_penalty', str(args.presence_penalty)]
    return cmd


def _run_command(cmd: List[str]) -> int:
    print('Running:', ' '.join(cmd))
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        print(f"  ↳ command failed with exit code {completed.returncode}")
    return completed.returncode


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=repo_root / 'data' / 'json_2.1.0' / 'atomic',
                        help='Root directory containing task folders')
    parser.add_argument('--pattern', type=str, default='candle',
                        help='Substring to match task folder names')
    parser.add_argument('--eval-script', type=Path, default=repo_root / 'models' / 'eval' / 'eval_llm_astar.py',
                        help='Path to eval_llm_astar.py')
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--max_fails', type=int, default=None)
    parser.add_argument('--smooth_nav', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--max_tokens', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--frequency_penalty', type=float, default=None)
    parser.add_argument('--presence_penalty', type=float, default=None)
    parser.add_argument('--dry_run', action='store_true',
                        help='Only print the commands that would be executed')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--vlm', action='store_true', help='Use VLM model for evaluation')

    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if args.vlm:
        args.eval_script = repo_root / 'models' / 'eval' / 'eval_vlm_step.py'
    eval_script = args.eval_script.resolve()

    if not data_root.exists():
        print(f"✗ data root not found: {data_root}")
        return 1
    if not eval_script.exists():
        print(f"✗ eval script not found: {eval_script}")
        return 1

    commands: List[List[str]] = []
    for task_dir in _iter_task_dirs(data_root, args.pattern):
        for traj_file in _iter_traj_files(task_dir):
                commands.append(_build_command(eval_script, traj_file, args, repo_root))

    if not commands:
        print('No matching trajectories found.')
        return 0

    if args.dry_run:
        for cmd in commands:
            print('DRY RUN:', ' '.join(cmd))
        print(f"Prepared {len(commands)} commands (dry run).")
        return 0

    failures = 0
    workers = max(1, args.workers)
    if workers == 1:
        for cmd in commands:
            ret = _run_command(cmd)
            if ret != 0:
                failures += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_cmd = {executor.submit(_run_command, cmd): cmd for cmd in commands}
            for future in as_completed(future_to_cmd):
                ret = future.result()
                if ret != 0:
                    failures += 1

    print(f"Completed {len(commands)} runs with {failures} failures.")
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
