#!/usr/bin/env python3
"""Normalize traj_data.json scene names and verify restoration.

For each traj_data.json under a data root, this script:
1) Resets the scene using the stored floor_plan.
2) Compares the initialized scene name returned by AI2-THOR with the JSON.
3) Updates the JSON if the names differ (unless --dry-run).
4) Attempts to restore the scene state from the JSON (object poses/toggles).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from env.thor_env import ThorEnv  # noqa: E402


TRAJ_JSON_NAME = "traj_data.json"


@dataclass
class Result:
    path: Path
    stored_name: Optional[str]
    initialized_name: Optional[str]
    name_changed: bool
    poses_updated: bool
    file_written: bool
    reset_error: Optional[str]
    restore_ok: bool
    restore_error: Optional[str]


def iter_traj_files(root: Path) -> Iterable[Path]:
    for path in root.rglob(TRAJ_JSON_NAME):
        if path.is_file():
            yield path


def try_restore_scene(env: ThorEnv, scene_info: dict) -> tuple[bool, Optional[str]]:
    """Restore poses/toggles from traj data; return (success, error)."""
    object_poses = scene_info.get("object_poses") or []
    object_toggles = scene_info.get("object_toggles") or []
    try:
        env.restore_scene(object_poses, object_toggles)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def process_traj(env: ThorEnv, path: Path, *, dry_run: bool) -> Result:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        return Result(
            path=path,
            stored_name=None,
            initialized_name=None,
            name_changed=False,
            poses_updated=False,
            file_written=False,
            reset_error=f"load failed: {exc}",
            restore_ok=False,
            restore_error=None,
        )

    scene_info = data.get("scene") or {}
    stored_name = scene_info.get("floor_plan")

    result = Result(
        path=path,
        stored_name=stored_name,
        initialized_name=None,
        name_changed=False,
        poses_updated=False,
        file_written=False,
        reset_error=None,
        restore_ok=False,
        restore_error=None,
    )

    if not stored_name:
        result.reset_error = "missing scene.floor_plan"
        return result

    try:
        event = env.reset(stored_name)
    except Exception as exc:  # noqa: BLE001
        result.reset_error = f"reset failed: {exc}"
        return result

    obj_list = (getattr(event, "metadata", {}) or {}).get("objects", [])
    object_names = {obj.get("name") for obj in obj_list if obj.get("name")}
    obj_poses = scene_info.get("object_poses") or []

    new_obj_poses = []
    if object_names and obj_poses:
        for obj in obj_poses:
            obj_name = obj.get("objectName")
            if obj_name in object_names:
                continue
            obj_name_split = obj_name.split("_")
            obj_type = obj_name_split[0]
            for name in object_names:
                if obj_type in name:
                    if obj_name != name:
                        obj["objectName"] = name
                        result.poses_updated = True
                    break
            new_obj_poses.append(obj)
    scene_info["object_poses"] = new_obj_poses
    task_desc_updated = False
    if not data.get("task_desc"):
        data["task_desc"] = data["turk_annotations"]["anns"][0]["task_desc"]
        task_desc_updated = True

    if (result.poses_updated or task_desc_updated) and not dry_run:
        data["scene"] = scene_info
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        result.file_written = True

    restore_ok, restore_error = try_restore_scene(env, scene_info)
    result.restore_ok = restore_ok
    result.restore_error = restore_error
    return result


def format_status(res: Result) -> str:
    if res.reset_error:
        return f"RESET ERROR ({res.reset_error})"
    pieces = []
    if res.name_changed:
        action = "updated" if res.file_written else "differs"
        pieces.append(f"{res.stored_name} -> {res.initialized_name} ({action})")
    else:
        pieces.append(f"{res.stored_name}")
    if res.poses_updated:
        pieces.append("poses=updated")

    if res.restore_ok:
        pieces.append("restore=ok")
    else:
        pieces.append(f"restore=failed ({res.restore_error})")
    return "; ".join(pieces)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directory to search for traj_data.json files (default: data/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write changes; only report differences and restore status",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optionally process only the first N traj files (for quick checks)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_root.exists():
        raise SystemExit(f"Data root not found: {args.data_root}")

    traj_files = list(iter_traj_files(args.data_root))
    if not traj_files:
        print(f"No {TRAJ_JSON_NAME} files found under {args.data_root}")
        return 0

    env = ThorEnv()
    results: list[Result] = []
    try:
        for idx, path in enumerate(traj_files, start=1):
            if args.limit and idx > args.limit:
                break
            res = process_traj(env, path, dry_run=args.dry_run)
            results.append(res)
            print(f"[{idx}] {path}: {format_status(res)}")
    finally:
        env.stop()

    total = len(results)
    reset_errors = sum(1 for r in results if r.reset_error)
    restore_failures = sum(1 for r in results if not r.restore_ok and not r.reset_error)
    written = sum(1 for r in results if r.file_written)
    name_diffs = sum(1 for r in results if r.name_changed)
    pose_updates = sum(1 for r in results if r.poses_updated)

    print("\nSummary:")
    print(f"  Processed: {total}")
    print(f"  Scene name diffs: {name_diffs} (written: {written})")
    print(f"  Pose name updates: {pose_updates}")
    print(f"  Reset errors: {reset_errors}")
    print(f"  Restore failures: {restore_failures}")

    if reset_errors or restore_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
