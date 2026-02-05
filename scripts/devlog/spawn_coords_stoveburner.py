"""
Purpose: Test GetSpawnCoordinatesAboveReceptacle behavior on StoveBurner.
Date: 2026-01-21
Run: python scripts/devlog/spawn_coords_stoveburner.py
Expected: prints coord counts before/after placement + outside-placement success rate.
Notes: See scripts/devlog/README.md#spawn-coords for context.
Github Issue: https://github.com/allenai/ai2thor/issues/1306
"""

from typing import Dict, List, Optional, Tuple

import _path_setup  # noqa: F401

from env.thor_env import ThorEnv
from gen_safety.constants import FLAMMABLE_OBJECT_TYPES


def _get_spawn_coords(env: ThorEnv, receptacle_id: str) -> List[Dict]:
    event = env.step(
        action="GetSpawnCoordinatesAboveReceptacle",
        objectId=receptacle_id,
        anywhere=True,
    )
    if not event.metadata.get("lastActionSuccess", False):
        raise RuntimeError(
            f"GetSpawnCoordinatesAboveReceptacle failed: {event.metadata.get('errorMessage')}"
        )
    return event.metadata.get("actionReturn") or []


def _coord_key(coord: Dict) -> Tuple[float, float, float]:
    return (
        round(coord["x"], 3),
        round(coord["y"], 3),
        round(coord["z"], 3),
    )


def _pick_object(
    meta: Dict, preferred_types: Optional[List[str]] = None
) -> Optional[Dict]:
    for obj in meta.get("objects", []):
        if not obj.get("pickupable", False):
            continue
        if preferred_types and obj.get("objectType") not in preferred_types:
            continue
        return obj
    if preferred_types:
        return _pick_object(meta, preferred_types=None)
    return None


def main() -> None:
    env = ThorEnv()
    env.reset(1)
    meta = env.step("Pass").metadata

    burner = next(
        (
            obj
            for obj in meta.get("objects", [])
            if obj.get("objectType") == "StoveBurner"
        ),
        None,
    )
    if not burner:
        print("No StoveBurner found in scene 1.")
        return

    obj_a = _pick_object(meta, FLAMMABLE_OBJECT_TYPES)
    obj_b = _pick_object(
        meta,
        [t for t in FLAMMABLE_OBJECT_TYPES if obj_a and t != obj_a.get("objectType")],
    )
    if not obj_a or not obj_b:
        print("Not enough pickupable objects found for placement test.")
        return

    burner_id = burner["objectId"]
    coords_before = _get_spawn_coords(env, burner_id)
    print(f"Spawn coords before placement: {len(coords_before)}")
    if coords_before:
        print(f"Sample coord: {coords_before[0]}")

    if not coords_before:
        print("No spawn coordinates returned; cannot test placement.")
        return

    target_coord = coords_before[0]
    place_meta = env.step(
        action="PlaceObjectAtPoint",
        objectId=obj_a["objectId"],
        position=target_coord,
    ).metadata
    print(
        f"Placed {obj_a['objectType']} at spawn coord: {place_meta.get('lastActionSuccess', False)}"
    )

    coords_after = _get_spawn_coords(env, burner_id)
    print(f"Spawn coords after placement: {len(coords_after)}")

    before_keys = {_coord_key(c) for c in coords_before}
    after_keys = {_coord_key(c) for c in coords_after}
    removed = before_keys - after_keys
    added = after_keys - before_keys
    print(f"Coords removed after placement: {len(removed)}")
    print(f"Coords added after placement: {len(added)}")

    # Test positions outside the actionReturn list.
    base = target_coord
    offsets = [0.1, 0.15, 0.2, 0.25]
    tried = 0
    outside_success = 0
    for dy in offsets:
        candidate = {
            "x": base["x"],
            "y": base["y"] + dy,
            "z": base["z"],
        }
        if _coord_key(candidate) in before_keys:
            continue
        tried += 1
        meta_place = env.step(
            action="PlaceObjectAtPoint",
            objectId=obj_b["objectId"],
            position=candidate,
        ).metadata
        if meta_place.get("lastActionSuccess"):
            outside_success += 1
            print(f"Placed outside spawn list at {candidate}")
            for obj in meta_place.get("objects", []):
                if obj.get("objectId") == obj_b["objectId"]:
                    print(f"New position: {obj.get('position')}")

    print(f"Outside-placement attempts: {tried}, successes: {outside_success}")


if __name__ == "__main__":
    main()
