#!/usr/bin/env python3
"""List objects available in an AI2-THOR scene."""

import argparse
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List

from ai2thor.controller import Controller


def _normalize_scene(scene: str) -> str:
    if scene.isdigit():
        return f"FloorPlan{scene}"
    return scene


def _format_object(obj: Dict[str, Any], fields: Iterable[str]) -> str:
    pieces: List[str] = []
    for name in fields:
        value = obj.get(name, "<missing>")
        pieces.append(f"{name}={value}")
    return ", ".join(pieces)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", help="Scene name or number (e.g. FloorPlan201 or 201)")
    parser.add_argument("--visible-only", action="store_true", help="Only include currently visible objects")
    parser.add_argument(
        "--fields",
        default="objectId,objectType,assetId,visible,temperature",
        help="Comma separated metadata keys to display for each object",
    )
    parser.add_argument("--quality", default="MediumCloseFitShadows", help="Quality level passed to Controller")
    parser.add_argument("--width", type=int, default=600, help="Render width")
    parser.add_argument("--height", type=int, default=600, help="Render height")
    parser.add_argument("--x-display", default=None, help="Optional X display to target")
    parser.add_argument("--local-build", default=None, help="Path to a local Unity build")
    args = parser.parse_args(argv)

    scene_name = _normalize_scene(args.scene)
    controller_kwargs: Dict[str, Any] = {
        "width": args.width,
        "height": args.height,
        "quality": args.quality,
    }
    if args.x_display is not None:
        controller_kwargs["x_display"] = args.x_display
    if args.local_build:
        controller_kwargs["local_executable_path"] = args.local_build

    requested_fields = [field.strip() for field in args.fields.split(",") if field.strip()]

    with Controller(**controller_kwargs) as controller:
        event = controller.reset(scene_name)
        objects = event.metadata.get("objects", [])
        if args.visible_only:
            objects = [obj for obj in objects if obj.get("visible")]

        print(f"Scene: {scene_name}")
        print(f"Total objects: {len(objects)}")
        type_counts = Counter(obj.get("objectType", "<unknown>") for obj in objects)
        if type_counts:
            print("\nObject types:")
            for object_type, count in sorted(type_counts.items()):
                print(f"  {object_type}: {count}")

        if requested_fields:
            print("\nObjects:")
            for obj in objects:
                line = _format_object(obj, requested_fields)
                print(f"  - {line}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
