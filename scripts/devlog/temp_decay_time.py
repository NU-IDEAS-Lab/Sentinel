"""
Purpose: Test if SetRoomTempDecayTimeForType works.
Date: 2026-01-22
Run: python scripts/devlog/temp_decay_time.py
Expected: Runs with no error.
Notes: See scripts/devlog/README.md#heatable-near-stove for context.
"""


import _path_setup  # noqa: F401

from env.thor_env import ThorEnv


def main() -> None:
    env = ThorEnv()
    scene_num = 26
    env.reset(scene_num)
    object_toggles = [
        {
            "action": "SetRoomTempDecayTimeForType",
            "objectType": "Apple",
            "TimeUntilRoomTemp": 60.0,
        }
    ]
    for step in object_toggles:
        metadata = env.step(**step).metadata
        if not metadata.get("lastActionSuccess"):
            raise RuntimeError(
                f"failed to perform action {step['action']} on {step['objectId']}"
            )


if __name__ == "__main__":
    main()
