"""Quick smoke test to ensure the Thor environment loads correctly."""

import os
import sys

# Allow running this file directly via `python scripts/check_thor.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai2thor.controller import Controller
from env.thor_env import ThorEnv

env = ThorEnv(headless=True)
event = env.step(dict(action="MoveAhead"))
assert event.metadata["lastActionSuccess"], "MoveAhead action failed in Thor environment."
print("Everything works!!!")