import argparse
import json
import os
import sys
from typing import Any, Dict, List, Union

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.thor_env import ThorEnv
from models.eval.eval_llm import EvalLLM
from models.model.llm import LLMAgent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reset AI2-THOR scenes, capture sanitized metadata, and query an LLM for safety constraints.",
    )
    parser.add_argument(
        "--floorplans",
        nargs="+",
        default=["1"],
        help="Scene identifiers to load (either integers or names like FloorPlan1).",
    )
    parser.add_argument(
        "--llm-model",
        default="deepseek/deepseek-chat-v3",
        help="LLM model identifier for OpenRouter (default: deepseek/deepseek-chat-v3).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Maximum number of tokens to request from the LLM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling parameter.",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        help="Frequency penalty for the LLM.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Presence penalty for the LLM.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional free-form notes to include in the LLM prompt.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save constraints and metadata as JSON.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print sanitized metadata before querying the LLM.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="If set, only print sanitized metadata without querying the LLM.",
    )
    return parser.parse_args()


def normalize_scene_names(floorplans: List[str]) -> List[str]:
    normalized = []
    for value in floorplans:
        if value.lower().startswith("floorplan"):
            normalized.append(value)
            continue
        try:
            normalized.append(f"FloorPlan{int(value)}")
        except ValueError:
            normalized.append(value)
    return normalized


def main():
    args = parse_args()
    scenes = normalize_scene_names(args.floorplans)

    # Ensure the args object has attributes expected by LLMAgent
    llm_args = argparse.Namespace(**vars(args))

    agent = LLMAgent(llm_args)
    agent.set_log_method(print if args.debug else lambda *_: None)

    env = ThorEnv()

    results: Dict[str, Dict[str, Any]] = {}

    try:
        for scene_name in scenes:
            print(f"\n=== {scene_name} ===")
            init_event = env.reset(scene_name)
            if not init_event.metadata.get("lastActionSuccess", False):
                error = init_event.metadata.get("errorMessage", "Initialize failed")
                print(f"Initialization failed: {error}")
                results[scene_name] = {"error": error}
                continue

            step_event = env.step({"action": "Pass"})
            metadata = step_event.metadata
            scene_info = EvalLLM.remove_useless_info(metadata)
            scene_info["sceneName"] = scene_name

            if args.debug:
                print("Sanitized metadata:")
                print(json.dumps(scene_info, indent=2, sort_keys=True))

            constraints: Union[Dict[str, Any], str, None]
            if args.skip_llm:
                constraints = "LLM query skipped."
            else:
                constraints = agent.generate_safety_constraints(
                    scene_info,
                    scene_name=scene_name,
                    notes=args.notes,
                )

            if constraints is None:
                print("No constraints generated (LLM returned no response).")
                results[scene_name] = {
                    "metadata": scene_info,
                    "constraints": None,
                }
                continue

            if isinstance(constraints, dict):
                print("Generated safety constraints:")
                print(json.dumps(constraints, indent=2))
            else:
                print("Generated safety constraints (raw response):")
                print(constraints)

            results[scene_name] = {
                "metadata": scene_info,
                "constraints": constraints,
            }
    finally:
        env.stop()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
