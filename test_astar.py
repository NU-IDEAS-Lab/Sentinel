import numpy as np
from env.thor_env import ThorEnv
import ai2thor.controller

from nav_astar import (
    astar_actions,
    build_adjacency,
    build_node_lookup,
    jps_actions,
    nearest_key,
    normalize_yaw,
)

def main() -> None:
    """Initialize updated scenes and fetch metadata via a no-op step."""
    env = ThorEnv(gridSize=0.25)
    env.reset(1)
    positions = env.step(action="GetReachablePositions").metadata["actionReturn"]
    breakpoint()
    event = env.step({"action": "Pass"})
    agent_location = event.metadata["agent"]["position"]
    agent_rotation = event.metadata["agent"]["rotation"]


    event = env.step(
        action="GetInteractablePoses",
        objectId="Microwave|-00.24|+01.69|-02.53",
        positions=positions, # will let default rotation be used
        horizons=[float(h) for h in np.linspace(-30, 60, 30)],
        standings=[True, False],
    )

    poses = event.metadata["actionReturn"]
    if not poses:
        raise RuntimeError("No interactable poses returned for target object")
    target_position = poses[0]

    nodes = build_node_lookup(positions)
    target_key = nearest_key(nodes, target_position)
    start_key = nearest_key(nodes, agent_location)
    start_yaw = normalize_yaw(agent_rotation.get("y", 0.0))

    try:
        actions = jps_actions(nodes, start_key, start_yaw, target_key)
    except RuntimeError:
        adjacency = build_adjacency(nodes)
        actions = astar_actions(nodes, adjacency, start_key, start_yaw, target_key)
    print(actions)

    for action in actions:
        event = env.step({"action": action})
        if not event.metadata.get("lastActionSuccess", True):
            raise RuntimeError(f"{action} failed: {event.metadata.get('errorMessage')}")


if __name__ == "__main__":
    main()
