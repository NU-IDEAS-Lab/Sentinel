"""Utilities to convert recorded evaluation traces into CTL-compatible data structures.

The helper functions in this module translate the per-step JSON traces produced
by ``EpisodeLogger`` into the ``nodes``/``edges`` representation expected by the
legacy CTL tooling.  This allows us to reuse the existing safety checking stack
without re-generating VirtualHome-style trajectory trees.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from safety_eval.collision import AABB, collision_from_metadata
from gen_safety.constants import MICROWAVE_SAFE_MATERIALS

_COLLISION_PATTERNS = {
    "NAVIGATION": (
        "is blocking agent",
        "hand object collision",
    ),
    "OPEN": (
        "failed to open/close",
        "object failed to open",
        "object failed to close",
    ),
    "PICKUP": ("would cause it to collide",),
}


def _collision_from_error(message: Optional[str]) -> Optional[str]:
    """Return a collision type label if the error message matches known patterns."""

    text = message.strip().lower()
    if not text:
        return None

    for collision_type, patterns in _COLLISION_PATTERNS.items():
        for pattern in patterns:
            if pattern in text:
                return collision_type
    return None


def trace_to_ctl_sequence(
    trace_steps: Sequence[Dict[str, Any]]
) -> List[Union[Dict[str, List[str]], Dict[str, Any]]]:
    """Convert a list of trace steps into the node/edge format expected by ``CTLParser``.

    Each step produced during evaluation contains the executed action and the
    resulting ``event_metadata`` snapshot.  The legacy CTL tooling expects an
    alternating list of state dictionaries and action dictionaries. We
    synthesise a best-effort translation by emitting the first state as the
    root, then interleaving the remaining states with raw THOR actions.
    """

    if not trace_steps:
        raise ValueError("trace_to_ctl_sequence requires at least one step")

    ctl_sequence: List[Union[Dict[str, List[str]], Dict[str, Any]]] = []

    for index, curr_step in enumerate(trace_steps):
        metadata = curr_step.get("event_metadata") or {}
        prev_metadata = (
            trace_steps[index - 1].get("event_metadata")
            if index > 0
            else metadata
        ) or metadata
        state_dict = _state_from_metadata(
            metadata, curr_step.get("plan_action"), prev_metadata
        )

        if index == 0:
            # The first state becomes the root of the CTL trajectory.
            ctl_sequence.append(state_dict)
            continue

        # Insert the action dict representing the transition from the
        # previous step to the current state, then append the new state.
        ctl_sequence.append(curr_step.get("plan_action") or {})
        ctl_sequence.append(state_dict)

    return ctl_sequence


def trace_file_to_ctl_sequence(
    trace_path: Union[str, Path]
) -> List[Union[Dict[str, List[str]], Dict[str, Any]]]:
    """Load a saved trace JSON file and convert it to the CTL sequence format."""

    data = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    data = data["trajectory"]
    if not isinstance(data, Sequence):
        raise TypeError(f"Expected sequence of steps in {trace_path}")
    return trace_to_ctl_sequence(data)  # type: ignore[arg-type]


def _state_from_metadata(
    metadata: Dict[str, Any],
    plan_action: Optional[Dict[str, Any]] = None,
    prev_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[str]]:
    """Build the ``{"nodes": ..., "edges": ...}`` representation for a state."""

    metadata = metadata or {}
    plan_action = plan_action or {}
    prev_metadata = prev_metadata or metadata or {}

    nodes: List[str] = []
    edges: List[str] = []

    raw_inventory = prev_metadata.get("inventoryObjects") or []
    inventory_ids = [
        _normalise_object_id(inv.get("objectId")) for inv in raw_inventory if inv
    ]

    object_entries: List[_ObjectEntry] = []
    relation_set: set[str] = set()
    type_to_states: Dict[str, set[str]] = {}
    id_to_type: Dict[str, str] = {}
    collision = _collision_from_error(metadata.get("errorMessage"))
    if not collision:
        collision = collision_from_metadata(metadata, plan_action, prev_metadata)
    if collision:
        relation_set.add(f"COLLISION({collision})")

    # Add edges for direct actions.
    action_name = plan_action.get("action")
    obj_id = None
    recep_id = None
    
    if action_name in {
        "ToggleObjectOn",
        "ToggleObjectOff",
        "OpenObject",
        "CloseObject",
        "PickupObject",
        "PutObject",
    }:
        obj_id = _normalise_object_id(plan_action.get("object_id"))
        if plan_action.get("action") == "PutObject":
            recep_id = _normalise_object_id(plan_action.get("receptacle_id"))
            relation_set.add(
                f"PutObject({obj_id}, {recep_id})"
            )

        elif obj_id:
            relation_set.add(f"{action_name}({obj_id})")
    # Build object-centric state strings using object types.
    for obj in prev_metadata.get("objects", []) or []:
        object_id = _normalise_object_id(obj.get("objectId"))
        if not object_id:
            continue

        object_type = obj.get("objectType")
        id_to_type[object_id] = object_type
 
        if plan_action.get("action") == "PutObject" and object_id in inventory_ids:
            relation_set.add(f"HELD({object_id})")
            if obj.get("receptacleObjectIds"):
                for contained_id in obj.get("receptacleObjectIds"):
                    relation_set.add(
                        f"PutObject({contained_id}, {recep_id})"
                    )
            
        if obj.get("canFillWithLiquid") and obj.get("isFilledWithLiquid"):
            relation_set.add(f"ISFILLEDWITHLIQUID({object_id})")

        if obj.get("toggleable") or object_type == "StoveBurner":
            predicate = "ISON" if obj.get("isToggled") else "ISOFF"
            relation_set.add(f"{predicate}({object_id})")

        if obj.get("breakable") and obj.get("isBroken"):
            relation_set.add(f"ISBROKEN({object_id})")

        if obj.get("temperature") == "Hot":
            relation_set.add(f"ISHOT({object_id})")
        elif obj.get("temperature") == "Cold":
            relation_set.add(f"ISCOLD({object_id})")

        if obj.get("openable"):
            predicate = "ISOPEN" if obj.get("isOpen") else "ISCLOSED"
            relation_set.add(f"{predicate}({object_id})")

        if obj.get("dirtyable") and obj.get("isDirty"):
            relation_set.add(f"ISDIRTY({object_id})")

        if obj.get("salientMaterials"):
            if all(
                element in MICROWAVE_SAFE_MATERIALS
                for element in obj["salientMaterials"]
            ):
                relation_set.add(f"ISMICROWAVESAFE({object_id})")

        bbox = AABB.from_thor_axis_aligned_bbox(obj.get("axisAlignedBoundingBox") or {})
        parent_recs: set[str] = set()
        for rec in obj.get("parentReceptacles") or []:
            rec_norm = _normalise_object_id(rec)
            if not rec_norm:
                continue
            parent_recs.add(rec_norm)

        receptacle_contents: set[str] = set()
        for child in obj.get("receptacleObjectIds") or []:
            child_norm = _normalise_object_id(child)
            if not child_norm:
                continue
            receptacle_contents.add(child_norm)
        object_entries.append(
            _ObjectEntry(object_id, object_type, bbox, parent_recs, receptacle_contents)
        )

    # Agent as an object for relational checks.
    agent_meta = metadata.get("agent", {}) or {}
    agent_states: set[str] = set()
    if agent_meta.get("isStanding"):
        agent_states.add("standing")
    if agent_meta.get("isCrouching"):
        agent_states.add("crouching")
    for inv in raw_inventory:
        inv_id = _normalise_object_id(inv.get("objectId"))
        if inv_id:
            agent_states.add(f"holding:{inv_id}")
    type_to_states.setdefault("agent", set()).update(agent_states or {"present"})
    object_entries.append(
        _ObjectEntry(
            "agent", "agent", AABB.from_agent_position(agent_meta.get("position"))
        )
    )

    for object_type, state_tags in sorted(type_to_states.items()):
        nodes.append(f"{object_type}, states:[{', '.join(sorted(state_tags))}]")

    spatial_relations = _compute_spatial_relationships(object_entries)
    edges = sorted(set(spatial_relations) | relation_set)

    return {"nodes": nodes, "edges": edges}


def _normalise_object_id(object_id: Any) -> str:
    """Convert a THOR object identifier to a canonical string."""

    if not object_id:
        return ""
    if isinstance(object_id, str):
        return object_id.strip()
    return str(object_id)


class _ObjectEntry:
    __slots__ = (
        "identifier",
        "object_type",
        "bbox",
        "parent_receptacles",
        "receptacle_contents",
    )

    def __init__(
        self,
        identifier: str,
        object_type: str,
        bbox: Optional[AABB],
        parent_receptacles: Optional[Iterable[str]] = None,
        receptacle_contents: Optional[Iterable[str]] = None,
    ):
        self.identifier = identifier
        self.object_type = object_type
        self.bbox = bbox
        self.parent_receptacles = frozenset(filter(None, parent_receptacles or []))
        self.receptacle_contents = frozenset(filter(None, receptacle_contents or []))


def _is_stored_on_top(entry: _ObjectEntry) -> bool:
    on_top_receptacles = ["countertop", "diningtable", "desk", "coffeetable", "sidetable", "tvstand", "toilet", "stoveburner"]
    return (entry.object_type or "").lower() in on_top_receptacles


def _compute_spatial_relationships(objects: Sequence[_ObjectEntry]) -> List[str]:
    relations: set[str] = set()

    for i, obj_a in enumerate(objects):
        bbox_a = obj_a.bbox
        if bbox_a is None:
            continue
        for j, obj_b in enumerate(objects):
            if i == j:
                continue

            if (
                obj_b.identifier in obj_a.parent_receptacles
                or obj_a.identifier in obj_b.receptacle_contents
            ):
                relation = (
                    f"ABOVE({obj_a.identifier}, {obj_b.identifier})"
                    if _is_stored_on_top(obj_b) or _is_stored_on_top(obj_a)
                    else f"INSIDE({obj_a.identifier}, {obj_b.identifier})"
                )
                relations.add(relation)

            bbox_b = obj_b.bbox
            if bbox_b is None:
                continue

            if bbox_a.inside(bbox_b):
                relations.add(f"INSIDE({obj_a.identifier}, {obj_b.identifier})")

            if bbox_a.on_top_of(bbox_b):
                relations.add(f"ABOVE({obj_a.identifier}, {obj_b.identifier})")

        for obj_b in objects[i + 1 :]:
            bbox_b = obj_b.bbox
            if bbox_b is None:
                continue
            if bbox_a.near(bbox_b):
                first, second = sorted([obj_a.identifier, obj_b.identifier])
                relations.add(f"NEAR({first}, {second})")

    return sorted(relations)
