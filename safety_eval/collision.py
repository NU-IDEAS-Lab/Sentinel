from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class AABB:
    """Axis-aligned bounding box backed by numpy arrays."""

    min_corners: np.ndarray  # shape (3,)
    max_corners: np.ndarray  # shape (3,)

    @staticmethod
    def from_corner_points(
        corner_points: Iterable[Iterable[float]]
    ) -> Optional["AABB"]:
        corners = np.asarray(list(corner_points), dtype=np.float32)
        if corners.shape != (8, 3):
            return None
        return AABB(min_corners=corners.min(axis=0), max_corners=corners.max(axis=0))

    @staticmethod
    def from_min_max(
        minimum: Tuple[float, float, float], maximum: Tuple[float, float, float]
    ) -> "AABB":
        return AABB(
            min_corners=np.asarray(minimum, dtype=np.float32),
            max_corners=np.asarray(maximum, dtype=np.float32),
        )

    @staticmethod
    def from_thor_axis_aligned_bbox(bbox: Dict[str, Any]) -> Optional["AABB"]:
        if not bbox:
            return None
        corner_points = bbox.get("cornerPoints")
        if not corner_points:
            return None
        return AABB.from_corner_points(corner_points)

    @staticmethod
    def from_agent_position(position: Optional[Dict[str, float]]) -> Optional["AABB"]:
        if not position:
            return None
        pos = np.array(
            [position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0)],
            dtype=np.float32,
        )
        half_size = np.array(
            [0.1, 0.9, 0.1], dtype=np.float32
        )  # approximate agent size
        return AABB(min_corners=pos - half_size, max_corners=pos + half_size)

    def overlaps(self, other: "AABB", padding: float = 0.0) -> bool:
        pad = np.array([padding, padding, padding], dtype=np.float32)
        return bool(
            np.all(self.min_corners - pad <= other.max_corners + pad)
            and np.all(self.max_corners + pad >= other.min_corners - pad)
        )

    def near(self, other: "AABB", threshold: float = 0.5) -> bool:
        sep = np.maximum(
            0.0,
            np.maximum(self.min_corners, other.min_corners)
            - np.minimum(self.max_corners, other.max_corners),
        )
        return float(np.linalg.norm(sep)) <= threshold

    def horizontal_overlap(self, other: "AABB") -> bool:
        overlap = self.max_corners[[0, 2]] >= other.min_corners[[0, 2]]
        overlap &= other.max_corners[[0, 2]] >= self.min_corners[[0, 2]]
        return bool(np.all(overlap))

    def on_top_of(self, other: "AABB", vertical_epsilon: float = 0.08) -> bool:
        y_ok = (
            other.max_corners[1] - vertical_epsilon
            <= self.min_corners[1]
            <= other.max_corners[1] + vertical_epsilon
        )
        return y_ok and self.horizontal_overlap(other)

    def inside(self, other: "AABB", margin: float = 0.02) -> bool:
        return bool(
            np.all(self.min_corners >= other.min_corners - margin)
            and np.all(self.max_corners <= other.max_corners + margin)
        )

    def union(self, other: "AABB") -> "AABB":
        return AABB(
            min_corners=np.minimum(self.min_corners, other.min_corners),
            max_corners=np.maximum(self.max_corners, other.max_corners),
        )

    def inflate(self, pad: float) -> "AABB":
        p = np.array([pad, pad, pad], dtype=np.float32)
        return AABB(self.min_corners - p, self.max_corners + p)


def collision_from_metadata(
    metadata: Dict[str, Any],
    action: Optional[Dict[str, Any]],
    previous_metadata: Optional[Dict[str, Any]] = None,
    padding: float = 0.05,
) -> Optional[str]:
    """Check agent/object overlap for open/close actions based on AABB geometry."""

    if not action:
        return None

    action_name = (action.get("action") or "").lower()
    if action_name not in {"openobject", "closeobject"}:
        return None

    target_id = action.get("objectId")
    if not target_id:
        return None

    agent_bbox = AABB.from_agent_position((metadata.get("agent") or {}).get("position"))
    if agent_bbox is None:
        return None

    def _get_target_bbox(source_meta: Dict[str, Any]) -> Optional[AABB]:
        for obj in source_meta.get("objects") or []:
            if obj.get("objectId") == target_id:
                return AABB.from_thor_axis_aligned_bbox(
                    obj.get("axisAlignedBoundingBox") or {}
                )
        return None

    current_bbox = _get_target_bbox(metadata)
    prev_bbox = _get_target_bbox(previous_metadata) if previous_metadata else None

    target_swept = current_bbox
    if current_bbox and prev_bbox:
        target_swept = current_bbox.union(prev_bbox).inflate(padding)

    if target_swept and agent_bbox.overlaps(target_swept):
        return "CLOSE" if action_name == "closeobject" else "OPEN"
    return None
