from env.thor_env import ThorEnv
from dataclasses import dataclass
from typing import Any, Dict, Iterable
import numpy as np


@dataclass(frozen=True)
class AABB:
    min_corners: np.ndarray  # shape (3,)
    max_corners: np.ndarray  # shape (3,)

    @staticmethod
    def from_corner_points(corner_points: Iterable[Iterable[float]]) -> "AABB":
        corners = np.asarray(list(corner_points), dtype=np.float32)
        if corners.shape != (8, 3):
            raise ValueError(f"Expected (8,3) cornerPoints, got {corners.shape}")
        return AABB(min_corners=corners.min(axis=0), max_corners=corners.max(axis=0))

    @staticmethod
    def from_thor_axis_aligned_bbox(bbox: Dict[str, Any]) -> "AABB":
        # AI2-THOR axisAlignedBoundingBox contains cornerPoints/center/size :contentReference[oaicite:1]{index=1}
        if "cornerPoints" not in bbox:
            raise KeyError("Expected bbox['cornerPoints'] in axisAlignedBoundingBox")
        return AABB.from_corner_points(bbox["cornerPoints"])

    def from_agent_position(position: Dict[str, float]) -> "AABB":
        pos = np.array([position["x"], position["y"], position["z"]], dtype=np.float32)
        half_size = np.array(
            [0.1, 0.9, 0.1], dtype=np.float32
        )  # approximate agent size
        return AABB(min_corners=pos - half_size, max_corners=pos + half_size)

    def inflate(self, pad: float) -> "AABB":
        p = np.array([pad, pad, pad], dtype=np.float32)
        return AABB(self.min_corners - p, self.max_corners + p)

    def intersects(self, other: "AABB", eps: float = 1e-6) -> bool:
        # touching counts as collision
        return bool(
            np.all(self.min_corners <= other.max_corners + eps)
            and np.all(self.max_corners + eps >= other.min_corners)
        )

    def union(self, other: "AABB") -> "AABB":
        return AABB(
            min_corners=np.minimum(self.min_corners, other.min_corners),
            max_corners=np.maximum(self.max_corners, other.max_corners),
        )


env = ThorEnv(gridSize=0.1)
env.reset("FloorPlan30")

before_open_meta = env.step(
    action="TeleportFull",
    position={"x": 2.5, "y": 0.9277887344360352, "z": -1.3},
    rotation={"x": -0.0, "y": 90.0, "z": 0.0},
    horizon=0,
    standing=True,
).metadata

cabinet_id = env.step("GetObjectInFrame", x=0.5, y=0.5).metadata["actionReturn"]
cabinet_name = [
    o for o in env.last_event.metadata["objects"] if o["objectId"] == cabinet_id
][0]["name"]

after_open_metadata = env.step(
    action="OpenObject",
    objectId=cabinet_id,
    forceAction=False,  # enable collision fails
    ignoreAgentInTransition=False,  # count agent body
    # stopAtNonStaticCol=True,         # count movable objects
    # raise_for_failure=True
).metadata
aabbox_closed = env.get_axis_aligned_bounding_box(cabinet_id, before_open_meta)
aabbox_opened = env.get_axis_aligned_bounding_box(cabinet_id, after_open_metadata)

agent_aabb = AABB.from_agent_position(after_open_metadata["agent"]["position"])
aabbox_closed = AABB.from_thor_axis_aligned_bbox(aabbox_closed)
aabbox_opened = AABB.from_thor_axis_aligned_bbox(aabbox_opened)

swept = aabbox_closed.union(aabbox_opened)
print("swept intersects agent:", agent_aabb.intersects(swept))
