import os
import sys
import math
import numpy as np
from typing import Dict, Optional, Tuple

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gen.constants as constants
from gen.graph.graph_obj import Graph
from models.eval.eval_llm import EvalLLM
from models.model.llm_astar import LLMAstar


class EvalLLMAstar(EvalLLM):
    """EvalLLM variant that expands GotoLocation into executable actions."""

    def __init__(self, args, manager=None):
        super().__init__(args, manager)
        self.llm_agent = LLMAstar(args)
        self.llm_agent.set_log_method(self.log)
        self._graph: Optional[Graph] = None
        self._graph_scene: Optional[int] = None

    def execute_action(self, env, action_dict, smooth_nav=False):  # type: ignore[override]
        action_name = action_dict.get('action')
        if action_name == 'GotoLocation':
            return self._execute_goto(env, action_dict, smooth_nav=smooth_nav)
        return super().execute_action(env, action_dict, smooth_nav=smooth_nav)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------
    def _execute_goto(self, env, action: Dict, smooth_nav: bool):
        metadata = env.last_event.metadata if env.last_event else {}
        target_position = self.llm_agent.get_navigation_target(action, metadata)
        if target_position is None:
            return False, env.last_event, 'GotoLocation missing valid target'

        self._ensure_graph(env)
        if self._graph is None:
            return False, env.last_event, 'Navigation graph unavailable'

        self._graph.update_map(env)
        if env.last_event:
            metadata = env.last_event.metadata or {}
            reachable = metadata.get('reachablePositions')
            if reachable is None:
                reachable = metadata.get('actionReturn', [])
        else:
            reachable = []
        nav_point = self._select_navigable_point(reachable, target_position)
        if nav_point is None:
            return False, env.last_event, 'No reachable navigation target'

        start_pose = self._get_agent_pose(env)
        goal_pose = self._build_goal_pose(nav_point, target_position, start_pose[3])

        max_iterations = 10
        while max_iterations > 0:
            try:
                actions, path = self._graph.get_shortest_path(start_pose, goal_pose)
            except Exception as exc:
                event = env.last_event
                return False, event, str(exc)
            if not actions:
                break

            segment_success = True
            for index, primitive in enumerate(actions):
                success, event, error = self._dispatch_nav_action(env, primitive, smooth_nav)
                if not success:
                    failure_pose = path[index + 1]
                    self._graph.add_impossible_spot(failure_pose)
                    start_pose = self._get_agent_pose(env)
                    segment_success = False
                    break
            if segment_success:
                start_pose = self._get_agent_pose(env)
                if start_pose[:3] == goal_pose[:3]:
                    break
            max_iterations -= 1

        final_pose = self._get_agent_pose(env)
        success = final_pose[:2] == goal_pose[:2]
        event = env.last_event
        target_object_id = action.get('object_id') if action else None
        if success and event and target_object_id:
            visible = self._is_object_visible(event.metadata, target_object_id)
            if not visible:
                visible = self._adjust_horizon_for_visibility(env, target_object_id, smooth_nav)
                event = env.last_event
            success = success and visible
        error = '' if success else (event.metadata.get('errorMessage', '') if event else '')
        return success, event, error

    def _dispatch_nav_action(self, env, primitive: Dict, smooth_nav: bool):
        action_name = primitive.get('action')
        if not action_name:
            return False, env.last_event, 'Invalid navigation primitive'
        action_dict = {'action': action_name}
        if 'objectId' in primitive:
            action_dict['object_id'] = primitive['objectId']
        use_smooth = smooth_nav if smooth_nav is not None else getattr(self.args, 'smooth_nav', False)
        return super().execute_action(env, action_dict, smooth_nav=use_smooth)

    def _ensure_graph(self, env):
        event = env.last_event
        if not event:
            return
        metadata = event.metadata or {}
        scene_name = metadata.get('sceneName')
        if not scene_name:
            scene_name = getattr(env, 'scene', None)
        if not scene_name:
            return
        try:
            scene_id = int(''.join(filter(str.isdigit, scene_name)))
        except ValueError:
            return
        if self._graph is None or self._graph_scene != scene_id:
            self._graph = Graph(use_gt=True, construct_graph=True, scene_id=scene_id)
            self._graph_scene = scene_id

    def _select_navigable_point(self, reachable, target_position):
        if not reachable:
            return None
        best = None
        best_dist = float('inf')
        for pos in reachable:
            dx = pos['x'] - target_position['x']
            dz = pos['z'] - target_position['z']
            dist = dx * dx + dz * dz
            if dist < best_dist:
                best = pos
                best_dist = dist
        return best

    def _build_goal_pose(self, nav_point, target_position, start_horizon):
        grid_x = self._world_to_grid(nav_point['x'])
        grid_z = self._world_to_grid(nav_point['z'])
        rotation = self._estimate_goal_rotation(nav_point, target_position)
        horizon = self._normalize_horizon(start_horizon)
        return (grid_x, grid_z, rotation, horizon)

    def _estimate_goal_rotation(self, nav_point, target_position):
        dx = target_position['x'] - nav_point['x']
        dz = target_position['z'] - nav_point['z']
        if abs(dx) < 1e-3 and abs(dz) < 1e-3:
            return 0
        angle = (math.degrees(math.atan2(dx, dz)) + 360.0) % 360.0
        return int(round(angle / 90.0)) % 4

    def _get_agent_pose(self, env) -> Tuple[int, int, int, int]:
        metadata = env.last_event.metadata if env.last_event else {}
        agent_meta = metadata.get('agent', {})
        position = agent_meta.get('position', {'x': 0.0, 'z': 0.0})
        rotation_y = agent_meta.get('rotation', {}).get('y', 0.0)
        horizon = agent_meta.get('cameraHorizon', 0.0)
        grid_x = self._world_to_grid(position.get('x', 0.0))
        grid_z = self._world_to_grid(position.get('z', 0.0))
        rotation = int(round(rotation_y / 90.0)) % 4
        norm_horizon = self._normalize_horizon(horizon)
        return (grid_x, grid_z, rotation, norm_horizon)

    def _adjust_horizon_for_visibility(self, env, target_object_id: str, smooth_nav: bool) -> bool:
        event = env.last_event
        if event is None or not getattr(event, 'metadata', None):
            return False

        min_horizon = -30.0
        max_horizon = 90.0

        latest_event = event
        agent_meta = latest_event.metadata.get('agent', {}) if latest_event.metadata else {}
        current_horizon = agent_meta.get('cameraHorizon', 0.0)

        if self._is_object_visible(latest_event.metadata, target_object_id):
            return True

        while current_horizon > min_horizon + 1e-3:
            lookup_success, latest_event, _ = super().execute_action(env, {'action': 'LookUp'}, smooth_nav=smooth_nav)
            if not lookup_success:
                raise RuntimeError('LookUp action failed during horizon adjustment')
            if latest_event and self._is_object_visible(latest_event.metadata, target_object_id):
                return True
            agent_meta = latest_event.metadata.get('agent', {}) if latest_event.metadata else {}
            current_horizon = agent_meta.get('cameraHorizon', current_horizon)
        if latest_event is None:
            latest_event = env.last_event

        while current_horizon < max_horizon - 1e-3:
            lookdown_success, latest_event, _ = super().execute_action(env, {'action': 'LookDown'}, smooth_nav=smooth_nav)
            if not lookdown_success:
                raise RuntimeError('LookDown action failed during horizon adjustment')
            if latest_event and self._is_object_visible(latest_event.metadata, target_object_id):
                return True
            agent_meta = latest_event.metadata.get('agent', {}) if latest_event.metadata else {}
            current_horizon = agent_meta.get('cameraHorizon', current_horizon)

        raise RuntimeError('Horizon adjustment exceeded limits without finding target object. This happened because an object being held is likely blocking the agent\'s view')

    @staticmethod
    def _is_object_visible(metadata: Optional[Dict], target_object_id: str) -> bool:
        if not metadata:
            return False
        objects = metadata.get('objects', [])
        for obj in objects:
            if obj.get('objectId') == target_object_id:
                return bool(obj.get('visible', False))
        return False

    @staticmethod
    def _normalize_horizon(horizon_value: float) -> int:
        step = max(constants.AGENT_HORIZON_ADJ, 1)
        return int(round(horizon_value / step)) * step

    @staticmethod
    def _world_to_grid(value: float) -> int:
        return int(np.round(value / constants.AGENT_STEP_SIZE))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_file', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--max_fails', type=int, default=5)
    parser.add_argument('--smooth_nav', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--llm_model', type=str, default='deepseek/deepseek-chat', help='LLM model to use')
    parser.add_argument('--max_tokens', type=int, default=10000, help='Max tokens for LLM response')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for LLM sampling')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p for LLM sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0.0, help='Frequency penalty for LLM')
    parser.add_argument('--presence_penalty', type=float, default=0.0, help='Presence penalty for LLM')
    parser.add_argument('--setup_debug', action='store_true', help='Log only setup issues for debugging scene restoration')

    
    args = parser.parse_args()

    evaluator = EvalLLMAstar(args)
    evaluator.test_single_trajectory(args.traj_file, goto=True)
