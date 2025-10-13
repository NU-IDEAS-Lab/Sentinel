import os
import sys
import math
from typing import Dict, Optional

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gen.constants as constants
from models.eval.eval_llm import EvalLLM
from models.model.llm_astar import LLMAstar
import numpy as np
from nav_astar import (
    astar_actions,
    build_adjacency,
    build_node_lookup,
    jps_actions,
    nearest_key,
    normalize_yaw as snap_yaw,
)


class EvalLLMAstar(EvalLLM):
    """EvalLLM variant that expands GotoLocation into executable actions."""

    def __init__(self, args, manager=None):
        super().__init__(args, manager)
        self.llm_agent = LLMAstar(args)
        self.llm_agent.set_log_method(self.log)

    def execute_action(self, env, action_dict, smooth_nav=False):  # type: ignore[override]
        action_name = action_dict.get('action')
        if action_name == 'GotoLocation':
            return self._execute_goto(env, action_dict, smooth_nav=smooth_nav)
        return super().execute_action(env, action_dict, smooth_nav=smooth_nav)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------
    def _execute_goto(self, env, action: Dict, smooth_nav: bool):
        target_object_id = action.get('object_id')
        
        reachable = env.step(action="GetReachablePositions")
        reach_meta = reachable.metadata
        if not reach_meta:
            return False, env.last_event, 'No reachable positions found'
        agent_meta = reach_meta['agent']
        reachable_positions = reach_meta["actionReturn"]

        event = env.step(
            action="GetInteractablePoses",
            objectId=target_object_id,
            positions=reachable_positions, # will let default 4 rotations be used
            horizons=[float(h) for h in np.linspace(-30, 60, 30)],
            standings=[True],
        )
        poses = event.metadata["actionReturn"]
        if not poses:
            raise RuntimeError("No interactable poses returned for target object")
        
        # TODO: Pick multiple poses and choose one that works in the future
        target_position = poses[0]
        if target_position is None:
            return False, env.last_event, 'GotoLocation missing valid target'

        nodes = build_node_lookup(reachable_positions)
        agent_position = agent_meta['position']

        start_key = nearest_key(nodes, agent_position)
        target_key = nearest_key(nodes, target_position)
        start_yaw = snap_yaw(agent_meta['rotation']['y'])

        try:
            action_names = jps_actions(nodes, start_key, start_yaw, target_key)
        except RuntimeError:
            adjacency = build_adjacency(nodes)
            action_names = astar_actions(nodes, adjacency, start_key, start_yaw, target_key)

        success = True
        event = env.last_event
        error = ''

        for action_name in action_names:
            success, event, error = self._dispatch_nav_action(env, {'action': action_name}, smooth_nav)
            if not success:
                break

        if not success:
            return success, event, error

        event = env.last_event

        agent_meta = event.metadata['agent']
        agent_position = agent_meta['position']
        current_yaw = agent_meta['rotation']['y']

        # the target pose already includes a desired yaw under rotation
        desired_yaw = target_position['rotation']
        for turn_name in self._turn_actions(current_yaw, desired_yaw):
            success, event, error = super().execute_action(env, {'action': turn_name}, smooth_nav=smooth_nav)
            if not success:
                return success, event, error

        event = env.last_event
        if success and event and target_object_id:
            visible = self._is_object_visible(event.metadata, target_object_id)
            if not visible:
                visible = self._adjust_horizon_for_visibility(env, target_object_id, smooth_nav)
                event = env.last_event
            success = success and visible
        error = '' if success else (event.metadata.get('errorMessage', '') if event else '')
        return success, event, error

    def _dispatch_nav_action(self, env, primitive: Dict, smooth_nav: bool):
        if isinstance(primitive, str):
            action_name = primitive
            action_dict = {'action': action_name}
        else:
            action_name = primitive.get('action')
            action_dict = {'action': action_name} if action_name else {}
        if not action_name:
            return False, env.last_event, 'Invalid navigation primitive'
        if isinstance(primitive, dict) and 'objectId' in primitive:
            action_dict['object_id'] = primitive['objectId']
        use_smooth = smooth_nav if smooth_nav is not None else getattr(self.args, 'smooth_nav', False)
        return super().execute_action(env, action_dict, smooth_nav=use_smooth)

    @staticmethod
    def _turn_actions(current_yaw: float, desired_yaw: Optional[float]):
        if desired_yaw is None:
            return []
        current = snap_yaw(current_yaw)
        desired = snap_yaw(desired_yaw)
        diff = (desired - current) % 360
        if diff == 0:
            return []
        if diff == 90:
            return ['RotateRight']
        if diff == 180:
            return ['RotateRight', 'RotateRight']
        if diff == 270:
            return ['RotateLeft']
        if diff < 180:
            steps = int(round(diff / 90.0))
            return ['RotateRight'] * steps
        steps = int(round((360.0 - diff) / 90.0))
        return ['RotateLeft'] * steps

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
