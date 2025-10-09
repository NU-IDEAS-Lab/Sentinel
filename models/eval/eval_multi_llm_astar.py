import os
import sys
import copy
from typing import Dict, Optional, List

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.eval.eval_llm_astar import EvalLLMAstar


class EvalMultiLLMAstar(EvalLLMAstar):
    """Multi-agent evaluator that keeps a primary agent active while others act as obstacles."""

    def __init__(
        self,
        args,
        manager=None,
        primary_agent_index: int = 0,
        obstacle_agent_ids: Optional[List[int]] = None,
    ):
        super().__init__(args, manager)
        self.primary_agent_index = primary_agent_index
        self.obstacle_agent_ids = (
            list(obstacle_agent_ids)
            if obstacle_agent_ids is not None
            else []
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _primary_event(self, event):
        if event is None:
            return None
        events = getattr(event, 'events', None)
        if isinstance(events, list) and events:
            if self.primary_agent_index < len(events):
                return events[self.primary_agent_index]
            return events[0]
        return event

    def _with_agent(self, action_dict: Optional[Dict]) -> Dict:
        result: Dict = {} if action_dict is None else copy.deepcopy(action_dict)
        if result.get('agentId') is None:
            result['agentId'] = self.primary_agent_index
        return result

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def execute_action(self, env, action_dict, smooth_nav=False):  # type: ignore[override]
        action_with_agent = self._with_agent(action_dict)
        success, event, error = super().execute_action(env, action_with_agent, smooth_nav=smooth_nav)
        event = self._primary_event(event)
        return success, event, error
    
    def _record_step(self, plan_action, thor_action, success, error, event):
        event = self._primary_event(event)
        return super()._record_step(plan_action, thor_action, success, error, event)

    def _dispatch_nav_action(self, env, primitive: Dict, smooth_nav: bool):
        action_name = primitive.get('action')
        if not action_name:
            return False, self._primary_event(env.last_event), 'Invalid navigation primitive'
        action_dict = {'action': action_name, 'agentId': self.primary_agent_index}
        if 'objectId' in primitive:
            action_dict['object_id'] = primitive['objectId']
        use_smooth = smooth_nav if smooth_nav is not None else getattr(self.args, 'smooth_nav', False)
        return self.execute_action(env, action_dict, smooth_nav=use_smooth)

    def _adjust_horizon_for_visibility(self, env, target_object_id: str, smooth_nav: bool) -> bool:
        event = self._primary_event(env.last_event)
        metadata = getattr(event, 'metadata', None)
        if event is None or metadata is None:
            return False

        min_horizon = -30.0
        max_horizon = 90.0

        latest_event = event
        agent_meta = metadata.get('agent', {}) if metadata else {}
        current_horizon = agent_meta.get('cameraHorizon', 0.0)

        if self._is_object_visible(metadata, target_object_id):
            return True

        while current_horizon > min_horizon + 1e-3:
            lookup_success, latest_event, _ = self.execute_action(
                env, {'action': 'LookUp'}, smooth_nav=smooth_nav
            )
            if not lookup_success:
                raise RuntimeError('LookUp action failed during horizon adjustment')
            latest_metadata = getattr(latest_event, 'metadata', None)
            if latest_event and self._is_object_visible(latest_metadata, target_object_id):
                return True
            agent_meta = latest_metadata.get('agent', {}) if latest_metadata else {}
            current_horizon = agent_meta.get('cameraHorizon', current_horizon)

        if latest_event is None:
            latest_event = self._primary_event(env.last_event)

        while current_horizon < max_horizon - 1e-3:
            lookdown_success, latest_event, _ = self.execute_action(
                env, {'action': 'LookDown'}, smooth_nav=smooth_nav
            )
            if not lookdown_success:
                raise RuntimeError('LookDown action failed during horizon adjustment')
            latest_metadata = getattr(latest_event, 'metadata', None)
            if latest_event and self._is_object_visible(latest_metadata, target_object_id):
                return True
            agent_meta = latest_metadata.get('agent', {}) if latest_metadata else {}
            current_horizon = agent_meta.get('cameraHorizon', current_horizon)

        raise RuntimeError(
            "Horizon adjustment exceeded limits without finding target object. "
            "This happened because an object being held is likely blocking the agent's view"
        )


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
    parser.add_argument('--primary_agent_index', type=int, default=0, help='Agent index to execute the plan with')
    parser.add_argument('--obstacle_agent_ids', type=str, default='', help='Comma separated list of obstacle agent indices')

    args = parser.parse_args()

    obstacle_ids = [int(x) for x in args.obstacle_agent_ids.split(',') if x.strip()]
    evaluator = EvalMultiLLMAstar(
        args,
        primary_agent_index=args.primary_agent_index,
        obstacle_agent_ids=obstacle_ids,
    )
    evaluator.test_single_trajectory(args.traj_file, goto=True)
