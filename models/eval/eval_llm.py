import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
import copy
import json
from datetime import datetime, time
from env.thor_env import ThorEnv
from models.model.llm import LLMAgent


class EpisodeTrace:
    """Collects per-step execution records for a single episode."""

    def __init__(self) -> None:
        self._steps = []
        self._step_index = 0

    def record(self, plan_action, thor_action, success, error, event_metadata) -> None:
        entry = {
            'step': self._step_index,
            'plan_action': self._sanitize(plan_action),
            'thor_action': self._sanitize(thor_action),
            'success': bool(success),
            'error': error or '',
            'event_metadata': self._sanitize(event_metadata) if event_metadata is not None else None,
        }
        self._steps.append(entry)
        self._step_index += 1

    def export(self):
        return list(self._steps)

    @staticmethod
    def _sanitize(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): EpisodeTrace._sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [EpisodeTrace._sanitize(v) for v in value]
        if hasattr(value, 'tolist'):
            return EpisodeTrace._sanitize(value.tolist())
        return str(value)


class EvalLLM:
    '''
    evaluate LLM performance on ALFRED tasks using zero-shot prompting - standalone version
    '''

    def __init__(self, args, manager=None):
        # Initialize without model loading since we're using LLM
        self.args = args
        self.manager = manager

        # Initialize OpenRouter client for DeepSeek
        # Initialize LLM agent
        self.llm_agent = LLMAgent(args)
        self.llm_agent.set_log_method(self.log)
        self.logging = args.debug if hasattr(args, 'debug') else False
        self.setup_debug = getattr(args, 'setup_debug', False)
        self._current_trace = None
        
        # Setup simple logging
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        traj_path = args.traj_file.replace("data/json_2.1.0/", f"{args.llm_model}/").replace("/traj_data.json", "")
        traj_log_file = os.path.join("logs", "trajectories", traj_path,f"{timestamp}.json")
        os.makedirs(os.path.dirname(traj_log_file), exist_ok=True)
        log_file = traj_log_file.replace(".json", ".txt")
        self.log_file = log_file
        self.trace_file = traj_log_file
        print(f"Logging to: {self.log_file}")

    def log(self, message):
        """Simple logging to text file"""
        should_log = self.logging
        if self.setup_debug and isinstance(message, str) and message.startswith('[setup_scene]'):
            should_log = True
        if should_log:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")

    def test_single_trajectory(self, traj_file_path=None, goto=False):
        """
        Test evaluation on a single trajectory
        """
        if not traj_file_path:
            traj_file_path = "data/json_2.1.0/train/look_at_obj_in_light-AlarmClock-None-DeskLamp-305/trial_T20190908_082723_323728/traj_data.json"
        
        # Load trajectory data
        with open(traj_file_path, 'r') as f:
            traj_data = json.load(f)
        
        # Create environment
        if "multiagent" in traj_data and traj_data["multiagent"]:
            env = ThorEnv(agentCount=traj_data["multiagent"]["agentCount"])
        else:
            env = ThorEnv()
        
        # Create simple objects (no multiprocessing needed for single test)
        successes = []
        failures = []
        results = {}
        
        # Simple lock class for single-threaded use
        class SimpleLock:
            def acquire(self): pass
            def release(self): pass
        
        lock = SimpleLock()  
        try:
            print(f"Testing single trajectory: {traj_file_path}")
            
            # Run evaluation on single trajectory
            self.evaluate(env, traj_data, self.args, lock, successes, failures, results, goto=goto)

        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            env.stop()
        
        print("\n=== FINAL RESULTS ===")
        print(f"Successes: {len(successes)}")
        print(f"Failures: {len(failures)}")
        if results:
            print(f"Results: {results}")
        
        return successes, failures, results

    def setup_scene(self, env, traj_data, args, reward_type='dense'):
        """
        Setup scene from trajectory data
        """
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        # reset env
        # TODO: adjust visibility distance as needed, default is 1.5
        env.reset(scene_name)
        
        # setup initial conditions
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        restore_event = env.last_event
        restore_issues = []
        if restore_event is not None:
            metadata = getattr(restore_event, 'metadata', {})
            if metadata and not metadata.get('lastActionSuccess', True):
                restore_issues.append(f"[setup_scene] SetObjectPoses failed: {metadata.get('errorMessage')}")
        else:
            restore_issues.append('[setup_scene] No event metadata available after restore_scene invocation.')

        post_restore_metadata = (restore_event.metadata if restore_event is not None and hasattr(restore_event, 'metadata') else None)
        if post_restore_metadata is None and env.last_event is not None:
            post_restore_metadata = getattr(env.last_event, 'metadata', None)
        scene_object_names = set()
        if isinstance(post_restore_metadata, dict):
            for obj in post_restore_metadata.get('objects', []) or []:
                name = obj.get('name')
                obj_id = obj.get('objectId')
                if name:
                    scene_object_names.add(name)
                if obj_id:
                    scene_object_names.add(obj_id)
        missing_objects = [pose['objectName'] for pose in object_poses
                           if pose.get('objectName') and pose['objectName'] not in scene_object_names]
        if missing_objects:
            restore_issues.append(f"[setup_scene] Missing objects after restoration: {missing_objects}")

        if restore_issues:
            for issue in restore_issues:
                print(issue)
                self.log(issue)
            general_msg = "[setup_scene] Scene restoration encountered issues. Check trajectory data or simulator state."
            print(general_msg)
            self.log(general_msg)
        else:
            print(f"[setup_scene] Restored {len(object_poses)} object poses successfully.")

        # initialize to start position
        event = env.step(dict(traj_data['scene']['init_action']))

        

        # print goal instr
        print("Task: %s" % (traj_data['task_desc']))

        # setup task for reward
        env.set_task(traj_data, args, reward_type=reward_type)

    def execute_action(self, env, action_dict, smooth_nav=False):
        """
        Execute action from dict format like {"action": "PickupObject", "object_id": "..."}
        Using direct scene interaction instead of va_interact
        """
        action_name = action_dict.get('action')
        object_id = action_dict.get('object_id', '')
        receptacle_id = action_dict.get('receptacle_id', '')
        agent_id = action_dict.get('agentId')
        if action_name == "PutObject":
            if not receptacle_id:
                raise ValueError("PutObject action requires a receptacle_id")
    
            # We only need the receptacle_id for PutObject (the object to put down is always the held object)
            object_id = receptacle_id

        plan_action_copy = copy.deepcopy(action_dict)
        try:
            # Use the same direct execution approach as eval_llm_step.py
            event, api_action = env.to_thor_api_exec(
                action_name,
                object_id,
                smooth_nav=smooth_nav,
                agent_id=agent_id,
            )
            success = event.metadata['lastActionSuccess']
            error = event.metadata.get('errorMessage', '') if not success else ''
            self.log(f"Action: {action_name}, Object ID: {object_id}, Success: {success}, Error: {error}")

            self._record_step(plan_action_copy, api_action, success, error, event)
            return success, event, error
        except Exception as e:
            self.log(f"Error: {e} during action {action_name} with object {object_id}")
            self._record_step(plan_action_copy, None, False, str(e), None)
            return False, None, str(e)
        

    def _record_step(self, plan_action, thor_action, success, error, event):
        if self._current_trace is None or event is None:
            return
        metadata = event.metadata
        metadata = self.remove_useless_info(metadata)
        self._current_trace.record(plan_action, thor_action, success, error, metadata)


    def evaluate(self, env, traj_data, args, lock, successes, failures, results, goto=False):
        EvalLLM.log_method = self.log
        trace = EpisodeTrace()
        previous_trace = self._current_trace
        self._current_trace = trace
        try:
            # setup scene
            reward_type = 'dense'
            self.setup_scene(env, traj_data, args, reward_type=reward_type)

            # goal instruction
            goal_instr = traj_data['task_desc']  # use more detailed task_desc if available

            print(f"Task description: {goal_instr}")

            # Log task info
            self.log(f"Task: {goal_instr}")
            self.log(f"Scene: {traj_data['scene']['scene_num']}")
            self.log(f"Objects: {list(traj_data['scene']['object_poses'])}")

            # Get scene information for LLM
            metadata = env.last_event.metadata 
            scene_info = self.remove_useless_info(metadata)

            # Test goal extraction
            subgoals = self.llm_agent.get_subgoals_from_scene(goal_instr, scene_info)

            # Generate LLM plan
            llm_plan = self.llm_agent.generate_plan(subgoals, scene_info, goto=goto)

            # Execute plan
            done, success = False, False
            fails = 0
            t = 0
            reward = 0
            action_idx = 0

            print(f"Generated plan with {len(llm_plan)} actions")

            while not done and action_idx < len(llm_plan):
                if t >= args.max_steps:
                    print("Max steps reached")
                    break

                action_data = llm_plan[action_idx]
                action = action_data.get('action')

                if not action:
                    print("Invalid action in plan")
                    break

                if action.lower() in ['stop', 'end', 'finish', 'done']:
                    print("\tLLM predicted STOP")
                    # Get final position of all the objects
                    self.execute_action(env, {'action': 'Done'}, smooth_nav=args.smooth_nav)
                    break

                if args.debug:
                    print(f"Step {t}: {action_data}")

                t_success, event, err = self.execute_action(env, action_data, smooth_nav=args.smooth_nav)

                if not t_success:
                    fails += 1
                    if fails >= args.max_fails:
                        print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                        break

                t_reward, t_done = env.get_transition_reward()
                reward += t_reward
                t += 1
                action_idx += 1

            goal_satisfied = env.get_goal_satisfied()
            if goal_satisfied:
                print("Goal Reached")
                success = True

            pcs = env.get_goal_conditions_met()
            goal_condition_success_rate = pcs[0] / float(pcs[1])

            path_len_weight = len(traj_data['plan']['low_actions'])
            s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(t)) if t > 0 else 0
            pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t)) if t > 0 else 0

            plw_s_spl = s_spl * path_len_weight
            plw_pc_spl = pc_spl * path_len_weight
            

            lock.acquire()
            log_entry = {
                'trial': traj_data['task_id'],
                'goal_instr': goal_instr,
                'completed_goal_conditions': int(pcs[0]),
                'total_goal_conditions': int(pcs[1]),
                'goal_condition_success': float(goal_condition_success_rate),
                'success_spl': float(s_spl),
                'path_len_weighted_success_spl': float(plw_s_spl),
                'goal_condition_spl': float(pc_spl),
                'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                'path_len_weight': int(path_len_weight),
                'reward': float(reward),
                'llm_plan_length': len(llm_plan),
                'steps_failed': int(fails),
                'trajectory': trace.export(),
            }
            # Write trace in a json file
            

            with open(self.trace_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'trajectory': trace.export(),
                    'success': bool(success)
                }, f, indent=2)
                print(f"Saved trajectory log to {self.trace_file}")

            if success:
                successes.append(log_entry)
            else:
                failures.append(log_entry)

            results['all'] = self.get_metrics(successes, failures)

            if results.get('all'):
                print("-------------")
                print("SR: %d/%d = %.3f" % (results['all']['success']['num_successes'],
                                            results['all']['success']['num_evals'],
                                            results['all']['success']['success_rate']))
                print("GC: %d/%d = %.3f" % (results['all']['goal_condition_success']['completed_goal_conditions'],
                                            results['all']['goal_condition_success']['total_goal_conditions'],
                                            results['all']['goal_condition_success']['goal_condition_success_rate']))
                print("-------------")

            lock.release()
        finally:
            self._current_trace = previous_trace

    @classmethod
    def remove_useless_info(cls, metadata):
        """
        Extract current scene information for LLM input
        Remove useless info to reduce token usage
        """
        scene_info = metadata.copy()
        useless_info = ['isSceneAtRest', 'fov', 'isStanding', 'cameraPosition', 'cameraOrthSize', 'thirdPartyCameras', 'sceneName', 'lastAction', 'errorCode', 'screenWidth', 'screenHeight', 'agentId', 'colors', 'colorBounds', 'reachablePositions', 'flatSurfacesOnGrid', 'normals', 'segmentedObjectIds', 'actionIntReturn', 'actionFloatReturn', 'actionStringsReturn', 'actionFloatsReturn', 'actionVector3sReturn', 'visibleRange', 'actionReturn']
        for key in useless_info:
            scene_info.pop(key, None)
        
        return scene_info

    @classmethod
    def get_metrics(cls, successes, failures):
        '''
        compute overall success and goal_condition success rates along with path-weighted metrics
        '''
        # stats
        num_successes, num_failures = len(successes), len(failures)
        num_evals = len(successes) + len(failures)
        
        if num_evals == 0:
            return {}
            
        total_path_len_weight = sum([entry['path_len_weight'] for entry in successes]) + \
                                sum([entry['path_len_weight'] for entry in failures])
        completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                   sum([entry['completed_goal_conditions'] for entry in failures])
        total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                               sum([entry['total_goal_conditions'] for entry in failures])

        # metrics
        sr = float(num_successes) / num_evals
        pc = completed_goal_conditions / float(total_goal_conditions) if total_goal_conditions > 0 else 0

        # result table
        res = dict()
        res['success'] = {'num_successes': num_successes,
                          'num_evals': num_evals,
                          'success_rate': sr}
        res['goal_condition_success'] = {'completed_goal_conditions': completed_goal_conditions,
                                        'total_goal_conditions': total_goal_conditions,
                                        'goal_condition_success_rate': pc}

        return res
    

    def get_trajectory_files(self, data_dir, split):
        """Get all trajectory files for a given split"""
        import glob
        pattern = f"{data_dir}/{split}/*/trial_*/traj_data.json"
        return glob.glob(pattern)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_file', type=str, default=None, help='Path to single trajectory JSON file for testing')
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum steps per episode')
    parser.add_argument('--max_fails', type=int, default=5, help='Maximum consecutive action fails before aborting')
    parser.add_argument('--smooth_nav', action='store_true', help='Use smooth navigation')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--llm_model', type=str, default='deepseek/deepseek-chat', help='LLM model to use')
    parser.add_argument('--max_tokens', type=int, default=10000, help='Max tokens for LLM response')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for LLM sampling')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p for LLM sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0.0, help='Frequency penalty for LLM')
    parser.add_argument('--presence_penalty', type=float, default=0.0, help='Presence penalty for LLM')
    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--batch', action='store_true', help='Run batch evaluation')
    parser.add_argument('--split', type=str, default='valid_seen', help='Data split to evaluate')
    parser.add_argument('--data_dir', type=str, default='data/json_2.1.0', help='Data directory')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs per trajectory')
    

    args = parser.parse_args()
    
    evaluator = EvalLLM(args)
    if args.batch:
        evaluator.test_batch(args.data_dir, args.split, args.num_runs)
    else:
        evaluator.test_single_trajectory(args.traj_file)
