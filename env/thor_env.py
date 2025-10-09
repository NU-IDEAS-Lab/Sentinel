import cv2
import copy
import gen.constants as constants
import numpy as np
from collections import Counter, OrderedDict
from env.tasks import get_task
from ai2thor.controller import Controller
import gen.utils.image_util as image_util
from gen.utils import game_util
from gen.utils.game_util import get_objects_of_type, get_obj_of_type_closest_to_obj


DEFAULT_RENDER_SETTINGS = {'renderImage': True,
                           'renderDepthImage': False,
                           'renderClassImage': False,
                           'renderObjectImage': False,
                           }

class ThorEnv(Controller):
    '''
    an extension of ai2thor.controller.Controller for ALFRED tasks
    '''
    def __init__(self, x_display=constants.X_DISPLAY,
                 player_screen_height=constants.DETECTION_SCREEN_HEIGHT,
                 player_screen_width=constants.DETECTION_SCREEN_WIDTH,
                 quality='MediumCloseFitShadows',
                 build_path=constants.BUILD_PATH,
                 headless=constants.HEADLESS,
                 agentCount=1):

        # ai2thor 5.x calls into reset() during Controller.__init__, so make sure
        # our subclass state exists before the super constructor runs.
        self.local_executable_path = build_path
        self.agent_count = agentCount
        self.task = None

        # internal states
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

        # intermediate states for CoolObject Subgoal
        self.cooled_reward = False
        self.reopen_reward = False

        controller_kwargs = {
            'quality': quality,
            'agentCount': agentCount,
        }
        if build_path is not None:
            controller_kwargs['local_executable_path'] = build_path
        if headless:
            super().__init__(
            headless=headless,
            platform='CloudRendering',
            **controller_kwargs
        )
        else:
            super().__init__(
                width=player_screen_width,
                height=player_screen_height,
                x_display=x_display,
                headless=headless,
                **controller_kwargs
            )

        print("ThorEnv started.")

    def reset(self, scene_name_or_num,
              grid_size=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
              camera_y=constants.CAMERA_HEIGHT_OFFSET,
              render_image=constants.RENDER_IMAGE,
              render_depth_image=constants.RENDER_DEPTH_IMAGE,
              render_class_image=constants.RENDER_CLASS_IMAGE,
              render_object_image=constants.RENDER_OBJECT_IMAGE,
              visibility_distance=constants.VISIBILITY_DISTANCE):
        '''
        reset scene and task states
        '''
        print("Resetting ThorEnv")

        if type(scene_name_or_num) == str:
            scene_name = scene_name_or_num
        else:
            scene_name = 'FloorPlan%d' % scene_name_or_num

        super().reset(scene_name)
        event = super().step(dict(
            action='Initialize',
            gridSize=grid_size,
            cameraY=camera_y,
            renderImage=render_image,
            renderDepthImage=render_depth_image,
            renderClassImage=render_class_image,
            renderObjectImage=render_object_image,
            visibilityDistance=visibility_distance,
            makeAgentsVisible=False,
            renderInstanceSegmentation=True,
            agentCount=self.agent_count,
        ))

        # reset task if specified
        if self.task is not None:
            self.task.reset()

        # clear object state changes
        self.reset_states()

        return event

    def reset_states(self):
        '''
        clear state changes
        '''
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

    def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
        '''
        restore object locations and states
        '''
        super().step(dict(
            action='Initialize',
            gridSize=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
            cameraY=constants.CAMERA_HEIGHT_OFFSET,
            renderImage=constants.RENDER_IMAGE,
            renderDepthImage=constants.RENDER_DEPTH_IMAGE,
            renderClassImage=constants.RENDER_CLASS_IMAGE,
            renderObjectImage=constants.RENDER_OBJECT_IMAGE,
            visibilityDistance=constants.VISIBILITY_DISTANCE,
            makeAgentsVisible=False,
            renderInstanceSegmentation=True,
            agentCount=self.agent_count,
        ))
        if len(object_toggles) > 0:
            for toggle in object_toggles:
                if toggle['action'] == 'PlaceObjectAtPoint':
                    super().step((dict(action=toggle['action'], objectId=toggle['objectId'], position=toggle['position'])))
                else:
                    super().step((dict(action=toggle['action'], objectId=toggle['objectId'], forceAction=True)))

        if len(object_poses) > 0:
            super().step((dict(action='SetObjectPoses', objectPoses=object_poses)))

    def set_task(self, traj, args, reward_type='sparse', max_episode_length=2000):
        '''
        set the current task type (one of 7 tasks)
        '''
        task_type = traj['task_type']
        self.task = get_task(task_type, traj, self, args, reward_type=reward_type, max_episode_length=max_episode_length)

    def step(self, action, smooth_nav=False, **kwargs):
        '''
        overrides ai2thor.controller.Controller.step() for smooth navigation and goal_condition updates
        '''
        if isinstance(action, str):
            action = {'action': action}
        elif not isinstance(action, dict):
            raise TypeError("action must be a dict or string understood by ai2thor")

        if smooth_nav:
            if "MoveAhead" in action['action']:
                self.smooth_move_ahead(action, **kwargs)
            elif "Rotate" in action['action']:
                self.smooth_rotate(action, **kwargs)
            elif "Look" in action['action']:
                self.smooth_look(action, **kwargs)
            else:
                super().step(self._sanitize_action(action), **kwargs)
        else:
            if "LookUp" in action['action']:
                self.look_angle(-constants.AGENT_HORIZON_ADJ, **kwargs)
            elif "LookDown" in action['action']:
                self.look_angle(constants.AGENT_HORIZON_ADJ, **kwargs)
            else:
                super().step(self._sanitize_action(action), **kwargs)

        event = self.update_states(action)
        self.check_post_conditions(action)
        return event

    def check_post_conditions(self, action):
        '''
        handle special action post-conditions
        '''
        if action['action'] == 'ToggleObjectOn':
            self.check_clean(action['objectId'])

    def update_states(self, action):
        '''
        extra updates to metadata after step
        '''
        # add 'cleaned' to all object that were washed in the sink
        event = self.last_event
        if event.metadata['lastActionSuccess']:
            # clean
            if action['action'] == 'ToggleObjectOn' and "Faucet" in action['objectId']:
                sink_basin = get_obj_of_type_closest_to_obj('SinkBasin', action['objectId'], event.metadata)
                cleaned_object_ids = sink_basin['receptacleObjectIds']
                self.cleaned_objects = self.cleaned_objects | set(cleaned_object_ids) if cleaned_object_ids is not None else set()
            # heat
            if action['action'] == 'ToggleObjectOn' and "Microwave" in action['objectId']:
                microwave = get_objects_of_type('Microwave', event.metadata)[0]
                heated_object_ids = microwave['receptacleObjectIds']
                self.heated_objects = self.heated_objects | set(heated_object_ids) if heated_object_ids is not None else set()
            # cool
            if action['action'] == 'CloseObject' and "Fridge" in action['objectId']:
                fridge = get_objects_of_type('Fridge', event.metadata)[0]
                cooled_object_ids = fridge['receptacleObjectIds']
                self.cooled_objects = self.cooled_objects | set(cooled_object_ids) if cooled_object_ids is not None else set()

        return event

    def get_transition_reward(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for transition_reward")
        else:
            return self.task.transition_reward(self.last_event)

    def get_goal_satisfied(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_satisfied(self.last_event)

    def get_goal_conditions_met(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_conditions_met(self.last_event)

    def get_subgoal_idx(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for subgoal_idx")
        else:
            return self.task.get_subgoal_idx()

    def noop(self):
        '''
        do nothing
        '''
        super().step(dict(action='Pass'))

    def smooth_move_ahead(self, action, render_settings=None, **step_kwargs):
        '''
        smoother MoveAhead
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        smoothing_factor = constants.RECORD_SMOOTHING_FACTOR
        new_action = copy.deepcopy(action)
        new_action['moveMagnitude'] = constants.AGENT_STEP_SIZE / smoothing_factor

        new_action['renderImage'] = render_settings['renderImage']
        new_action['renderClassImage'] = render_settings['renderClassImage']
        new_action['renderObjectImage'] = render_settings['renderObjectImage']
        new_action['renderDepthImage'] = render_settings['renderDepthImage']

        events = []
        for xx in range(smoothing_factor - 1):
            event = super().step(self._sanitize_action(new_action), **step_kwargs)
            if event.metadata['lastActionSuccess']:
                events.append(event)

        event = super().step(self._sanitize_action(new_action), **step_kwargs)
        if event.metadata['lastActionSuccess']:
            events.append(event)
        return events

    def smooth_rotate(self, action, render_settings=None, **step_kwargs):
        '''
        smoother RotateLeft and RotateRight
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        if action['action'] == 'RotateLeft':
            end_rotation = (start_rotation - 90)
        else:
            end_rotation = (start_rotation + 90)

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            rotation_step = np.round(start_rotation * (1 - xx) + end_rotation * xx, 3)
            teleport_action = self._build_teleport_action(
                position,
                rotation_step,
                horizon,
            )
            event = super().step(self._sanitize_action(teleport_action), **step_kwargs)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def smooth_look(self, action, render_settings=None, **step_kwargs):
        '''
        smoother LookUp and LookDown
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + constants.AGENT_HORIZON_ADJ * (1 - 2 * int(action['action'] == 'LookUp'))
        position = event.metadata['agent']['position']

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            horizon_step = np.round(start_horizon * (1 - xx) + end_horizon * xx, 3)
            teleport_action = self._build_teleport_action(
                position,
                rotation,
                horizon_step,
            )
            event = super().step(self._sanitize_action(teleport_action), **step_kwargs)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def look_angle(self, angle, render_settings=None, **step_kwargs):
        '''
        look at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + angle
        position = event.metadata['agent']['position']

        teleport_action = self._build_teleport_action(
            position,
            rotation,
            np.round(end_horizon, 3),
        )
        event = super().step(self._sanitize_action(teleport_action), **step_kwargs)
        return event

    def rotate_angle(self, angle, render_settings=None, **step_kwargs):
        '''
        rotate at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        end_rotation = start_rotation + angle

        teleport_action = self._build_teleport_action(
            position,
            np.round(end_rotation, 3),
            horizon,
        )
        event = super().step(self._sanitize_action(teleport_action), **step_kwargs)
        return event

    def _sanitize_action(self, action):
        action = copy.deepcopy(action)
        unsupported_keys = {'rotateOnTeleport'}
        deprecated_render_keys = {
            'tempRenderChange',
            'renderImage',
            'renderDepthImage',
            'renderClassImage',
            'renderObjectImage',
            'renderNormalsImage',
        }
        if action.get('action') == 'PutObject':
            receptacle_object_id = action.pop('receptacleObjectId', None)
            if receptacle_object_id is not None:
                action['objectId'] = receptacle_object_id
        if action.get('action') == 'TeleportFull':
            for key in unsupported_keys | deprecated_render_keys:
                action.pop(key, None)
            rotation = action.get('rotation')
            if isinstance(rotation, (int, float)):
                action['rotation'] = {'x': 0, 'y': rotation, 'z': 0}
            if 'standing' not in action:
                standing = True
                if getattr(self, 'last_event', None) is not None:
                    metadata = getattr(self.last_event, 'metadata', {}) or {}
                    agent_meta = metadata.get('agent') or {}
                    standing = agent_meta.get('isStanding', standing)
                action['standing'] = standing
        return action

    def _build_teleport_action(self, position, rotation, horizon):
        """Construct a TeleportFull action that is compatible with AI2-THOR 5.x."""
        return {
            'action': 'TeleportFull',
            'x': position['x'],
            'y': position['y'],
            'z': position['z'],
            'rotation': rotation,
            'horizon': horizon,
        }

    def to_thor_api_exec(self, action, object_id="", smooth_nav=False):
        # TODO: parametrized navigation commands

        if "RotateLeft" in action:
            action = dict(action="RotateLeft",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "RotateRight" in action:
            action = dict(action="RotateRight",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "MoveAhead" in action:
            action = dict(action="MoveAhead",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "LookUp" in action:
            action = dict(action="LookUp",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "LookDown" in action:
            action = dict(action="LookDown",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "OpenObject" in action:
            action = dict(action="OpenObject",
                          objectId=object_id,
                          moveMagnitude=1.0)
            event = self.step(action)
        elif "CloseObject" in action:
            action = dict(action="CloseObject",
                          objectId=object_id,
                          forceAction=True)
            event = self.step(action)
        elif "PickupObject" in action:
            action = dict(action="PickupObject",
                          objectId=object_id)
            event = self.step(action)
        elif "PutObject" in action:
            action = dict(action="PutObject",
                          objectId=object_id,
                          forceAction=True,
                          placeStationary=True)
            event = self.step(action)
        elif "ToggleObjectOn" in action:
            action = dict(action="ToggleObjectOn",
                          objectId=object_id)
            event = self.step(action)

        elif "ToggleObjectOff" in action:
            action = dict(action="ToggleObjectOff",
                          objectId=object_id)
            event = self.step(action)
        elif "SliceObject" in action:
            # check if agent is holding knife in hand
            inventory_objects = self.last_event.metadata['inventoryObjects']
            if len(inventory_objects) == 0 or 'Knife' not in inventory_objects[0]['objectType']:
                raise Exception("Agent should be holding a knife before slicing.")

            action = dict(action="SliceObject",
                          objectId=object_id)
            event = self.step(action)
        else:
            raise Exception("Invalid action. Conversion to THOR API failed! (action='" + str(action) + "')")

        return event, action

    def check_clean(self, object_id):
        '''
        Handle special case when Faucet is toggled on.
        In this case, we need to execute a `CleanAction` in the simulator on every object in the corresponding
        basin. This is to clean everything in the sink rather than just things touching the stream.
        '''
        event = self.last_event
        if event.metadata['lastActionSuccess'] and 'Faucet' in object_id:
            # Need to delay one frame to let `isDirty` update on stream-affected.
            event = self.step({'action': 'Pass'})
            sink_basin_obj = game_util.get_obj_of_type_closest_to_obj("SinkBasin", object_id, event.metadata)
            for in_sink_obj_id in sink_basin_obj['receptacleObjectIds']:
                if (game_util.get_object(in_sink_obj_id, event.metadata)['dirtyable']
                        and game_util.get_object(in_sink_obj_id, event.metadata)['isDirty']):
                    event = self.step({'action': 'CleanObject', 'objectId': in_sink_obj_id})
        return event

    def prune_by_any_interaction(self, instances_ids):
        '''
        ignores any object that is not interactable in anyway
        '''
        pruned_instance_ids = []
        for obj in self.last_event.metadata['objects']:
            obj_id = obj['objectId']
            if obj_id in instances_ids:
                if obj['pickupable'] or obj['receptacle'] or obj['openable'] or obj['toggleable'] or obj['sliceable']:
                    pruned_instance_ids.append(obj_id)

        ordered_instance_ids = [id for id in instances_ids if id in pruned_instance_ids]
        return ordered_instance_ids

    @staticmethod
    def bbox_to_mask(bbox):
        return image_util.bbox_to_mask(bbox)

    @staticmethod
    def point_to_mask(point):
        return image_util.point_to_mask(point)

    @staticmethod
    def decompress_mask(compressed_mask):
        return image_util.decompress_mask(compressed_mask)
