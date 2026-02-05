import json
import numpy as np
from gen.utils.game_util import get_objects_with_name_and_prop


class BaseTask(object):
    """
    base class for tasks
    """

    def __init__(self, traj, env, args, reward_type="sparse", max_episode_length=2000):
        # settings
        self.traj = traj
        self.env = env
        self.args = args
        self.task_type = self.traj["task_type"]
        self.max_episode_length = max_episode_length
        self.reward_type = reward_type
        self.step_num = 0
        self.goal_finished = False

        # reward config
        self.reward_config = {}
        self.load_reward_config(args.reward_config)

    def load_reward_config(self, config_file):
        """
        load json file with reward values
        """
        with open(config_file, "r") as rc:
            reward_config = json.load(rc)
        self.reward_config = reward_config

    def goal_satisfied(self, state):
        """
        check if the overall task goal was satisfied.
        """
        raise NotImplementedError

    def transition_reward(self, state):
        """
        immediate reward given the current state
        """
        reward = 0
        metadata = getattr(state, "metadata", {}) if state is not None else {}
        last_action_success = metadata.get("lastActionSuccess", False)
        generic_rewards = self.reward_config.get("Generic", {})

        if not self.goal_finished and self.goal_satisfied(state):
            reward += generic_rewards.get("goal_reward", 0)
            self.goal_finished = True

        if "success" in self.reward_type and last_action_success:
            reward += generic_rewards.get("success", 0)

        if "failure" in self.reward_type and not last_action_success:
            reward += generic_rewards.get("failure", 0)

        if self.step_num > len(self.traj["plan"]["low_actions"]):
            reward += generic_rewards.get("step_penalty", 0)

        self.step_num += 1
        done = self.goal_finished or self.step_num >= self.max_episode_length
        return reward, done

    def reset(self):
        """
        Reset internal states
        """
        self.step_num = 0
        self.goal_finished = False

    def get_target(self, var):
        """
        returns the object type of a task param
        """
        return (
            self.traj["pddl_params"][var]
            if self.traj["pddl_params"][var] is not None
            else None
        )

    def _get_target_from_state_map(self, map_key):
        """
        Return all target -> desired_state pairs from a map like {"Microwave": true}.
        """
        pddl_params = self.traj.get("pddl_params", {})
        state_map = pddl_params.get(map_key) or {}
        if not isinstance(state_map, dict) or not state_map:
            return None
        return state_map

    def get_targets(self):
        """
        returns a dictionary of all targets for the task
        """
        toggle_targets = self._get_target_from_state_map("object_toggled")
        open_targets = self._get_target_from_state_map("object_opened")
        targets = {
            "object": self.get_target("object_target"),
            "parent": self.get_target("parent_target"),
            "toggle_targets": toggle_targets,
            "open_targets": open_targets,
            "mrecep": self.get_target("mrecep_target"),
        }

        # slice exception
        if (
            "object_sliced" in self.traj["pddl_params"]
            and self.traj["pddl_params"]["object_sliced"]
        ):
            targets[
                "object"
            ] += "Sliced"  # Change, e.g., "Apple" -> "AppleSliced" as pickup target.

        return targets


class PickAndPlaceSimpleTask(BaseTask):
    """
    pick_and_place task
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def goal_satisfied(self, state):
        # check if any object of 'object' class is inside any receptacle of 'parent' class
        pcs = self.goal_conditions_met(state)
        return pcs[0] == pcs[1]

    def goal_conditions_met(self, state):
        ts = 1
        s = 0

        targets = self.get_targets()
        receptacles = get_objects_with_name_and_prop(
            targets["parent"], "receptacle", state.metadata
        )
        pickupables = get_objects_with_name_and_prop(
            targets["object"], "pickupable", state.metadata
        )

        # check if object needs to be sliced
        if "Sliced" in targets["object"]:
            ts += 1
            if len([p for p in pickupables if "Sliced" in p["objectId"]]) >= 1:
                s += 1

        if np.any(
            [
                np.any(
                    [
                        p["objectId"] in r["receptacleObjectIds"]
                        for r in receptacles
                        if r["receptacleObjectIds"] is not None
                    ]
                )
                for p in pickupables
            ]
        ):
            s += 1

        return s, ts

    def reset(self):
        super().reset()


class ToggleSimpleTask(BaseTask):
    """
    toggle_simple task – succeed once the specified toggle target is on.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def goal_satisfied(self, state):
        satisfied, total = self.goal_conditions_met(state)
        return satisfied == total

    def goal_conditions_met(self, state):
        targets = self.get_targets()
        toggle_targets = targets.get("toggle_targets")
        if not toggle_targets:
            return 0, 1

        ts = len(toggle_targets)
        s = 0
        for name, desired_state in toggle_targets.items():
            toggleables = get_objects_with_name_and_prop(
                name, "toggleable", state.metadata
            )
            if desired_state is None:
                desired_state = True
            if any(obj.get("isToggled") == desired_state for obj in toggleables):
                s += 1

        return s, ts

    def reset(self):
        super().reset()


class OpenCloseSimpleTask(BaseTask):
    """
    open_close_simple task – succeed once the specified open target is opened/closed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def goal_satisfied(self, state):
        satisfied, total = self.goal_conditions_met(state)
        return satisfied == total

    def goal_conditions_met(self, state):
        targets = self.get_targets()
        open_targets = targets.get("open_targets")
        if not open_targets:
            return 0, 1

        ts = len(open_targets)
        s = 0
        for name, desired_state in open_targets.items():
            openables = get_objects_with_name_and_prop(
                name, "openable", state.metadata
            )
            if desired_state is None:
                desired_state = True
            if any(obj.get("isOpen") == desired_state for obj in openables):
                s += 1

        return s, ts

    def reset(self):
        super().reset()


class PickTwoObjAndPlaceTask(BaseTask):
    """
    pick_two_obj_and_place task
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def goal_satisfied(self, state):
        # check if two objects of 'object' class are in any receptacle of 'parent' class
        pcs = self.goal_conditions_met(state)
        return pcs[0] == pcs[1]

    def goal_conditions_met(self, state):
        ts = 2
        s = 0

        targets = self.get_targets()
        receptacles = get_objects_with_name_and_prop(
            targets["parent"], "receptacle", state.metadata
        )
        pickupables = get_objects_with_name_and_prop(
            targets["object"], "pickupable", state.metadata
        )

        # check if object needs to be sliced
        if "Sliced" in targets["object"]:
            ts += 2
            s += min(len([p for p in pickupables if "Sliced" in p["objectId"]]), 2)

        # placing each object counts as a goal_condition
        s += min(
            np.max(
                [
                    sum(
                        [
                            1
                            if r["receptacleObjectIds"] is not None
                            and p["objectId"] in r["receptacleObjectIds"]
                            else 0
                            for p in pickupables
                        ]
                    )
                    for r in receptacles
                ]
            ),
            2,
        )
        return s, ts

    def reset(self):
        super().reset()


class LookAtObjInLightTask(BaseTask):
    """
    look_at_obj_in_light task
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def goal_satisfied(self, state):
        # check if any object of 'object' class is being held in front of 'toggle' object that is turned on
        pcs = self.goal_conditions_met(state)
        return pcs[0] == pcs[1]

    def goal_conditions_met(self, state):
        ts = 2
        s = 0

        targets = self.get_targets()
        toggleables = get_objects_with_name_and_prop(
            targets["toggle"], "toggleable", state.metadata
        )
        pickupables = get_objects_with_name_and_prop(
            targets["object"], "pickupable", state.metadata
        )
        inventory_objects = state.metadata["inventoryObjects"]

        # check if object needs to be sliced
        if "Sliced" in targets["object"]:
            ts += 1
            if len([p for p in pickupables if "Sliced" in p["objectId"]]) >= 1:
                s += 1

        # check if the right object is in hand
        if len(inventory_objects) > 0 and inventory_objects[0]["objectId"] in [
            p["objectId"] for p in pickupables
        ]:
            s += 1
        # check if the lamp is visible and turned on
        if np.any([t["isToggled"] and t["visible"] for t in toggleables]):
            s += 1

        return s, ts

    def reset(self):
        super().reset()


class PickHeatThenPlaceInRecepTask(BaseTask):
    """
    pick_heat_then_place_in_recep task
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def goal_satisfied(self, state):
        # check if any object of 'object' class inside receptacle of 'parent' class is hot
        pcs = self.goal_conditions_met(state)
        return pcs[0] == pcs[1]

    def goal_conditions_met(self, state):
        ts = 3
        s = 0

        targets = self.get_targets()
        receptacles = get_objects_with_name_and_prop(
            targets["parent"], "receptacle", state.metadata
        )
        pickupables = get_objects_with_name_and_prop(
            targets["object"], "pickupable", state.metadata
        )

        # check if object needs to be sliced
        if "Sliced" in targets["object"]:
            ts += 1
            if len([p for p in pickupables if "Sliced" in p["objectId"]]) >= 1:
                s += 1

        objs_in_place = [
            p["objectId"]
            for p in pickupables
            for r in receptacles
            if r["receptacleObjectIds"] is not None
            and p["objectId"] in r["receptacleObjectIds"]
        ]
        objs_heated = [
            p["objectId"]
            for p in pickupables
            if p["objectId"] in self.env.heated_objects
        ]

        # check if object is in the receptacle
        if len(objs_in_place) > 0:
            s += 1
        # check if some object was heated
        if len(objs_heated) > 0:
            s += 1
        # check if the object is both in the receptacle and hot
        if np.any([obj_id in objs_heated for obj_id in objs_in_place]):
            s += 1

        return s, ts

    def reset(self):
        super().reset()


class PickCoolThenPlaceInRecepTask(BaseTask):
    """
    pick_cool_then_place_in_recep task
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def goal_satisfied(self, state):
        # check if any object of 'object' class inside receptacle of 'parent' class is cold
        pcs = self.goal_conditions_met(state)
        return pcs[0] == pcs[1]

    def goal_conditions_met(self, state):
        ts = 3
        s = 0

        targets = self.get_targets()
        receptacles = get_objects_with_name_and_prop(
            targets["parent"], "receptacle", state.metadata
        )
        pickupables = get_objects_with_name_and_prop(
            targets["object"], "pickupable", state.metadata
        )

        if "Sliced" in targets["object"]:
            ts += 1
            if len([p for p in pickupables if "Sliced" in p["objectId"]]) >= 1:
                s += 1

        objs_in_place = [
            p["objectId"]
            for p in pickupables
            for r in receptacles
            if r["receptacleObjectIds"] is not None
            and p["objectId"] in r["receptacleObjectIds"]
        ]
        objs_cooled = [
            p["objectId"]
            for p in pickupables
            if p["objectId"] in self.env.cooled_objects
        ]

        # check if object is in the receptacle
        if len(objs_in_place) > 0:
            s += 1
        # check if some object was cooled
        if len(objs_cooled) > 0:
            s += 1
        # check if the object is both in the receptacle and cold
        if np.any([obj_id in objs_cooled for obj_id in objs_in_place]):
            s += 1

        return s, ts

    def reset(self):
        super().reset()


class PickCleanThenPlaceInRecepTask(BaseTask):
    """
    pick_clean_then_place_in_recep task
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def goal_satisfied(self, state):
        # check if any object of 'object' class inside receptacle of 'parent' class is clean
        pcs = self.goal_conditions_met(state)
        return pcs[0] == pcs[1]

    def goal_conditions_met(self, state):
        ts = 3
        s = 0

        targets = self.get_targets()
        receptacles = get_objects_with_name_and_prop(
            targets["parent"], "receptacle", state.metadata
        )
        pickupables = get_objects_with_name_and_prop(
            targets["object"], "pickupable", state.metadata
        )

        if "Sliced" in targets["object"]:
            ts += 1
            if len([p for p in pickupables if "Sliced" in p["objectId"]]) >= 1:
                s += 1

        objs_in_place = [
            p["objectId"]
            for p in pickupables
            for r in receptacles
            if r["receptacleObjectIds"] is not None
            and p["objectId"] in r["receptacleObjectIds"]
        ]
        objs_cleaned = [
            p["objectId"]
            for p in pickupables
            if p["objectId"] in self.env.cleaned_objects
        ]

        # check if object is in the receptacle
        if len(objs_in_place) > 0:
            s += 1
        # check if some object was cleaned
        if len(objs_cleaned) > 0:
            s += 1
        # check if the object is both in the receptacle and clean
        if np.any([obj_id in objs_cleaned for obj_id in objs_in_place]):
            s += 1

        return s, ts

    def reset(self):
        super().reset()


class PickAndPlaceWithMovableRecepTask(BaseTask):
    """
    pick_and_place_with_movable_recep task
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def goal_satisfied(self, state):
        # check if any object of 'object' class is inside any movable receptacle of 'mrecep' class at receptacle of 'parent' class
        pcs = self.goal_conditions_met(state)
        return pcs[0] == pcs[1]

    def goal_conditions_met(self, state):
        ts = 3
        s = 0

        targets = self.get_targets()
        receptacles = get_objects_with_name_and_prop(
            targets["parent"], "receptacle", state.metadata
        )
        pickupables = get_objects_with_name_and_prop(
            targets["object"], "pickupable", state.metadata
        )
        movables = get_objects_with_name_and_prop(
            targets["mrecep"], "pickupable", state.metadata
        )

        # check if object needs to be sliced
        if "Sliced" in targets["object"]:
            ts += 1
            if len([p for p in pickupables if "Sliced" in p["objectId"]]) >= 1:
                s += 1

        pickup_in_place = [
            p
            for p in pickupables
            for m in movables
            if "receptacleObjectIds" in p
            and m["receptacleObjectIds"] is not None
            and p["objectId"] in m["receptacleObjectIds"]
        ]
        movable_in_place = [
            m
            for m in movables
            for r in receptacles
            if "receptacleObjectIds" in r
            and r["receptacleObjectIds"] is not None
            and m["objectId"] in r["receptacleObjectIds"]
        ]
        # check if the object is in the final receptacle
        if len(pickup_in_place) > 0:
            s += 1
        # check if the movable receptacle is in the final receptacle
        if len(movable_in_place) > 0:
            s += 1
        # check if both the object and movable receptacle stack is in the final receptacle
        if np.any(
            [
                np.any([p["objectId"] in m["receptacleObjectIds"] for p in pickupables])
                and np.any(
                    [r["objectId"] in m["parentReceptacles"] for r in receptacles]
                )
                for m in movables
                if m["parentReceptacles"] is not None
                and m["receptacleObjectIds"] is not None
            ]
        ):
            s += 1

        return s, ts

    def reset(self):
        super().reset()


def get_task(task_type, traj, env, args, reward_type="sparse", max_episode_length=2000):
    task_class_str = task_type.replace("_", " ").title().replace(" ", "") + "Task"

    if task_class_str in globals():
        task = globals()[task_class_str]
        return task(
            traj,
            env,
            args,
            reward_type=reward_type,
            max_episode_length=max_episode_length,
        )
    else:
        raise Exception("Invalid task_type %s" % task_class_str)
