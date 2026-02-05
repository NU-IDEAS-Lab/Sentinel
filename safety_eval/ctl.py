from dataclasses import dataclass
from typing import Optional, Union, Sequence, List, Dict
from .tree_traj import Proposition, Action, State, TrajectoryTree
from collections import deque
import re


def build_id_to_name_dict(objs: List[str]):
    import re

    # an object is in the form of "name.id"
    pattern = re.compile(r"(\w+)\.(\d+)")
    id_to_name = {}
    for obj in objs:
        match = pattern.search(obj)
        assert match, f"Failed to match pattern in {obj}"
        id_to_name[match.group(2)] = match.group(1)
    return id_to_name


def build_name_to_id_list_dict(objs: List[str]):
    import re

    # an object is in the form of "name.id"
    pattern = re.compile(r"(\w+)\.(\d+)")
    name_to_id_list = {}
    for obj in objs:
        match = pattern.search(obj)
        assert match, f"Failed to match pattern in {obj}"
        if match.group(1) not in name_to_id_list:
            name_to_id_list[match.group(1)] = []
        name_to_id_list[match.group(1)].append(match.group(2))

    # sort the list for each name
    for name, id_list in name_to_id_list.items():
        name_to_id_list[name] = sorted(id_list)
    return name_to_id_list


def get_first_object_id(name_to_id_list: Dict[str, List[str]], obj_name):
    assert obj_name in name_to_id_list, f"{obj_name} not in name_to_id_list"
    assert len(name_to_id_list[obj_name]) > 0, f"{obj_name} has no id"
    return name_to_id_list[obj_name][0]


def get_all_object_ids(name_to_id_list: Dict[str, List[str]], obj_name):
    assert obj_name in name_to_id_list, f"{obj_name} not in name_to_id_list"
    return name_to_id_list[obj_name]


def get_random_object_id(name_to_id_list: Dict[str, List[str]], obj_name):
    import random

    assert obj_name in name_to_id_list, f"{obj_name} not in name_to_id_list"
    assert len(name_to_id_list[obj_name]) > 0, f"{obj_name} has no id"
    return random.choice(name_to_id_list[obj_name])


def get_id_name(id_to_name: Dict[str, str], obj_id):
    assert obj_id in id_to_name, f"{obj_id} not in id_to_name"
    return id_to_name[obj_id]


def has_id(object: str) -> bool:
    import re

    # an object can be in the form of "name.id" or just "name"
    pattern = re.compile(r"(\w+)\.(\d+)")
    match = pattern.search(object)
    # if the object has an id, return True, if the object only has a name, return False
    return bool(match)


def full_id_objects(objects: List[str]) -> bool:
    """
    This function assesses whether there is any object in the list of objects that does not have an id.
    """
    for obj in objects:
        if not has_id(obj):
            return False
    return True


def parse_propositions(
    props: Union[Sequence[Proposition], Proposition], objects: List[str]
):
    if not full_id_objects(objects):
        return props if not isinstance(props, Proposition) else [props]
    name_to_id_dict = build_name_to_id_list_dict(objects)
    new_props = list()
    if isinstance(props, Proposition):
        props = [props]
    for prop in props:
        prop_name = prop.name
        prop_args = prop.args
        parsed_args = []
        for arg in prop_args:
            if has_id(arg):
                parsed_args.append(arg)
            else:
                first_id = get_first_object_id(name_to_id_dict, arg)
                object_str = f"{arg}.{first_id}"
                parsed_args.append(object_str)
        new_props.append(Proposition(prop_name, parsed_args))
    return new_props


@dataclass
class EvaluationResult(object):
    rv: bool
    """The result of the evaluation."""

    shortest_prefix: int
    """The length of the shortest prefix that satisfies the expression. If the expression is never satisfied, this should be -1."""


class CTLExpression(object):
    """A simple CTL expression.

    There are two types of expressions: state goals and temporal goals.

    - State goals are evaluated on a single state-action pair. When we evaluate a state-goal expression on a trajectory,
        we return True if there exists a state-action pair in the trajectory that satisfies the expression.
    - Temporal goals are evaluated on a trajectory. It can only be evaluated on a trajectory, but not on a single state-action pair.

    :meth:`eval_state` is used to evaluate the expression on a single state-action pair.
    :meth:`eval` is used to evaluate the expression on a trajectory.

    To handle free variables (in forall and exists), we use a variable mapping to map variables to our "guessed" grounding.
    """

    def __init__(self, is_state_goal: bool):
        self.is_state_goal = is_state_goal
        self.is_temporal_goal = not is_state_goal

    def eval_state(
        self, state: State, action: Optional[Action], variable_mapping: Dict[str, str]
    ) -> bool:
        """Evaluate the expression on the given state-action pair.

        Args:
            state: the state to evaluate the expression on.
            action: the action to evaluate the expression on.
            variable_mapping: a mapping from variables to their values.

        Returns:
            bool: the return value of the expression.
        """
        raise NotImplementedError("eval_state is not implemented.")

    def eval(
        self, trajectory: TrajectoryTree, variable_mapping: Dict[str, str]
    ) -> EvaluationResult:
        """Evaluate the expression on the given trajectory.

        Args:
            trajectory: the trajectory to evaluate the expression on.
            variable_mapping: a mapping from variables to their values.

        Returns:
            EvaluationResult: the evaluation result, including the boolean value and the shortest prefix length.
        """
        raise NotImplementedError("eval is not implemented.")


class CTLPrimitive(CTLExpression):
    def __init__(self, prop_or_action: Union[Proposition, Action]):
        super().__init__(is_state_goal=True)
        if isinstance(prop_or_action, Proposition):
            self.prop_or_action = prop_or_action
            self.is_proposition = True
            self.is_action = False
        elif isinstance(prop_or_action, Action):
            self.prop_or_action = prop_or_action
            self.is_proposition = False
            self.is_action = True
        elif hasattr(prop_or_action, "name") and hasattr(prop_or_action, "args"):
            self.prop_or_action = Proposition(
                prop_or_action.name, list(prop_or_action.args)
            )
            self.is_proposition = True
            self.is_action = False
        else:
            self.prop_or_action = prop_or_action
            self.is_proposition = False
            self.is_action = False

    def __str__(self):
        return str(self.prop_or_action)

    _SUFFIX_VAR_RE = re.compile(r"^(?P<base>.+)_(?P<suffix>[0-9]+)$")

    def ground(self, variable_mapping: Dict[str, str]):
        def _bind_arg(arg: str) -> str:
            match = self._SUFFIX_VAR_RE.match(arg)
            if match:
                suffix = match.group("suffix")
                return variable_mapping.get(suffix, arg)
            return variable_mapping.get(arg, arg)

        if self.is_proposition:
            base = (
                self.prop_or_action
                if isinstance(self.prop_or_action, Proposition)
                else Proposition(
                    self.prop_or_action.name, list(self.prop_or_action.args)
                )
            )
            bound_args = [_bind_arg(arg) for arg in base.args]
            if base.name == "NEAR":
                bound_args = sorted(bound_args)
            return CTLPrimitive(Proposition(base.name, bound_args))
        elif self.is_action:
            return CTLPrimitive(
                Action(
                    self.prop_or_action.name,
                    [_bind_arg(arg) for arg in self.prop_or_action.args],
                )
            )

    def eval_state(
        self, state: State, action: Optional[Action], variable_mapping: Dict[str, str]
    ) -> bool:
        ground_self = self.ground(variable_mapping)
        if self.is_proposition:
            # Handle both dictionary and list-based propositions
            return self._eval_proposition_in_state(state, ground_self.prop_or_action)
        elif self.is_action:
            if action is None:
                return False
            return action.equals(ground_self.prop_or_action)
        else:
            raise ValueError("Unknown prop_or_action type.")

    def _eval_proposition_in_state(
        self, state: State, proposition: Proposition
    ) -> bool:
        """Evaluate a proposition in a state, handling both dict and list formats"""

        # Create the string representation of the proposition
        prop_str = str(proposition)

        # Check if propositions is a dictionary (original format)
        if isinstance(state.propositions, dict):
            return prop_str in state.propositions

        # Check if propositions is a list (new format from parser)
        elif isinstance(state.propositions, list):
            for prop in state.propositions:
                if hasattr(prop, "name") and hasattr(prop, "args"):
                    # Create string representation from Proposition object
                    existing_prop_str = f"{prop.name}({', '.join(prop.args)})"
                    if existing_prop_str == prop_str:
                        return True
                elif isinstance(prop, str):
                    # Direct string comparison
                    if prop == prop_str:
                        return True

            # If not found in propositions list, check objects_state for single-argument propositions
            # This handles cases like OPEN(kitchen_cabinet.1000) where OPEN is a state of kitchen_cabinet.1000
            if hasattr(state, "objects_state") and len(proposition.args) == 1:
                obj_id = proposition.args[0]
                prop_name = proposition.name

                if obj_id in state.objects_state:
                    obj_states = state.objects_state[obj_id]
                    if isinstance(obj_states, list) and prop_name in obj_states:
                        return True
                    elif isinstance(obj_states, str) and prop_name == obj_states:
                        return True

            return False

        else:
            # Fallback: try the original eval method
            try:
                return state.eval(proposition)
            except Exception:
                print(
                    f"Warning: Unknown propositions format: {type(state.propositions)}"
                )
                return False

    def eval(
        self, trajectory: TrajectoryTree, variable_mapping: Dict[str, str]
    ) -> EvaluationResult:
        for i, (s, a) in enumerate(trajectory.iter_sa_pairs()):
            if self.eval_state(s, a, variable_mapping):
                return EvaluationResult(rv=True, shortest_prefix=i + 1)
        return EvaluationResult(rv=False, shortest_prefix=-1)


# TODO: Change the following statements to CTL structure
class CTLNot(CTLExpression):
    def __init__(self, child: CTLExpression):
        super().__init__(is_state_goal=child.is_state_goal)
        self.child = child

    def __str__(self):
        return "Not({})".format(self.child)

    def eval_state(
        self, state: State, action: Optional[Action], variable_mapping: Dict[str, str]
    ) -> bool:
        return not self.child.eval_state(state, action, variable_mapping)

    def eval(
        self, trajectory: TrajectoryTree, variable_mapping: Dict[str, str]
    ) -> EvaluationResult:
        if self.is_state_goal:
            node_queue = deque([trajectory.root])
            node_idx = -1
            while node_queue:
                node = node_queue.popleft()
                if self.eval_state(node.state, node.action, variable_mapping):
                    node_idx = len(node.get_path_from_root()) - 1
                    return EvaluationResult(rv=True, shortest_prefix=node_idx)
                for ch in node.children:
                    node_queue.append(ch)
            return EvaluationResult(rv=False, shortest_prefix=node_idx)
        else:
            result = self.child.eval(trajectory, variable_mapping)
            shortest_prefix = -1 if result.rv else result.shortest_prefix
            return EvaluationResult(rv=not result.rv, shortest_prefix=shortest_prefix)


# TODO: Change the following statements to CTL structure
class CTLAnd(CTLExpression):
    def __init__(self, children: List[CTLExpression]):
        super().__init__(is_state_goal=all(child.is_state_goal for child in children))
        self.children = children

    def __str__(self):
        return "And({})".format(", ".join(map(str, self.children)))

    def eval_state(
        self, state: State, action: Optional[Action], variable_mapping: Dict[str, str]
    ) -> bool:
        return all(
            child.eval_state(state, action, variable_mapping) for child in self.children
        )

    def eval(
        self, trajectory: TrajectoryTree, variable_mapping: Dict[str, str]
    ) -> EvaluationResult:
        if self.is_state_goal:
            for i, (s, a) in enumerate(trajectory.iter_sa_pairs()):
                if self.eval_state(s, a, variable_mapping):
                    return EvaluationResult(rv=True, shortest_prefix=i + 1)
            return EvaluationResult(rv=False, shortest_prefix=-1)
        else:
            results = [
                child.eval(trajectory, variable_mapping) for child in self.children
            ]
            all_true = all(result.rv for result in results)

            if all_true:
                max_prefix = max(
                    result.shortest_prefix
                    for result in results
                    if result.shortest_prefix >= 0
                )
                return EvaluationResult(rv=True, shortest_prefix=max_prefix)
            else:
                return EvaluationResult(rv=False, shortest_prefix=-1)


# TODO: Change the following statements to CTL structure
class CTLOr(CTLExpression):
    def __init__(self, children: List[CTLExpression]):
        super().__init__(is_state_goal=all(child.is_state_goal for child in children))
        self.children = children

    def __str__(self):
        return "Or({})".format(", ".join(map(str, self.children)))

    def eval_state(
        self, state: State, action: Optional[Action], variable_mapping: Dict[str, str]
    ) -> bool:
        return any(
            child.eval_state(state, action, variable_mapping) for child in self.children
        )

    def eval(
        self, trajectory: TrajectoryTree, variable_mapping: Dict[str, str]
    ) -> EvaluationResult:
        if self.is_state_goal:
            node_queue = deque([trajectory.root])
            node_idx = -1  # use bfs distance or node to root distance
            while node_queue:
                node = node_queue.popleft()
                if any(
                    child.eval_state(node.state, node.action, variable_mapping)
                    for child in self.children
                ):
                    node_idx = len(node.get_path_from_root()) - 1
                    return EvaluationResult(rv=True, shortest_prefix=node_idx)
                for ch in node.children:
                    node_queue.append(ch)
            return EvaluationResult(rv=False, shortest_prefix=node_idx)
        else:
            results = [
                child.eval(trajectory, variable_mapping) for child in self.children
            ]
            all_true = any(result.rv for result in results)
            node_idx = (
                min([result.shortest_prefix for result in results]) if all_true else -1
            )
            return EvaluationResult(rv=all_true, shortest_prefix=node_idx)


# TODO: Change the following statements to CTL structure
class CTLAllThen(CTLExpression):
    def __init__(self, child: CTLExpression):  # changed to child: CTLExpression
        super().__init__(is_state_goal=False)  # default to False
        self.child = child

    def __str__(self):
        return "AX({})".format(", ".join(map(str, self.children)))  # AllThen

    def eval(
        self, trajectory: TrajectoryTree, variable_mapping: Dict[str, str]
    ) -> EvaluationResult:
        if len(trajectory.root.children) == 0:
            return EvaluationResult(rv=False, shortest_prefix=-1)

        for child_node in trajectory.root.children:
            if self.child.is_state_goal:
                if not self.child.eval_state(
                    child_node.state, child_node.action, variable_mapping
                ):
                    return EvaluationResult(rv=False, shortest_prefix=-1)
            else:
                subtree = TrajectoryTree(child_node.state)
                subtree.root.children = child_node.children
                child_result = self.child.eval(subtree, variable_mapping)
                if not child_result.rv:
                    return EvaluationResult(rv=False, shortest_prefix=-1)
        return EvaluationResult(rv=True, shortest_prefix=1)


# TODO: Change the following statements to CTL structure
class CTLAllEventually(CTLExpression):
    def __init__(self, child: CTLExpression):
        super().__init__(is_state_goal=False)
        self.child = child

    def __str__(self):
        return "AF({})".format(str(self.child))

    def eval(
        self, trajectory: TrajectoryTree, variable_mapping: Dict[str, str]
    ) -> EvaluationResult:
        if self.child.is_state_goal:
            if self.child.eval_state(
                trajectory.root.state, trajectory.root.action, variable_mapping
            ):
                return EvaluationResult(rv=True, shortest_prefix=0)
        else:
            child_result = self.child.eval(trajectory, variable_mapping)
            if child_result.rv:
                return EvaluationResult(
                    rv=True, shortest_prefix=child_result.shortest_prefix
                )

        if len(trajectory.root.children) == 0:
            return EvaluationResult(rv=False, shortest_prefix=-1)

        all_paths_satisfy = True
        max_prefix_length = 0

        for child_node in trajectory.root.children:
            subtree = TrajectoryTree(child_node.state)
            subtree.root.action = child_node.action
            subtree.root.children = child_node.children
            subtree_result = self.eval(subtree, variable_mapping)

            if not subtree_result.rv:
                all_paths_satisfy = False
                break

            if subtree_result.shortest_prefix >= 0:
                max_prefix_length = max(
                    max_prefix_length, subtree_result.shortest_prefix + 1
                )

        if all_paths_satisfy:
            return EvaluationResult(rv=True, shortest_prefix=max_prefix_length)
        else:
            return EvaluationResult(rv=False, shortest_prefix=-1)


# TODO: Change the following statements to CTL structure
class CTLAllAlways(CTLExpression):
    def __init__(self, child: CTLExpression):
        super().__init__(is_state_goal=child.is_state_goal)
        self.child = child

    def __str__(self):
        return "AG({})".format(", ".join(map(str, self.children)))

    def eval(
        self, trajectory: TrajectoryTree, variable_mapping: Dict[str, str]
    ) -> EvaluationResult:
        if self.child.is_state_goal:
            if not self.child.eval_state(
                trajectory.root.state, trajectory.root.action, variable_mapping
            ):
                return EvaluationResult(rv=False, shortest_prefix=-1)
        else:
            child_result = self.child.eval(trajectory, variable_mapping)
            if not child_result.rv:
                return EvaluationResult(rv=False, shortest_prefix=-1)

        if len(trajectory.root.children) == 0:
            return EvaluationResult(rv=True, shortest_prefix=0)

        for child_node in trajectory.root.children:
            subtree = TrajectoryTree(child_node.state)
            subtree.root.action = child_node.action
            subtree.root.children = child_node.children

            subtree_result = self.eval(subtree, variable_mapping)
            if not subtree_result.rv:
                return EvaluationResult(rv=False, shortest_prefix=-1)

        return EvaluationResult(rv=True, shortest_prefix=0)


# TODO: Change the following statements to CTL structure
class CTLAllUntil(CTLExpression):
    def __init__(self, left: CTLExpression, right: CTLExpression):
        super().__init__(is_state_goal=left.is_state_goal and right.is_state_goal)
        self.left = left
        self.right = right

    def __str__(self):
        return "AU({}, {})".format(str(self.left), str(self.right))

    def eval(
        self, trajectory: TrajectoryTree, variable_mapping: Dict[str, str]
    ) -> EvaluationResult:
        if self.right.is_state_goal:
            if self.right.eval_state(
                trajectory.root.state, trajectory.root.action, variable_mapping
            ):
                return EvaluationResult(rv=True, shortest_prefix=0)
        else:
            right_result = self.right.eval(trajectory, variable_mapping)
            if right_result.rv:
                return EvaluationResult(
                    rv=True, shortest_prefix=right_result.shortest_prefix
                )

        if self.left.is_state_goal:
            if not self.left.eval_state(
                trajectory.root.state, trajectory.root.action, variable_mapping
            ):
                return EvaluationResult(rv=False, shortest_prefix=-1)
        else:
            left_result = self.left.eval(trajectory, variable_mapping)
            if not left_result.rv:
                return EvaluationResult(rv=False, shortest_prefix=-1)

        if len(trajectory.root.children) == 0:
            return EvaluationResult(rv=False, shortest_prefix=-1)

        all_paths_satisfy = True
        max_prefix_length = 0

        for child_node in trajectory.root.children:
            subtree = TrajectoryTree(child_node.state)
            subtree.root.action = child_node.action
            subtree.root.children = child_node.children

            subtree_result = self.eval(subtree, variable_mapping)

            if not subtree_result.rv:
                all_paths_satisfy = False
                break

            if subtree_result.shortest_prefix >= 0:
                max_prefix_length = max(
                    max_prefix_length, subtree_result.shortest_prefix + 1
                )

        if all_paths_satisfy:
            return EvaluationResult(rv=True, shortest_prefix=max_prefix_length)
        else:
            return EvaluationResult(rv=False, shortest_prefix=-1)
