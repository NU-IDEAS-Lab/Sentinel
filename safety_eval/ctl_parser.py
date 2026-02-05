import json
from typing import Any, Dict, List, Optional, Union

try:
    from .tree_traj import (
        Action,
        Proposition,
        State,
        TrajectoryTree,
        visualize_trajectory_tree,
    )
except ImportError:  # pragma: no cover - fallback for script execution
    from tree_traj import (
        Action,
        Proposition,
        State,
        TrajectoryTree,
        visualize_trajectory_tree,
    )


class CTLParser:
    """Parser for trajectory data into TrajectoryTree objects."""

    def __init__(self):
        pass

    def parse_proposition(self, prop_string: str) -> Proposition:
        """Parse a proposition string into a Proposition object."""
        name, rest = prop_string.split("(", 1)
        args_part = rest.rstrip(")")
        args = [arg.strip() for arg in args_part.split(",")] if args_part else []
        return Proposition(name=name, args=args)

    def parse_action(self, action_dict: Optional[Dict[str, Any]]) -> Action:
        """Parse an action dictionary into an Action object."""
        if not action_dict:
            return Action(name="NoOp", args=[])
        if not isinstance(action_dict, dict):
            raise ValueError(f"Invalid action dict: {action_dict}")

        action_name = action_dict.get("action") or "NoOp"
        args: List[str] = []

        object_id = action_dict.get("objectId")
        if object_id:
            args.append(str(object_id).strip())

        return Action(name=action_name, args=args)

    def parse_state(self, state_dict: Dict[str, List[str]]) -> State:
        """Parse a state dictionary into a State object."""
        nodes = state_dict["nodes"]
        edges = state_dict["edges"]

        objects_state = {}
        proposition_list = []

        for node in nodes:
            state_components = node.split(", states:")
            object_name = state_components[0].split(", ")[0]
            object_state = [
                item.strip().strip("'") for item in state_components[1][1:-1].split(",")
            ]
            objects_state[object_name] = sorted(object_state)

        for edge in edges:
            proposition_list.append(self.parse_proposition(edge))

        return State(objects_state, proposition_list)

    def to_tree_traj(
        self, traj_data: List[Union[Dict[str, List[str]], Dict[str, Any]]]
    ) -> TrajectoryTree:
        """Convert trajectory data into a TrajectoryTree object."""
        root_state = self.parse_state(traj_data[0])
        tree = TrajectoryTree(root_state)
        current_node = tree.root

        for i in range(1, len(traj_data), 2):
            if i + 1 >= len(traj_data):
                break

            node_state = self.parse_state(traj_data[i + 1])
            node_action = self.parse_action(traj_data[i])
            next_node_id = tree.add_node(
                parent_id=current_node.node_id, state=node_state, action=node_action
            )
            current_node = tree.get_node(next_node_id)
        return tree

    def _sort_json_data(self, data):
        """
        Sort nodes and edges alphabetically in the JSON data.

        Args:
            data (list): List containing alternating state dictionaries and action dicts

        Returns:
            list: Updated data with sorted nodes and edges
        """
        sorted_data = []

        for item in data:
            if isinstance(item, dict) and ("nodes" in item or "edges" in item):
                # This is a state dictionary
                sorted_item = {}

                # Sort nodes alphabetically if present
                if "nodes" in item:
                    sorted_item["nodes"] = sorted(item["nodes"])

                # Sort edges alphabetically if present
                if "edges" in item:
                    sorted_item["edges"] = sorted(item["edges"])

                sorted_data.append(sorted_item)
            else:
                # This is an action dict, keep as is
                sorted_data.append(item)

        return sorted_data

    def parse_from_file(self, file_path: str) -> TrajectoryTree:
        """Parse trajectory data from a JSON file."""
        with open(file_path, "r") as file:
            traj_data = self._sort_json_data(json.load(file))
        return self.to_tree_traj(traj_data)

    def parse_from_data(
        self, traj_data: List[Union[Dict[str, List[str]], Dict[str, Any]]]
    ) -> TrajectoryTree:
        """Parse trajectory data from loaded data."""
        return self.to_tree_traj(traj_data)


def merge_trees(*trees: "TrajectoryTree") -> "TrajectoryTree":
    """
    Merge any number of trajectory trees into a single tree.
    Trees are expected to have the same initial state.
    Nodes with identical states are merged, while different states create separate branches.

    Args:
        *trees: Variable number of TrajectoryTree objects to merge

    Returns:
        A new merged TrajectoryTree

    Raises:
        ValueError: If no trees are provided or if trees don't have the same initial state
    """
    if not trees:
        raise ValueError("At least one tree must be provided")

    if len(trees) == 1:
        # If only one tree, create a copy and return it
        return _copy_tree(trees[0])

    # Helper function to check if two states are equal
    def states_equal(state1: State, state2: State) -> bool:
        """Check if two states are equal by comparing their objects_state and propositions."""
        # Compare objects_state dictionaries
        if state1.objects_state != state2.objects_state:
            return False

        # Compare propositions - convert to sets of string representations for comparison
        props1 = {str(prop) for prop in state1.propositions}
        props2 = {str(prop) for prop in state2.propositions}

        return props1 == props2

    # Helper function to create a unique signature for a state
    def create_state_signature(state: State) -> str:
        """Create a unique signature for a state that can be used for comparison."""
        # Sort objects_state items for consistent comparison
        objects_items = sorted(state.objects_state.items())
        objects_str = str(objects_items)

        # Sort propositions by string representation for consistent comparison
        props_strs = sorted([str(prop) for prop in state.propositions])
        props_str = str(props_strs)

        return f"{objects_str}|{props_str}"

    # Verify all trees have the same initial state
    first_tree = trees[0]
    for i, tree in enumerate(trees[1:], 1):
        if not states_equal(first_tree.root.state, tree.root.state):
            raise ValueError(
                f"Tree {i} has a different initial state than tree 0. All trees must have the same initial state to be merged"
            )

    # Create a new tree with the same initial state as the first tree
    merged_tree = TrajectoryTree(first_tree.root.state)

    # Use a queue to traverse all trees simultaneously
    # Each item in queue is (merged_node, [(tree_index, tree_node), ...])
    queue = [(merged_tree.root, [(i, tree.root) for i, tree in enumerate(trees)])]

    while queue:
        merged_node, tree_nodes = queue.pop(0)

        # Collect all unique children states from all trees at this level
        children_by_state = {}  # state_signature -> [(tree_index, tree_node, child_node), ...]

        for tree_index, tree_node in tree_nodes:
            for child in tree_node.children:
                # Create a signature for the state to group identical states
                state_sig = create_state_signature(child.state)

                if state_sig not in children_by_state:
                    children_by_state[state_sig] = []
                children_by_state[state_sig].append((tree_index, tree_node, child))

        # For each unique state, create a merged child
        for state_sig, child_groups in children_by_state.items():
            # Use the first child's state and action as the representative
            representative_child = child_groups[0][2]

            # Create merged child node
            merged_child_id = merged_tree.add_node(
                parent_id=merged_node.node_id,
                state=representative_child.state,
                action=representative_child.action,
            )
            merged_child = merged_tree.get_node(merged_child_id)

            # Add to queue for further processing
            # Only include tree nodes that actually have this child state
            next_tree_nodes = [
                (tree_index, child) for tree_index, tree_node, child in child_groups
            ]
            queue.append((merged_child, next_tree_nodes))

    return merged_tree


def _copy_tree(original_tree: "TrajectoryTree") -> "TrajectoryTree":
    """
    Create a deep copy of a TrajectoryTree.

    Args:
        original_tree: The tree to copy

    Returns:
        A new TrajectoryTree that is a copy of the original
    """
    # Create new tree with same initial state
    new_tree = TrajectoryTree(original_tree.root.state)

    # Use a queue to copy all nodes
    queue = [(original_tree.root, new_tree.root)]

    while queue:
        original_node, new_node = queue.pop(0)

        for child in original_node.children:
            # Add child to new tree
            new_child_id = new_tree.add_node(
                parent_id=new_node.node_id, state=child.state, action=child.action
            )
            new_child = new_tree.get_node(new_child_id)

            # Add to queue for further processing
            queue.append((child, new_child))

    return new_tree


if __name__ == "__main__":
    parser = CTLParser()

    # Parse your trajectory files
    tree1 = parser.parse_from_file("../test1.json")
    tree2 = parser.parse_from_file("../test.json")
    tree3 = parser.parse_from_file("../test.json")

    # Example 1: Merge all three trees at once
    merged_tree = merge_trees(tree1, tree2, tree3)
    print("Merging completed successfully!")
    breakpoint()

    # Visualize the results
    print(f"Original tree1 has {len(tree1._nodes)} nodes")
    # print(f"Original tree2 has {len(tree2._nodes)} nodes")
    # print(f"Original tree3 has {len(tree3._nodes)} nodes")
    print(f"Merged tree has {len(merged_tree._nodes)} nodes")
    print(f"Merged tree has {len(merged_tree.get_leaves())} leaf nodes")
    print(f"Merged tree has {len(merged_tree.get_paths_to_leaves())} possible paths")

    # Visualize the merged tree
    print("\nMerged tree structure:")
    merged_tree.visualize()

    # Save to file
    visualize_trajectory_tree(merged_tree, "merged_trajectory_tree.txt")
    print("Merged tree saved to 'merged_trajectory_tree.txt'")
