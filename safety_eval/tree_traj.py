from dataclasses import dataclass
from typing import Optional, Union, Sequence, List, Set, Dict, Tuple, Any, Iterator
from collections import deque
from treelib.tree import Tree, Node
import uuid

@dataclass
class Proposition(object):
    name: str
    args: List[str]

    def __str__(self):
        return '{}({})'.format(self.name, ', '.join(self.args))
    
def build_id_to_name_dict(objs: List[str]):
    import re
    # an object is in the form of "name.id"
    pattern = re.compile(r'(\w+)\.(\d+)')
    id_to_name = {}
    for obj in objs:
        match = pattern.search(obj)
        assert match, f'Failed to match pattern in {obj}'
        id_to_name[match.group(2)] = match.group(1)
    return id_to_name

def build_name_to_id_list_dict(objs: List[str]):
    import re
    # an object is in the form of "name.id"
    pattern = re.compile(r'(\w+)\.(\d+)')
    name_to_id_list = {}
    for obj in objs:
        match = pattern.search(obj)
        assert match, f'Failed to match pattern in {obj}'
        if match.group(1) not in name_to_id_list:
            name_to_id_list[match.group(1)] = []
        name_to_id_list[match.group(1)].append(match.group(2))
    
    # sort the list for each name
    for name, id_list in name_to_id_list.items():
        name_to_id_list[name] = sorted(id_list)
    return name_to_id_list

def get_first_object_id(name_to_id_list: Dict[str, List[str]], obj_name):
    assert obj_name in name_to_id_list, f'{obj_name} not in name_to_id_list'
    assert len(name_to_id_list[obj_name]) > 0, f'{obj_name} has no id'
    return name_to_id_list[obj_name][0]

def get_all_object_ids(name_to_id_list: Dict[str, List[str]], obj_name):
    assert obj_name in name_to_id_list, f'{obj_name} not in name_to_id_list'
    return name_to_id_list[obj_name]

def get_random_object_id(name_to_id_list: Dict[str, List[str]], obj_name):
    import random
    assert obj_name in name_to_id_list, f'{obj_name} not in name_to_id_list'
    assert len(name_to_id_list[obj_name]) > 0, f'{obj_name} has no id'
    return random.choice(name_to_id_list[obj_name])

def get_id_name(id_to_name: Dict[str, str], obj_id):
    assert obj_id in id_to_name, f'{obj_id} not in id_to_name'
    return id_to_name[obj_id]

def has_id(object:str) -> bool:
    import re
    # an object can be in the form of "name.id" or just "name"
    pattern = re.compile(r'(\w+)\.(\d+)')
    match = pattern.search(object)
    # if the object has an id, return True, if the object only has a name, return False
    return bool(match)

def full_id_objects(objects: List[str]) -> bool:
    '''
    This function assesses whether there is any object in the list of objects that does not have an id.
    ''' 
    for obj in objects:
        if not has_id(obj):
            return False
    return True

def parse_propositions(props: Union[Sequence[Proposition], Proposition], objects: List[str]):
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
                object_str = f'{arg}.{first_id}'
                parsed_args.append(object_str)
        new_props.append(Proposition(prop_name, parsed_args))
    return new_props

@dataclass
class State(object):
    objects_state: Dict[str, List[str]]
    propositions: Sequence[Proposition]

    def __str__(self):
        props_str = ', '.join(map(str, self.propositions))
        objects_str = ', '.join(f"{name}: [{', '.join(states)}]" for name, states in self.objects_state.items())
        return f'State(propositions=[{props_str}], objects_state=[{objects_str}])'

    def eval(self, proposition: Proposition):
        return str(proposition) in self.propositions


@dataclass
class Action(object):
    name: str
    args: List[str]

    def __str__(self):
        return '{}({})'.format(self.name, ', '.join(self.args))

    def equals(self, other: 'Action'):
        # this part needs further modification, but not now.
        # when I start to write runtime checking, I will get back to this part to see what can I do to make this work
        return self.name == other.name and tuple(self.args) == tuple(other.args)
    

class StateActionSequence(object):
    def __init__(self, states: List[State], actions: List[Action]):
        self.states = states
        self.actions = actions

        assert len(states) == len(actions) + 1 or len(states) == len(actions) == 0, 'The number of states and actions should differ by 1.'

    def iter_sa_pairs(self):
        for state, action in zip(self.states, self.actions):
            yield state, action
        if len(self.states) > 0:
            yield self.states[-1], None

    def exclude_prefix(self, prefix_length: int):
        if prefix_length >= len(self.states):
            return StateActionSequence([], [])
        return StateActionSequence(self.states[prefix_length:], self.actions[prefix_length:])

    def __str__(self):
        return 'StateActionSequence({}, {})'.format(self.states, self.actions)
    

class TrajectoryNode:
    """
    A node in the trajectory tree, representing a state in the environment.
    Each node can have multiple children representing possible next states.
    """
    def __init__(
        self, 
        state: State,
        action: Optional[Action] = None, 
        parent: Optional['TrajectoryNode'] = None,
        node_id: Optional[str] = None
    ):
        self.state = state
        self.action = action  # Action that led to this state
        self.parent = parent
        self.children: List['TrajectoryNode'] = []
        self.node_id = node_id if node_id else str(uuid.uuid4())
        
    def add_child(self, state: State, action: Optional[Action] = None) -> 'TrajectoryNode':
        """
        Add a child node with the given state and action.
        
        Args:
            state: The state of the child node
            action: The action that led to this state
            
        Returns:
            The newly created child node
        """
        child = TrajectoryNode(state=state, action=action, parent=self)
        self.children.append(child)
        return child
        
    def is_leaf(self) -> bool:
        """Check if this node is a leaf node (has no children)."""
        return len(self.children) == 0
    
    def get_path_from_root(self) -> List['TrajectoryNode']:
        """Get the path from root to this node."""
        path = []
        current = self
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def get_state_action_sequence(self) -> List[Tuple[State, Optional[Action]]]:
        """Convert the path from root to this node into a state-action sequence."""
        path = self.get_path_from_root()
        return [(node.state, node.action) for node in path]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary for serialization."""
        return {
            "state": self.state.to_dict() if hasattr(self.state, 'to_dict') else str(self.state),
            "action": str(self.action) if self.action else None,
            "node_id": self.node_id,
            "children": [child.node_id for child in self.children]
        }

class TrajectoryTree:
    """
    A tree structure for capturing different possible trajectories in LLM planning.
    Unlike a linear StateActionSequence, this allows for branching paths.
    """
    def __init__(self, initial_state: State):
        self.root = TrajectoryNode(state=initial_state)
        self._nodes: Dict[str, TrajectoryNode] = {self.root.node_id: self.root}
        
    def add_node(self, parent_id: str, state: State, action: Optional[Action] = None) -> str:
        """
        Add a new node to the tree under the specified parent.
        
        Args:
            parent_id: ID of the parent node
            state: State for the new node
            action: Action that led to this state
            
        Returns:
            ID of the newly created node
        """
        if parent_id not in self._nodes:
            raise ValueError(f"Parent node with ID {parent_id} does not exist")
            
        parent = self._nodes[parent_id]
        child = parent.add_child(state=state, action=action)
        self._nodes[child.node_id] = child
        return child.node_id
    
    def get_node(self, node_id: str) -> Optional[TrajectoryNode]:
        """Get a node by its ID."""
        return self._nodes.get(node_id)
    
    def get_leaves(self) -> List[TrajectoryNode]:
        """Get all leaf nodes in the tree."""
        return [node for node in self._nodes.values() if node.is_leaf()]
    
    def get_paths_to_leaves(self) -> List[List[TrajectoryNode]]:
        """Get all paths from root to leaves."""
        leaves = self.get_leaves()
        return [leaf.get_path_from_root() for leaf in leaves]
    
    def get_state_action_sequences(self) -> List[List[Tuple[State, Optional[Action]]]]:
        """Get all possible state-action sequences in the tree."""
        paths = self.get_paths_to_leaves()
        return [[(node.state, node.action) for node in path] for path in paths]
    
    def to_treelib(self) -> Tree:
        """Convert to a treelib.Tree for visualization."""
        tree = Tree()
        
        # Add the root node first
        tree.create_node(
            tag=str(self.root.state),
            identifier=self.root.node_id,
            data=self.root
        )
        
        # Use a queue for breadth-first traversal
        queue = [self.root]
        while queue:
            current = queue.pop(0)
            
            for child in current.children:
                # Add the child node to the tree
                action_str = f" <- {child.action}" if child.action else ""
                tree.create_node(
                    tag=f"{str(child.state)}{action_str}",
                    identifier=child.node_id,
                    parent=current.node_id,
                    data=child
                )
                queue.append(child)
                
        return tree
    
    def visualize(self) -> None:
        """Visualize the tree using treelib."""
        tree = self.to_treelib()
        tree.show()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tree to a dictionary for serialization."""
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self._nodes.items()},
            "root": self.root.node_id
        }

    def find_paths_satisfying(self, condition_func) -> List[List[TrajectoryNode]]:
        """
        Find all paths from root to leaves that satisfy a given condition.
        
        Args:
            condition_func: A function that takes a path and returns True if the path satisfies the condition
            
        Returns:
            List of paths that satisfy the condition
        """
        paths = self.get_paths_to_leaves()
        return [path for path in paths if condition_func(path)]
    
    def prune(self, node_id: str) -> None:
        """
        Prune the tree by removing the specified node and all its children.
        
        Args:
            node_id: ID of the node to prune
        """
        if node_id == self.root.node_id:
            raise ValueError("Cannot prune the root node")
            
        node = self._nodes.get(node_id)
        if not node:
            return
            
        # Get all descendant nodes
        to_remove = set()
        queue = [node]
        while queue:
            current = queue.pop(0)
            to_remove.add(current.node_id)
            queue.extend(current.children)
            
        # Remove node from parent's children list
        if node.parent:
            node.parent.children = [child for child in node.parent.children if child.node_id != node_id]
            
        # Remove all nodes from the _nodes dictionary
        for nid in to_remove:
            self._nodes.pop(nid, None)
    
    def merge(self, other: 'TrajectoryTree', node_id: str) -> None:
        """
        Merge another trajectory tree into this one at the specified node.
        
        Args:
            other: The TrajectoryTree to merge
            node_id: ID of the node where the other tree should be merged
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node with ID {node_id} does not exist")
            
        node = self._nodes[node_id]
        
        # Function to recursively copy nodes from other tree
        def copy_subtree(parent: TrajectoryNode, other_node: TrajectoryNode) -> None:
            for child in other_node.children:
                new_child = parent.add_child(state=child.state, action=child.action)
                self._nodes[new_child.node_id] = new_child
                copy_subtree(new_child, child)
                
        # Copy nodes from other tree
        copy_subtree(node, other.root)

    def iter_sa_pairs(self) -> Iterator[Tuple['State', Optional['Action']]]:
        """
        Iterate over all state-action pairs in the tree using breadth-first traversal.
        This provides a way to examine all states in the tree for CTL evaluation.
        """
        queue = deque([self.root])
        visited = set()
        
        while queue:
            node = queue.popleft()
            
            # Avoid revisiting nodes (though in a tree this shouldn't happen)
            if node.node_id in visited:
                continue
            visited.add(node.node_id)
            
            # Yield the state-action pair for this node
            yield node.state, node.action
            
            # Add children to queue for further exploration
            for child in node.children:
                queue.append(child)

def visualize_trajectory_tree(tree: TrajectoryTree, output_file="trajectory_tree.txt"):
    """
    Visualizes a TrajectoryTree by writing its structure to a text file.
    
    Args:
        tree: TrajectoryTree to visualize
        output_file: Path to the output text file (default: "trajectory_tree.txt")
    """
    with open(output_file, 'w') as f:
        def _visualize_node(node, depth=0):
            indent = "  " * depth
            state_str = str(node.state)
            action_str = f" (via {node.action})" if node.action else ""
            f.write(f"{indent}Node {node.node_id}: {state_str}{action_str}\n")
            for child in node.children:
                _visualize_node(child, depth + 1)
        
        _visualize_node(tree.root)
    
    print(f"Trajectory tree visualization written to {output_file}")