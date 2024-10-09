class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state  # State object
        self.parent = parent  # Parent Node
        self.action = action  # Action taken to reach this node
        self.path_cost = path_cost  # Total cost from the root node to this node
        self.depth = 0 if parent is None else parent.depth + 1

    def __repr__(self):
        return (
            f"Node(State={self.state}, Path cost={self.path_cost}, Depth={self.depth})"
        )

    def expand(self, problem):
        """Generates the successors of this node."""
        return [
            Node(next_state, self, action, self.path_cost + action.cost())
            for next_state, action in self.state.neighbors
        ]
