class Node:

    node_counter = 0  # Contador para saber la order de generacion

    def __init__(self, state, parent=None, action=None, path_cost=0, heuristic_cost=0):
        self.state = state  # State object
        self.parent = parent  # Parent Node
        self.action = action  # Action taken to reach this node
        self.path_cost = (
            path_cost  # Total cost from the root node to this node (g_cost)
        )
        self.heuristic_cost = heuristic_cost  # Heuristic estimate to the goal (h_cost)
        self.f_cost = (
            self.path_cost + self.heuristic_cost
        )  # Total cost for A* (f = g + h)
        self.depth = 0 if parent is None else parent.depth + 1
        self.order = Node.node_counter  # Para ordenar
        Node.node_counter += 1

    def __lt__(self, other):
        # Comparison based on `f_cost` and then `id` for tie-breaking
        return (self.f_cost, self.state.order) < (other.f_cost, other.order)

    def __repr__(self):
        return f"Node(State={self.state.identifier}, Path cost={self.path_cost}, Heuristic={self.heuristic_cost}, Depth={self.depth})"
