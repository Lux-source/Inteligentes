class Problem:
    def __init__(self, initial_state, goal_state, intersections, segments):
        self.initial_state = initial_state  # Starting State
        self.goal_state = goal_state  # Goal State
        self.intersections = (
            intersections  # Dictionary of State objects (key: identifier)
        )
        self.segments = segments  # List of Action objects

    def actions(self, state):
        """Returns the actions available from a given state."""
        return state.neighbors

    def result(self, state, action):
        """Returns the resulting state after taking an action."""
        return action.destination

    def goal_test(self, state):
        """Checks if the given state is the goal state."""
        return state == self.goal_state

    def path_cost(self, cost_so_far, state1, action, state2):
        """Returns the cost of a solution path that arrives at state2 from state1."""
        return cost_so_far + action.cost()
