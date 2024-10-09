# Inteligentes
Here’s the documentation for each of the files in your project, detailing why we use them and explaining their methods/attributes.

---

### **Action.py**

```python
class Action:
    def __init__(self, origin, destination, distance, speed):
        self.origin = origin  # State object
        self.destination = destination  # State object
        self.distance = distance  # Distance between origin and destination
        self.speed = speed  # Speed limit on this segment

    def cost(self):
        # Calculate travel time as the cost
        return self.distance / self.speed

    def __repr__(self):
        return f"Action({self.origin.identifier} -> {self.destination.identifier}, distance={self.distance}, speed={self.speed})"
```

**Purpose**:
- Represents the action or road segment connecting two intersections (states) in the city. It includes information about distance and speed.

**Attributes**:
- `origin`: The starting state (intersection) for this action.
- `destination`: The ending state (intersection) for this action.
- `distance`: The distance between `origin` and `destination`.
- `speed`: The speed limit on this segment.

**Methods**:
- `cost(self)`: Returns the travel time as the cost by dividing `distance` by `speed`.
- `__repr__(self)`: Provides a string representation of the `Action` object for easy debugging.

---

### **ImportJSON.py**

```python
import json
from State import State
from Action import Action
from Problem import Problem

def loadJSON(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    intersections = {}
    for i_data in data["intersections"]:
        inter = State(
            identifier=i_data["identifier"],
            latitude=i_data["latitude"],
            longitude=i_data["longitude"],
        )
        intersections[inter.identifier] = inter

    segments = []
    for seg_data in data["segments"]:
        origin = intersections[seg_data["origin"]]
        destination = intersections[seg_data["destination"]]
        segment = Action(
            origin=origin,
            destination=destination,
            distance=seg_data["distance"],
            speed=seg_data["speed"],
        )
        segments.append(segment)
        origin.neighbors.append((destination, segment))

    initial_state = intersections[data["initial"]]
    goal_state = intersections[data["final"]]

    return Problem(initial_state, goal_state, intersections, segments)
```

**Purpose**:
- Loads and parses the JSON file containing information about intersections and road segments. It creates `State` and `Action` objects based on the data and returns a `Problem` instance.

**Function**:
- `loadJSON(file_path)`: Reads the JSON file and initializes the data into `State` and `Action` objects, linking intersections with their neighbors. It returns a `Problem` instance with all necessary information for the search algorithms.

---

### **Node.py**

```python
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
```

**Purpose**:
- Represents a node in the search tree. It contains information about the state, the action taken to reach it, and the cost and depth of the node.

**Attributes**:
- `state`: The current state (intersection) this node represents.
- `parent`: The parent node leading to this node.
- `action`: The action taken to reach this node from its parent.
- `path_cost`: The cumulative cost from the root node to this node.
- `depth`: The depth of this node in the search tree.

**Methods**:
- `expand(self, problem)`: Generates and returns a list of successor nodes by expanding this node’s neighbors.
- `__repr__(self)`: Provides a string representation for debugging purposes.

---

### **Problem.py**

```python
class Problem:
    def __init__(self, initial_state, goal_state, intersections, segments):
        self.initial_state = initial_state  # Starting State
        self.goal_state = goal_state  # Goal State
        self.intersections = intersections  # Dictionary of State objects (key: identifier)
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
```

**Purpose**:
- Represents the problem of navigating the city by defining the initial state, goal state, and the methods to interact with states and actions.

**Attributes**:
- `initial_state`: The starting state for the search.
- `goal_state`: The target state that the search aims to reach.
- `intersections`: Dictionary of all `State` objects, keyed by their identifier.
- `segments`: List of all `Action` objects (road segments).

**Methods**:
- `actions(self, state)`: Returns all available actions (segments) from the given state.
- `result(self, state, action)`: Returns the state resulting from taking an action.
- `goal_test(self, state)`: Checks if a state is the goal state.
- `path_cost(self, cost_so_far, state1, action, state2)`: Computes the cumulative cost of a path.

---

### **Search.py**

```python
from collections import deque
from abc import ABC, abstractmethod
from Node import Node
import heapq

# Helper function to extract the solution path
def solution(node):
    path = []
    while node.parent is not None:
        path.append(node.action)
        node = node.parent
    path.reverse()
    return path

# Breadth-First Search (BFS) function
def breadth_first_search(problem):
    ...

# Depth-First Search (DFS) function
def depth_first_search(problem):
    ...

# A* Search function
def a_star_search(problem, heuristic):
    ...

# Heuristic function for A*
def heuristic(state, goal_state):
    dx = state.latitude - goal_state.latitude
    dy = state.longitude - goal_state.longitude
    return (dx**2 + dy**2) ** 0.5

# Abstract Base Class for Search Algorithms
class SearchAlgorithm(ABC):
    @abstractmethod
    def search(self, problem):
        pass

# Breadth-First Search class
class BreadthFirstSearch(SearchAlgorithm):
    def search(self, problem):
        return breadth_first_search(problem)

# Depth-First Search class
class DepthFirstSearch(SearchAlgorithm):
    def search(self, problem):
        return depth_first_search(problem)

# A* Search class
class AStarSearch(SearchAlgorithm):
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def search(self, problem):
        return a_star_search(problem, self.heuristic)
```

**Purpose**:
- Contains implementations of search algorithms (BFS, DFS, and A*) and the heuristic for A* search. It also defines a base class for search algorithms for consistent structure.

**Classes and Methods**:
- `solution(node)`: Helper function that reconstructs the solution path from a node.
- `breadth_first_search(problem)`: Implements BFS using a queue.
- `depth_first_search(problem)`: Implements DFS using a stack.
- `a_star_search(problem, heuristic)`: Implements A* using a priority queue and heuristic function.
- `heuristic(state, goal_state)`: Calculates the Euclidean distance as the heuristic.
- `SearchAlgorithm(ABC)`: Abstract base class for consistency across search algorithms.
- `BreadthFirstSearch`, `DepthFirstSearch`, `AStarSearch`: Classes implementing specific search strategies.

---

### **State.py**

```python
class State:
    def __init__(self, identifier, latitude, longitude):
        self.identifier = identifier
        self.latitude = latitude
        self.longitude = longitude
        self.neighbors = []  # List of (neighbor_state, action/segment) tuples

    def __eq__(self, other):
        return isinstance(other, State) and self.identifier == other.identifier

    def __hash__(self):
        # Hash method to allow using State instances in sets or as dictionary keys
        return hash(self.identifier)

    def __repr__(self):
        return f"State({self.identifier}, {self.latitude}, {self.longitude})"
```

**Purpose**:
- Represents an intersection (node) in the city graph, storing its identifier, location, and neighbors.

**Attributes**:
- `identifier`: Unique ID of the state (intersection).
- `latitude`, `longitude`: Coordinates of the intersection.
- `neighbors`: List of connected intersections and actions.

**Methods**:
- `__eq__(self, other

)`: Checks equality based on the identifier.
- `__hash__(self)`: Allows using `State` instances in sets or as dictionary keys.
- `__repr__(self)`: Provides a string representation for debugging.

---

### **Testing.py**

```python
from ImportJSON import loadJSON
from Search import BreadthFirstSearch, DepthFirstSearch, AStarSearch, heuristic

# Load the problem from the JSON file
json_file_path = r"your_file_path_here"
problem = loadJSON(json_file_path)

# Test BFS
...

# Test DFS
...

# Test A* Search
...
```

**Purpose**:
- Tests each search algorithm using a problem loaded from a JSON file.

**Methods**:
- Runs BFS, DFS, and A* and prints the results and statistics for verification.

---

This documentation provides a concise explanation of why each part of the code is used and what each method or attribute represents. Let me know if you'd like further details!
