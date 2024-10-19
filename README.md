---
# Inteligentes

Welcome to the **Inteligentes** project! This repository contains implementations of various search algorithms to navigate through a city represented as a graph. Below is the detailed documentation for each file in the project, explaining their purposes, methods, and attributes.
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

- Represents a road segment connecting two intersections (states) in the city. It encapsulates information about the distance and speed limit of the segment.

**Attributes**:

- `origin`: The starting state (intersection) for this action.
- `destination`: The ending state (intersection) for this action.
- `distance`: The distance between `origin` and `destination`.
- `speed`: The speed limit on this segment.

**Methods**:

- `cost(self)`: Calculates and returns the travel time as the cost by dividing `distance` by `speed`.
- `__repr__(self)`: Provides a string representation of the `Action` object for easy debugging and logging.

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
        # Add neighbors to the origin state
        origin.neighbors.append((destination, segment))

    # Pre-sort neighbors for each state to eliminate sorting during search
    for state in intersections.values():
        state.neighbors.sort(key=lambda x: x[0].identifier, reverse=True)

    initial_state = intersections[data["initial"]]
    goal_state = intersections[data["final"]]

    return Problem(initial_state, goal_state, intersections, segments)
```

**Purpose**:

- Loads and parses the JSON file containing information about intersections and road segments. It creates `State` and `Action` objects based on the data and returns a `Problem` instance.

**Function**:

- `loadJSON(file_path)`: Reads the JSON file, initializes `State` and `Action` objects, links intersections with their neighbors, pre-sorts the neighbors for each state, and returns a `Problem` instance containing all necessary information for the search algorithms.

**Key Operations**:

1. **Parsing Intersections**: Creates `State` objects for each intersection and stores them in a dictionary for easy access.
2. **Parsing Segments**: Creates `Action` objects for each road segment, linking origin and destination `State` objects.
3. **Linking Neighbors**: Appends each destination and corresponding action to the `neighbors` list of the origin state.
4. **Pre-sorting Neighbors**: Sorts the `neighbors` list for each state based on the destination identifier in reverse order to maintain consistent traversal order during searches.

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
        return f"Node(State={self.state.identifier}, Path cost={self.path_cost}, Depth={self.depth})"
```

**Purpose**:

- Represents a node in the search tree. It contains information about the current state, the action taken to reach it, and the cumulative cost and depth of the node.

**Attributes**:

- `state`: The current state (intersection) this node represents.
- `parent`: The parent node leading to this node.
- `action`: The action taken to reach this node from its parent.
- `path_cost`: The cumulative cost from the root node to this node.
- `depth`: The depth of this node in the search tree.

**Methods**:

- `__repr__(self)`: Provides a string representation of the `Node` object, displaying the state identifier, path cost, and depth for debugging purposes.

**Notes**:

- The `expand` method has been removed/commented out to streamline node creation directly within the search functions.

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

- Represents the problem of navigating the city by defining the initial state, goal state, and methods to interact with states and actions.

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
import time
import heapq
import itertools
from datetime import timedelta
from collections import deque
from abc import ABC, abstractmethod
from Node import Node

# Helper function to extract the solution path
def solution(node):
    path = []
    while node.parent is not None:
        path.append(node.action)
        node = node.parent
    path.reverse()
    return path

# Formatting the output for easier readability
def format_solution_details(solution_path, stats):
    formatted_solution = []
    for action in solution_path:
        formatted_solution.append(
            f"{action.origin.identifier} → {action.destination.identifier}, {action.distance}"
        )
    formatted_solution_str = "[" + ", ".join(formatted_solution) + "]"

    formatted_output = (
        f"Generated nodes: {stats['nodes_generated']}\n"
        f"Expanded nodes: {stats['nodes_explored']}\n"
        f"Execution time: {stats['execution_time']}\n"
        f"Solution length: {stats['solution_depth']}\n"
        f"Solution cost: {stats['solution_cost']}\n"
        f"Solution: {formatted_solution_str}"
    )
    return formatted_output

# Convert time to the desired format days:hours:min:seconds
def format_time(seconds):
    td = timedelta(seconds=seconds)
    return str(td)

# Breadth-First Search (BFS) function
def breadth_first_search(problem):
    start_time = time.time()
    frontier = deque([Node(problem.initial_state)])
    explored = set()
    nodes_generated = 1
    nodes_explored = 0

    while frontier:
        node = frontier.popleft()
        nodes_explored += 1

        if problem.goal_test(node.state):
            solution_path = solution(node)
            execution_time = format_time(time.time() - start_time)
            solution_cost = node.path_cost
            return solution_path, {
                "nodes_generated": nodes_generated,
                "nodes_explored": nodes_explored,
                "solution_depth": node.depth,
                "solution_cost": solution_cost,
                "execution_time": execution_time,
            }

        explored.add(node.state.identifier)

        for next_state, action in node.state.neighbors:
            if (
                next_state.identifier not in explored
                and all(front_node.state.identifier != next_state.identifier for front_node in frontier)
            ):
                child = Node(next_state, node, action, node.path_cost + action.cost())
                frontier.append(child)
                nodes_generated += 1

    return None, {
        "nodes_generated": nodes_generated,
        "nodes_explored": nodes_explored,
        "solution_depth": None,
        "solution_cost": None,
        "execution_time": format_time(time.time() - start_time),
    }

# Depth-First Search (DFS) function
def depth_first_search(problem):
    start_time = time.time()
    frontier = [Node(problem.initial_state)]
    frontier_states = set([problem.initial_state.identifier])  # Track states in frontier
    explored = set()
    nodes_generated = 1
    nodes_explored = 0

    while frontier:
        node = frontier.pop()
        frontier_states.remove(node.state.identifier)  # Remove from frontier states
        nodes_explored += 1

        if problem.goal_test(node.state):
            solution_path = solution(node)
            execution_time = format_time(time.time() - start_time)
            solution_cost = node.path_cost
            return solution_path, {
                "nodes_generated": nodes_generated,
                "nodes_explored": nodes_explored,
                "solution_depth": node.depth,
                "solution_cost": solution_cost,
                "execution_time": execution_time,
            }

        explored.add(node.state.identifier)

        for next_state, action in node.state.neighbors:
            if (
                next_state.identifier not in explored
                and next_state.identifier not in frontier_states
            ):
                child = Node(next_state, node, action, node.path_cost + action.cost())
                frontier.append(child)
                frontier_states.add(next_state.identifier)
                nodes_generated += 1

    execution_time = format_time(time.time() - start_time)

    return None, {
        "nodes_generated": nodes_generated,
        "nodes_explored": nodes_explored,
        "solution_depth": None,
        "solution_cost": None,
        "execution_time": execution_time,
    }

# A* Search function
def a_star_search(problem, heuristic):
    start_time = time.time()
    frontier = []
    counter = itertools.count()
    start_node = Node(problem.initial_state)
    f_cost = heuristic(start_node.state, problem.goal_state)
    heapq.heappush(frontier, (f_cost, next(counter), start_node))
    explored = set()
    frontier_state_costs = {problem.initial_state.identifier: 0}
    nodes_generated = 1
    nodes_explored = 0

    while frontier:
        _, _, node = heapq.heappop(frontier)
        nodes_explored += 1

        if problem.goal_test(node.state):
            solution_path = solution(node)
            execution_time = format_time(time.time() - start_time)
            solution_cost = node.path_cost
            return solution_path, {
                "nodes_generated": nodes_generated,
                "nodes_explored": nodes_explored,
                "solution_depth": node.depth,
                "solution_cost": solution_cost,
                "execution_time": execution_time,
            }

        explored.add(node.state.identifier)

        for next_state, action in node.state.neighbors:
            child_cost = node.path_cost + action.cost()
            if next_state.identifier not in explored and (
                next_state.identifier not in frontier_state_costs
                or child_cost < frontier_state_costs[next_state.identifier]
            ):
                child = Node(next_state, node, action, child_cost)
                f_cost = child_cost + heuristic(child.state, problem.goal_state)
                heapq.heappush(frontier, (f_cost, next(counter), child))
                frontier_state_costs[next_state.identifier] = child_cost
                nodes_generated += 1

    return None, {
        "nodes_generated": nodes_generated,
        "nodes_explored": nodes_explored,
        "solution_depth": None,
        "solution_cost": None,
        "execution_time": format_time(time.time() - start_time),
    }

# Best-First Search function
def best_first_search(problem, heuristic):
    start_time = time.time()
    frontier = []
    counter = itertools.count()
    start_node = Node(problem.initial_state)
    h_cost = heuristic(start_node.state, problem.goal_state)
    heapq.heappush(frontier, (h_cost, next(counter), start_node))
    explored = set()
    nodes_generated = 1
    nodes_explored = 0

    while frontier:
        _, _, node = heapq.heappop(frontier)
        nodes_explored += 1

        if problem.goal_test(node.state):
            solution_path = solution(node)
            execution_time = format_time(time.time() - start_time)
            solution_cost = node.path_cost
            return solution_path, {
                "nodes_generated": nodes_generated,
                "nodes_explored": nodes_explored,
                "solution_depth": node.depth,
                "solution_cost": solution_cost,
                "execution_time": execution_time,
            }

        explored.add(node.state.identifier)

        for next_state, action in node.state.neighbors:
            if next_state.identifier not in explored:
                child = Node(next_state, node, action, node.path_cost + action.cost())
                h_cost = heuristic(child.state, problem.goal_state)
                heapq.heappush(frontier, (h_cost, next(counter), child))
                nodes_generated += 1

    return None, {
        "nodes_generated": nodes_generated,
        "nodes_explored": nodes_explored,
        "solution_depth": None,
        "solution_cost": None,
        "execution_time": format_time(time.time() - start_time),
    }

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

# Best-First Search class
class BestFirstSearch(SearchAlgorithm):
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def search(self, problem):
        return best_first_search(problem, self.heuristic)
```

**Purpose**:

- Implements various search algorithms (BFS, DFS, A\*, Best-First) to navigate through the city graph. It also includes helper functions for solution path reconstruction and output formatting.

**Functions and Classes**:

1. **Helper Functions**:

   - `solution(node)`: Reconstructs the solution path by traversing from the goal node back to the initial node.
   - `format_solution_details(solution_path, stats)`: Formats the solution path and performance statistics for easy readability.
   - `format_time(seconds)`: Converts elapsed time in seconds to a `days:hours:min:seconds` format.

2. **Search Algorithms**:

   - `breadth_first_search(problem)`: Implements the BFS algorithm using a queue (`deque`). It tracks nodes generated and explored, and returns the solution path and statistics upon finding the goal.
   - `depth_first_search(problem)`: Implements the DFS algorithm using a stack (list). It uses a `frontier_states` set for efficient duplicate checks and returns the solution path and statistics upon finding the goal.
   - `a_star_search(problem, heuristic)`: Implements the A\* search algorithm using a priority queue (`heapq`). It utilizes a heuristic function to guide the search towards the goal efficiently.
   - `best_first_search(problem, heuristic)`: Implements the Best-First Search algorithm using a priority queue, guided solely by the heuristic function.

3. **Heuristic Function**:

   - `heuristic(state, goal_state)`: Calculates the Euclidean distance between the current state and the goal state as the heuristic.

4. **Abstract Base Class**:

   - `SearchAlgorithm(ABC)`: Defines an abstract base class for all search algorithms, ensuring consistency in their implementation.

5. **Search Algorithm Classes**:
   - `BreadthFirstSearch`, `DepthFirstSearch`, `AStarSearch`, `BestFirstSearch`: Concrete classes inheriting from `SearchAlgorithm`, each implementing their respective search strategies.

**Key Optimizations**:

- **Pre-sorted Neighbors**: Neighbors are sorted during problem initialization in `ImportJSON.py` to eliminate the need for sorting during each node expansion, enhancing performance.
- **Efficient Explored Set Management**: In DFS, a separate `frontier_states` set is maintained for **O(1)** duplicate checks, reducing computational overhead.
- **Lightweight Node Creation**: Nodes are created efficiently without unnecessary methods or attributes, ensuring faster node expansions.

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

- Represents an intersection (state) in the city graph, storing its unique identifier, geographical coordinates, and connected neighbors.

**Attributes**:

- `identifier`: Unique identifier of the state (e.g., intersection name or number).
- `latitude`, `longitude`: Geographical coordinates of the intersection.
- `neighbors`: List of tuples containing neighboring `State` objects and the corresponding `Action` objects (road segments).

**Methods**:

- `__eq__(self, other)`: Determines equality based on the state's identifier.
- `__hash__(self)`: Enables `State` objects to be used in sets and as dictionary keys by returning a hash of the identifier.
- `__repr__(self)`: Provides a string representation of the `State` object for debugging purposes.

**Notes**:

- Ensures immutability post-creation to maintain consistent hashing and equality checks.
- Neighbors are pre-sorted during initialization for efficient traversal during searches.

---

### **Testing.py**

```python
from ImportJSON import loadJSON
from Search import (
    BreadthFirstSearch,
    DepthFirstSearch,
    AStarSearch,
    BestFirstSearch,
    heuristic,
    format_solution_details,
)

# Load the problem from the JSON file
json_file_path = r"C:\Users\andre\DocumentsCopia\ACurso_2425\Primer Cuatri\INTELIGENTES\Lab\Entrega 1\Lab 1. space state search-20241016\examples_with_solutions\problems\huge\calle_agustina_aroca_albacete_5000_0.json"
problem = loadJSON(json_file_path)

# Test BFS
print("Testing BFS:")
bfs_search = BreadthFirstSearch()
bfs_result, bfs_stats = bfs_search.search(problem)
if bfs_result:
    output = format_solution_details(bfs_result, bfs_stats)
    print(output)
else:
    print("No solution found")

# Test DFS
print("\nTesting DFS:")
dfs_search = DepthFirstSearch()
dfs_result, dfs_stats = dfs_search.search(problem)
if dfs_result:
    output = format_solution_details(dfs_result, dfs_stats)
    print(output)
else:
    print("No solution found")

# Test A* Search
print("\nTesting A* Search:")
a_star_search = AStarSearch(heuristic)
a_star_result, a_star_stats = a_star_search.search(problem)
if a_star_result:
    output = format_solution_details(a_star_result, a_star_stats)
    print(output)
else:
    print("No solution found")

# Test Best-First Search
print("\nTesting Best-First Search:")
best_first_search = BestFirstSearch(heuristic)
best_first_result, best_first_stats = best_first_search.search(problem)
if best_first_result:
    output = format_solution_details(best_first_result, best_first_stats)
    print(output)
else:
    print("No solution found")
```

**Purpose**:

- Tests each search algorithm (BFS, DFS, A\*, Best-First) using a problem instance loaded from a JSON file. It runs the algorithms, captures their results and statistics, and prints formatted outputs for verification.

**Key Operations**:

1. **Loading the Problem**:

   - Utilizes `loadJSON` from `ImportJSON.py` to parse the JSON file and initialize the `Problem` instance containing states and actions.

2. **Executing Search Algorithms**:

   - **Breadth-First Search (BFS)**:
     - Instantiates `BreadthFirstSearch` and executes the `search` method.
   - **Depth-First Search (DFS)**:
     - Instantiates `DepthFirstSearch` and executes the `search` method.
   - **A\* Search**:
     - Instantiates `AStarSearch` with the provided `heuristic` function and executes the `search` method.
   - **Best-First Search**:
     - Instantiates `BestFirstSearch` with the provided `heuristic` function and executes the `search` method.

3. **Output Formatting**:
   - Uses `format_solution_details` to neatly format and display the solution paths along with performance statistics such as nodes generated, nodes explored, solution depth, solution cost, and execution time.

**Notes**:

- Ensure that the `json_file_path` points to a valid JSON file containing the problem instance.
- The printed outputs provide insights into the efficiency and effectiveness of each search algorithm.
