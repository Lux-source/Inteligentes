import time
import itertools
import heapq
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
            f"{action.origin.identifier} → {action.destination.identifier}, {action.cost():.5f}"
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
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    microseconds = td.microseconds

    if days > 0:
        return f"{days}:{hours}:{minutes:02}:{sec:02}.{microseconds:06}"
    else:
        return f"{hours}:{minutes:02}:{sec:02}.{microseconds:06}"


# Breadth-First Search (BFS) function
def breadth_first_search(problem):
    start_time = time.perf_counter()
    frontier = deque([Node(problem.initial_state)])
    explored = set()
    nodes_generated = 1
    nodes_explored = 0

    while frontier:
        node = frontier.popleft()
        nodes_explored += 1

        if problem.goal_test(node.state):
            solution_path = solution(node)
            execution_time = format_time(time.perf_counter() - start_time)
            solution_cost = format_time(node.path_cost)
            return solution_path, {
                "nodes_generated": nodes_generated,
                "nodes_explored": nodes_explored,
                "solution_depth": node.depth,
                "solution_cost": solution_cost,
                "execution_time": execution_time,
            }

        explored.add(node.state)

        for next_state, action in node.state.neighbors:
            if next_state not in explored and all(
                front_node.state != next_state for front_node in frontier
            ):
                child = Node(next_state, node, action, node.path_cost + action.cost())
                frontier.append(child)
                nodes_generated += 1

    return None, {
        "nodes_generated": nodes_generated,
        "nodes_explored": nodes_explored,
        "solution_depth": None,
        "solution_cost": None,
        "execution_time": format_time(time.perf_counter() - start_time),
    }


# Depth-First Search (DFS) function
def depth_first_search(problem):
    start_time = time.perf_counter()
    frontier = [Node(problem.initial_state)]
    explored = set()
    nodes_generated = 1
    nodes_explored = 0

    while frontier:
        node = frontier.pop()
        nodes_explored += 1

        if problem.goal_test(node.state):
            solution_path = solution(node)
            execution_time = format_time(time.perf_counter() - start_time)
            solution_cost = format_time(node.path_cost)
            return solution_path, {
                "nodes_generated": nodes_generated,
                "nodes_explored": nodes_explored,
                "solution_depth": node.depth,
                "solution_cost": solution_cost,
                "execution_time": execution_time,
            }

        explored.add(node.state)

        for next_state, action in node.state.neighbors:
            if next_state not in explored:
                child = Node(next_state, node, action, node.path_cost + action.cost())
                frontier.append(child)
                nodes_generated += 1

    execution_time = format_time(time.perf_counter() - start_time)

    return None, {
        "nodes_generated": nodes_generated,
        "nodes_explored": nodes_explored,
        "solution_depth": None,
        "solution_cost": None,
        "execution_time": execution_time,
    }


# A* Search function
def a_star_search(problem, heuristic):
    start_time = time.perf_counter()
    frontier = []
    start_node = Node(
        problem.initial_state,
        heuristic_cost=heuristic(problem.initial_state, problem.goal_state),
    )
    start_node.f_cost = (
        start_node.path_cost
        + start_node.heuristic_cost  # no sumar seg y metros, solo coste segundos en tiempo
    )  # Initial f_cost
    heapq.heappush(
        frontier, (start_node.f_cost, start_node.state.identifier, start_node)
    )

    explored = set()
    frontier_state_costs = {
        problem.initial_state: 0
    }  # Track minimum costs to each state
    nodes_generated = 1
    nodes_explored = 0

    while frontier:
        _, _, node = heapq.heappop(frontier)
        nodes_explored += 1

        if problem.goal_test(node.state):
            solution_path = solution(node)
            execution_time = format_time(time.perf_counter() - start_time)
            solution_cost = format_time(node.path_cost)
            return solution_path, {
                "nodes_generated": nodes_generated,
                "nodes_explored": nodes_explored,
                "solution_depth": node.depth,
                "solution_cost": solution_cost,
                "execution_time": execution_time,
            }

        explored.add(node.state)

        for next_state, action in node.state.neighbors:
            child_cost = node.path_cost + action.cost()
            if next_state not in explored and (
                next_state not in frontier_state_costs
                or child_cost < frontier_state_costs[next_state]
            ):
                child = Node(
                    next_state,
                    node,
                    action,
                    child_cost,
                    heuristic_cost=heuristic(next_state, problem.goal_state),
                )
                child.f_cost = (
                    child.path_cost + child.heuristic_cost
                )  # Calculate f_cost

                # Push to priority queue based on f_cost and id
                heapq.heappush(frontier, (child.f_cost, next_state.identifier, child))
                frontier_state_costs[next_state] = child_cost
                nodes_generated += 1

    execution_time = format_time(time.perf_counter() - start_time)
    return None, {
        "nodes_generated": nodes_generated,
        "nodes_explored": nodes_explored,
        "solution_depth": None,
        "solution_cost": None,
        "execution_time": execution_time,
    }


def best_first_search(problem, heuristic):
    start_time = time.perf_counter()
    frontier = []
    start_node = Node(problem.initial_state)
    start_node.h_cost = heuristic(
        start_node.state, problem.goal_state
    )  # Set heuristic cost for start node

    # push tuples con h_cost e id
    heapq.heappush(
        frontier,
        (
            start_node.h_cost,
            start_node.state.identifier,
            start_node,
        ),  # Cambiar id, por tiempo mas viejo (edad)
    )

    explored = set()
    nodes_generated = 1
    nodes_explored = 0

    while frontier:
        _, _, node = heapq.heappop(frontier)
        nodes_explored += 1

        if problem.goal_test(node.state):
            solution_path = solution(node)
            execution_time = format_time(time.perf_counter() - start_time)
            solution_cost = format_time(node.path_cost)
            return solution_path, {
                "nodes_generated": nodes_generated,
                "nodes_explored": nodes_explored,
                "solution_depth": node.depth,
                "solution_cost": solution_cost,
                "execution_time": execution_time,
            }

        explored.add(node.state)

        for next_state, action in node.state.neighbors:
            if next_state not in explored:
                child = Node(next_state, node, action, node.path_cost + action.cost())
                child.h_cost = heuristic(
                    child.state, problem.goal_state
                )  # Set heuristic cost

                # Push with h_cost and id
                heapq.heappush(frontier, (child.h_cost, next_state.identifier, child))
                nodes_generated += 1

    execution_time = format_time(time.perf_counter() - start_time)
    return None, {
        "nodes_generated": nodes_generated,
        "nodes_explored": nodes_explored,
        "solution_depth": None,
        "solution_cost": None,
        "execution_time": execution_time,
    }


# Heuristic function for A*
def heuristic(
    state, goal_state
):  # Aqu no debemos usar la distancia sino el tiempo geopy.distance
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
