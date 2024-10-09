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
    frontier = deque([Node(problem.initial_state)])
    explored = set()
    nodes_generated = 1
    nodes_explored = 0

    while frontier:
        node = frontier.popleft()
        nodes_explored += 1

        if problem.goal_test(node.state):
            return solution(node), {
                "nodes_generated": nodes_generated,
                "nodes_explored": nodes_explored,
                "path_cost": node.path_cost,
                "solution_depth": node.depth,
            }

        explored.add(node.state)

        for next_state, action in sorted(
            node.state.neighbors, key=lambda x: x[0].identifier
        ):
            if next_state not in explored and all(
                front_node.state != next_state for front_node in frontier
            ):
                child = Node(next_state, node, action, node.path_cost + action.cost())
                frontier.append(child)
                nodes_generated += 1

    return None, {
        "nodes_generated": nodes_generated,
        "nodes_explored": nodes_explored,
        "path_cost": None,
        "solution_depth": None,
    }


# Depth-First Search (DFS) function
def depth_first_search(problem):
    frontier = [Node(problem.initial_state)]
    explored = set()
    nodes_generated = 1
    nodes_explored = 0

    while frontier:
        node = frontier.pop()
        nodes_explored += 1

        if problem.goal_test(node.state):
            return solution(node), {
                "nodes_generated": nodes_generated,
                "nodes_explored": nodes_explored,
                "path_cost": node.path_cost,
                "solution_depth": node.depth,
            }

        explored.add(node.state)

        for next_state, action in sorted(
            node.state.neighbors, key=lambda x: x[0].identifier, reverse=True
        ):
            if next_state not in explored and all(
                front_node.state != next_state for front_node in frontier
            ):
                child = Node(next_state, node, action, node.path_cost + action.cost())
                frontier.append(child)
                nodes_generated += 1

    return None, {
        "nodes_generated": nodes_generated,
        "nodes_explored": nodes_explored,
        "path_cost": None,
        "solution_depth": None,
    }


# A* Search function
def a_star_search(problem, heuristic):
    frontier = []
    heapq.heappush(frontier, (0, Node(problem.initial_state)))
    explored = set()
    nodes_generated = 1
    nodes_explored = 0

    while frontier:
        _, node = heapq.heappop(frontier)
        nodes_explored += 1

        if problem.goal_test(node.state):
            return solution(node), {
                "nodes_generated": nodes_generated,
                "nodes_explored": nodes_explored,
                "path_cost": node.path_cost,
                "solution_depth": node.depth,
            }

        explored.add(node.state)

        for next_state, action in node.state.neighbors:
            if next_state not in explored:
                child = Node(next_state, node, action, node.path_cost + action.cost())
                f_cost = child.path_cost + heuristic(child.state, problem.goal_state)
                heapq.heappush(frontier, (f_cost, child))
                nodes_generated += 1

    return None, {
        "nodes_generated": nodes_generated,
        "nodes_explored": nodes_explored,
        "path_cost": None,
        "solution_depth": None,
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
