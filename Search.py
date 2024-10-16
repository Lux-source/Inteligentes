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

def GreedybestFirstSearch(problem, heuristic):
    """Implements the Greedy Best-First Search algorithm."""
    # Initialize the priority queue with the start node
    start_node = Node(problem.initial_state)
    frontier = []
    heapq.heappush(frontier, (heuristic(start_node.state, problem.goal_state), start_node))
    nodes_generated = 1
    nodes_explored = 0
    explored = set()  # Keep track of explored states

    while frontier:
        # Get the node with the lowest heuristic value (closest to the goal)
        _, current_node = heapq.heappop(frontier)
        nodes_explored += 1 
        
        # If the goal is reached, return the solution path
        if problem.goal_test(current_node.state):
            return solution(current_node), {
                "nodes_generated": nodes_generated,
                "nodes_explored": nodes_explored,
                "path_cost": current_node.path_cost,
                "solution_depth": current_node.depth,
            }

        explored.add(current_node.state)

        # Expand the current node and add its neighbors to the priority queue
        for action in problem.actions(current_node.state):
            child = current_node.expand(problem)
            for child_node in child:
                if child_node.state not in explored:
                    child_node.heuristic = heuristic(child_node.state, problem.goal_state)
                    heapq.heappush(frontier, (child_node.heuristic, child_node))
                    nodes_generated += 1

    return None, {
        "nodes_generated: ": nodes_generated,
        "nodes_explored: ": nodes_explored, 
        "path_cost: ": None,
        "Solution depth: ":None,
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
    
class GreedyBestFirstSearch(SearchAlgorithm):
    def __init__(self, heuristic):
        self.heuristic = heuristic
    
    def search(self, problem):
        return GreedybestFirstSearch(problem, self.heuristic)
