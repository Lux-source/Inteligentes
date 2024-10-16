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
