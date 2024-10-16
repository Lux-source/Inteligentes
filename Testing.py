from ImportJSON import loadJSON
from Search import BreadthFirstSearch, DepthFirstSearch, AStarSearch, heuristic, GreedyBestFirstSearch

# Load the problem from the JSON file
json_file_path = r"/Users/carmen/repository/Inteligentes/problems/huge/calle_agustina_aroca_albacete_5000_0.json"
problem = loadJSON(json_file_path)

# Test BFS
print("Testing BFS:")
bfs_search = BreadthFirstSearch()
bfs_result, bfs_stats = bfs_search.search(problem)
print("BFS Path:", bfs_result if bfs_result else "No solution found")
print("BFS Stats:", bfs_stats)

# Test DFS
print("\nTesting DFS:")
dfs_search = DepthFirstSearch()
dfs_result, dfs_stats = dfs_search.search(problem)
print("DFS Path:", dfs_result if dfs_result else "No solution found")
print("DFS Stats:", dfs_stats)

# Test A* Search
print("\nTesting A* Search:")
a_star_search = AStarSearch(heuristic)
a_star_result, a_star_stats = a_star_search.search(problem)
print("A* Path:", a_star_result if a_star_result else "No solution found")
print("A* Stats:", a_star_stats)

#Testing Greedy Best-First Search
print("Greedy Best-First Search solution: ")
gbfs = GreedyBestFirstSearch(heuristic)
gbfs_solution_path, gbfs_solution_stats = gbfs.search(problem)
print(gbfs_solution_path if gbfs_solution_path else "No solution found")
#print(gbfs_solition_stats)