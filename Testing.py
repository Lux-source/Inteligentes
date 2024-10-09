from ImportJSON import loadJSON
from Search import BreadthFirstSearch, DepthFirstSearch, AStarSearch, heuristic

# Load the problem from the JSON file
json_file_path = r"C:\Users\andre\DocumentsCopia\ACurso_2425\Primer Cuatri\INTELIGENTES\Lab\Entrega 1\pr1_SSII_English\problems\large\calle_agustina_aroca_albacete_1000_0.json"
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
