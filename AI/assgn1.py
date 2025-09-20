from collections import deque
import copy

class AI_Assgn_1:
    """
    AI Assignment 1 Solution class.

    Solves a 4x4 grid coloring problem with connectivity and adjacency constraints.
    """
    
    def __init__(self):
        """
        Initialize the problem with initial state and data structures.
        """
        self.grid_size = 4 
        self.colors = ['R', 'G']  # Red, Green

        self.initial_state = self.create_initial_state() 

        self.graph = {} 
        self.adjacency_matrix = []
        
    def create_initial_state(self):
        """
        Create initial 4x4 grid with (2,3) colored Red.
        """
        state = [[None for _ in range(4)] for _ in range(4)]  # Empty 4x4 grid
        state[1][2] = 'R'  # Initial state as defined in question
        return state
    
    def state_to_string(self, state):
        """
        Convert 2D grid state to string for hashing and storage.
        Improves performance by reducing memory footprint.
        """
        result = "" 
        for row in state:
            for cell in row:
                result += str(cell) if cell else "."
        return result
    
    def string_to_state(self, state_str):
        """
        Convert string representation back to 2D grid state.
        """
        state = [[None for _ in range(4)] for _ in range(4)]
        idx = 0
        for i in range(4):
            for j in range(4):
                if state_str[idx] != '.' and state_str[idx] != 'N':
                    state[i][j] = state_str[idx]
                idx += 1
        return state
    
    def get_neighbors(self, row, col):
        """
        Get all valid 4-connected neighbors of a grid position.
        """
        neighbors = [] 
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        for dr, dc in directions:
            nr, nc = row + dr, col + dc 
            if 0 <= nr < 4 and 0 <= nc < 4:
                neighbors.append((nr, nc))
        return neighbors
    
    def get_colored_positions(self, state):
        """
        Get all positions that are already colored in the state.
        """
        colored = []
        for i in range(4):
            for j in range(4):
                if state[i][j] is not None:
                    colored.append((i, j))
        return colored
    
    def get_valid_next_positions(self, state):
        """
        Get positions that can be colored next (neighbors of ANY colored cell).
        """
        colored = self.get_colored_positions(state)
        valid_positions = set()
        
        for row, col in colored:
            neighbors = self.get_neighbors(row, col)
            for nr, nc in neighbors:
                if state[nr][nc] is None:
                    valid_positions.add((nr, nc))
        
        return list(valid_positions)
    
    def is_valid_coloring(self, state, row, col, color):
        """
        Check if coloring a position with given color violates adjacency constraint.
        """
        neighbors = self.get_neighbors(row, col)
        for nr, nc in neighbors:
            if state[nr][nc] == color:
                return False
        return True
    
    def is_goal_state(self, state):
        """
        Check if all cells in the grid are colored (goal achieved).
        """
        for i in range(4):
            for j in range(4):
                if state[i][j] is None:
                    return False
        return True
    
    def generate_successors(self, state):
        """
        Generate all valid successor states from current state.
        """
        successors = []
        valid_positions = self.get_valid_next_positions(state)
        
        for row, col in valid_positions:
            for color in self.colors:
                if self.is_valid_coloring(state, row, col, color):
                    new_state = copy.deepcopy(state)
                    new_state[row][col] = color
                    successors.append(new_state)
        
        return successors
    
    #------------------------------------------------------------------------------------------------
    def build_problem_graph(self):
        """
        Build complete problem graph using BFS to explore all reachable states.
        """
        visited = set()
        queue = deque([self.initial_state])
        visited.add(self.state_to_string(self.initial_state))
        self.graph = {}
        
        while queue:
            current_state = queue.popleft()
            current_key = self.state_to_string(current_state)
            self.graph[current_key] = [] 
            
            successors = self.generate_successors(current_state)
            for successor in successors:
                successor_key = self.state_to_string(successor)
                self.graph[current_key].append(successor_key) 
                
                if successor_key not in visited:
                    visited.add(successor_key)
                    queue.append(successor)
        
        return self.graph
    #------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------
    def build_adjacency_matrix(self):
        """
        Build adjacency matrix representation of the problem graph.
        """
        if not self.graph:
            self.build_problem_graph()
        
        states = list(self.graph.keys())
        n = len(states)
        self.adjacency_matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        state_to_index = {state: i for i, state in enumerate(states)}
        
        for i, state in enumerate(states):
            for successor in self.graph[state]:
                j = state_to_index[successor]
                self.adjacency_matrix[i][j] = 1 
        
        return self.adjacency_matrix, states
    #------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------
    def dfs(self):
        """
        Depth-First Search to find solution path.
        """
        if not self.graph:
            self.build_problem_graph()
        
        start_state = self.state_to_string(self.initial_state)
        visited = set()
        
        def dfs_recursive(state_key, path):
            """
            Recursive DFS helper function.
            """
            if state_key in visited:  # Cycle detection
                return False, []
            
            visited.add(state_key)
            path.append(state_key)
            
            state = self.string_to_state(state_key) 
            if self.is_goal_state(state):
                return True, path.copy()
            
            for successor in self.graph.get(state_key, []):
                found, result_path = dfs_recursive(successor, path)
                if found:
                    return True, result_path
            
            path.pop() 
            return False, []
        
        found, solution_path = dfs_recursive(start_state, [])
        return solution_path if found else []
    #------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------    
    def bfs(self):
        """
        Breadth-First Search to find solution path.
        """
        if not self.graph:
            self.build_problem_graph()
        
        start_state = self.state_to_string(self.initial_state)
        queue = deque([(start_state, [start_state])]) 
        visited = {start_state}
        
        while queue:
            current_state, path = queue.popleft()
            
            state = self.string_to_state(current_state) 
            if self.is_goal_state(state):
                return path
            
            for successor in self.graph.get(current_state, []):
                if successor not in visited:
                    visited.add(successor)
                    new_path = path + [successor]
                    queue.append((successor, new_path))
        
        return []
    #------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------   
    def depth_limited_search(self, depth_limit):
        """
        Depth-Limited Search with specified depth limit.
        """
        if not self.graph:
            self.build_problem_graph()
        
        start_state = self.state_to_string(self.initial_state)
        def dls_recursive(state_key, path, depth):
            """
            Recursive DLS helper function.
            """
            if depth > depth_limit:
                return False, [], "cutoff"
            
            state = self.string_to_state(state_key)
            if self.is_goal_state(state):
                return True, path + [state_key], "found"
            
            cutoff_occurred = False
            
            for successor in self.graph.get(state_key, []):
                if successor not in path:
                    found, result_path, result = dls_recursive(successor, path + [state_key], depth + 1)
                    if found:
                        return True, result_path, "found"
                    elif result == "cutoff":
                        cutoff_occurred = True
            
            return False, [], "cutoff" if cutoff_occurred else "failure"
        
        found, solution_path, result = dls_recursive(start_state, [], 0)
        return solution_path if found else []
    #------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------    
    def iterative_deepening(self, initial_depth=3, increment=2, max_depth=7):
        """
        Iterative Deepening Search with increasing depth limits.
        """
        depth = initial_depth  # Current depth limit
        while depth <= max_depth:
            result = self.depth_limited_search(depth)
            if result:
                return result
            depth += increment 
        return []
    #------------------------------------------------------------------------------------------------    

    def print_state(self, state_str):
        """
        Print a state in readable grid format.
        """
        state = self.string_to_state(state_str)
        for row in state:
            print([cell if cell else '.' for cell in row])

def main():
    problem = AI_Assgn_1()
    
    print("Initial State:")
    problem.print_state(problem.state_to_string(problem.initial_state))
    
    # Q1.problem graph
    problem.build_problem_graph()
    print(f"Problem graph built with {len(problem.graph)} states")
    
    # Q2.adjacency matrix
    adj_matrix, states = problem.build_adjacency_matrix()
    print(f"Adjacency matrix: {len(states)}x{len(states)}")
    
    # Q3.DFS
    dfs_solution = problem.dfs()
    print(f"DFS: {'Solution found' if dfs_solution else 'No solution'} - Length: {len(dfs_solution)}")
    
    # Q4.BFS
    bfs_solution = problem.bfs()
    print(f"BFS: {'Solution found' if bfs_solution else 'No solution'} - Length: {len(bfs_solution)}")
    
    # Q5.Depth Limited Search with limit 3
    dls_solution = problem.depth_limited_search(3)
    print(f"DLS (limit=3): {'Solution found' if dls_solution else 'No solution'} - Length: {len(dls_solution)}")
    
    # Q6.Iterative Deepening starting at depth 3, increment 2
    iddfs_solution = problem.iterative_deepening(3, 2)
    print(f"IDDFS: {'Solution found' if iddfs_solution else 'No solution'} - Length: {len(iddfs_solution)}")

if __name__ == "__main__":
    main()