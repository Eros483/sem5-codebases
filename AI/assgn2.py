import heapq
import math
from typing import List, Tuple, Set, Optional, Dict

class AI_Assgn_2:
    """
    Solves robot path finding problem as per specified conditions, allowing for dynamic grid selection.

    Assumes cost to be 1 for each movement across tiles in the grid
    """
    def __init__(self, grid: List[List[str]]):
        """
        Initialises solution class for specified graph input

        Args:
            grid: Robot movement problem to be solved
        """
        self.grid = grid

        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0

        self.start = self._finder('S')
        self.end = self._finder('E')

        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
    def _finder(self, target: str) -> Tuple[int, int]:
        """
        Find the position of start, end or X in the grid
        
        Args:
            target: type of position to be found in problem grind
        
        Returns:
            Tuple: tuple of x, y coordinates of position
        """
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == target:
                    return (i, j)
        raise ValueError(f"{target} position not found in grid")
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        """
        Check if the position is valid and not blocked

        Args:
            row: Row of specified coordinate
            col: Column of specified coordinate
        
        Returns:
            bool: True or false position for if valid move
        """
        return (
            0 <= row < self.rows and 
                0 <= col < self.cols and 
                self.grid[row][col] != 'X'
        )
    
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate Euclidean distance between two positions as determined heuristic for movement

        Args:
            pos1: first point of consideration
            pos2: second point of consideration
        
        Returns:
            float: value of heuristic function to determine cost
        """
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) #euclidean distance formula
    
    #------------------------------------------------------------------------------------------------
    # Q1 Solution
    def construct_problem_graph(self) -> Dict[Tuple[int, int], Dict[str, any]]:
        """
        Solves Q1 and constructs problem graph.
        """
        graph = {}

        for i in range(self.rows):
            for j in range(self.cols):
                if self._is_valid_position(i, j):
                    pos = (i, j)
                    graph[pos] = {
                        'neighbors': [],
                        'heuristic': self.heuristic(pos, self.end),
                        'position': pos
                    }

        for pos in graph:
            row, col = pos
            for dr, dc in self.directions:
                new_row, new_col = row + dr, col + dc
                neighbor = (new_row, new_col)
                if neighbor in graph:  # If neighbor is a valid node, add connection
                    graph[pos]['neighbors'].append(neighbor)
        
        return graph
    #------------------------------------------------------------------------------------------------
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighboring positions for a given position.

        Args:
            pos: Tuple of row, column coordinates of selected position
        
        Returns:
            list: List of tuples containing viable coordinates to which robot can move
        """
        neighbors = []
        row, col = pos
        
        for dr, dc in self.directions: #as defined initially, up, left, right, down movements
            new_row, new_col = row + dr, col + dc
            if self._is_valid_position(new_row, new_col):
                neighbors.append((new_row, new_col))
        
        return neighbors

    #------------------------------------------------------------------------------------------------
    #Q2 Solution
    def greedy_search(self) -> Optional[List[Tuple[int, int]]]:
        """
        Implement Greedy Best-First Search using Euclidean distance heuristic as asked in q2.
        """
        print("\n=== GREEDY SEARCH ===\n")
        
        #Priority queue for heuristic value and position
        open_list = [(self.heuristic(self.start, self.end), self.start)]

        #defining closed list
        closed_set: Set[Tuple[int, int]] = set()

        #records traversed node for path reconstruction
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {self.start: None}
        
        while open_list:
            #node with least heuristic value
            current_h, current_pos = heapq.heappop(open_list)
            
            if current_pos in closed_set:
                continue
                
            closed_set.add(current_pos)
            
            print(f"Visiting: {current_pos}, Heuristic: {current_h:.2f}")
            
            #checking if we reached the goal
            if current_pos == self.end:
                print("Goal reached!")
                return self._reconstruct_path(parent, current_pos)
            
            #explore neighbors and add accordingly to open list
            for neighbor in self.get_neighbors(current_pos):
                if neighbor not in closed_set:
                    heuristic = self.heuristic(neighbor, self.end)
                    heapq.heappush(open_list, (heuristic, neighbor))
                    
                    if neighbor not in parent:
                        parent[neighbor] = current_pos
        
        #if no path found and open list covered completely
        print("No path found!")
        return None
    #------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------
    #Q3 Solution
    def a_star_search(self) -> Optional[List[Tuple[int, int]]]:
        """
        Implement A* algorithm using Euclidean distance heuristic for q3 of assignment
        """
        print("\n=== A* SEARCH ===\n")
        
        #defining open list and closed list for nodes to be visited and visited nodes respectively
        open_list = []
        closed_set: Set[Tuple[int, int]] = set()

        start_g = 0 #g score is actual cost from start to end, initialises at 0 as we are at start
        start_h = self.heuristic(self.start, self.end) #heuristic estimate of cost to goal node from current node
        start_f = start_g + start_h #combined score as per A* conventions
        
        g_score: Dict[Tuple[int, int], float] = {self.start: start_g}        
        h_score: Dict[Tuple[int, int], float] = {self.start: start_h}
        f_score: Dict[Tuple[int, int], float] = {self.start: start_f}
        
        heapq.heappush(open_list, (start_f, self.start))

        #parent to retrace path taken by a star search
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {self.start: None}
        
        print(f"Starting node: {self.start}")
        print(f"Initial - g={start_g:.2f}, h={start_h:.2f}, f={start_f:.2f}")
        
        while open_list:
            #obtaining node with lowest cost
            _, current_pos = heapq.heappop(open_list)
            
            if current_pos in closed_set:
                continue
                
            closed_set.add(current_pos)
            
            current_g = g_score[current_pos]
            current_h = h_score[current_pos]
            current_f_calculated = current_g + current_h
            
            print(f"Visiting: {current_pos}")
            print(f"  g(n) = {current_g:.2f} (actual cost from start)")
            print(f"  h(n) = {current_h:.2f} (heuristic to goal)")
            print(f"  f(n) = g(n) + h(n) = {current_g:.2f} + {current_h:.2f} = {current_f_calculated:.2f}")
            
            if current_pos == self.end:
                print("Goal reached!")
                print(f"Final path cost: g({self.end}) = {current_g:.2f}")
                return self._reconstruct_path(parent, current_pos)
            
            #Explore neighbors and add to open list
            for neighbor in self.get_neighbors(current_pos):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current_pos] + 1
                neighbor_h = self.heuristic(neighbor, self.end)
                tentative_f = tentative_g + neighbor_h
                
                print(f"  Exploring neighbor {neighbor}:")
                print(f"    tentative_g = {current_g:.2f} + 1 = {tentative_g:.2f}")
                print(f"    h = {neighbor_h:.2f}")
                print(f"    tentative_f = {tentative_g:.2f} + {neighbor_h:.2f} = {tentative_f:.2f}")
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    print(f"    -> Better path found! Updating neighbor {neighbor}")
                    parent[neighbor] = current_pos
                    g_score[neighbor] = tentative_g
                    h_score[neighbor] = neighbor_h
                    f_score[neighbor] = tentative_f
                    
                    heapq.heappush(open_list, (tentative_f, neighbor))
                else:
                    if neighbor in g_score:
                        print(f"    -> Path not better (existing g={g_score[neighbor]:.2f})")
        
        print("No path found!")
        return None
    #------------------------------------------------------------------------------------------------
    
    def _reconstruct_path(self, parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]], 
                         end_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct path from parent dictionary created during construction of either searches.
        """
        path = []
        current = end_pos
        
        while current is not None:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        return path
    
    def print_grid_with_path(self, path: List[Tuple[int, int]], title: str):
        """
        Print the grid with the path marked for visual reference.
        """
        print(f"\n{title}")
        print("=" * len(title))

        display_grid = [row[:] for row in self.grid]
    
        for i, pos in enumerate(path):
            row, col = pos
            if display_grid[row][col] not in ['Start', 'End']:
                display_grid[row][col] = str(i)
        
        #printing the grid
        for row in display_grid:
            print(' | '.join(f'{cell:^4}' for cell in row))
        
        print(f"\nPath length: {len(path)}")
        print(f"Path: {' -> '.join([f'({r},{c})' for r, c in path])}")

def main():
    #defining grid as defined in assignment
    grid = [
        ['',     '',     'X',   'E'],
        ['',     'X',    '',    ''],
        ['X',    '',     '',    ''],
        ['S','',     '',    'X']
    ]
    
    print("\n" + "="*60)
    print("Problem Grid")
    print("="*60)
    for row in grid:
        print(' | '.join(f'{cell if cell else " ":^4}' for cell in row))

    pathfinder = AI_Assgn_2(grid)
    
    print(f"\nStart position: {pathfinder.start}")
    print(f"End position: {pathfinder.end}")

    #------------------------------------------------------------------------------------------------
    #1.Construct the problem graph as defined in question 1
    problem_graph = pathfinder.construct_problem_graph()
    #------------------------------------------------------------------------------------------------ 


    #------------------------------------------------------------------------------------------------ 
    #2.Applying Greedy Search
    print("\n" + "="*60)
    print("Answer 2) Greedy Search")
    print("="*60)
    greedy_path = pathfinder.greedy_search()
    
    if greedy_path:
        pathfinder.print_grid_with_path(greedy_path, "Greedy Search Result")
    #------------------------------------------------------------------------------------------------ 
    

    #------------------------------------------------------------------------------------------------ 
    #3.Applying A* Search
    print("\n" + "="*60)
    print("Answer 3) A* Search")
    print("="*60)
    astar_path = pathfinder.a_star_search()
    
    if astar_path:
        pathfinder.print_grid_with_path(astar_path, "A* Search Result")
    #------------------------------------------------------------------------------------------------ 

if __name__ == "__main__":
    main()