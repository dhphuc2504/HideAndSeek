import heapq

import numpy as np
from collections import deque
from typing import Tuple, Optional
from environment import Move
from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost

    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "A* Pacman"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move.

        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Ghost's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)

        Returns:
            Move or (Move, steps): Direction to move (optionally with step count)
        """
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        # Use current sighting, fallback to last known, or explore
        target = enemy_position or self.last_known_enemy_pos

        if target is None:
            # No information about enemy - explore randomly
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    return (move, 1)
            return (Move.STAY, 1)

        # Retrieve path
        path = self.a_star(my_position, target, map_state)
        step1 = path[1]
        step2 = path[2]

        move = Move.STAY
        step = 1

        if len(path) > 0:
            if step1[0] == my_position[0]:
                if step1[1] < my_position[1]:
                    move = Move.LEFT
                elif step1[1] > my_position[1]:
                    move = Move.RIGHT
                if step2[0] == step1[0]:
                    step = 2
            elif step1[1] == my_position[1]:
                if step1[0] < my_position[0]:
                    move = Move.UP
                elif step1[0] > my_position[0]:
                    move = Move.DOWN
                if step2[1] == step1[1]:
                    step = 2

        return (move, step)

    # A* PATHFINDING ALGORITHM
    def a_star(self, start_pos: tuple, end_pos: tuple, map_state: np.ndarray):
        frontier_heap = []  # Priority queue (heapq)
        frontier = set()  # Sort of a tracker for the frontier. Needed because we use lazy deletion when updating the heapq.
        explored = set()

        # Track parents and costs using dictionaries (i don't know if we're allowed to add a Node class)
        parent = {start_pos: None}
        g_cost = {start_pos: 0}
        h_cost = {start_pos: self._manhattan_distance(start_pos, end_pos)}
        f_cost = {start_pos: g_cost[start_pos] + h_cost[start_pos]}

        # Push start pos into frontier. Use tuple for priority order (f-cost -> h-cost -> coordinate itself as fallback)
        heapq.heappush(frontier_heap, (f_cost[start_pos], h_cost[start_pos], start_pos))
        frontier.add(start_pos)

        # Loop through frontier
        iterations = 0
        while frontier and iterations < 128:
            current_node = heapq.heappop(frontier_heap)[2]

            # Handle phantom nodes left behind by lazy deletion.
            if current_node not in frontier:
                continue

            # Found target
            if current_node == end_pos:
                path = []
                while current_node in parent:
                    path.append(current_node)
                    current_node = parent[current_node]
                path.reverse()
                return path

            # Move node to explored
            explored.add(current_node)
            frontier.remove(current_node)

            # Process neighbors
            neighbors = self._get_neighbors(current_node, map_state)
            for neighbor in neighbors:
                if neighbor in explored:
                    continue

                new_g_cost = g_cost[current_node] + 1
                if neighbor not in frontier:
                    h_cost[neighbor] = self._manhattan_distance(neighbor, end_pos)
                    frontier.add(neighbor)
                    g_cost[neighbor] = 1024 # Placeholder g-cost for the second if

                if new_g_cost < g_cost[neighbor]:
                    parent[neighbor] = current_node
                    g_cost[neighbor] = new_g_cost
                    f_cost[neighbor] = g_cost[neighbor] + h_cost[neighbor]
                    heapq.heappush(frontier_heap, (f_cost[neighbor], h_cost[neighbor], neighbor))

            iterations += 1

        # If no path is found
        print("PACMAN A* FAILED")
        return []

    # Helper methods
    def _apply_move(self, pos, move):
        """Apply a move to a position, return new position."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _get_neighbors(self, pos, map_state):
        """Get all valid neighboring positions and their moves."""
        neighbors = []

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append(next_pos)

        return neighbors

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid for at least one step."""
        return self._max_valid_steps(pos, move, map_state, 1) == 1

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        return map_state[row, col] == 0

class GhostAgent(BaseGhostAgent):
    """
    The Ghost Agent utilizing BFS safe-space heuristics and Manhattan distance.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_depth = 4
        self.memo = {}

    def _evaluate_state(self, ghost_pos: Tuple[int, int], pacman_pos: Tuple[int, int], map_state: np.ndarray) -> float:
        """Calculates the desirability of a board state for the Ghost."""
        distance = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
        
        # If Pacman is close enough to catch us, this is the worst possible state
        if distance < 2:
            return -float('inf')
            
        safe_space = self._get_safe_space_score(ghost_pos, pacman_pos, map_state)
        
        # Combine metrics: Distance is good, but mobility (safe space) prevents traps
        return (distance * 10) + safe_space
    
    def _minimax(self, ghost_pos, pacman_pos, depth, is_maximizing, map_state, alpha, beta) -> float:
        # Create a unique key for the current state
        state_key = (ghost_pos, pacman_pos, depth, is_maximizing)
        if state_key in self.memo:
            return self.memo[state_key]
        distance = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
        if distance < 2:
            return -99999 + (depth * 1000) 

        # Base case evaluation (if depth == 0)
        if depth == 0:
            return self._evaluate_state(ghost_pos, pacman_pos, map_state)
        """Recursive Minimax with Alpha-Beta Pruning."""
        
        if is_maximizing:
            # GHOST'S TURN: Maximize the score
            max_eval = -float('inf')
            for next_pos, _ in self._get_neighbors(ghost_pos, map_state):
                eval_score = self._minimax(next_pos, pacman_pos, depth - 1, False, map_state, alpha, beta)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta pruning: Pacman would never allow this branch
            self.memo[state_key] = max_eval
            return max_eval
            
        else:
            # PACMAN'S TURN: Minimize the score
            min_eval = float('inf')
            height, width = map_state.shape
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
                dr, dc = move.value
                
                # Check 1-step move
                r1, c1 = pacman_pos[0] + dr, pacman_pos[1] + dc
                
                if (0 <= r1 < height) and (0 <= c1 < width) and map_state[r1, c1] != 1:
                    # Simulate Pacman taking just 1 step
                    eval_score1 = self._minimax(ghost_pos, (r1, c1), depth - 1, True, map_state, alpha, beta)
                    min_eval = min(min_eval, eval_score1)
                    
                    # Check 2-step straight-line sprint
                    r2, c2 = r1 + dr, c1 + dc
                    if (0 <= r2 < height) and (0 <= c2 < width) and map_state[r2, c2] != 1:
                        # Simulate Pacman taking the full 2 steps
                        eval_score2 = self._minimax(ghost_pos, (r2, c2), depth - 1, True, map_state, alpha, beta)
                        min_eval = min(min_eval, eval_score2)
                
                # Standard Alpha-Beta Pruning
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break
                    
            self.memo[state_key] = min_eval
            return min_eval
    
    def _get_neighbors(self, pos: Tuple[int, int], map_state: np.ndarray):
        neighbors = []
        height, width = map_state.shape
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            r = pos[0] + move.value[0]
            c = pos[1] + move.value[1]
            
            if 0 <= r < height and 0 <= c < width and map_state[r, c] != 1:
                neighbors.append(((r, c), move))
                
        return neighbors

    def _get_safe_space_score(self, ghost_pos, pacman_pos, map_state) -> int:
        distance_to_pacman = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
        if distance_to_pacman > 6:
            return 50 # Arbitrary high score for being safe

        # The queue stores a tuple of (position, last_move_direction)
        queue = deque([(ghost_pos, None)])
        visited = {ghost_pos} 
        
        safe_score = 0
        tiles_explored = 0     # <--- SEPARATE COUNTER FOR THE LIMIT
        max_search_limit = 20 
        
        danger_zone = {pacman_pos}
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            r1, c1 = pacman_pos[0] + dr, pacman_pos[1] + dc
            danger_zone.add((r1, c1))
            danger_zone.add((r1 + dr, c1 + dc)) 

        # <--- LIMIT BASED ON TILES EXPLORED, NOT THE SCORE
        while queue and tiles_explored < max_search_limit: 
            current_pos, last_move = queue.popleft()
            tiles_explored += 1 
            
            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited and next_pos not in danger_zone:
                    visited.add(next_pos)
                    queue.append((next_pos, move))
                    
                    # --- THE CORNER BONUS LOGIC ---
                    tile_value = 1  
                    if last_move is not None and move != last_move:
                        tile_value += 2  # Reward corners!
                        
                    safe_score += tile_value # Add to score, but doesn't trigger the limit early
                    
        return safe_score

    def step(self, map_state: np.ndarray, 
             my_position: Tuple[int, int], 
             enemy_position: Optional[Tuple[int, int]],
             step_number: int) -> Move:
        self.memo.clear()
        if enemy_position is None:
            return Move.STAY
        
        best_move = Move.STAY
        highest_score = -float('inf')
        
        # Initialize Alpha and Beta limits
        alpha = -float('inf')
        beta = float('inf')
        
        # Evaluate all 4 possible movement directions
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            next_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            
            height, width = map_state.shape
            if (0 <= next_pos[0] < height) and (0 <= next_pos[1] < width) and map_state[next_pos[0], next_pos[1]] != 1:
                
                # Ghost just moved, so the next layer of the tree is Pacman's turn (is_maximizing = False)
                score = self._minimax(next_pos, enemy_position, self.max_depth - 1, False, map_state, alpha, beta)
                
                if score > highest_score:
                    highest_score = score
                    best_move = move
                
                # Update alpha for the root layer
                alpha = max(alpha, highest_score)
                        
        return best_move