import numpy as np
from collections import deque
from typing import Tuple, Optional
from environment import Move
from agent_interface import GhostAgent as BaseGhostAgent

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
            # Return a massive penalty, but add depth so it prefers dying LATER
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
        
        # Add STAY as a valid move
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            r = pos[0] + move.value[0]
            c = pos[1] + move.value[1]
            
            if 0 <= r < height and 0 <= c < width and map_state[r, c] != 1:
                neighbors.append(((r, c), move))
                
        return neighbors

    def _get_safe_space_score(self, ghost_pos, pacman_pos, map_state) -> int:
        # Optimization: Only calculate safe space if Pacman is close
        distance_to_pacman = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
        if distance_to_pacman > 6:
            return 20 # Arbitrary high score for being safe

        queue = deque([ghost_pos])
        visited = {ghost_pos} 
        safe_tiles_count = 0
        max_search_limit = 20 # Limit BFS to save time
        
        # Include Pacman's 2-step reach in danger zone!
        danger_zone = {pacman_pos}
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            r1, c1 = pacman_pos[0] + dr, pacman_pos[1] + dc
            danger_zone.add((r1, c1))
            danger_zone.add((r1 + dr, c1 + dc)) # 2-step sprint danger

        while queue and safe_tiles_count < max_search_limit:
            current_pos = queue.popleft()
            safe_tiles_count += 1
            
            for next_pos, _ in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited and next_pos not in danger_zone:
                    visited.add(next_pos)
                    queue.append(next_pos)
                    
        return safe_tiles_count

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