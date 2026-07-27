import numpy as np
from collections import deque
from typing import Tuple, Optional
from environment import Move
from agent_interface import GhostAgent as BaseGhostAgent

class GhostAgent(BaseGhostAgent):    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The Global Memory Map (Starts completely unknown: -1)
        # We assume max size is 21x21 based on the PDF.
        self.memory_map = np.full((21, 21), -1, dtype=int)
        
        # State tracking
        self.last_known_pacman = None
        self.turns_since_seen = 999
        self.current_hideout = None

        self.last_pos = None

    def _update_memory(self, local_map_state: np.ndarray):
        """
        Overlays the visible local map onto the global memory map.
        Everything the Ghost sees is remembered permanently.
        """
        # Find all cells in the local map that are not fog (-1)
        # and copy them into our persistent memory.
        visible_mask = local_map_state != -1
        self.memory_map[visible_mask] = local_map_state[visible_mask]

    def _is_in_line_of_sight(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """
        Checks if two positions share an unbroken straight line in our memory map.
        Crucial for knowing if Pacman can see us.
        """
        r1, c1 = pos1
        r2, c2 = pos2

        if abs(r1 - r2) > 5 or abs(c1 - c2) > 5:
            return False
        
        if r1 == r2:
            c_min, c_max = min(c1, c2), max(c1, c2)
            for c in range(c_min + 1, c_max):
                if self.memory_map[r1, c] == 1: # Wall blocks vision
                    return False
            return True
            
        if c1 == c2:
            r_min, r_max = min(r1, r2), max(r1, r2)
            for r in range(r_min + 1, r_max):
                if self.memory_map[r, c1] == 1:
                    return False
            return True
            
        return False

    def step(self, map_state: np.ndarray, 
             my_position: Tuple[int, int], 
             enemy_position: Optional[Tuple[int, int]],
             step_number: int) -> Move:
             
        # Update our mental map with what we can currently see
        self._update_memory(map_state)

        is_visible = False
        if enemy_position is not None:
            is_visible = self._is_in_line_of_sight(my_position, enemy_position)

            self.last_known_pacman = enemy_position

            if is_visible:
                # PACMAN IS LOOKING AT US
                self.turns_since_seen = 0
            else:
                # PACMAN IS BEHIND A WALL!
                # Calculate how close he is using Manhattan distance
                dist = abs(my_position[0] - enemy_position[0]) + abs(my_position[1] - enemy_position[1])

                if dist <= 8:
                    # TREMORSENSE TRIGGERED: We hear his footsteps
                    # Wake up from Deep Sleep and start sneaking away
                    self.turns_since_seen = 1 
                else:
                    # He is far away. Safe to sleep.
                    self.turns_since_seen = 999
        else:
            # The Arena is working properly and not leaking coordinates.
            self.turns_since_seen += 1

        if self.turns_since_seen == 0:
            best_move = self._execute_panic_flee(my_position, enemy_position)
        elif self.turns_since_seen < 4:
            best_move = self._execute_relocate(my_position)
        else:
            best_move = self._execute_hide(my_position)

        self.last_pos = my_position
        return best_move

    def _execute_panic_flee(self, my_pos: Tuple[int, int], pacman_pos: Tuple[int, int]) -> Move:
        """
        Emergency evasion. Pacman sees us.
        Goal: Step to the nearest tile that breaks Line of Sight.
        """
        queue = deque([(my_pos, [])])
        visited = {my_pos}
        
        best_move = Move.STAY
        
        while queue:
            curr, path = queue.popleft()
            
            # Check if this tile breaks LOS from Pacman's current position
            if not self._is_in_line_of_sight(curr, pacman_pos):
                if path:
                    return path[0] # Return the very first move to get on this escape path
            
            # Continue searching
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = curr[0] + move.value[0], curr[1] + move.value[1]
                if 0 <= nr < 21 and 0 <= nc < 21:
                    # Treat unknown (-1) as walls during panic to be safe, only walk on known empty (0)
                    if self.memory_map[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [move]))
                        
        # Fallback if trapped: run away from Pacman using Manhattan distance
        return self._fallback_flee(my_pos, pacman_pos)
        
    def _execute_relocate(self, my_pos: Tuple[int, int]) -> Move:
        """
        Pacman just lost sight of us, but he is investigating our Last Known Position
        Goal: Run away from his last known position, prioritizing taking turns to confuse him.
        """
        if self.last_known_pacman is None:
            return Move.STAY
            
        best_move = Move.STAY
        best_score = -float('inf')
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = my_pos[0] + move.value[0], my_pos[1] + move.value[1]
            
            if 0 <= nr < 21 and 0 <= nc < 21 and self.memory_map[nr, nc] == 0:
                # Get further away from Pacman's last known location
                dist_from_pacman = abs(nr - self.last_known_pacman[0]) + abs(nc - self.last_known_pacman[1])
                
                # Count valid moves to find intersections
                valid_moves = 0
                for next_m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    r2, c2 = nr + next_m.value[0], nc + next_m.value[1]
                    if 0 <= r2 < 21 and 0 <= c2 < 21 and self.memory_map[r2, c2] in [0, -1]:
                        valid_moves += 1
                
                score = (dist_from_pacman * 10)
                
                # Got a bonus when we hit the intersection
                if valid_moves >= 3:
                    score += 50
                    
                # Small penalty for backtracking
                if self.last_pos is not None and (nr, nc) == self.last_pos:
                    score -= 100
                    
                if score > best_score:
                    best_score = score
                    best_move = move
                    
        return best_move

    def _execute_hide(self, my_pos: Tuple[int, int]) -> Move:
        """
        Active Hide: Seek the deep corners of the map! 
        Avoids straight hallways, defuses dead ends, and creeps away from the center.
        """
        structural_moves = [] 
        safe_moves = []       
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = my_pos[0] + move.value[0], my_pos[1] + move.value[1]
            if 0 <= nr < 21 and 0 <= nc < 21:
                if self.memory_map[nr, nc] in [0, -1]:
                    structural_moves.append(move)
                if self.memory_map[nr, nc] == 0:
                    safe_moves.append(move)
                    
        # Check if we are in a straight kill-zone
        is_straight_hallway = False
        if len(structural_moves) == 2:
            m1, m2 = structural_moves
            if m1.value[0] + m2.value[0] == 0 and m1.value[1] + m2.value[1] == 0:
                is_straight_hallway = True
                
        # Calculate how far we are from the exact center of the map (10, 10)
        dist_from_center = abs(my_pos[0] - 10) + abs(my_pos[1] - 10)
        
        # If we are NOT in a straight hallway, and NOT in a dead end...
        if not is_straight_hallway and len(structural_moves) >= 2:
            # AND we are deep in the outer perimeter of the map (Threshold: 12+ tiles away)
            if dist_from_center >= 12:
                # We have reached the ultimate hiding spot
                return Move.STAY

        # We need to keep moving (either to escape the center, a hallway, or a dead end)
        best_move = Move.STAY
        best_score = -float('inf')
        
        for move in safe_moves:
            nr, nc = my_pos[0] + move.value[0], my_pos[1] + move.value[1]
            
            # Prevent vibrating back and forth (unless we are in a dead end and MUST turn around)
            if len(safe_moves) > 1 and self.last_pos is not None and (nr, nc) == self.last_pos:
                continue
                
            # Score this move based on how far it pushes us into the map corners!
            score = abs(nr - 10) + abs(nc - 10)
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move

    def _fallback_flee(self, my_pos: Tuple[int, int], pacman_pos: Tuple[int, int]) -> Move:
        """
        If we can't break Line of Sight, run away through confirmed safe paths.
        """
        best_move = Move.STAY
        best_dist = -1
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = my_pos[0] + move.value[0], my_pos[1] + move.value[1]
            if 0 <= nr < 21 and 0 <= nc < 21:
                if self.memory_map[nr, nc] == 0:
                    dist = abs(nr - pacman_pos[0]) + abs(nc - pacman_pos[1])
                    if dist > best_dist:
                        # Prevent 180-degree backtracking into Pacman
                        if self.last_pos is None or (nr, nc) != self.last_pos:
                            best_dist = dist
                            best_move = move
                            
        return best_move
