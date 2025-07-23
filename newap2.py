from helper.game import Game
from lib.interact.tile import Tile
from lib.interface.events.moves.move_place_tile import MovePlaceTile
from lib.interface.events.moves.move_place_meeple import (
    MovePlaceMeeple,
    MovePlaceMeeplePass,
)
from lib.interface.queries.typing import QueryType
from lib.interface.queries.query_place_tile import QueryPlaceTile
from lib.interface.queries.query_place_meeple import QueryPlaceMeeple
from lib.interface.events.moves.typing import MoveType
from lib.config.map_config import MAX_MAP_LENGTH
from lib.config.map_config import MONASTARY_IDENTIFIER
from lib.interact.structure import StructureType
from collections import defaultdict
import math
import random
import copy

class MeepleInfo:
    """Track information about placed meeples"""
    def __init__(self, tile: Tile, structure_type: StructureType, edge: str, move_number: int):
        self.tile = tile
        self.structure_type = structure_type
        self.edge = edge
        self.move_number = move_number
        self.position = tile.placed_pos


class DomainExpansionBotState:
    """Enhanced bot state for domain expansion strategy"""

    def __init__(self):
        self.last_tile: Tile | None = None
        self.meeples_placed: int = 0
        self.placed_meeples: list[MeepleInfo] = []
        self.move_count = 0
        self.opponent_meeples: dict[int, list[MeepleInfo]] = {1: [], 2: [], 3: []}  # Track opponents' meeples
        self.completed_structures: set[str] = set()  # Track completed structures we've seen
        self.structure_ownership: dict[str, int] = {}  # Track which player owns which structures
    
    def track_opponent_meeple(self, player_id: int, tile: Tile, structure_type: StructureType, edge: str, move_number: int):
        """Track an opponent's meeple placement"""
        if player_id != 0 and player_id in self.opponent_meeples:
            meeple_info = MeepleInfo(tile, structure_type, edge, move_number)
            self.opponent_meeples[player_id].append(meeple_info)
            
            # Update structure ownership
            structure_key = f"{structure_type}_{tile.placed_pos}_{edge}"
            self.structure_ownership[structure_key] = player_id
    
    def is_structure_claimed(self, tile: Tile, edge: str) -> bool:
        """Check if a structure is already claimed by any player"""
        if not tile.placed_pos:
            return False
        
        structure_type = tile.internal_edges.get(edge)
        if not structure_type:
            return False
            
        structure_key = f"{structure_type}_{tile.placed_pos}_{edge}"
        return structure_key in self.structure_ownership
    
    def get_structure_owner(self, tile: Tile, edge: str) -> int | None:
        """Get the owner of a structure, returns None if unclaimed"""
        if not tile.placed_pos:
            return None
        
        structure_type = tile.internal_edges.get(edge)
        if not structure_type:
            return None
            
        structure_key = f"{structure_type}_{tile.placed_pos}_{edge}"
        return self.structure_ownership.get(structure_key)
    
    def mark_structure_completed(self, tile: Tile, edge: str):
        """Mark a structure as completed"""
        if not tile.placed_pos:
            return
        
        structure_type = tile.internal_edges.get(edge)
        if not structure_type:
            return
            
        structure_key = f"{structure_type}_{tile.placed_pos}_{edge}"
        self.completed_structures.add(structure_key)
    
    def is_structure_completed(self, tile: Tile, edge: str) -> bool:
        """Check if a structure is completed"""
        if not tile.placed_pos:
            return False
        
        structure_type = tile.internal_edges.get(edge)
        if not structure_type:
            return False
            
        structure_key = f"{structure_type}_{tile.placed_pos}_{edge}"
        return structure_key in self.completed_structures
    
    def get_meeple_count_by_structure(self, structure_type: StructureType) -> dict[int, int]:
        """Get count of meeples each player has on a specific structure type"""
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        # Count our meeples
        for meeple in self.placed_meeples:
            if meeple.structure_type == structure_type:
                counts[0] += 1
        
        # Count opponents' meeples
        for player_id, meeples in self.opponent_meeples.items():
            for meeple in meeples:
                if meeple.structure_type == structure_type:
                    counts[player_id] += 1
        
        return counts

class LightweightGameState:
    """Lightweight representation of game state for MCTS"""
    
    def __init__(self, game: Game):
        # Extract only serializable game state information
        self.map_grid = [[None for _ in range(MAX_MAP_LENGTH)] for _ in range(MAX_MAP_LENGTH)]
        self.placed_tiles = []
        self.my_tiles = []
        self.player_scores = [0, 0, 0, 0]
        self.player_meeples = [[], [], [], []]  # Track meeples for all 4 players
        self.completed_structures = set()  # Track completed structures to avoid meeple placement
        self.current_player = 0
        
        # Copy placed tiles with meeple information
        for tile in game.state.map.placed_tiles:
            if hasattr(tile, 'placed_pos') and tile.placed_pos:
                x, y = tile.placed_pos
                tile_data = {
                    'tile_type': tile.tile_type,
                    'rotation': tile.rotation,
                    'internal_edges': tile.internal_edges.copy(),
                    'position': (x, y),
                    'meeples': {}  # {edge: player_id} for meeples on this tile
                }
                
                # Extract meeple information from the tile
                for edge in tile.get_edges():
                    claims = game.state._get_claims(tile, edge)
                    if claims:
                        # Assume first claim represents the meeple owner
                        tile_data['meeples'][edge] = 0  # Simplified: assign to player 0 for now
                
                self.map_grid[y][x] = tile_data
                self.placed_tiles.append(tile_data)
        
        # Extract meeple positions for all players
        self._extract_meeple_positions(game)
        
        # Identify completed structures
        self._identify_completed_structures(game)
        
        # Copy available tiles (simplified)
        for tile in game.state.my_tiles:
            self.my_tiles.append({
                'tile_type': tile.tile_type,
                'rotation': tile.rotation,
                'internal_edges': tile.internal_edges.copy()
            })
    
    def _extract_meeple_positions(self, game: Game):
        """Extract meeple positions for all players"""
        # This is a simplified extraction - in a real implementation,
        # you'd need to track meeples more comprehensively
        for tile_data in self.placed_tiles:
            for edge, player_id in tile_data['meeples'].items():
                meeple_info = {
                    'tile_position': tile_data['position'],
                    'edge': edge,
                    'structure_type': tile_data['internal_edges'][edge]
                }
                self.player_meeples[player_id].append(meeple_info)
    
    def _identify_completed_structures(self, game: Game):
        """Identify completed structures where meeples cannot be placed"""
        # This is a simplified implementation - you'd need more sophisticated 
        # structure completion detection based on the actual game rules
        for tile_data in self.placed_tiles:
            x, y = tile_data['position']
            
            # Check if roads/cities are completed (simplified heuristic)
            for edge, structure_type in tile_data['internal_edges'].items():
                if structure_type in [StructureType.ROAD, StructureType.CITY]:
                    structure_key = f"{structure_type}_{x}_{y}_{edge}"
                    
                    # Simple completion check: if structure connects to multiple tiles
                    connected_count = self._count_connected_structures(tile_data, edge, structure_type)
                    if connected_count >= 2:  # Simplified: assume completed if connects to 2+ tiles
                        self.completed_structures.add(structure_key)
    
    def _count_connected_structures(self, tile_data, edge, structure_type):
        """Count connected structures of the same type"""
        x, y = tile_data['position']
        count = 0
        
        # Check adjacent tiles
        adjacent_positions = {
            'top_edge': (x, y-1),
            'right_edge': (x+1, y),
            'bottom_edge': (x, y+1),
            'left_edge': (x-1, y)
        }
        
        if edge in adjacent_positions:
            adj_x, adj_y = adjacent_positions[edge]
            if (0 <= adj_x < MAX_MAP_LENGTH and 0 <= adj_y < MAX_MAP_LENGTH and 
                self.map_grid[adj_y][adj_x] is not None):
                
                adj_tile = self.map_grid[adj_y][adj_x]
                opposite_edge = self._get_opposite_edge(edge)
                
                if (opposite_edge in adj_tile['internal_edges'] and 
                    adj_tile['internal_edges'][opposite_edge] == structure_type):
                    count += 1
        
        return count
    
    def _get_opposite_edge(self, edge):
        """Get the opposite edge for connection checking"""
        opposites = {
            'top_edge': 'bottom_edge',
            'bottom_edge': 'top_edge',
            'left_edge': 'right_edge',
            'right_edge': 'left_edge'
        }
        return opposites.get(edge, edge)
    
    def is_structure_completed(self, tile_position, edge, structure_type):
        """Check if a structure is completed and cannot have meeples placed"""
        x, y = tile_position
        structure_key = f"{structure_type}_{x}_{y}_{edge}"
        return structure_key in self.completed_structures
    
    def can_place_meeple(self, tile_position, edge):
        """Check if a meeple can be placed on a specific tile edge"""
        x, y = tile_position
        if self.map_grid[y][x] is None:
            return False
        
        tile_data = self.map_grid[y][x]
        
        # Check if edge already has a meeple
        if edge in tile_data['meeples']:
            return False
        
        # Check if structure is completed
        structure_type = tile_data['internal_edges'].get(edge)
        if structure_type and self.is_structure_completed(tile_position, edge, structure_type):
            return False
        
        return True

class GameStateNode:
    """Represents a game state node for MCTS tree search"""
    
    def __init__(self, lightweight_state: LightweightGameState, current_player: int = 0, parent_move=None, move_type="tile"):
        self.state = lightweight_state
        self.current_player = current_player
        self.parent_move = parent_move  # The move that led to this state
        self.move_type = move_type  # "tile" or "meeple"
        self._children = None
        self._is_terminal = None
        
    def __hash__(self):
        # Create a hash based on game state for MCTS node identification
        state_str = f"{self.current_player}_{len(self.state.placed_tiles)}"
        for tile_data in self.state.placed_tiles:
            if tile_data and 'position' in tile_data:
                x, y = tile_data['position']
                state_str += f"_{x}_{y}_{tile_data['rotation']}"
        return hash(state_str)
    
    def __eq__(self, other):
        return isinstance(other, GameStateNode) and hash(self) == hash(other)
    
    def is_terminal(self):
        """Check if this is a terminal game state"""
        if self._is_terminal is None:
            # Terminal if no more tiles to place
            self._is_terminal = (
                len(self.state.my_tiles) == 0 or
                any(score >= 50 for score in self.state.player_scores)
            )
        return self._is_terminal
    
    def get_final_scores(self):
        """Get final scores for all players"""
        if not self.is_terminal():
            # If not terminal, estimate current scores
            return self._evaluate_current_scores()
        return self.state.player_scores
    
    def _evaluate_current_scores(self):
        """Evaluate current game state and estimate player scores"""
        scores = [0, 0, 0, 0]
        
        # Simple heuristic scoring based on placed tiles
        for tile_data in self.state.placed_tiles:
            if not tile_data:
                continue
            
            # Basic scoring: reward tile placement
            scores[0] += 1  # Each placed tile gives 1 point
            
            # Bonus for different structure types
            for edge, structure_type in tile_data['internal_edges'].items():
                if structure_type == StructureType.CITY:
                    scores[0] += 2  # Cities are valuable
                elif structure_type == StructureType.ROAD:
                    scores[0] += 1  # Roads are moderate
                elif structure_type == StructureType.MONASTERY:
                    scores[0] += 3  # Monasteries are good
        
        # Add some randomness for other players
        for i in range(1, 4):
            scores[i] = len(self.state.placed_tiles) * random.uniform(0.5, 1.5)
        
        return scores
    
    def find_children(self):
        """Generate child states for both tile and meeple placement"""
        if self._children is not None:
            return self._children
            
        children = set()
        
        if self.is_terminal():
            self._children = children
            return children
        
        if self.move_type == "tile":
            # Generate tile placement children
            children = self._generate_tile_placement_children()
        elif self.move_type == "meeple":
            # Generate meeple placement children
            children = self._generate_meeple_placement_children()
        
        self._children = children
        return children
    
    def _generate_tile_placement_children(self):
        """Generate children for tile placement decisions"""
        children = set()
        max_children = 71  # Reduced for performance
        
        # Only try first tile with different rotations for speed
        for rotation in [0, 1, 2, 3]:  
            child_state = copy.deepcopy(self.state)
            
            # Simulate placing a tile (simplified)
            if child_state.my_tiles:
                child_state.my_tiles.pop(0)
            
            move_info = {
                'tile_idx': 0,
                'rotation': rotation,
                'simplified': True
            }
            
            # Next decision is meeple placement by same player
            child_node = GameStateNode(child_state, self.current_player, move_info, "meeple")
            children.add(child_node)
            
            if len(children) >= max_children:
                break
        
        return children
    
    def _generate_meeple_placement_children(self):
        """Generate children for meeple placement decisions"""
        children = set()
        
        if not self.state.placed_tiles:
            return children
        
        # Get the most recently placed tile for meeple placement
        last_tile = self.state.placed_tiles[-1] if self.state.placed_tiles else None
        if not last_tile:
            return children
        
        tile_position = last_tile['position']
        next_player = (self.current_player + 1) % 4
        
        # Option 1: Pass on meeple placement
        child_state_pass = copy.deepcopy(self.state)
        pass_move = {'action': 'pass', 'meeple': False}
        child_pass = GameStateNode(child_state_pass, next_player, pass_move, "tile")
        children.add(child_pass)
        
        # Option 2: Place meeple on first available edge (simplified for performance)
        for edge in ['top_edge', 'right_edge', 'bottom_edge', 'left_edge']:
            if self.state.can_place_meeple(tile_position, edge):
                child_state = copy.deepcopy(self.state)
                
                # Simulate meeple placement
                x, y = tile_position
                if child_state.map_grid[y][x]:
                    child_state.map_grid[y][x]['meeples'][edge] = self.current_player
                
                move_info = {
                    'action': 'place_meeple',
                    'edge': edge,
                    'tile_position': tile_position
                }
                
                child_node = GameStateNode(child_state, next_player, move_info, "tile")
                children.add(child_node)
                break  # Only add first valid option for speed
        
        return children
    
    def find_random_child(self):
        """Find a random child for simulation"""
        children = self.find_children()
        if not children:
            return None
        return random.choice(list(children))
    
    def reward(self):
        """Calculate reward for terminal state"""
        if not self.is_terminal():
            return 0.5  # Neutral reward for non-terminal states
        
        scores = self.get_final_scores()
        player1_score = scores[0]
        opponents_max = max(scores[1], scores[2], scores[3])
        
        raw_reward = player1_score - opponents_max
        # Normalize to [0, 1]
        return 1.0 / (1.0 + math.exp(-raw_reward / 10.0))

class MCTS:
    def __init__(self, exploration_weight=1, max_simulation_depth=20):
        self.Q = defaultdict(float)    # total reward of each node
        self.N = defaultdict(int)      # visit count of each node
        self.children = dict()         # children of each node
        self.exploration_weight = exploration_weight
        self.max_simulation_depth = max_simulation_depth  # NEW: Control prediction depth

    def _select(self, node):
        "Selection: descend until a leaf or unexpanded node is found"
        path = []
        while True:
            path.append(node)
            # if node is not expanded or is terminal, stop
            if node not in self.children or not self.children[node]:
                return path
            # choose any unexplored child if available
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                path.append(unexplored.pop())
                return path
            # otherwise use UCT to select among children
            node = self._uct_select(node)

    def _uct_select(self, node):
        "Upper Confidence Bound selection among children"
        log_n_vertex = math.log(self.N[node])
        
        def uct_value(child):
            if self.N[child] == 0:
                return float('inf')  # prioritize unvisited nodes
            return self.Q[child] / self.N[child] + self.exploration_weight * math.sqrt(log_n_vertex / self.N[child])
        
        return max(self.children[node], key=uct_value)

    def _expand(self, node):
        "Expansion: add all legal children of node to the tree"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Simulation: play random moves with depth limit for 20-state prediction"
        current_player = node.current_player
        depth = 0
        
        while depth < self.max_simulation_depth:  # NEW: Depth-limited simulation
            if node.is_terminal():
                # Get final scores for all 4 players
                scores = node.get_final_scores()
                
                # Calculate reward for player 1
                player1_score = scores[0]
                opponents_max_score = max(scores[1], scores[2], scores[3])
                raw_reward = player1_score - opponents_max_score
                
                # Normalize reward to [0, 1] range
                reward = 1.0 / (1.0 + math.exp(-raw_reward / 10.0))
                
                # Return reward from perspective of current player
                if current_player == 0:
                    return reward
                else:
                    return 1.0 - reward
            
            next_node = node.find_random_child()
            if next_node is None:
                break
                
            node = next_node
            depth += 1
            
        # If we reach max depth without terminal, evaluate current position
        scores = node.get_final_scores()
        player1_score = scores[0]
        opponents_max_score = max(scores[1], scores[2], scores[3])
        raw_reward = player1_score - opponents_max_score
        reward = 1.0 / (1.0 + math.exp(-raw_reward / 10.0))
        
        return reward if current_player == 0 else 1.0 - reward

    def _backpropagate(self, path, reward):
        "Backpropagation: update statistics along the path for 4-player game"
        for i, node in enumerate(reversed(path)):
            self.N[node] += 1
            
            # For 4-player games, alternate perspective based on whose turn it was
            # The reward should reflect the perspective of the player who made the move at this node
            if i % 4 == 0:  # same player as simulation start
                self.Q[node] += reward
            elif i % 4 == 1:  # next player
                self.Q[node] += 1.0 - reward
            elif i % 4 == 2:  # third player  
                self.Q[node] += 1.0 - reward
            else:  # fourth player
                self.Q[node] += 1.0 - reward

    def do_rollout(self, node, num_rollouts=100):
        "Execute MCTS rollouts to find the best move"
        for _ in range(num_rollouts):
            path = self._select(node)
            leaf = path[-1]
            self._expand(leaf)
            reward = self._simulate(leaf)
            self._backpropagate(path, reward)

    def best_child(self, node, exploration_weight=0):
        "Select the best child based on visit counts and rewards"
        def score(child):
            if self.N[child] == 0:
                return float('-inf')
            return self.Q[child] / self.N[child] + exploration_weight * math.sqrt(math.log(self.N[node]) / self.N[child])
        
        return max(self.children[node], key=score)

def main():
    game = Game()
    bot_state = DomainExpansionBotState()

    while True:
        query = game.get_next_query()

        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryPlaceTile() as q:
                    print("placing tile")
                    bot_state.move_count += 1
                    return handle_place_tile(game, bot_state, q)

                case QueryPlaceMeeple() as q:
                    print("meeple")
                    return handle_place_meeple(game, bot_state, q)
                case _:
                    assert False

        print("sending move")
        game.send_move(choose_move(query))

def handle_river_phase(
    game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceTile
) -> MovePlaceTile:
    """
    Copy complex.py's exact river handling logic with U-turn detection
    """
    grid = game.state.map._grid

    # The direction of placing the tile in reference to the last placed tile (from complex.py)
    directions = {
        (1, 0): "left_edge",  # if we place on the right of the target tile, we will have to consider our left_edge of the tile in our hand
        (0, 1): "top_edge",   # if we place at the bottom of the target tile, we will have to consider the top_edge of
        (-1, 0): "right_edge", # left
        (0, -1): "bottom_edge", # top
    }
    
    # Will either be the latest tile
    latest_tile = game.state.map.placed_tiles[-1]
    latest_pos = latest_tile.placed_pos

    print(game.state.my_tiles)
    assert latest_pos

    # Try to place a tile adjacent to the latest tile (copied from complex.py)
    for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
        river_flag = False
        for find_edge in directions.values():
            if tile_in_hand.internal_edges[find_edge] == StructureType.RIVER:
                river_flag = True
                print("river on tile")
                break

        # Looking at each edge of the target tile and seeing if we can match it
        for (dx, dy), edge in directions.items():
            target_x = latest_pos[0] + dx
            target_y = latest_pos[1] + dy

            # Check bounds
            if not (0 <= target_x < MAX_MAP_LENGTH and 0 <= target_y < MAX_MAP_LENGTH):
                continue

            # Check if position is empty
            if grid[target_y][target_x] is not None:
                continue

            if game.can_place_tile_at(tile_in_hand, target_x, target_y):
                if river_flag:
                    uturn_check = False
                    print(tile_in_hand.internal_edges[edge])
                    if tile_in_hand.internal_edges[edge] != StructureType.RIVER:
                        continue

                    for tile_edge in tile_in_hand.get_edges():
                        if (
                            tile_edge == edge
                            or tile_in_hand.internal_edges[tile_edge]
                            != StructureType.RIVER
                        ):
                            continue
                        forcast_coordinates_one = {
                            "top_edge": (0, -1),
                            "right_edge": (1, 0),
                            "bottom_edge": (0, 1),
                            "left_edge": (-1, 0),
                        }

                        extension = forcast_coordinates_one[tile_edge]
                        forecast_x = target_x + extension[0]
                        forecast_y = target_y + extension[1]
                        print(forecast_x, forecast_y)
                        for coords in forcast_coordinates_one.values():
                            checking_x = forecast_x + coords[0]
                            checking_y = forecast_y + coords[1]
                            if checking_x != target_x or checking_y != target_y:
                                if grid[checking_y][checking_x] is not None:
                                    print("direct uturn")
                                    uturn_check = True

                        forcast_coordinates_two = {
                            "top_edge": (0, -2),
                            "right_edge": (2, 0),
                            "bottom_edge": (0, 2),
                            "left_edge": (-2, 0),
                        }
                        extension = forcast_coordinates_two[tile_edge]

                        forecast_x = target_x + extension[0]
                        forecast_y = target_y + extension[1]
                        for coords in forcast_coordinates_one.values():
                            checking_x = forecast_x + coords[0]
                            checking_y = forecast_y + coords[1]
                            if grid[checking_y][checking_x] is not None:
                                print("future uturn")
                                uturn_check = True

                    if uturn_check:
                        tile_in_hand.rotate_clockwise(1)
                        if tile_in_hand.internal_edges[edge] != StructureType.RIVER:
                            tile_in_hand.rotate_clockwise(2)

                bot_state.last_tile = tile_in_hand
                bot_state.last_tile.placed_pos = (target_x, target_y)
                print(
                    bot_state.last_tile.placed_pos,
                    tile_hand_index,
                    tile_in_hand.rotation,
                    tile_in_hand.tile_type,
                    flush=True,
                )

                return game.move_place_tile(
                    query, tile_in_hand._to_model(), tile_hand_index
                )
    
    print("could not find with heuristic")
    return brute_force_tile(game, bot_state, query)

def is_river_phase(game: Game) -> bool:
    """Check if we're still in the river phase"""
    # River phase ends when we have river tiles in our hand OR 
    # when we can't find any river tiles in our hand and placed tiles indicate river completion
    
    # First check if we have river tiles in hand
    for tile in game.state.my_tiles:
        for edge in tile.get_edges():
            if tile.internal_edges[edge] == StructureType.RIVER:
                return True  # We have river tiles, so still in river phase
    
    # If no river tiles in hand, river phase is over
    return False

def brute_force_tile(
    game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceTile
) -> MovePlaceTile:
    """Brute force tile placement (from complex.py)"""
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    directions = {
        (0, 1): "top",
        (1, 0): "right",
        (0, -1): "bottom",
        (-1, 0): "left",
    }

    print("Cards", game.state.my_tiles)

    for y in range(height):
        for x in range(width):
            if grid[y][x] is not None:
                print(f"Checking if tile can be placed near tile - {grid[y][x]}")
                for tile_index, tile in enumerate(game.state.my_tiles):
                    for direction in directions:
                        dx, dy = direction
                        x1, y1 = (x + dx, y + dy)

                        if game.can_place_tile_at(tile, x1, y1):
                            bot_state.last_tile = tile
                            bot_state.last_tile.placed_pos = x1, y1
                            return game.move_place_tile(
                                query, tile._to_model(), tile_index
                            )

def handle_place_tile(game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Enhanced tile placement with domain expansion strategy"""
    # Check for river phase first
    if is_river_phase(game):
        return handle_river_phase(game, bot_state, query)
    
    # Try domain expansion strategy first (like 2_bot_domainexpansion)
    print("Attempting domain expansion strategy")
    
    # Define structure type priority (same as domain expansion bot)
    structure_priority = [
        StructureType.MONASTARY,
        StructureType.CITY,
        StructureType.ROAD,
        StructureType.ROAD_START
    ]
    
    # Try to expand existing structures in priority order
    for priority in structure_priority:
        for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
            # Find expansion opportunities for this structure type
            expansion_pos = find_expansion_opportunities(game, bot_state, tile_in_hand, priority)
            if expansion_pos:
                target_x, target_y = expansion_pos
                if is_valid_placement(game, tile_in_hand, target_x, target_y):
                    bot_state.last_tile = tile_in_hand
                    bot_state.last_tile.placed_pos = (target_x, target_y)
                    print(f"Expanding {priority.name} at {target_x}, {target_y}")
                    return game.move_place_tile(query, tile_in_hand._to_model(), tile_hand_index)
    
    # Fallback to regular tile placement if no expansion opportunities
    print("No expansion opportunities, using fallback")
    return fallback_tile_placement(game, bot_state, query)

def brute_force_tile_with_rotation(game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceTile, preferred_rotation: int = 0) -> MovePlaceTile:
    """Brute force tile placement with preferred rotation"""
    grid = game.state.map._grid
    
    for tile_idx, tile in enumerate(game.state.my_tiles):
        # Try preferred rotation first
        for rotation in [preferred_rotation, (preferred_rotation + 1) % 4, (preferred_rotation + 2) % 4, (preferred_rotation + 3) % 4]:
            tile.rotation = rotation
            for y in range(MAX_MAP_LENGTH):
                for x in range(MAX_MAP_LENGTH):
                    if grid[y][x] is None and game.can_place_tile_at(tile, x, y):
                        tile.placed_pos = (x, y)
                        bot_state.last_tile = tile
                        return game.move_place_tile(query, tile._to_model(), tile_idx)

    # Final fallback
    return brute_force_tile(game, bot_state, query)

def handle_place_meeple(game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceMeeple) -> MovePlaceMeeple | MovePlaceMeeplePass:
    """Handle meeple placement using enhanced heuristics for speed"""
    if not bot_state.last_tile:
        return game.move_place_meeple_pass(query)
    
    # Use fast heuristic-based placement for performance
    print("Using fast heuristic meeple placement")
    return _fallback_meeple_placement(game, bot_state, query)

def _fallback_meeple_placement(game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceMeeple) -> MovePlaceMeeple | MovePlaceMeeplePass:
    """Enhanced heuristic meeple placement with proper completion checking and aggressive strategy"""
    last_tile = bot_state.last_tile
    
    # More aggressive meeple placement - use more meeples early game
    if bot_state.meeples_placed >= 7:
        return game.move_place_meeple_pass(query)
    
    # Enhanced priority system: Monastery > City > Road, but also consider strategic value
    def get_enhanced_edge_priority(edge: str) -> tuple[int, float]:
        if edge == MONASTARY_IDENTIFIER:
            return (0, 9.0)  # Highest priority, high score
            
        structure_type = last_tile.internal_edges[edge]
        base_priority = 3
        base_score = 1.0
        
        if structure_type == StructureType.CITY:
            base_priority = 1
            base_score = 6.0  # Higher value for cities
        elif structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
            base_priority = 2
            base_score = 3.0  # Moderate value for roads
            
        # Bonus for early game aggressive expansion
        if bot_state.move_count <= 10:
            base_score *= 1.5  # 50% bonus for early aggressive play
            
        return (base_priority, base_score)
    
    # Create enhanced placement priorities
    base_edges = ["top_edge", "right_edge", "bottom_edge", "left_edge"]
    edges_with_priority = [(get_enhanced_edge_priority(edge), edge) for edge in base_edges]
    edges_with_priority.sort(key=lambda x: (x[0][0], -x[0][1]))  # Sort by priority, then by score desc
    
    placement_priorities = [MONASTARY_IDENTIFIER] + [edge for (_, edge) in edges_with_priority]
    
    for edge in placement_priorities:
        if edge == MONASTARY_IDENTIFIER:
            # Check for monastery
            if (hasattr(last_tile, "modifiers") and 
                any(mod.name == "MONESTERY" for mod in last_tile.modifiers)):
                
                # Use proper completion check
                is_monastery_completed = game.state._check_completed_component(last_tile, MONASTARY_IDENTIFIER)
                if is_monastery_completed:
                    print("monastery already completed, cannot place meeple")
                    continue
                    
                print("Found monastery, placing meeple (HIGH PRIORITY)")
                meeple_info = MeepleInfo(last_tile, StructureType.MONASTARY, MONASTARY_IDENTIFIER, bot_state.move_count)
                bot_state.placed_meeples.append(meeple_info)
                bot_state.meeples_placed += 1
                
                return game.move_place_meeple(query, last_tile._to_model(), placed_on=MONASTARY_IDENTIFIER)
        else:
            # Check regular edges with enhanced strategy
            structure_type = last_tile.internal_edges[edge]
            
            # Use proper completion check from game engine
            is_completed = game.state._check_completed_component(last_tile, edge)
            
            # Enhanced validation with strategic considerations
            if (not game.state._get_claims(last_tile, edge) and
                structure_type != StructureType.RIVER and
                structure_type != StructureType.GRASS and
                not is_completed):
                
                # Additional strategic check - prefer structures we can expand later
                strategic_value = 1.0
                if structure_type == StructureType.CITY:
                    strategic_value = 2.0  # Cities are very valuable
                elif structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
                    strategic_value = 1.5  # Roads are moderately valuable
                    
                # Early game bonus
                if bot_state.move_count <= 8:
                    strategic_value *= 1.3
                
                if strategic_value >= 1.2:  # Lower threshold for more aggressive play
                    print(f"Placing meeple on {edge} with structure {structure_type} (strategic value: {strategic_value:.1f})")
                    meeple_info = MeepleInfo(last_tile, structure_type, edge, bot_state.move_count)
                    bot_state.placed_meeples.append(meeple_info)
                    bot_state.meeples_placed += 1
                    
                    return game.move_place_meeple(query, last_tile._to_model(), placed_on=edge)
    
    print("No strategically valuable meeple placement found, passing")
    return game.move_place_meeple_pass(query)

def _evaluate_completion_potential(game: Game, tile: Tile, edge: str, structure_type: StructureType) -> float:
    """Evaluate how likely a structure is to be completed for scoring"""
    if not tile.placed_pos:
        return 0.0
    
    x, y = tile.placed_pos
    grid = game.state.map._grid
    
    # Count connected structures of the same type
    connected_count = 0
    open_ends = 0
    
    # Check adjacent tiles for structure connections
    adjacent_positions = {
        'top_edge': (x, y-1),
        'right_edge': (x+1, y),
        'bottom_edge': (x, y+1),
        'left_edge': (x-1, y)
    }
    
    if edge in adjacent_positions:
        adj_x, adj_y = adjacent_positions[edge]
        if (0 <= adj_x < MAX_MAP_LENGTH and 0 <= adj_y < MAX_MAP_LENGTH):
            if grid[adj_y][adj_x] is not None:
                connected_count += 1
            else:
                open_ends += 1
    
    # Bonus for structures that are more likely to be completed
    completion_bonus = 0.0
    if structure_type == StructureType.CITY:
        # Cities with fewer open ends are more likely to be completed
        completion_bonus = max(0, 1.5 - open_ends * 0.5)
    elif structure_type == StructureType.ROAD:
        # Roads with connections are valuable
        completion_bonus = connected_count * 0.5
    elif structure_type == StructureType.MONASTARY:
        # Monasteries surrounded by more tiles are better
        surrounding_tiles = sum(1 for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                              if dx != 0 or dy != 0 
                              if 0 <= x+dx < MAX_MAP_LENGTH and 0 <= y+dy < MAX_MAP_LENGTH 
                              and grid[y+dy][x+dx] is not None)
        completion_bonus = surrounding_tiles * 0.3
    
    return completion_bonus

def _evaluate_blocking_potential(game: Game, bot_state: DomainExpansionBotState, tile: Tile, edge: str) -> float:
    """Evaluate the strategic value of blocking opponents"""
    # This is a simplified implementation - could be enhanced with more sophisticated analysis
    
    # If we have many meeples placed, consider blocking more valuable
    if bot_state.meeples_placed >= 3:
        return 0.5  # Small bonus for potential blocking moves
    
    return 0.0

def find_expansion_opportunities(game: Game, bot_state: DomainExpansionBotState, tile_in_hand: Tile, structure_type: StructureType) -> tuple[int, int] | None:
    """Find positions to expand specific structure types"""
    # Filter meeples by requested structure type
    relevant_meeples = [m for m in bot_state.placed_meeples if m.structure_type == structure_type]
    
    # Try most recent meeples first
    for meeple_info in sorted(relevant_meeples, key=lambda m: -m.move_number):
        if not meeple_info.position:
            continue
            
        # Try adjacent positions
        directions = {
            (1, 0): "left_edge",
            (0, 1): "top_edge", 
            (-1, 0): "right_edge",
            (0, -1): "bottom_edge",
        }
        
        for (dx, dy), edge in directions.items():
            target_x = meeple_info.position[0] + dx
            target_y = meeple_info.position[1] + dy
            
            # Check bounds and empty position
            if not (0 <= target_x < MAX_MAP_LENGTH and 0 <= target_y < MAX_MAP_LENGTH):
                continue
            if game.state.map._grid[target_y][target_x] is not None:
                continue
                
            # Check if tile can expand this specific structure
            if can_expand_structure(game, tile_in_hand, target_x, target_y, meeple_info):
                return (target_x, target_y)
    
    return None

def can_expand_structure(game: Game, tile: Tile, x: int, y: int, meeple_info: MeepleInfo) -> bool:
    """Check if placing a tile at (x,y) would expand the structure where we have a meeple"""
    
    # First check if any valid placement exists at this position
    if not is_valid_placement(game, tile, x, y):
        return False
    
    # After validation, check if any edge matches the structure type of our meeple
    for edge in tile.get_edges():
        if tile.internal_edges[edge] == meeple_info.structure_type:
            # Check if this edge would connect to the meeple's structure
            if would_connect_to_structure(game, tile, x, y, edge, meeple_info):
                return True
    
    return False

def would_connect_to_structure(game: Game, tile: Tile, x: int, y: int, edge: str, meeple_info: MeepleInfo) -> bool:
    """Check if placing this tile's edge would connect to our meeple's structure"""
    
    # Get the adjacent position for this edge
    directions = {
        "left_edge": (-1, 0),
        "right_edge": (1, 0), 
        "top_edge": (0, -1),
        "bottom_edge": (0, 1),
    }
    
    if edge not in directions:
        return False
        
    dx, dy = directions[edge]
    adj_x, adj_y = x + dx, y + dy
    
    # Check if the adjacent position has our meeple's tile
    grid = game.state.map._grid
    if not (0 <= adj_y < len(grid) and 0 <= adj_x < len(grid[0])):
        return False
        
    adjacent_tile = grid[adj_y][adj_x]
    if adjacent_tile is None:
        return False
        
    # Check if this is the tile where we have a meeple
    return (adjacent_tile.placed_pos == meeple_info.position and 
            adjacent_tile.tile_type == meeple_info.tile.tile_type)

def is_valid_placement(game: Game, tile: Tile, x: int, y: int) -> bool:
    """Simple validation that uses the game's own can_place_tile_at function"""
    return game.can_place_tile_at(tile, x, y)

def fallback_tile_placement(game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Fallback strategy: iterate through recently placed tiles until a legal placement is found"""
    directions = {
        (1, 0): "left_edge",
        (0, 1): "top_edge", 
        (-1, 0): "right_edge",
        (0, -1): "bottom_edge",
    }
    
    grid = game.state.map._grid
    placed_tiles = game.state.map.placed_tiles
    
    # Start with most recent tiles and work backwards
    for tile_index in range(len(placed_tiles) - 1, -1, -1):
        reference_tile = placed_tiles[tile_index]
        if not reference_tile.placed_pos:
            continue
            
        reference_pos = reference_tile.placed_pos
        
        # Try each tile in hand
        for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
            # Try each adjacent position
            for (dx, dy), edge in directions.items():
                target_x = reference_pos[0] + dx
                target_y = reference_pos[1] + dy

                # Check bounds
                if not (0 <= target_x < MAX_MAP_LENGTH and 0 <= target_y < MAX_MAP_LENGTH):
                    continue

                # Check if position is empty
                if grid[target_y][target_x] is not None:
                    continue
                
                # Use can_place_tile_at which handles rotation validation internally
                if is_valid_placement(game, tile_in_hand, target_x, target_y):
                    bot_state.last_tile = tile_in_hand
                    bot_state.last_tile.placed_pos = (target_x, target_y)
                    print(f"Fallback placement at {target_x}, {target_y}")
                    
                    return game.move_place_tile(query, tile_in_hand._to_model(), tile_hand_index)
    
    # Final fallback - brute force
    print("Using brute force as final fallback")
    return brute_force_tile(game, bot_state, query)

def _evaluate_blocking_potential_enhanced(game: Game, bot_state: DomainExpansionBotState, tile: Tile, edge: str, structure_type: StructureType) -> float:
    """Enhanced evaluation of blocking potential using opponent meeple tracking"""
    blocking_bonus = 0.0
    
    # Check if placing here would block or compete with opponents
    meeple_counts = bot_state.get_meeple_count_by_structure(structure_type)
    
    # If opponents have more meeples on this structure type, blocking is more valuable
    our_count = meeple_counts[0]
    max_opponent_count = max(meeple_counts[1], meeple_counts[2], meeple_counts[3])
    
    if max_opponent_count > our_count:
        blocking_bonus += 1.0  # Blocking is valuable when we're behind
    elif max_opponent_count == our_count and our_count > 0:
        blocking_bonus += 0.5  # Competing is moderately valuable
    
    # Additional bonus if this is a high-value structure and opponents are active
    if structure_type == StructureType.CITY and max_opponent_count > 0:
        blocking_bonus += 0.5
    elif structure_type == StructureType.MONASTARY and max_opponent_count > 0:
        blocking_bonus += 0.3
    
    return blocking_bonus


if __name__ == "__main__":
    main()
