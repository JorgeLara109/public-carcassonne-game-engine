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

class LightweightGameState:
    """Lightweight representation of game state for MCTS"""
    
    def __init__(self, game: Game):
        # Extract only serializable game state information
        self.map_grid = [[None for _ in range(MAX_MAP_LENGTH)] for _ in range(MAX_MAP_LENGTH)]
        self.placed_tiles = []
        self.my_tiles = []
        self.player_scores = [0, 0, 0, 0]
        
        # Copy placed tiles
        for tile in game.state.map.placed_tiles:
            if hasattr(tile, 'placed_pos') and tile.placed_pos:
                x, y = tile.placed_pos
                self.map_grid[y][x] = {
                    'tile_type': tile.tile_type,
                    'rotation': tile.rotation,
                    'internal_edges': tile.internal_edges.copy(),
                    'position': (x, y)
                }
                self.placed_tiles.append(self.map_grid[y][x])
        
        # Copy available tiles (simplified)
        for tile in game.state.my_tiles:
            self.my_tiles.append({
                'tile_type': tile.tile_type,
                'rotation': tile.rotation,
                'internal_edges': tile.internal_edges.copy()
            })

class GameStateNode:
    """Represents a game state node for MCTS tree search"""
    
    def __init__(self, lightweight_state: LightweightGameState, current_player: int = 0, parent_move=None):
        self.state = lightweight_state
        self.current_player = current_player
        self.parent_move = parent_move  # The move that led to this state
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
        """Generate simplified child states (for basic MCTS functionality)"""
        if self._children is not None:
            return self._children
            
        children = set()
        
        if self.is_terminal():
            self._children = children
            return children
        
        # Generate simplified children - just a few basic moves
        # This is a simplified version to avoid complex game state manipulation
        max_children = 10  # Limit number of children for performance
        
        for i in range(min(max_children, len(self.state.my_tiles))):
            for rotation in [0, 1, 2, 3]:  # Try different rotations
                # Create a simple child state
                child_state = copy.deepcopy(self.state)
                
                # Simulate placing a tile (simplified)
                if child_state.my_tiles:
                    child_state.my_tiles.pop(0)  # Remove first tile
                
                move_info = {
                    'tile_idx': i,
                    'rotation': rotation,
                    'simplified': True
                }
                
                next_player = (self.current_player + 1) % 4
                child_node = GameStateNode(child_state, next_player, move_info)
                children.add(child_node)
                
                if len(children) >= max_children:
                    break
            
            if len(children) >= max_children:
                break
        
        self._children = children
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
    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(float)    # total reward of each node
        self.N = defaultdict(int)      # visit count of each node
        self.children = dict()         # children of each node
        self.exploration_weight = exploration_weight

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
        "Simulation: play random moves to the end of the game"
        current_player = node.current_player
        while True:
            if node.is_terminal():
                # Get final scores for all 4 players
                scores = node.get_final_scores()  # Should return [score_p1, score_p2, score_p3, score_p4]
                
                # Calculate reward for player 1: value_player1 - max(value_player2, value_player3, value_player4)
                player1_score = scores[0]
                opponents_max_score = max(scores[1], scores[2], scores[3])
                raw_reward = player1_score - opponents_max_score
                
                # Normalize reward to [0, 1] range for MCTS
                # Positive means player 1 is winning, negative means losing
                reward = 1.0 / (1.0 + math.exp(-raw_reward / 10.0))  # sigmoid normalization
                
                # Return reward from perspective of current player
                if current_player == 0:  # player 1
                    return reward
                else:
                    return 1.0 - reward  # opponents want opposite outcome
            
            node = node.find_random_child()

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

    def do_rollout(self, node, num_rollouts=1000):
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
    # River phase typically ends when all river tiles are placed
    for tile in game.state.map.placed_tiles:
        for edge in tile.get_edges():
            if tile.internal_edges[edge] == StructureType.RIVER:
                return True
    return len(game.state.map.placed_tiles) < 12  # Assume river phase lasts ~12 tiles

def brute_force_tile(game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Fallback brute force tile placement"""
    grid = game.state.map._grid
    
    for tile_idx, tile in enumerate(game.state.my_tiles):
        for y in range(MAX_MAP_LENGTH):
            for x in range(MAX_MAP_LENGTH):
                if grid[y][x] is not None:
                    continue
                    
                for rotation in range(4):
                    tile.rotation = rotation
                    if game.can_place_tile_at(tile, x, y):
                        tile.placed_pos = (x, y)
                        return game.move_place_tile(query, tile._to_model(), tile_idx)
    
    # If no valid placement found, return first tile at first valid position
    if game.state.my_tiles:
        tile = game.state.my_tiles[0]
        return game.move_place_tile(query, tile._to_model(), 0)

def handle_place_tile(game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Main tile placement handler using MCTS for decision making"""
    # Check for river phase first
    if is_river_phase(game):
        return handle_river_phase(game, bot_state, query)
    
    # Use simplified MCTS for strategic tile placement
    try:
        lightweight_state = LightweightGameState(game)
        mcts = MCTS(exploration_weight=1.4)
        root_node = GameStateNode(lightweight_state, current_player=0)
        
        # Run MCTS simulations (reduce rollouts for performance)
        mcts.do_rollout(root_node, num_rollouts=20)  
        
        # Get best move from MCTS
        if root_node in mcts.children and mcts.children[root_node]:
            best_child = mcts.best_child(root_node)
            move_info = best_child.parent_move
            
            if move_info and not move_info.get('simplified', False):
                tile_idx = move_info['tile_idx']
                rotation = move_info['rotation']
                
                # Apply the move using brute force to find valid position
                if tile_idx < len(game.state.my_tiles):
                    tile = game.state.my_tiles[tile_idx]
                    tile.rotation = rotation
                    
                    # Find a valid position for this tile
                    grid = game.state.map._grid
                    for y in range(MAX_MAP_LENGTH):
                        for x in range(MAX_MAP_LENGTH):
                            if grid[y][x] is None and game.can_place_tile_at(tile, x, y):
                                tile.placed_pos = (x, y)
                                bot_state.last_tile = tile
                                print(f"MCTS-guided placement: pos=({x}, {y}), rot={rotation}, tile_idx={tile_idx}")
                                return game.move_place_tile(query, tile._to_model(), tile_idx)
        
        # If MCTS suggests simplified move, use first available tile with MCTS rotation
        if root_node in mcts.children and mcts.children[root_node]:
            best_child = mcts.best_child(root_node)
            move_info = best_child.parent_move
            if move_info:
                suggested_rotation = move_info.get('rotation', 0)
                return brute_force_tile_with_rotation(game, bot_state, query, suggested_rotation)
                
    except Exception as e:
        print(f"MCTS error: {e}")
    
    # Fallback to brute force if MCTS fails
    print("MCTS failed, using brute force")
    return brute_force_tile(game, bot_state, query)

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
    """Handle meeple placement with strategic heuristics"""
    if not bot_state.last_tile:
        return game.move_place_meeple_pass(query)
    
    last_tile = bot_state.last_tile
    
    # Limit meeple placement based on game phase
    if bot_state.meeples_placed >= 5:  # Don't place too many meeples early
        return game.move_place_meeple_pass(query)
    
    # Find the best edge to place a meeple, checking if structures are available
    best_edge = None
    best_score = 0
    
    for edge in last_tile.get_edges():
        # Check if this structure is already claimed by someone
        if game.state._get_claims(last_tile, edge):
            continue  # Skip edges that are already claimed
        
        structure_type = last_tile.internal_edges[edge]
        
        # Score different structure types
        score = 0
        if structure_type == StructureType.CITY:
            score = 4  # Cities are valuable
        elif structure_type == StructureType.ROAD:
            score = 2  # Roads are moderate
        elif structure_type == StructureType.MONASTARY:
            score = 3  # Monasteries are good
        else:
            continue  # Skip other structure types
            
        # Add randomness for variety
        score += random.uniform(-0.5, 0.5)
            
        if score > best_score:
            best_score = score
            best_edge = edge
    
    # Only place meeple if we found a good, unclaimed structure
    if best_edge and best_score > 1.5:  # Lower threshold since we're checking claims
        print(f"Placing meeple on {best_edge} with structure {last_tile.internal_edges[best_edge]}")
        meeple_info = MeepleInfo(last_tile, last_tile.internal_edges[best_edge], best_edge, bot_state.move_count)
        bot_state.placed_meeples.append(meeple_info)
        bot_state.meeples_placed += 1
        
        return game.move_place_meeple(query, last_tile._to_model(), placed_on=best_edge)
    
    # No suitable unclaimed structure found, pass
    return game.move_place_meeple_pass(query)

if __name__ == "__main__":
    main()

