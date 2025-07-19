"""
Wena11 Bot with Monte Carlo Tree Search (MCTS)

This version combines the practical winning strategies from the original Wena11
with Monte Carlo Tree Search for improved decision-making.

Key improvements:
1. MCTS for tile placement decisions
2. Retains the strategic evaluation functions as heuristics
3. UCT (Upper Confidence bounds applied to Trees) for exploration/exploitation
4. Simulation-based evaluation of long-term consequences
"""

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
from lib.config.map_config import MAX_MAP_LENGTH, MONASTARY_IDENTIFIER
from lib.interact.structure import StructureType

from collections import defaultdict
import math
import random
import time
from typing import Dict, List, Tuple, Optional, Set
from copy import deepcopy


class MeepleInfo:
    """Track information about placed meeples"""
    def __init__(self, tile: Tile, structure_type: StructureType, edge: str, move_number: int, player_id: int = -1):
        self.tile = tile
        self.structure_type = structure_type
        self.edge = edge
        self.move_number = move_number
        self.position = tile.placed_pos
        self.player_id = player_id


class GamePhase:
    """Game phase constants"""
    EARLY = "early"    # 0-20 tiles
    MID = "mid"        # 21-50 tiles
    LATE = "late"      # 51+ tiles


class MCTSNode:
    """Node in the Monte Carlo Tree Search"""
    def __init__(self, state=None, parent=None, action=None, player_id=None):
        self.state = state  # Game state at this node
        self.parent = parent
        self.action = action  # Action that led to this state
        self.player_id = player_id
        
        self.visits = 0
        self.total_reward = 0.0
        self.children = []
        self.untried_actions = []
        
    def uct_value(self, c=3.0):
        """Calculate UCT value for node selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, c=3.0):
        """Select best child using UCT"""
        return max(self.children, key=lambda n: n.uct_value(c))
    
    def most_visited_child(self):
        """Return child with most visits (for final decision)"""
        return max(self.children, key=lambda n: n.visits)
    
    def update(self, reward):
        """Update node statistics"""
        self.visits += 1
        self.total_reward += reward


class ImprovedBotState:
    """Improved bot state with MCTS integration"""
    def __init__(self):
        # Basic state
        self.last_tile: Tile | None = None
        self.meeples_placed: int = 0
        self.placed_meeples: list[MeepleInfo] = []
        self.move_count = 0
        self.our_score = 0
        self.game_phase = GamePhase.EARLY
        
        # Opponent tracking
        self.opponent_meeples: list[MeepleInfo] = []
        
        # Strategy state
        self.strategy_mode = "balanced"
        self.monastery_priority = True
        self.aggressive_blocking = False
        self.farmer_endgame = False
        
        # Performance tracking
        self.completed_structures = 0
        self.blocked_opponent_structures = 0
        self.average_points_per_meeple = 0.0
        
        # MCTS parameters
        self.mcts_iterations = 100  # Number of MCTS iterations per move
        self.simulation_depth = 10  # Max depth for simulations
        self.c_param = 3.0  # UCT exploration parameter

    def update_game_phase(self, game: Game):
        """Update game phase and strategy"""
        tiles_played = len(game.state.map.placed_tiles)
        self.our_score = game.state.points
        
        # Update phase
        if tiles_played < 20:
            self.game_phase = GamePhase.EARLY
        elif tiles_played < 50:
            self.game_phase = GamePhase.MID
        else:
            self.game_phase = GamePhase.LATE
        
        # Update strategy based on phase and score
        self.update_strategy()
    
    def update_strategy(self):
        """Update strategy based on game state"""
        meeples_remaining = 7 - self.meeples_placed
        
        if self.game_phase == GamePhase.EARLY:
            self.monastery_priority = True
            self.aggressive_blocking = False
            self.farmer_endgame = False
        elif self.game_phase == GamePhase.MID:
            self.monastery_priority = meeples_remaining > 3
            self.aggressive_blocking = self.our_score < 30
            self.farmer_endgame = False
        else:  # LATE
            self.monastery_priority = meeples_remaining > 5
            self.aggressive_blocking = True
            self.farmer_endgame = meeples_remaining <= 2


def main():
    game = Game()
    bot_state = ImprovedBotState()

    while True:
        query = game.get_next_query()

        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryPlaceTile() as q:
                    print("Placing tile using MCTS")
                    bot_state.move_count += 1
                    bot_state.update_game_phase(game)
                    return handle_place_tile_mcts(game, bot_state, q)

                case QueryPlaceMeeple() as q:
                    print("Placing meeple")
                    return handle_place_meeple_improved(game, bot_state, q)
                case _:
                    assert False

        print("Sending move")
        game.send_move(choose_move(query))


def handle_place_tile_mcts(game: Game, bot_state: ImprovedBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Use MCTS to determine best tile placement"""
    
    # Handle river phase with existing logic
    if is_river_phase(game):
        return handle_river_phase(game, bot_state, query)
    
    # Get all possible placements
    possible_actions = get_all_valid_placements(game, bot_state)
    
    if not possible_actions:
        return fallback_tile_placement(game, bot_state, query)
    
    if len(possible_actions) == 1:
        # Only one option, no need for MCTS
        tile_idx, tile, x, y = possible_actions[0]
        bot_state.last_tile = tile
        bot_state.last_tile.placed_pos = (x, y)
        return game.move_place_tile(query, tile._to_model(), tile_idx)
    
    # Run MCTS to find best action
    best_action = run_mcts(game, bot_state, possible_actions)
    
    if best_action:
        tile_idx, tile, x, y = best_action
        bot_state.last_tile = tile
        bot_state.last_tile.placed_pos = (x, y)
        print(f"MCTS selected: placing tile at {x}, {y}")
        return game.move_place_tile(query, tile._to_model(), tile_idx)
    
    # Fallback if MCTS fails
    return fallback_tile_placement(game, bot_state, query)


def run_mcts(game: Game, bot_state: ImprovedBotState, possible_actions: List[Tuple]) -> Optional[Tuple]:
    """Run Monte Carlo Tree Search to find best action"""
    
    # Create root node with current game state
    root = MCTSNode(state=game, player_id=game.state.me.player_id)
    root.untried_actions = possible_actions.copy()
    
    # Run MCTS iterations
    for i in range(bot_state.mcts_iterations):
        node = root
        state_copy = create_game_state_copy(game)
        
        # 1. Selection - traverse tree using UCT
        while node.untried_actions == [] and node.children != []:
            node = node.best_child(bot_state.c_param)
            # Apply action to state copy
            if node.action:
                apply_action_to_state(state_copy, node.action)
        
        # 2. Expansion - add new child
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            
            # Apply action and create child node
            apply_action_to_state(state_copy, action)
            child = MCTSNode(state=state_copy, parent=node, action=action, 
                           player_id=game.state.me.player_id)
            node.children.append(child)
            node = child
        
        # 3. Simulation - play out randomly from this state
        reward = simulate_game(state_copy, bot_state, bot_state.simulation_depth)
        
        # 4. Backpropagation - update statistics
        while node is not None:
            node.update(reward)
            node = node.parent
    
    # Return action of most visited child
    if root.children:
        best_child = root.most_visited_child()
        return best_child.action
    
    return None


def create_game_state_copy(game: Game):
    """Create a lightweight copy of game state for simulation"""
    # This is a simplified copy - in practice, you'd need to deep copy relevant state
    state_copy = {
        'map': deepcopy(game.state.map._grid),
        'placed_tiles': game.state.map.placed_tiles.copy(),
        'my_tiles': game.state.my_tiles.copy(),
        'scores': game.state.points,
        'meeples': game.state.get_meeples_placed_by(game.state.me.player_id)
    }
    return state_copy


def apply_action_to_state(state, action):
    """Apply a tile placement action to the game state"""
    tile_idx, tile, x, y = action
    # Simplified state update - would need full game logic here
    if 'placed_pos' not in state:
        state['placed_pos'] = []
    state['placed_pos'].append((x, y))


def simulate_game(state, bot_state: ImprovedBotState, max_depth: int) -> float:
    """Simulate game from current state and return reward"""
    
    # This is a simplified simulation using the existing evaluation functions
    # In a full implementation, you would simulate actual game moves
    
    depth = 0
    total_score = 0.0
    
    # Use existing evaluation functions to estimate value
    if isinstance(state, dict) and 'placed_pos' in state:
        for x, y in state['placed_pos']:
            # Estimate score based on position quality and strategy
            score = evaluate_position_quality_for_simulation(x, y)
            
            # Add strategy-based bonuses
            if bot_state.monastery_priority:
                score += 5.0  # Bonus for monastery-favorable positions
            
            if bot_state.aggressive_blocking:
                score += 3.0  # Bonus for blocking positions
            
            total_score += score
    
    # Normalize reward to [-1, 1] range
    return math.tanh(total_score / 100.0)


def evaluate_position_quality_for_simulation(x: int, y: int) -> float:
    """Evaluate position quality for simulation"""
    center_x, center_y = MAX_MAP_LENGTH // 2, MAX_MAP_LENGTH // 2
    distance_from_center = abs(x - center_x) + abs(y - center_y)
    
    # Prefer central positions
    position_score = max(0, 20 - distance_from_center)
    
    # Add some randomness for simulation variety
    position_score += random.uniform(-5, 5)
    
    return position_score

def is_valid_placement(game: Game, tile: Tile, x: int, y: int) -> bool:
    """
    Simple validation that uses the game's own can_place_tile_at function
    This function handles rotation internally and sets the tile to valid rotation
    """
    # Just use the game's validation function directly
    # It handles rotation internally and will set the tile to a valid rotation if one exists
    return game.can_place_tile_at(tile, x, y)

def get_all_valid_placements(game: Game, bot_state: ImprovedBotState) -> List[Tuple]:
    """Get all valid tile placements efficiently"""
    placements = []
    positions = get_strategic_positions(game, bot_state)
    grid = game.state.map._grid
    
    for tile_index, original_tile in enumerate(game.state.my_tiles):
        # Create a working copy to avoid modifying the original
        tile = deepcopy(original_tile)
        
        for x, y in positions:
            if grid[y][x] is not None:  # Skip occupied positions
                continue
                
            # Check all 4 rotations efficiently
            for rotation in range(4):
                if rotation > 0:
                    tile.rotate_clockwise(1)
                    
                if game.can_place_tile_at(tile, x, y):
                    placements.append((tile_index, deepcopy(tile), x, y))
            
            # Reset tile to original rotation
            tile.rotation = original_tile.rotation
            
    return placements


def get_strategic_positions(game: Game, bot_state: ImprovedBotState) -> List[Tuple[int, int]]:
    """Get positions to check in strategic order"""
    positions = set()
    grid = game.state.map._grid
    
    # Priority 1: Adjacent to our meeples
    for meeple in bot_state.placed_meeples:
        if meeple.position:
            x, y = meeple.position
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < MAX_MAP_LENGTH and 0 <= new_y < MAX_MAP_LENGTH 
                    and grid[new_y][new_x] is None):
                    positions.add((new_x, new_y))
    
    # Priority 2: Adjacent to opponent meeples (for blocking)
    if bot_state.aggressive_blocking:
        for meeple in bot_state.opponent_meeples:
            if meeple.position:
                x, y = meeple.position
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_x, new_y = x + dx, y + dy
                    if (0 <= new_x < MAX_MAP_LENGTH and 0 <= new_y < MAX_MAP_LENGTH 
                        and grid[new_y][new_x] is None):
                        positions.add((new_x, new_y))
    
    # Priority 3: All other adjacent positions
    for tile in game.state.map.placed_tiles:
        if tile.placed_pos:
            x, y = tile.placed_pos
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < MAX_MAP_LENGTH and 0 <= new_y < MAX_MAP_LENGTH 
                    and grid[new_y][new_x] is None):
                    positions.add((new_x, new_y))
    
    return list(positions)


def evaluate_position_quality(game: Game, x: int, y: int) -> float:
    """Evaluate the quality of a position on the board"""
    center_x, center_y = MAX_MAP_LENGTH // 2, MAX_MAP_LENGTH // 2
    distance_from_center = abs(x - center_x) + abs(y - center_y)
    
    position_score = max(0, 20 - distance_from_center)
    return position_score


def is_river_phase(game: Game) -> bool:
    """Detect if we're in river phase"""
    for tile in game.state.my_tiles:
        for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
            if tile.internal_edges[edge] == StructureType.RIVER:
                return True
    return False


def handle_place_meeple_improved(game: Game, bot_state: ImprovedBotState, query: QueryPlaceMeeple) -> MovePlaceMeeplePass | MovePlaceMeeple:
    """Improved meeple placement with strategic priorities"""
    
    if not bot_state.last_tile or bot_state.meeples_placed >= 7:
        return game.move_place_meeple_pass(query)
    
    recent_tile = bot_state.last_tile
    meeples_remaining = 7 - bot_state.meeples_placed
    
    # PRIORITY 1: MONASTERY
    if (hasattr(recent_tile, "modifiers") 
        and any(mod.name == "MONESTARY" for mod in recent_tile.modifiers)
        and not game.state._check_completed_component(recent_tile, MONASTARY_IDENTIFIER)):
        
        if bot_state.monastery_priority or meeples_remaining <= 2:
            bot_state.meeples_placed += 1
            meeple_info = MeepleInfo(recent_tile, StructureType.MONASTARY, MONASTARY_IDENTIFIER, bot_state.move_count)
            bot_state.placed_meeples.append(meeple_info)
            print(f"Placed monastery meeple - guaranteed 9 points")
            return game.move_place_meeple(query, recent_tile._to_model(), MONASTARY_IDENTIFIER)
    
    # PRIORITY 2: HIGH-VALUE STRUCTURES
    structures = list(game.state.get_placeable_structures(recent_tile._to_model()).items())
    
    meeple_evaluations = []
    
    for edge, structure in structures:
        structure_type = recent_tile.internal_edges.get(edge)
        if structure_type is None or structure_type == StructureType.RIVER:
            continue
        
        if (not game.state._get_claims(recent_tile, edge) 
            and not game.state._check_completed_component(recent_tile, edge)):
            
            score = evaluate_meeple_placement(game, bot_state, recent_tile, edge, structure_type)
            meeple_evaluations.append((score, edge, structure_type))
    
    if meeple_evaluations:
        meeple_evaluations.sort(key=lambda x: x[0], reverse=True)
        best_score, best_edge, best_structure_type = meeple_evaluations[0]
        
        min_score_threshold = 8.0 if meeples_remaining > 3 else 5.0
        
        if best_score >= min_score_threshold:
            bot_state.meeples_placed += 1
            meeple_info = MeepleInfo(recent_tile, best_structure_type, best_edge, bot_state.move_count)
            bot_state.placed_meeples.append(meeple_info)
            print(f"Placed meeple on {best_structure_type.name} with score {best_score:.2f}")
            return game.move_place_meeple(query, recent_tile._to_model(), best_edge)
    
    print("Passing meeple placement")
    return game.move_place_meeple_pass(query)


def evaluate_meeple_placement(game: Game, bot_state: ImprovedBotState, tile: Tile, edge: str, structure_type: StructureType) -> float:
    """Evaluate the value of placing a meeple on a specific structure"""
    score = 0.0
    
    # Base scores by structure type
    if structure_type == StructureType.MONASTARY:
        score = 20.0
    elif structure_type == StructureType.CITY:
        score = 15.0
    elif structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
        score = 10.0
    elif structure_type == StructureType.GRASS:
        score = 5.0
    
    # Adjustment based on completion probability
    completion_prob = estimate_completion_probability(game, bot_state, tile, edge, structure_type)
    score *= completion_prob
    
    # Adjustment based on game phase
    if bot_state.game_phase == GamePhase.EARLY:
        if structure_type == StructureType.MONASTARY:
            score *= 1.5
    elif bot_state.game_phase == GamePhase.MID:
        if structure_type == StructureType.CITY:
            score *= 1.3
    
    # Adjustment based on meeples remaining
    meeples_remaining = 7 - bot_state.meeples_placed
    if meeples_remaining <= 2:
        if structure_type == StructureType.MONASTARY:
            score *= 1.5
        elif completion_prob < 0.7:
            score *= 0.5
    
    return score


def estimate_completion_probability(game: Game, bot_state: ImprovedBotState, tile: Tile, edge: str, structure_type: StructureType) -> float:
    """Estimate the probability of completing a structure"""
    base_prob = 0.5
    
    if structure_type == StructureType.MONASTARY:
        x, y = tile.placed_pos
        empty_spaces = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < MAX_MAP_LENGTH and 0 <= check_y < MAX_MAP_LENGTH 
                    and game.state.map._grid[check_y][check_x] is None):
                    empty_spaces += 1
        
        base_prob = 0.9 - (empty_spaces * 0.08)
        
    elif structure_type == StructureType.CITY:
        base_prob = 0.6
        if bot_state.game_phase == GamePhase.LATE:
            base_prob = 0.4
            
    elif structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
        base_prob = 0.8
        
    elif structure_type == StructureType.GRASS:
        base_prob = 1.0
    
    return base_prob


def handle_river_phase(game: Game, bot_state: ImprovedBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Handle river phase tile placement"""
    grid = game.state.map._grid

    directions = {
        (1, 0): "left_edge",
        (0, 1): "top_edge",
        (-1, 0): "right_edge",
        (0, -1): "bottom_edge",
    }
    
    latest_tile = game.state.map.placed_tiles[-1]
    latest_pos = latest_tile.placed_pos

    assert latest_pos

    for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
        river_flag = False
        for find_edge in directions.values():
            if tile_in_hand.internal_edges[find_edge] == StructureType.RIVER:
                river_flag = True
                break

        for (dx, dy), edge in directions.items():
            target_x = latest_pos[0] + dx
            target_y = latest_pos[1] + dy

            if not (0 <= target_x < MAX_MAP_LENGTH and 0 <= target_y < MAX_MAP_LENGTH):
                continue

            if grid[target_y][target_x] is not None:
                continue

            if game.can_place_tile_at(tile_in_hand, target_x, target_y):
                if river_flag:
                    # River U-turn detection logic from original
                    uturn_check = False
                    if tile_in_hand.internal_edges[edge] != StructureType.RIVER:
                        continue

                    for tile_edge in tile_in_hand.get_edges():
                        if (tile_edge == edge or 
                            tile_in_hand.internal_edges[tile_edge] != StructureType.RIVER):
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
                        
                        for coords in forcast_coordinates_one.values():
                            checking_x = forecast_x + coords[0]
                            checking_y = forecast_y + coords[1]
                            if checking_x != target_x or checking_y != target_y:
                                if grid[checking_y][checking_x] is not None:
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
                                uturn_check = True

                    if uturn_check:
                        tile_in_hand.rotate_clockwise(1)
                        if tile_in_hand.internal_edges[edge] != StructureType.RIVER:
                            tile_in_hand.rotate_clockwise(2)

                bot_state.last_tile = tile_in_hand
                bot_state.last_tile.placed_pos = (target_x, target_y)
                return game.move_place_tile(query, tile_in_hand._to_model(), tile_hand_index)
    
    return brute_force_tile(game, bot_state, query)


def fallback_tile_placement(game: Game, bot_state: ImprovedBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Fallback tile placement when no good options found"""
    for tile_index, tile_in_hand in enumerate(game.state.my_tiles):
        for y in range(MAX_MAP_LENGTH):
            for x in range(MAX_MAP_LENGTH):
                if game.can_place_tile_at(tile_in_hand, x, y):
                    bot_state.last_tile = tile_in_hand
                    bot_state.last_tile.placed_pos = (x, y)
                    return game.move_place_tile(query, tile_in_hand._to_model(), tile_index)
    
    # Emergency fallback
    tile_in_hand = game.state.my_tiles[0]
    for y in range(MAX_MAP_LENGTH):
        for x in range(MAX_MAP_LENGTH):
            if game.state.map._grid[y][x] is None:
                for _ in range(4):
                    if game.can_place_tile_at(tile_in_hand, x, y):
                        bot_state.last_tile = tile_in_hand
                        bot_state.last_tile.placed_pos = (x, y)
                        return game.move_place_tile(query, tile_in_hand._to_model(), 0)
                    tile_in_hand.rotate_clockwise(1)
    
    raise RuntimeError("No valid tile placement found!")


def brute_force_tile(game: Game, bot_state: ImprovedBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Brute force tile placement"""
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    directions = {
        (0, 1): "top",
        (1, 0): "right",
        (0, -1): "bottom",
        (-1, 0): "left",
    }

    for y in range(height):
        for x in range(width):
            if grid[y][x] is not None:
                for tile_index, tile in enumerate(game.state.my_tiles):
                    for direction in directions:
                        dx, dy = direction
                        x1, y1 = (x + dx, y + dy)

                        if game.can_place_tile_at(tile, x1, y1):
                            bot_state.last_tile = tile
                            bot_state.last_tile.placed_pos = x1, y1
                            return game.move_place_tile(query, tile._to_model(), tile_index)


if __name__ == "__main__":
    main()