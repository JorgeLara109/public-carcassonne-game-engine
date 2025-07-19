"""
Improved Wena10 Bot - Practical Winning Strategies

Focus on proven, high-impact strategies rather than theoretical complexity:
1. Aggressive Monastery Strategy - Guaranteed 9 points
2. Smart City Expansion - High-value completions
3. Opponent Blocking - Deny opponent points
4. Farmer Strategy - End-game point multiplication
5. Tile Efficiency - Maximize points per meeple
6. Adaptive Strategy - Change tactics based on game state

This version prioritizes practical winning over theoretical elegance.
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
from lib.config.map_config import MAX_MAP_LENGTH
from lib.config.map_config import MONASTARY_IDENTIFIER
from lib.interact.structure import StructureType

from collections import defaultdict
import math
import random
from typing import Dict, List, Tuple, Optional


class MeepleInfo:
    """Track information about placed meeples"""
    def __init__(self, tile: Tile, structure_type: StructureType, edge: str, move_number: int, player_id: int = -1):
        self.tile = tile
        self.structure_type = structure_type
        self.edge = edge
        self.move_number = move_number
        self.position = tile.placed_pos
        self.player_id = player_id  # -1 for our meeples, 0-3 for opponents


class GamePhase:
    """Game phase constants"""
    EARLY = "early"    # 0-20 tiles
    MID = "mid"        # 21-50 tiles
    LATE = "late"      # 51+ tiles


class OpponentModel:
    """Simple opponent modeling"""
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.monastery_preference = 0.5
        self.city_preference = 0.5
        self.blocking_behavior = 0.5
        self.moves_observed = 0
        self.monastery_placements = 0
        self.city_placements = 0
        self.blocking_moves = 0
        
    def update_from_move(self, move_type: str):
        """Update model based on observed move"""
        self.moves_observed += 1
        
        if move_type == "monastery":
            self.monastery_placements += 1
        elif move_type == "city":
            self.city_placements += 1
        elif move_type == "blocking":
            self.blocking_moves += 1
        
        # Update preferences
        if self.moves_observed > 0:
            self.monastery_preference = self.monastery_placements / self.moves_observed
            self.city_preference = self.city_placements / self.moves_observed
            self.blocking_behavior = self.blocking_moves / self.moves_observed


class ImprovedBotState:
    """Improved bot state focusing on practical strategies"""

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
        self.opponent_models = [OpponentModel(i) for i in range(4)]
        
        # Strategy state
        self.strategy_mode = "balanced"
        self.monastery_priority = True  # Always prioritize monasteries early
        self.aggressive_blocking = False
        self.farmer_endgame = False
        
        # Performance tracking
        self.completed_structures = 0
        self.blocked_opponent_structures = 0
        self.average_points_per_meeple = 0.0
        
        # Tile analysis cache
        self.tile_value_cache = {}
        self.completion_probability_cache = {}

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
            # Early game: Focus on monasteries and safe investments
            self.monastery_priority = True
            self.aggressive_blocking = False
            self.farmer_endgame = False
            
        elif self.game_phase == GamePhase.MID:
            # Mid game: Balance expansion and blocking
            self.monastery_priority = meeples_remaining > 3
            self.aggressive_blocking = self.our_score < 30  # Block if behind
            self.farmer_endgame = False
            
        else:  # LATE
            # Late game: Aggressive completion and farmer strategy
            self.monastery_priority = meeples_remaining > 5
            self.aggressive_blocking = True
            self.farmer_endgame = meeples_remaining <= 2
        
        # Adaptive strategy based on performance
        if self.average_points_per_meeple < 6.0 and self.game_phase != GamePhase.EARLY:
            self.strategy_mode = "conservative"  # Focus on guaranteed points
        elif self.our_score > 35:
            self.strategy_mode = "defensive"  # Protect lead
        elif self.our_score < 20 and self.game_phase == GamePhase.MID:
            self.strategy_mode = "aggressive"  # Catch up
        else:
            self.strategy_mode = "balanced"


def main():
    game = Game()
    bot_state = ImprovedBotState()

    while True:
        query = game.get_next_query()

        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryPlaceTile() as q:
                    print("placing tile")
                    bot_state.move_count += 1
                    bot_state.update_game_phase(game)
                    return handle_place_tile_improved(game, bot_state, q)

                case QueryPlaceMeeple() as q:
                    print("meeple")
                    return handle_place_meeple_improved(game, bot_state, q)
                case _:
                    assert False

        print("sending move")
        game.send_move(choose_move(query))


def analyze_opponent_meeples(game: Game, bot_state: ImprovedBotState):
    """Analyze opponent meeple placements"""
    bot_state.opponent_meeples.clear()
    
    for tile in game.state.map.placed_tiles:
        if not tile.placed_pos:
            continue
            
        # Check for opponent meeples
        for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
            if tile.internal_claims.get(edge) is not None:
                # This is an opponent meeple (simplified assumption)
                structure_type = tile.internal_edges[edge]
                meeple_info = MeepleInfo(
                    tile, structure_type, edge, -1, 0  # player_id=0 for opponent
                )
                bot_state.opponent_meeples.append(meeple_info)
        
        # Check monastery
        if tile.internal_claims.get(MONASTARY_IDENTIFIER) is not None:
            meeple_info = MeepleInfo(
                tile, StructureType.MONASTARY, MONASTARY_IDENTIFIER, -1, 0
            )
            bot_state.opponent_meeples.append(meeple_info)


def is_river_phase(game: Game) -> bool:
    """
    Detect if we're in river phase by checking if any tiles in hand have river structures.
    """
    for tile in game.state.my_tiles:
        for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
            if tile.internal_edges[edge] == StructureType.RIVER:
                return True
    return False

def handle_place_tile_improved(game: Game, bot_state: ImprovedBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Improved tile placement with practical strategies"""
    
    # Update opponent analysis
    analyze_opponent_meeples(game, bot_state)
    
    # Handle river phase
    if is_river_phase(game):
        return handle_river_phase(game, bot_state, query)
    
    # Get all possible placements and evaluate them
    best_placements = []
    
    for tile_index, tile_in_hand in enumerate(game.state.my_tiles):
        positions_to_check = get_strategic_positions(game, bot_state)
        
        for x, y in positions_to_check:
            if not game.can_place_tile_at(tile_in_hand, x, y):
                continue
            
            # Evaluate this placement
            score = evaluate_tile_placement_improved(game, bot_state, tile_in_hand, x, y)
            best_placements.append((score, tile_index, tile_in_hand, x, y))
    
    # Sort by score and pick the best
    if best_placements:
        best_placements.sort(key=lambda x: x[0], reverse=True)
        score, tile_index, tile_in_hand, x, y = best_placements[0]
        
        if game.can_place_tile_at(tile_in_hand, x, y):
            bot_state.last_tile = tile_in_hand
            bot_state.last_tile.placed_pos = (x, y)
            print(f"Placing tile at {x}, {y} with score {score:.2f}")
            return game.move_place_tile(query, tile_in_hand._to_model(), tile_index)
    
    # Fallback
    return fallback_tile_placement(game, bot_state, query)


def get_strategic_positions(game: Game, bot_state: ImprovedBotState) -> List[Tuple[int, int]]:
    """Get positions to check in strategic order"""
    positions = set()
    grid = game.state.map._grid
    
    # Priority 1: Adjacent to our meeples (expansion opportunities)
    for meeple in bot_state.placed_meeples:
        if meeple.position:
            x, y = meeple.position
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < MAX_MAP_LENGTH and 0 <= new_y < MAX_MAP_LENGTH 
                    and grid[new_y][new_x] is None):
                    positions.add((new_x, new_y))
    
    # Priority 2: Adjacent to opponent meeples (blocking opportunities)
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


def evaluate_tile_placement_improved(game: Game, bot_state: ImprovedBotState, tile: Tile, x: int, y: int) -> float:
    """Improved tile placement evaluation focusing on practical strategies"""
    score = 0.0
    
    # Strategy 1: Monastery Priority (Guaranteed 9 points)
    monastery_score = evaluate_monastery_placement(game, bot_state, tile, x, y)
    if monastery_score > 0 and bot_state.monastery_priority:
        score += monastery_score * 3.0  # High priority multiplier
    else:
        score += monastery_score
    
    # Strategy 2: Structure Expansion (Connect to our existing meeples)
    expansion_score = evaluate_structure_expansion_improved(game, bot_state, tile, x, y)
    score += expansion_score * 2.0
    
    # Strategy 3: Quick Completion Opportunities
    completion_score = evaluate_quick_completion_improved(game, bot_state, tile, x, y)
    score += completion_score * 1.5
    
    # Strategy 4: Opponent Blocking
    if bot_state.aggressive_blocking:
        blocking_score = evaluate_opponent_blocking_improved(game, bot_state, tile, x, y)
        score += blocking_score * 1.8
    
    # Strategy 5: City Value (High point potential)
    city_score = evaluate_city_placement(game, bot_state, tile, x, y)
    score += city_score * 1.3
    
    # Strategy 6: Farmer Preparation (Late game)
    '''if bot_state.farmer_endgame:
        farmer_score = evaluate_farmer_potential(game, bot_state, tile, x, y)
        score += farmer_score * 2.0'''
    
    # Strategy 7: Tile Efficiency (Points per meeple)
    efficiency_score = evaluate_tile_efficiency(game, bot_state, tile, x, y)
    score += efficiency_score
    
    # Strategy 8: Position Quality (Central vs edge)
    position_score = evaluate_position_quality(game, x, y)
    score += position_score * 0.5
    
    return score


def evaluate_monastery_placement(game: Game, bot_state: ImprovedBotState, tile: Tile, x: int, y: int) -> float:
    """Evaluate monastery placement potential"""
    if not (hasattr(tile, "modifiers") and any(mod.name == "MONESTARY" for mod in tile.modifiers)):
        return 0.0
    
    # Count surrounding tiles
    surrounding_tiles = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            check_x, check_y = x + dx, y + dy
            if (0 <= check_x < MAX_MAP_LENGTH and 0 <= check_y < MAX_MAP_LENGTH 
                and game.state.map._grid[check_y][check_x] is not None):
                surrounding_tiles += 1
    
    # Higher score for more surrounded monasteries (easier to complete)
    base_score = 30.0  # Base monastery value
    completion_bonus = surrounding_tiles * 5.0  # Bonus for existing surrounding tiles
    
    # Phase-based adjustment
    if bot_state.game_phase == GamePhase.EARLY:
        phase_multiplier = 1.5  # Prioritize early
    elif bot_state.game_phase == GamePhase.MID:
        phase_multiplier = 1.2
    else:
        phase_multiplier = 0.8  # Lower priority late game
    
    return (base_score + completion_bonus) * phase_multiplier


def evaluate_structure_expansion_improved(game: Game, bot_state: ImprovedBotState, tile: Tile, x: int, y: int) -> float:
    """Evaluate how well this tile expands our existing structures"""
    expansion_score = 0.0
    
    # Check each edge of the tile
    for edge in tile.get_edges():
        structure_type = tile.internal_edges[edge]
        if structure_type in [StructureType.RIVER, StructureType.GRASS]:
            continue
        
        # Check if this edge connects to our existing meeples
        for meeple in bot_state.placed_meeples:
            if meeple.structure_type == structure_type and meeple.position:
                # Calculate distance to our meeple
                meeple_x, meeple_y = meeple.position
                distance = abs(x - meeple_x) + abs(y - meeple_y)
                
                if distance == 1:  # Adjacent connection
                    if structure_type == StructureType.CITY:
                        expansion_score += 25.0  # High value for city expansion
                    elif structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
                        expansion_score += 15.0  # Medium value for road expansion
                    elif structure_type == StructureType.MONASTARY:
                        expansion_score += 20.0  # Good value for monastery connection
                elif distance <= 3:  # Nearby connection potential
                    expansion_score += 5.0
    
    return expansion_score


def evaluate_quick_completion_improved(game: Game, bot_state: ImprovedBotState, tile: Tile, x: int, y: int) -> float:
    """Evaluate potential for quick structure completion"""
    completion_score = 0.0
    
    # Check adjacent tiles for completion opportunities
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    adjacent_structures = []
    
    for dx, dy in directions:
        adj_x, adj_y = x + dx, y + dy
        if (0 <= adj_x < MAX_MAP_LENGTH and 0 <= adj_y < MAX_MAP_LENGTH):
            adjacent_tile = game.state.map._grid[adj_y][adj_x]
            if adjacent_tile:
                adjacent_structures.append(adjacent_tile)
    
    # Bonus for positions that can complete multiple structures
    if len(adjacent_structures) >= 3:
        completion_score += 20.0  # High completion potential
    elif len(adjacent_structures) >= 2:
        completion_score += 10.0  # Medium completion potential
    
    # Check for specific completion patterns
    for adj_tile in adjacent_structures:
        # Look for our meeples on adjacent tiles
        for meeple in bot_state.placed_meeples:
            if meeple.position == adj_tile.placed_pos:
                if meeple.structure_type == StructureType.CITY:
                    completion_score += 15.0  # City completion bonus
                elif meeple.structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
                    completion_score += 10.0  # Road completion bonus
    
    return completion_score


def evaluate_opponent_blocking_improved(game: Game, bot_state: ImprovedBotState, tile: Tile, x: int, y: int) -> float:
    """Evaluate how well this placement blocks opponents"""
    blocking_score = 0.0
    
    # Block opponent monasteries
    for opponent_meeple in bot_state.opponent_meeples:
        if opponent_meeple.structure_type == StructureType.MONASTARY and opponent_meeple.position:
            monastery_x, monastery_y = opponent_meeple.position
            
            # Check if we're placing adjacent to their monastery
            if abs(x - monastery_x) <= 1 and abs(y - monastery_y) <= 1:
                # Count how complete their monastery is
                surrounding_tiles = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        check_x, check_y = monastery_x + dx, monastery_y + dy
                        if (0 <= check_x < MAX_MAP_LENGTH and 0 <= check_y < MAX_MAP_LENGTH 
                            and game.state.map._grid[check_y][check_x] is not None):
                            surrounding_tiles += 1
                
                # Higher blocking value for more complete monasteries
                completion_percentage = surrounding_tiles / 8.0
                blocking_score += completion_percentage * 25.0
    
    # Block opponent structure expansions
    for opponent_meeple in bot_state.opponent_meeples:
        if opponent_meeple.position:
            distance = abs(x - opponent_meeple.position[0]) + abs(y - opponent_meeple.position[1])
            if distance == 1:  # Adjacent blocking
                if opponent_meeple.structure_type == StructureType.CITY:
                    blocking_score += 20.0
                elif opponent_meeple.structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
                    blocking_score += 15.0
    
    return blocking_score


def evaluate_city_placement(game: Game, bot_state: ImprovedBotState, tile: Tile, x: int, y: int) -> float:
    """Evaluate city placement potential"""
    city_score = 0.0
    
    # Check if tile has city structures
    has_city = False
    city_edges = 0
    
    for edge in tile.get_edges():
        if tile.internal_edges[edge] == StructureType.CITY:
            has_city = True
            city_edges += 1
    
    if not has_city:
        return 0.0
    
    # Base city value
    base_score = 20.0
    
    # Bonus for multiple city edges (easier to expand)
    if city_edges >= 2:
        base_score += 10.0
    
    # Check for banner tiles (extra points)
    if hasattr(tile, "modifiers"):
        for modifier in tile.modifiers:
            if "BANNER" in modifier.name or "banner" in modifier.name.lower():
                base_score += 15.0  # Banner bonus
    
    # Check adjacency to existing cities
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    adjacent_cities = 0
    
    for dx, dy in directions:
        adj_x, adj_y = x + dx, y + dy
        if (0 <= adj_x < MAX_MAP_LENGTH and 0 <= adj_y < MAX_MAP_LENGTH):
            adjacent_tile = game.state.map._grid[adj_y][adj_x]
            if adjacent_tile:
                for edge in adjacent_tile.get_edges():
                    if adjacent_tile.internal_edges[edge] == StructureType.CITY:
                        adjacent_cities += 1
                        break
    
    # Bonus for connecting to existing cities
    city_score = base_score + (adjacent_cities * 8.0)
    
    return city_score


def evaluate_farmer_potential(game: Game, bot_state: ImprovedBotState, tile: Tile, x: int, y: int) -> float:
    """Evaluate farmer placement potential for end game"""
    farmer_score = 0.0
    
    # Only consider farmer strategy in late game
    if bot_state.game_phase != GamePhase.LATE:
        return 0.0
    
    # Check if tile has grass areas
    has_grass = False
    for edge in tile.get_edges():
        if tile.internal_edges[edge] == StructureType.GRASS:
            has_grass = True
            break
    
    if not has_grass:
        return 0.0
    
    # Count nearby completed cities (farmers get points for adjacent completed cities)
    nearby_cities = 0
    search_radius = 3
    
    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            check_x, check_y = x + dx, y + dy
            if (0 <= check_x < MAX_MAP_LENGTH and 0 <= check_y < MAX_MAP_LENGTH):
                tile_at_pos = game.state.map._grid[check_y][check_x]
                if tile_at_pos:
                    # Check if this tile has completed cities
                    for edge in tile_at_pos.get_edges():
                        if tile_at_pos.internal_edges[edge] == StructureType.CITY:
                            # Simplified check for completed city
                            if game.state._check_completed_component(tile_at_pos, edge):
                                nearby_cities += 1
    
    # Farmer value based on nearby completed cities
    farmer_score = nearby_cities * 10.0
    
    return farmer_score


def evaluate_tile_efficiency(game: Game, bot_state: ImprovedBotState, tile: Tile, x: int, y: int) -> float:
    """Evaluate points per meeple efficiency"""
    efficiency_score = 0.0
    
    # Calculate expected points for structures on this tile
    expected_points = 0.0
    placeable_structures = 0
    
    for edge in tile.get_edges():
        structure_type = tile.internal_edges[edge]
        if structure_type not in [StructureType.RIVER, StructureType.GRASS]:
            placeable_structures += 1
            
            if structure_type == StructureType.MONASTARY:
                expected_points += 9.0  # Guaranteed monastery points
            elif structure_type == StructureType.CITY:
                expected_points += 12.0  # Average city points
            elif structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
                expected_points += 6.0  # Average road points
    
    # Efficiency = expected points / meeples needed
    if placeable_structures > 0:
        efficiency_score = expected_points / placeable_structures
    
    return efficiency_score


def evaluate_position_quality(game: Game, x: int, y: int) -> float:
    """Evaluate the quality of the position on the board"""
    # Prefer central positions over edge positions
    center_x, center_y = MAX_MAP_LENGTH // 2, MAX_MAP_LENGTH // 2
    distance_from_center = abs(x - center_x) + abs(y - center_y)
    
    # Closer to center = higher score
    position_score = max(0, 20 - distance_from_center)
    
    return position_score


def handle_place_meeple_improved(game: Game, bot_state: ImprovedBotState, query: QueryPlaceMeeple) -> MovePlaceMeeplePass | MovePlaceMeeple:
    """Improved meeple placement with strategic priorities"""
    
    if not bot_state.last_tile or bot_state.meeples_placed >= 7:
        return game.move_place_meeple_pass(query)
    
    recent_tile = bot_state.last_tile
    meeples_remaining = 7 - bot_state.meeples_placed
    
    # PRIORITY 1: MONASTERY (Guaranteed 9 points)
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
    
    # Evaluate each possible meeple placement
    meeple_evaluations = []
    
    for edge, structure in structures:
        structure_type = recent_tile.internal_edges.get(edge)
        if structure_type is None or structure_type == StructureType.RIVER:
            continue
        
        if (not game.state._get_claims(recent_tile, edge) 
            and not game.state._check_completed_component(recent_tile, edge)):
            
            score = evaluate_meeple_placement_improved(game, bot_state, recent_tile, edge, structure_type)
            meeple_evaluations.append((score, edge, structure_type))
    
    # Sort by score and place best meeple
    if meeple_evaluations:
        meeple_evaluations.sort(key=lambda x: x[0], reverse=True)
        best_score, best_edge, best_structure_type = meeple_evaluations[0]
        
        # Only place meeple if score is above threshold
        min_score_threshold = 8.0 if meeples_remaining > 3 else 5.0
        
        if best_score >= min_score_threshold:
            bot_state.meeples_placed += 1
            meeple_info = MeepleInfo(recent_tile, best_structure_type, best_edge, bot_state.move_count)
            bot_state.placed_meeples.append(meeple_info)
            print(f"Placed meeple on {best_structure_type.name} with score {best_score:.2f}")
            return game.move_place_meeple(query, recent_tile._to_model(), best_edge)
    
    # PRIORITY 3: FARMER STRATEGY (Late game only)
    '''if bot_state.farmer_endgame and meeples_remaining <= 2:
        for edge, structure in structures:
            structure_type = recent_tile.internal_edges.get(edge)
            if structure_type == StructureType.GRASS:
                if (not game.state._get_claims(recent_tile, edge) 
                    and not game.state._check_completed_component(recent_tile, edge)):
                    
                    # Place farmer for end-game points
                    bot_state.meeples_placed += 1
                    meeple_info = MeepleInfo(recent_tile, StructureType.GRASS, edge, bot_state.move_count)
                    bot_state.placed_meeples.append(meeple_info)
                    print(f"Placed farmer for end-game strategy")
                    return game.move_place_meeple(query, recent_tile._to_model(), edge)
    '''
    print("Passing meeple placement")
    return game.move_place_meeple_pass(query)


def evaluate_meeple_placement_improved(game: Game, bot_state: ImprovedBotState, tile: Tile, edge: str, structure_type: StructureType) -> float:
    """Evaluate the value of placing a meeple on a specific structure"""
    score = 0.0
    
    # Base scores by structure type
    if structure_type == StructureType.MONASTARY:
        score = 20.0  # High base value for monastery
    elif structure_type == StructureType.CITY:
        score = 15.0  # Good base value for city
    elif structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
        score = 10.0  # Medium base value for road
    elif structure_type == StructureType.GRASS:
        score = 5.0   # Low base value for grass (unless farmer strategy)
    
    # Adjustment based on completion probability
    completion_prob = estimate_completion_probability(game, bot_state, tile, edge, structure_type)
    score *= completion_prob
    
    # Adjustment based on game phase
    if bot_state.game_phase == GamePhase.EARLY:
        if structure_type == StructureType.MONASTARY:
            score *= 1.5  # Prioritize monasteries early
    elif bot_state.game_phase == GamePhase.MID:
        if structure_type == StructureType.CITY:
            score *= 1.3  # Cities become more valuable
    '''else:  # LATE
        if structure_type == StructureType.GRASS and bot_state.farmer_endgame:
            score *= 3.0  # Farmers very valuable late game'''
    
    # Adjustment based on meeples remaining
    meeples_remaining = 7 - bot_state.meeples_placed
    if meeples_remaining <= 2:
        # Be more selective with last meeples
        if structure_type == StructureType.MONASTARY:
            score *= 1.5  # Prioritize guaranteed points
        elif completion_prob < 0.7:
            score *= 0.5  # Avoid risky placements
    
    return score


def estimate_completion_probability(game: Game, bot_state: ImprovedBotState, tile: Tile, edge: str, structure_type: StructureType) -> float:
    """Estimate the probability of completing a structure"""
    # Simple estimation based on structure type and game phase
    base_prob = 0.5
    
    if structure_type == StructureType.MONASTARY:
        # Count empty spaces around monastery
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
        
        # More empty spaces = harder to complete
        base_prob = 0.9 - (empty_spaces * 0.08)
        
    elif structure_type == StructureType.CITY:
        # Cities are generally harder to complete
        base_prob = 0.6
        if bot_state.game_phase == GamePhase.LATE:
            base_prob = 0.4  # Even harder late game
            
    elif structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
        # Roads are easier to complete
        base_prob = 0.8
        
    elif structure_type == StructureType.GRASS:
        # Farmers don't need completion
        base_prob = 1.0
    
    return base_prob


def handle_river_phase(
    game: Game, bot_state: ImprovedBotState, query: QueryPlaceTile  # Change from DomainExpansionBotState
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


def fallback_tile_placement(game: Game, bot_state: ImprovedBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Fallback tile placement when no good options found"""
    # Try to place any tile anywhere valid
    for tile_index, tile_in_hand in enumerate(game.state.my_tiles):
        for y in range(MAX_MAP_LENGTH):
            for x in range(MAX_MAP_LENGTH):
                if game.can_place_tile_at(tile_in_hand, x, y):
                    bot_state.last_tile = tile_in_hand
                    bot_state.last_tile.placed_pos = (x, y)
                    print(f"Fallback placement at {x}, {y}")
                    return game.move_place_tile(query, tile_in_hand._to_model(), tile_index)
    
    # If absolutely nothing works, place first tile at first valid position
    tile_in_hand = game.state.my_tiles[0]
    for y in range(MAX_MAP_LENGTH):
        for x in range(MAX_MAP_LENGTH):
            if game.state.map._grid[y][x] is None:
                # Try all rotations
                for _ in range(4):
                    if game.can_place_tile_at(tile_in_hand, x, y):
                        bot_state.last_tile = tile_in_hand
                        bot_state.last_tile.placed_pos = (x, y)
                        print(f"Emergency placement at {x}, {y}")
                        return game.move_place_tile(query, tile_in_hand._to_model(), 0)
                    tile_in_hand.rotate_clockwise(1)
    
    # This should never happen
    raise RuntimeError("No valid tile placement found!")


def brute_force_tile(
    game: Game, bot_state: ImprovedBotState, query: QueryPlaceTile
) -> MovePlaceTile:
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

if __name__ == "__main__":
    main()