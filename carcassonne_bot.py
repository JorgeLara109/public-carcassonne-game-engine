"""
Intelligent Carcassonne Bot - Iterative Strategy Development
Module 1: Structure Origin Tracking

This bot implements strategic modules incrementally, starting with structure origin tracking.
"""

from helper.game import Game
from lib.interact.tile import Tile, TileModifier
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
from typing import Dict, List, Tuple, Set
import uuid


class OriginStructure:
    """Represents a structure that originated from a tile placement"""
    def __init__(self, structure_id: str, structure_type: StructureType, 
                 origin_pos: Tuple[int, int], origin_edge: str, initial_value: int):
        self.structure_id = structure_id
        self.structure_type = structure_type
        self.origin_pos = origin_pos
        self.origin_edge = origin_edge
        self.value = initial_value
        self.expansion_edges: List[Tuple[int, int, str]] = []  # (x, y, edge)
        self.blocked_by_enemy = False
        
    def __repr__(self):
        return f"Structure({self.structure_id[:8]}, {self.structure_type}, value={self.value}, blocked={self.blocked_by_enemy})"


class BotState:
    """Bot state with structure origin tracking"""
    
    def __init__(self):
        self.last_tile: Tile | None = None
        self.meeples_placed: int = 0
        self.round_count: int = 0
        
        # Module 1: Structure Origin Tracking
        self.origin_structures: Dict[str, OriginStructure] = {}
        self.turn_log: List[str] = []
        
        # Module 2: Structure Completion & Expansion Detection
        self.completed_structures: Set[str] = set()
        self.extended_structures: Dict[str, int] = {}  # structure_id -> times_extended
        
    def log_action(self, action: str):
        """Log an action for debugging"""
        self.turn_log.append(f"Round {self.round_count}: {action}")
        print(f"[STRUCTURE_TRACKER] {action}")


def main():
    game = Game()
    bot_state = BotState()
    
    while True:
        query = game.get_next_query()
        
        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryPlaceTile() as q:
                    bot_state.round_count += 1
                    return handle_place_tile(game, bot_state, q)
                    
                case QueryPlaceMeeple() as q:
                    return handle_place_meeple(game, bot_state, q)
                case _:
                    assert False
                    
        game.send_move(choose_move(query))


def handle_place_tile(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile:
    """
    Module 1: Structure Origin Tracking
    - Place tile using basic logic
    - Track new structures created by this placement
    """
    bot_state.log_action("Starting tile placement")
    
    # Basic tile placement (similar to existing bots but simplified)
    placed_tile_info = place_tile_basic(game, bot_state, query)
    
    # Module 1: Track structures originated from this placement
    if bot_state.last_tile and bot_state.last_tile.placed_pos:
        track_new_structures(game, bot_state, bot_state.last_tile)
    
    # Module 2: Check for structure completion and expansion
    if bot_state.last_tile and bot_state.last_tile.placed_pos:
        analyze_structure_changes(game, bot_state, bot_state.last_tile)
    
    return placed_tile_info


def place_tile_basic(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile:
    """MODULE 6: Advanced tile placement priority strategy WITH SUB-20 ROUND OPTIMIZATION"""
    
    bot_state.log_action("🎯 MODULE 6: Advanced strategic tile placement")
    
    # SUB-20 OPTIMIZATION: Force aggressive end-game completion in final rounds
    if 16 <= bot_state.round_count <= 19:
        bot_state.log_action("⚡ SUB-19 MODE: FORCE END-GAME completion strategy")
        
        # ULTRA-PRIORITY: Complete ANY available structure immediately to force game end
        completion_move = find_high_priority_completion(game, bot_state, query)
        if completion_move:
            bot_state.log_action("🔥 SUB-19: FORCE COMPLETE to end game!")
            return completion_move
            
        # ULTRA-PRIORITY 2: Place monasteries for quick points to finish game
        monastery_move = try_monastery_placement(game, bot_state, query)
        if monastery_move:
            bot_state.log_action("🏛️ SUB-19: EMERGENCY monastery to finish!")
            return monastery_move
    
    # PRIORITY 1: Complete unclaimed structures (cities heavily prioritized)
    completion_move = find_high_priority_completion(game, bot_state, query)
    if completion_move:
        bot_state.log_action("✅ MODULE 6: Found high-priority completion move!")
        return completion_move
    
    # PRIORITY 2: Extend valuable unclaimed structures (cities over roads)
    extension_move = find_strategic_extension(game, bot_state, query)
    if extension_move:
        bot_state.log_action("📈 MODULE 6: Found strategic extension move!")
        return extension_move
    
    # PRIORITY 3: Monastery placement (density-based after river phase)
    monastery_move = try_monastery_placement(game, bot_state, query)
    if monastery_move:
        bot_state.log_action("🏛️ MODULE 5: Placed monastery tile!")
        return monastery_move
    
    # PRIORITY 4: General structure completion (existing logic)
    legacy_completion = find_structure_completion_move(game, bot_state, query)
    if legacy_completion:
        bot_state.log_action("✅ LEGACY: Found structure completion move!")
        return legacy_completion
    
    # PRIORITY 5: General structure extension (existing logic)  
    legacy_extension = find_structure_extension_move(game, bot_state, query)
    if legacy_extension:
        bot_state.log_action("📈 LEGACY: Found structure extension move!")
        return legacy_extension
    
    # PRIORITY 6: Basic placement near existing tiles (fallback)
    bot_state.log_action("⚠️ MODULE 6: Using fallback placement logic")
    return fallback_tile_placement(game, bot_state, query)


def find_structure_completion_move(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile | None:
    """Find a move that completes one of our tracked structures"""
    grid = game.state.map._grid
    
    for structure_id, origin_struct in bot_state.origin_structures.items():
        if structure_id in bot_state.completed_structures:
            continue
            
        # Check if this structure has exactly 1 expansion left
        remaining_expansions = []
        for exp_x, exp_y, exp_edge in origin_struct.expansion_edges:
            if (0 <= exp_x < MAX_MAP_LENGTH and 0 <= exp_y < MAX_MAP_LENGTH and 
                grid[exp_y][exp_x] is None):
                remaining_expansions.append((exp_x, exp_y, exp_edge))
        
        if len(remaining_expansions) == 1:
            exp_x, exp_y, exp_edge = remaining_expansions[0]
            bot_state.log_action(f"🎯 Found structure {structure_id[:8]} with 1 expansion at ({exp_x}, {exp_y})")
            
            # Try each tile to see if it can complete this structure
            for tile_index, tile in enumerate(game.state.my_tiles):
                if game.can_place_tile_at(tile, exp_x, exp_y):
                    # Check river validation if needed
                    has_river = has_river_edge(tile)
                    if has_river:
                        try:
                            river_validation = game.state.map.river_validation(tile, exp_x, exp_y)
                            if river_validation != "pass":
                                continue
                        except:
                            continue
                    
                    # Check if this tile has matching structure type to complete it
                    opposite_edges = {
                        "top_edge": "bottom_edge",
                        "right_edge": "left_edge",
                        "bottom_edge": "top_edge",
                        "left_edge": "right_edge"
                    }
                    
                    if exp_edge in opposite_edges:
                        connecting_edge = opposite_edges[exp_edge]
                        tile_structure = tile.internal_edges.get(connecting_edge)
                        
                        if tile_structure == origin_struct.structure_type:
                            bot_state.last_tile = tile
                            bot_state.last_tile.placed_pos = (exp_x, exp_y)
                            bot_state.log_action(f"🏆 COMPLETING structure {structure_id[:8]} (value: {origin_struct.value}) at ({exp_x}, {exp_y})")
                            return game.move_place_tile(query, tile._to_model(), tile_index)
    
    return None


def find_structure_extension_move(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile | None:
    """Find a move that extends one of our high-value structures"""
    grid = game.state.map._grid
    
    # Sort structures by value (prioritize higher value structures)
    sorted_structures = sorted(bot_state.origin_structures.items(), 
                              key=lambda x: x[1].value, reverse=True)
    
    for structure_id, origin_struct in sorted_structures:
        if structure_id in bot_state.completed_structures:
            continue
            
        # Try to extend this structure
        for exp_x, exp_y, exp_edge in origin_struct.expansion_edges:
            if (0 <= exp_x < MAX_MAP_LENGTH and 0 <= exp_y < MAX_MAP_LENGTH and 
                grid[exp_y][exp_x] is None):
                
                # Try each tile to see if it can extend this structure
                for tile_index, tile in enumerate(game.state.my_tiles):
                    if game.can_place_tile_at(tile, exp_x, exp_y):
                        # Check river validation if needed
                        has_river = has_river_edge(tile)
                        if has_river:
                            try:
                                river_validation = game.state.map.river_validation(tile, exp_x, exp_y)
                                if river_validation != "pass":
                                    continue
                            except:
                                continue
                        
                        # Check if this tile can extend the structure
                        opposite_edges = {
                            "top_edge": "bottom_edge",
                            "right_edge": "left_edge",
                            "bottom_edge": "top_edge",
                            "left_edge": "right_edge"
                        }
                        
                        if exp_edge in opposite_edges:
                            connecting_edge = opposite_edges[exp_edge]
                            tile_structure = tile.internal_edges.get(connecting_edge)
                            
                            if tile_structure == origin_struct.structure_type:
                                # Check that this won't be immediately claimed by opponents
                                bot_state.last_tile = tile
                                bot_state.last_tile.placed_pos = (exp_x, exp_y)
                                bot_state.log_action(f"📈 EXTENDING structure {structure_id[:8]} (value: {origin_struct.value}) at ({exp_x}, {exp_y})")
                                return game.move_place_tile(query, tile._to_model(), tile_index)
    
    return None


def find_high_priority_completion(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile | None:
    """
    IMPROVED: Find unclaimed structures with only 1 expansion left and complete them
    HEAVILY prioritize cities over roads (cities are 2-4x more valuable)
    """
    completion_candidates = []
    
    # Scan all placed tiles for structures with exactly 1 expansion left
    for placed_tile in game.state.map.placed_tiles:
        if not placed_tile.placed_pos:
            continue
            
        edges = ["top_edge", "right_edge", "bottom_edge", "left_edge"]
        
        for edge in edges:
            structure_type = placed_tile.internal_edges.get(edge)
            
            # Only consider valuable structures (Cities and Roads)
            if structure_type not in [StructureType.CITY, StructureType.ROAD]:
                continue
            
            # Check if this edge is unclaimed
            if game.state._get_claims(placed_tile, edge):
                continue  # Already claimed
                
            # Check if structure is already completed
            if game.state._check_completed_component(placed_tile, edge):
                continue  # Already completed
            
            # Find all expansion positions for this structure
            expansion_positions = find_structure_expansions(game, placed_tile, edge, structure_type)
            
            # Only interested in structures with exactly 1 expansion left
            if len(expansion_positions) == 1:
                exp_x, exp_y, exp_edge = expansion_positions[0]
                
                # Calculate structure value with HEAVY city bias
                base_value = calculate_structure_value(game, placed_tile, edge, structure_type)
                
                # ULTIMATE CITY PRIORITY: Cities are almost always better than monasteries
                if structure_type == StructureType.CITY:
                    # Cities get massive priority boost - they're typically worth 2x+ points
                    if bot_state.round_count <= 5:
                        early_game_multiplier = 100  # Extreme city focus - even small cities are valuable
                    elif bot_state.round_count <= 10:
                        early_game_multiplier = 60   # Very high priority
                    else:
                        early_game_multiplier = 25   # Still much higher than monastery priority
                    
                    # Additional bonus for immediate completion (1 expansion)
                    if len(expansion_positions) == 1:
                        early_game_multiplier *= 2  # Double priority for immediate completion
                    
                    priority_value = base_value * early_game_multiplier
                    bot_state.log_action(f"[ULTIMATE-CITY] City priority: {priority_value} (base: {base_value}, mult: {early_game_multiplier}x)")
                else:
                    # Roads get decent priority but much less than cities
                    early_game_multiplier = 8 if bot_state.round_count <= 10 else 3
                    if len(expansion_positions) == 1:
                        early_game_multiplier *= 1.5  # Modest bonus for immediate completion
                    priority_value = base_value * early_game_multiplier
                
                completion_candidates.append((exp_x, exp_y, exp_edge, priority_value, structure_type, placed_tile, edge, base_value))
    
    # Sort by priority value (cities heavily favored)
    completion_candidates.sort(key=lambda x: x[3], reverse=True)
    
    # Try to complete the highest priority structures
    for exp_x, exp_y, exp_edge, priority_value, structure_type, source_tile, source_edge, actual_value in completion_candidates[:5]:
        bot_state.log_action(f"[IMPROVED] Trying to complete {structure_type} (value: {actual_value}, priority: {priority_value}) at ({exp_x}, {exp_y})")
        
        # Try each tile in hand
        for tile_index, tile in enumerate(game.state.my_tiles):
            if game.can_place_tile_at(tile, exp_x, exp_y):
                # Check river validation if needed
                if has_river_edge(tile):
                    try:
                        river_validation = game.state.map.river_validation(tile, exp_x, exp_y)
                        if river_validation != "pass":
                            continue
                    except:
                        continue
                
                # Check if this tile can complete the structure
                if can_complete_structure(tile, exp_edge, structure_type):
                    bot_state.last_tile = tile
                    bot_state.last_tile.placed_pos = (exp_x, exp_y)
                    bot_state.log_action(f"[IMPROVED] COMPLETING {structure_type} (value: {actual_value}) at ({exp_x}, {exp_y})")
                    return game.move_place_tile(query, tile._to_model(), tile_index)
    
    return None


def find_strategic_extension(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile | None:
    """
    MODULE 6: Extend valuable unclaimed structures
    Only extend if structure is unclaimed and has good potential value
    """
    grid = game.state.map._grid
    extension_candidates = []
    
    # Find unclaimed structures worth extending
    for placed_tile in game.state.map.placed_tiles:
        if not placed_tile.placed_pos:
            continue
            
        tile_x, tile_y = placed_tile.placed_pos
        edges = ["top_edge", "right_edge", "bottom_edge", "left_edge"]
        
        for edge in edges:
            structure_type = placed_tile.internal_edges.get(edge)
            
            # Only consider valuable structures
            if structure_type not in [StructureType.CITY, StructureType.ROAD]:
                continue
            
            # Check if this edge is unclaimed
            if game.state._get_claims(placed_tile, edge):
                continue  # Already claimed
                
            # Check if structure is already completed
            if game.state._check_completed_component(placed_tile, edge):
                continue  # Already completed
            
            # Find expansion positions
            expansion_positions = find_structure_expansions(game, placed_tile, edge, structure_type)
            
            # Only extend structures with 2+ expansions (avoid interfering with completion logic)
            if len(expansion_positions) < 2:
                continue
            
            # Calculate structure value potential
            structure_value = calculate_structure_value(game, placed_tile, edge, structure_type)
            
            for exp_x, exp_y, exp_edge in expansion_positions[:2]:  # Limit to 2 best positions
                extension_candidates.append((exp_x, exp_y, exp_edge, structure_value, structure_type, placed_tile, edge))
    
    # Sort by structure value (highest first)
    extension_candidates.sort(key=lambda x: x[3], reverse=True)
    
    # Try to extend highest value structures
    for exp_x, exp_y, exp_edge, value, structure_type, source_tile, source_edge in extension_candidates[:5]:
        bot_state.log_action(f"[MODULE 6] Trying to extend {structure_type} (value: {value}) at ({exp_x}, {exp_y})")
        
        # Try each tile in hand
        for tile_index, tile in enumerate(game.state.my_tiles):
            if game.can_place_tile_at(tile, exp_x, exp_y):
                # Check river validation if needed
                if has_river_edge(tile):
                    try:
                        river_validation = game.state.map.river_validation(tile, exp_x, exp_y)
                        if river_validation != "pass":
                            continue
                    except:
                        continue
                
                # Check if this tile can extend the structure
                if can_extend_structure(tile, exp_edge, structure_type):
                    bot_state.last_tile = tile
                    bot_state.last_tile.placed_pos = (exp_x, exp_y)
                    bot_state.log_action(f"[MODULE 6] EXTENDING {structure_type} (value: {value}) at ({exp_x}, {exp_y})")
                    return game.move_place_tile(query, tile._to_model(), tile_index)
    
    return None


def find_structure_expansions(game: Game, tile: Tile, edge: str, structure_type: StructureType) -> list:
    """Find all possible expansion positions for a structure"""
    if not tile.placed_pos:
        return []
        
    grid = game.state.map._grid
    expansions = []
    
    # Get all positions connected to this structure
    visited = set()
    structure_edges = []
    
    def explore_structure(current_tile, current_edge):
        if not current_tile.placed_pos:
            return
        pos_key = (current_tile.placed_pos[0], current_tile.placed_pos[1], current_edge)
        if pos_key in visited:
            return
        visited.add(pos_key)
        
        # Check if this edge has the matching structure type
        if current_tile.internal_edges.get(current_edge) == structure_type:
            structure_edges.append((current_tile, current_edge))
            
            # Find adjacent tiles and continue exploration
            x, y = current_tile.placed_pos
            edge_directions = {
                "top_edge": (0, -1),
                "right_edge": (1, 0),
                "bottom_edge": (0, 1),
                "left_edge": (-1, 0)
            }
            
            if current_edge in edge_directions:
                dx, dy = edge_directions[current_edge]
                adj_x, adj_y = x + dx, y + dy
                
                # Check bounds and get adjacent tile
                if (0 <= adj_x < MAX_MAP_LENGTH and 0 <= adj_y < MAX_MAP_LENGTH):
                    adj_tile = grid[adj_y][adj_x]
                    if adj_tile is not None:
                        # Find connecting edge on adjacent tile
                        opposite_edges = {
                            "top_edge": "bottom_edge",
                            "right_edge": "left_edge",
                            "bottom_edge": "top_edge",
                            "left_edge": "right_edge"
                        }
                        opposite_edge = opposite_edges[current_edge]
                        if adj_tile.internal_edges.get(opposite_edge) == structure_type:
                            explore_structure(adj_tile, opposite_edge)
    
    # Start exploration from the given edge
    explore_structure(tile, edge)
    
    # Find expansion positions (empty positions adjacent to structure edges)
    edge_directions = {
        "top_edge": (0, -1),
        "right_edge": (1, 0), 
        "bottom_edge": (0, 1),
        "left_edge": (-1, 0)
    }
    
    expansion_set = set()
    for struct_tile, struct_edge in structure_edges:
        if not struct_tile.placed_pos:
            continue
        x, y = struct_tile.placed_pos
        
        if struct_edge in edge_directions:
            dx, dy = edge_directions[struct_edge]
            exp_x, exp_y = x + dx, y + dy
            
            # Check if position is empty and in bounds
            if (0 <= exp_x < MAX_MAP_LENGTH and 0 <= exp_y < MAX_MAP_LENGTH and
                grid[exp_y][exp_x] is None):
                expansion_set.add((exp_x, exp_y, struct_edge))
    
    return list(expansion_set)


def calculate_structure_value(game: Game, tile: Tile, edge: str, structure_type: StructureType) -> int:
    """Calculate the potential value of a structure"""
    if structure_type == StructureType.ROAD:
        return 1  # Roads are worth 1 point per tile
    elif structure_type == StructureType.CITY:
        # Cities are worth 2 points per tile (4 when completed)
        # Check for emblems/banners for bonus
        has_emblem = False
        if hasattr(tile, 'modifiers'):
            has_emblem = any(mod.name == "PENNANT" for mod in tile.modifiers)
        return 4 if has_emblem else 2
    
    return 0


def can_complete_structure(tile: Tile, expansion_edge: str, structure_type: StructureType) -> bool:
    """Check if a tile can complete a structure at the given expansion edge"""
    opposite_edges = {
        "top_edge": "bottom_edge",
        "right_edge": "left_edge",
        "bottom_edge": "top_edge",
        "left_edge": "right_edge"
    }
    
    if expansion_edge not in opposite_edges:
        return False
    
    connecting_edge = opposite_edges[expansion_edge]
    tile_structure = tile.internal_edges.get(connecting_edge)
    
    return tile_structure == structure_type


def can_extend_structure(tile: Tile, expansion_edge: str, structure_type: StructureType) -> bool:
    """Check if a tile can extend a structure at the given expansion edge"""
    return can_complete_structure(tile, expansion_edge, structure_type)


def fallback_tile_placement(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Fallback tile placement logic"""
    grid = game.state.map._grid
    
    # Try to place near the most recent tile (original logic)
    if game.state.map.placed_tiles:
        latest_tile = game.state.map.placed_tiles[-1]
        latest_pos = latest_tile.placed_pos
        
        if latest_pos:
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            
            for tile_index, tile in enumerate(game.state.my_tiles):
                has_river = has_river_edge(tile)
                
                for dx, dy in directions:
                    target_x = latest_pos[0] + dx
                    target_y = latest_pos[1] + dy
                    
                    # Check bounds
                    if not (0 <= target_x < MAX_MAP_LENGTH and 0 <= target_y < MAX_MAP_LENGTH):
                        continue
                        
                    # Check if position is empty
                    if grid[target_y][target_x] is not None:
                        continue
                    
                    if game.can_place_tile_at(tile, target_x, target_y):
                        # Check river validation if needed
                        if has_river:
                            try:
                                river_validation = game.state.map.river_validation(tile, target_x, target_y)
                                if river_validation != "pass":
                                    continue
                            except:
                                continue
                        
                        bot_state.last_tile = tile
                        bot_state.last_tile.placed_pos = (target_x, target_y)
                        bot_state.log_action(f"⚠️ FALLBACK: Placed tile {tile.tile_type} at ({target_x}, {target_y})")
                        return game.move_place_tile(query, tile._to_model(), tile_index)
    
    # Ultimate fallback: brute force search
    return brute_force_tile_placement(game, bot_state, query)


def brute_force_tile_placement(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Brute force tile placement as fallback with river handling"""
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for y in range(height):
        for x in range(width):
            if grid[y][x] is not None:
                for tile_index, tile in enumerate(game.state.my_tiles):
                    # Check if this is a river tile
                    has_river = has_river_edge(tile)
                    
                    for dx, dy in directions:
                        x1, y1 = (x + dx, y + dy)
                        
                        if game.can_place_tile_at(tile, x1, y1):
                            # For river tiles, check river validation
                            if has_river:
                                try:
                                    river_validation = game.state.map.river_validation(tile, x1, y1)
                                    if river_validation != "pass":
                                        continue
                                except:
                                    continue
                                    
                            bot_state.last_tile = tile
                            bot_state.last_tile.placed_pos = (x1, y1)
                            
                            if has_river:
                                bot_state.log_action(f"Brute force placed river tile {tile.tile_type} at ({x1}, {y1})")
                            else:
                                bot_state.log_action(f"Brute force placed tile {tile.tile_type} at ({x1}, {y1})")
                                
                            return game.move_place_tile(query, tile._to_model(), tile_index)
    
    raise ValueError("No valid tile placement found")


def track_new_structures(game: Game, bot_state: BotState, placed_tile: Tile):
    """
    Module 1: Structure Origin Tracking
    Detect and track new structures created by placing this tile
    """
    if not placed_tile.placed_pos:
        return
        
    x, y = placed_tile.placed_pos
    bot_state.log_action(f"Tracking structures on tile {placed_tile.tile_type} at ({x}, {y})")
    
    # Check each edge of the placed tile for structures
    edges = ["top_edge", "right_edge", "bottom_edge", "left_edge"]
    
    for edge in edges:
        structure_type = placed_tile.internal_edges.get(edge)
        
        # Only track Cities and Roads (not Grass, River, etc.)
        if structure_type in [StructureType.CITY, StructureType.ROAD]:
            # Check if this creates a new structure or extends existing one
            if is_new_origin_structure(game, bot_state, placed_tile, edge, structure_type):
                create_origin_structure(bot_state, placed_tile, edge, structure_type)


def is_new_origin_structure(game: Game, bot_state: BotState, tile: Tile, edge: str, 
                          structure_type: StructureType) -> bool:
    """
    Determine if this edge creates a new origin structure or extends an existing one
    For Module 1, we'll mark it as new if no adjacent structures of same type
    """
    if not tile.placed_pos:
        return False
        
    x, y = tile.placed_pos
    
    # Check adjacent position for this edge
    edge_directions = {
        "top_edge": (0, -1),
        "right_edge": (1, 0),
        "bottom_edge": (0, 1),
        "left_edge": (-1, 0)
    }
    
    if edge not in edge_directions:
        return False
        
    dx, dy = edge_directions[edge]
    adj_x, adj_y = x + dx, y + dy
    
    # Check bounds
    if not (0 <= adj_x < MAX_MAP_LENGTH and 0 <= adj_y < MAX_MAP_LENGTH):
        return True  # Edge of map, so this is a new origin
    
    grid = game.state.map._grid
    adjacent_tile = grid[adj_y][adj_x]
    
    # If no adjacent tile, this is a new origin structure
    if adjacent_tile is None:
        return True
        
    # If adjacent tile has matching structure type on connecting edge, this extends existing structure
    opposite_edges = {
        "top_edge": "bottom_edge",
        "right_edge": "left_edge",
        "bottom_edge": "top_edge", 
        "left_edge": "right_edge"
    }
    
    opposite_edge = opposite_edges[edge]
    adjacent_structure_type = adjacent_tile.internal_edges.get(opposite_edge)
    
    # If adjacent has same structure type, this extends existing structure
    if adjacent_structure_type == structure_type:
        return False
        
    # Otherwise, this is a new origin structure
    return True


def create_origin_structure(bot_state: BotState, tile: Tile, edge: str, structure_type: StructureType):
    """Create and track a new origin structure"""
    if not tile.placed_pos:
        return
        
    structure_id = str(uuid.uuid4())
    x, y = tile.placed_pos
    
    # Determine initial value based on structure type
    initial_value = 1  # Road default
    if structure_type == StructureType.CITY:
        # Check for emblem (banner) - simplified check
        has_emblem = False
        if hasattr(tile, 'modifiers'):
            has_emblem = any(mod.name == "PENNANT" for mod in tile.modifiers)
        initial_value = 4 if has_emblem else 2
    
    # Create origin structure
    origin_struct = OriginStructure(
        structure_id=structure_id,
        structure_type=structure_type,
        origin_pos=(x, y),
        origin_edge=edge,
        initial_value=initial_value
    )
    
    # Calculate expansion edges (where this structure could extend)
    calculate_expansion_edges(origin_struct, tile, edge)
    
    bot_state.origin_structures[structure_id] = origin_struct
    bot_state.log_action(f"Created origin structure: {origin_struct}")


def calculate_expansion_edges(origin_struct: OriginStructure, tile: Tile, origin_edge: str):
    """Calculate potential expansion edges for this structure"""
    if not tile.placed_pos:
        return
        
    x, y = tile.placed_pos
    structure_type = origin_struct.structure_type
    
    # For Module 1, we'll add basic expansion tracking
    # Check other edges of the same tile that have the same structure type
    edges = ["top_edge", "right_edge", "bottom_edge", "left_edge"]
    
    edge_directions = {
        "top_edge": (0, -1),
        "right_edge": (1, 0), 
        "bottom_edge": (0, 1),
        "left_edge": (-1, 0)
    }
    
    for edge in edges:
        if edge == origin_edge:
            continue
            
        # Check if this edge has the same structure type
        if tile.internal_edges.get(edge) == structure_type:
            # This edge could be an expansion point
            if edge in edge_directions:
                dx, dy = edge_directions[edge]
                exp_x, exp_y = x + dx, y + dy
                origin_struct.expansion_edges.append((exp_x, exp_y, edge))


def analyze_structure_changes(game: Game, bot_state: BotState, placed_tile: Tile):
    """
    Module 2: Structure Completion & Expansion Detection
    Analyze the board after tile placement to detect completed/extended structures
    """
    if not placed_tile.placed_pos:
        return
    
    x, y = placed_tile.placed_pos
    bot_state.log_action(f"[MODULE 2] Analyzing structure changes for tile at ({x}, {y})")
    
    # Check all existing origin structures for completion or extension
    structures_to_remove = []
    
    for structure_id, origin_struct in bot_state.origin_structures.items():
        if structure_id in bot_state.completed_structures:
            continue  # Already marked as completed
        
        # Check if this structure is now completed
        if is_structure_completed(game, bot_state, origin_struct, placed_tile):
            bot_state.completed_structures.add(structure_id)
            bot_state.log_action(f"[MODULE 2] Structure {structure_id[:8]} COMPLETED (value: {origin_struct.value})")
            structures_to_remove.append(structure_id)
            
        # Check if this structure was extended (increased in value)
        elif is_structure_extended(game, bot_state, origin_struct, placed_tile):
            # Increase structure value based on extension
            extension_value = calculate_extension_value(origin_struct.structure_type, placed_tile)
            origin_struct.value += extension_value
            
            # Track extensions
            if structure_id not in bot_state.extended_structures:
                bot_state.extended_structures[structure_id] = 0
            bot_state.extended_structures[structure_id] += 1
            
            bot_state.log_action(f"[MODULE 2] Structure {structure_id[:8]} EXTENDED (new value: {origin_struct.value}, extensions: {bot_state.extended_structures[structure_id]})")
            
            # Update expansion edges for the extended structure
            update_expansion_edges(game, bot_state, origin_struct, placed_tile)
    
    # Remove completed structures from active tracking
    for structure_id in structures_to_remove:
        if structure_id in bot_state.origin_structures:
            del bot_state.origin_structures[structure_id]


def is_structure_completed(game: Game, bot_state: BotState, origin_struct: OriginStructure, placed_tile: Tile) -> bool:
    """Check if a structure is now completed after this tile placement"""
    try:
        # Use the game's completion check mechanism
        origin_x, origin_y = origin_struct.origin_pos
        grid = game.state.map._grid
        
        if not (0 <= origin_x < len(grid[0]) and 0 <= origin_y < len(grid)):
            return False
            
        origin_tile = grid[origin_y][origin_x]
        if origin_tile is None:
            return False
        
        # Check if the structure at the origin is now completed
        completed = game.state._check_completed_component(origin_tile, origin_struct.origin_edge)
        return completed
        
    except Exception as e:
        bot_state.log_action(f"[MODULE 2] Error checking completion: {e}")
        return False


def is_structure_extended(game: Game, bot_state: BotState, origin_struct: OriginStructure, placed_tile: Tile) -> bool:
    """Check if a structure was extended by this tile placement"""
    if not placed_tile.placed_pos:
        return False
        
    placed_x, placed_y = placed_tile.placed_pos
    
    # Check if the placed tile is adjacent to any expansion edges of this structure
    for exp_x, exp_y, exp_edge in origin_struct.expansion_edges:
        if (exp_x, exp_y) == (placed_x, placed_y):
            # Check if the placed tile has a matching structure type on the connecting edge
            opposite_edges = {
                "top_edge": "bottom_edge",
                "right_edge": "left_edge", 
                "bottom_edge": "top_edge",
                "left_edge": "right_edge"
            }
            
            if exp_edge in opposite_edges:
                connecting_edge = opposite_edges[exp_edge]
                placed_structure = placed_tile.internal_edges.get(connecting_edge)
                
                if placed_structure == origin_struct.structure_type:
                    return True
    
    return False


def calculate_extension_value(structure_type: StructureType, placed_tile: Tile) -> int:
    """Calculate the value added by extending a structure"""
    if structure_type == StructureType.ROAD:
        return 1  # Each road segment adds 1 point
    elif structure_type == StructureType.CITY:
        # Check for emblem/banner on the new tile segment
        has_emblem = False
        if hasattr(placed_tile, 'modifiers'):
            has_emblem = any(mod.name == "PENNANT" for mod in placed_tile.modifiers)
        return 4 if has_emblem else 2  # City segment adds 2 (4 with emblem)
    
    return 0


def update_expansion_edges(game: Game, bot_state: BotState, origin_struct: OriginStructure, placed_tile: Tile):
    """Update expansion edges after a structure is extended"""
    if not placed_tile.placed_pos:
        return
        
    x, y = placed_tile.placed_pos
    structure_type = origin_struct.structure_type
    
    # Remove the expansion edge that was just filled
    origin_struct.expansion_edges = [
        (exp_x, exp_y, exp_edge) for exp_x, exp_y, exp_edge in origin_struct.expansion_edges
        if (exp_x, exp_y) != (x, y)
    ]
    
    # Add new expansion edges from this tile
    edges = ["top_edge", "right_edge", "bottom_edge", "left_edge"]
    edge_directions = {
        "top_edge": (0, -1),
        "right_edge": (1, 0),
        "bottom_edge": (0, 1), 
        "left_edge": (-1, 0)
    }
    
    for edge in edges:
        if placed_tile.internal_edges.get(edge) == structure_type:
            if edge in edge_directions:
                dx, dy = edge_directions[edge]
                new_exp_x, new_exp_y = x + dx, y + dy
                
                # Check if this position is empty (potential expansion)
                grid = game.state.map._grid
                if (0 <= new_exp_x < len(grid[0]) and 0 <= new_exp_y < len(grid) and 
                    grid[new_exp_y][new_exp_x] is None):
                    origin_struct.expansion_edges.append((new_exp_x, new_exp_y, edge))



def has_river_edge(tile) -> bool:
    """Check if tile has any river edges"""
    try:
        for edge in ["left_edge", "right_edge", "top_edge", "bottom_edge"]:
            if (hasattr(tile, 'internal_edges') and 
                edge in tile.internal_edges and 
                tile.internal_edges[edge] == StructureType.RIVER):
                return True
        return False
    except:
        return False


def is_river_phase(game: Game) -> bool:
    """Check if we're still in the river phase"""
    try:
        # River phase is complete when we find the river end tile (RE)
        for tile in game.state.map.placed_tiles:
            if hasattr(tile, 'tile_type') and tile.tile_type == 'RE':
                return False  # River phase completed
        return True  # Still in river phase
    except:
        return False  # Assume main game if error


def calculate_simple_density(game: Game, x: int, y: int) -> int:
    """Calculate simple density - count adjacent (not diagonal) tiles only"""
    grid = game.state.map._grid
    density = 0
    
    # Check only adjacent positions (not diagonal to reduce complexity)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for dx, dy in directions:
        check_x, check_y = x + dx, y + dy
        
        # Check bounds and if tile exists
        if (0 <= check_x < MAX_MAP_LENGTH and 0 <= check_y < MAX_MAP_LENGTH):
            if grid[check_y][check_x] is not None:
                density += 1
    
    return density


def find_dense_monastery_positions(game: Game, tile: Tile) -> list:
    """Find positions with high tile density - simplified approach"""
    grid = game.state.map._grid
    positions_with_density = []
    
    # Only check positions adjacent to recently placed tiles (last 5 tiles)
    recent_tiles = game.state.map.placed_tiles[-5:] if game.state.map.placed_tiles else []
    
    if not recent_tiles:
        return positions_with_density
    
    checked_positions = set()
    
    for placed_tile in recent_tiles:
        if not placed_tile.placed_pos:
            continue
            
        base_x, base_y = placed_tile.placed_pos
        
        # Only check immediately adjacent positions (not diagonal)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dx, dy in directions:
            target_x, target_y = base_x + dx, base_y + dy
            
            # Skip if already checked or out of bounds
            if ((target_x, target_y) in checked_positions or 
                not (0 <= target_x < MAX_MAP_LENGTH and 0 <= target_y < MAX_MAP_LENGTH)):
                continue
                
            checked_positions.add((target_x, target_y))
            
            # Skip if position is already occupied
            if grid[target_y][target_x] is not None:
                continue
            
            # CRITICAL: Validate placement first
            if game.can_place_tile_at(tile, target_x, target_y):
                # Check river validation if needed
                if has_river_edge(tile):
                    try:
                        river_validation = game.state.map.river_validation(tile, target_x, target_y)
                        if river_validation != "pass":
                            continue
                    except:
                        continue
                
                # Calculate simple density
                density = calculate_simple_density(game, target_x, target_y)
                positions_with_density.append((target_x, target_y, density))
    
    # Sort by density (highest first) 
    positions_with_density.sort(key=lambda x: x[2], reverse=True)
    return positions_with_density


def try_monastery_placement(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile | None:
    """
    Module 5: AGGRESSIVE monastery placement for sub-20 round games
    Only applies AFTER river phase - during river phase, use existing river strategy
    """
    # IMPORTANT: Skip density strategy during river phase - use existing river placement
    if is_river_phase(game):
        bot_state.log_action("[MODULE 5] River phase active - using existing river strategy")
        return None
    
    # Find monastery tiles in hand
    for tile_index, tile in enumerate(game.state.my_tiles):
        if has_monastery(tile):
            bot_state.log_action(f"[MODULE 5] Found monastery tile: {tile.tile_type}")
            
            # BALANCED RUSH: In rounds 8-12, place monasteries more aggressively for mid-game points
            if 8 <= bot_state.round_count <= 12:
                bot_state.log_action("[BALANCED RUSH] Strategic monastery placement for steady points!")
                
                # Find positions with decent density (slightly relaxed criteria)
                dense_positions = find_dense_monastery_positions(game, tile)
                
                if dense_positions:
                    # Try the top 2 positions for good balance of speed and density
                    for target_x, target_y, density in dense_positions[:2]:
                        bot_state.log_action(f"[BALANCED RUSH] Trying density {density} position ({target_x}, {target_y})")
                        
                        # Double-check placement validation
                        if game.can_place_tile_at(tile, target_x, target_y):
                            # Final river validation
                            if has_river_edge(tile):
                                try:
                                    river_validation = game.state.map.river_validation(tile, target_x, target_y)
                                    if river_validation != "pass":
                                        continue
                                except:
                                    continue
                            
                            bot_state.last_tile = tile
                            bot_state.last_tile.placed_pos = (target_x, target_y)
                            bot_state.log_action(f"[BALANCED RUSH] Strategic monastery placement at ({target_x}, {target_y}) density {density}")
                            return game.move_place_tile(query, tile._to_model(), tile_index)
            
            # Normal density-based placement for late game
            dense_positions = find_dense_monastery_positions(game, tile)
            
            if dense_positions:
                # Try the highest density positions first (limit to top 2 to reduce complexity)
                for target_x, target_y, density in dense_positions[:2]:
                    bot_state.log_action(f"[MODULE 5] Trying density {density} position ({target_x}, {target_y})")
                    
                    # Double-check placement validation
                    if game.can_place_tile_at(tile, target_x, target_y):
                        # Final river validation
                        if has_river_edge(tile):
                            try:
                                river_validation = game.state.map.river_validation(tile, target_x, target_y)
                                if river_validation != "pass":
                                    continue
                            except:
                                continue
                        
                        bot_state.last_tile = tile
                        bot_state.last_tile.placed_pos = (target_x, target_y)
                        bot_state.log_action(f"[MODULE 5] DENSITY PLACEMENT: monastery at ({target_x}, {target_y}) with density {density}")
                        return game.move_place_tile(query, tile._to_model(), tile_index)
            
            # Fallback: simple adjacent placement if no dense positions found
            bot_state.log_action("[MODULE 5] No dense positions found, using simple placement")
            if game.state.map.placed_tiles:
                latest_tile = game.state.map.placed_tiles[-1]
                if latest_tile.placed_pos:
                    x, y = latest_tile.placed_pos
                    
                    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                    for dx, dy in directions:
                        target_x, target_y = x + dx, y + dy
                        
                        if game.can_place_tile_at(tile, target_x, target_y):
                            if has_river_edge(tile):
                                try:
                                    river_validation = game.state.map.river_validation(tile, target_x, target_y)
                                    if river_validation != "pass":
                                        continue
                                except:
                                    continue
                            
                            bot_state.last_tile = tile
                            bot_state.last_tile.placed_pos = (target_x, target_y)
                            bot_state.log_action(f"[MODULE 5] FALLBACK: monastery at ({target_x}, {target_y})")
                            return game.move_place_tile(query, tile._to_model(), tile_index)
    
    return None


def analyze_tile_completely(tile: Tile) -> dict:
    """Comprehensive tile analysis - detect ALL edges and structures"""
    analysis = {
        'tile_type': getattr(tile, 'tile_type', 'UNKNOWN'),
        'has_monastery': False,
        'edges': {},
        'modifiers': [],
        'claimable_edges': 0,
        'structure_types': set()
    }
    
    try:
        # Analyze modifiers
        if hasattr(tile, 'modifiers'):
            analysis['modifiers'] = list(tile.modifiers) if hasattr(tile.modifiers, '__iter__') else []
            analysis['has_monastery'] = TileModifier.MONASTARY in tile.modifiers
        
        # Analyze all edges
        edges = ["top_edge", "right_edge", "bottom_edge", "left_edge"]
        for edge in edges:
            if hasattr(tile, 'internal_edges') and edge in tile.internal_edges:
                structure_type = tile.internal_edges[edge]
                analysis['edges'][edge] = structure_type
                analysis['structure_types'].add(structure_type)
                
                # Count claimable edges
                if structure_type in [StructureType.CITY, StructureType.ROAD]:
                    analysis['claimable_edges'] += 1
        
        # Additional monastery detection for tile types A and B
        if tile.tile_type in ['A', 'B']:
            analysis['has_monastery'] = True
            
        print(f"[TILE_ANALYSIS] {analysis['tile_type']}: monastery={analysis['has_monastery']}, edges={analysis['claimable_edges']}, structures={list(analysis['structure_types'])}", flush=True)
        
    except Exception as e:
        print(f"[TILE_ANALYSIS] Error analyzing tile: {e}", flush=True)
    
    return analysis


def has_monastery(tile: Tile) -> bool:
    """Enhanced monastery detection with comprehensive analysis"""
    analysis = analyze_tile_completely(tile)
    return analysis['has_monastery']


# Removed problematic has_unclaimed_cities function that caused timeouts





def handle_place_meeple(game: Game, bot_state: BotState, query: QueryPlaceMeeple) -> MovePlaceMeeple | MovePlaceMeeplePass:
    """
    MODULE 6: Strategic meeple placement with priority system
    Priority: Monastery > City > Road
    Only place meeples on unclaimed structures
    """
    if not bot_state.last_tile:
        bot_state.log_action("No last tile, passing meeple placement")
        return game.move_place_meeple_pass(query)
    
    if bot_state.meeples_placed >= 7:
        bot_state.log_action("All meeples used, passing")
        return game.move_place_meeple_pass(query)
    
    recent_tile = bot_state.last_tile
    bot_state.log_action("[MODULE 6] Strategic meeple placement analysis")
    
    # ENHANCED PRIORITY 1: MONASTERY - DELAYED GRATIFICATION STRATEGY
    analysis = analyze_tile_completely(recent_tile)
    if analysis['has_monastery'] and bot_state.meeples_placed <= 5:  # More aggressive monastery targeting
        bot_state.log_action(f"[ENHANCED] Found monastery on {analysis['tile_type']}! Long-term investment strategy.")
        try:
            bot_state.meeples_placed += 1
            bot_state.log_action(f"[ENHANCED] PLACING meeple on MONASTERY for delayed gratification (6-9 points)")
            return game.move_place_meeple(query, recent_tile._to_model(), MONASTARY_IDENTIFIER)
        except Exception as e:
            bot_state.log_action(f"[ENHANCED] Failed to place monastery meeple: {e}")
            bot_state.meeples_placed -= 1  # Revert counter
    
    # PRIORITY 2: CITIES (2-4 points per segment, higher when completed)
    # PRIORITY 3: ROADS (1 point per segment)
    edges = ["top_edge", "right_edge", "bottom_edge", "left_edge"]
    city_options = []
    road_options = []
    
    for edge in edges:
        structure_type = recent_tile.internal_edges.get(edge)
        
        if structure_type == StructureType.CITY:
            # Only place on unclaimed and incomplete structures
            if (not game.state._get_claims(recent_tile, edge) and
                not game.state._check_completed_component(recent_tile, edge)):
                
                # Calculate city value potential
                has_emblem = False
                if hasattr(recent_tile, 'modifiers'):
                    has_emblem = any(mod.name == "PENNANT" for mod in recent_tile.modifiers)
                
                city_value = 4 if has_emblem else 2
                city_options.append((edge, city_value, has_emblem))
        
        elif structure_type == StructureType.ROAD:
            # Only place on unclaimed and incomplete structures
            if (not game.state._get_claims(recent_tile, edge) and
                not game.state._check_completed_component(recent_tile, edge)):
                
                road_options.append((edge, 1))
    
    # Place on highest value city first
    if city_options:
        # Sort by value (emblem cities prioritized)
        city_options.sort(key=lambda x: x[1], reverse=True)
        edge, city_value, has_emblem = city_options[0]
        
        bot_state.meeples_placed += 1
        emblem_msg = " (EMBLEM)" if has_emblem else ""
        bot_state.log_action(f"[MODULE 6] Placed meeple on CITY{emblem_msg} (value: {city_value}) at {edge}")
        return game.move_place_meeple(query, recent_tile._to_model(), edge)
    
    # Place on roads only if no other options - IMPROVED SELECTIVITY
    if road_options and bot_state.meeples_placed <= 4:  # Only place on roads if we have meeples to spare
        edge, road_value = road_options[0]
        bot_state.meeples_placed += 1
        bot_state.log_action(f"[MODULE 6] Placed meeple on ROAD (value: {road_value}) at {edge}")
        return game.move_place_meeple(query, recent_tile._to_model(), edge)
    
    bot_state.log_action("[MODULE 6] No unclaimed structures available, passing")
    return game.move_place_meeple_pass(query)


if __name__ == "__main__":
    main()