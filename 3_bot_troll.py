"""
Troll Bot - Sabotage Strategy

Strategy:
- Prioritize completing monasteries when present
- Sabotage incomplete structures with opponent meeples by outnumbering them
- Place city and road tiles near opponent structures to expand and claim them
- Use defensive tile placement to avoid expanding unoccupied opponent structures
- Always place meeples aggressively (don't hold back)
- Use river logic from complex_rotate.py for river phase

Features:
- Monastery completion priority
- City sabotage with meeple outnumbering
- Defensive tile placement strategy
- Aggressive meeple placement
- Traceback-based tile placement instead of full grid scanning
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


class OpponentStructure:
    """Track opponent structures that can be sabotaged"""
    def __init__(self, tile: Tile, edge: str, structure_type: StructureType, meeple_count: int):
        self.tile = tile
        self.edge = edge
        self.structure_type = structure_type
        self.meeple_count = meeple_count
        self.position = tile.placed_pos


class TrollBotState:
    """Bot state for troll sabotage strategy"""
    
    def __init__(self):
        self.last_tile: Tile | None = None
        self.meeples_placed: int = 0
        self.move_count = 0
        self.opponent_structures: list[OpponentStructure] = []


def main():
    game = Game()
    bot_state = TrollBotState()

    while True:
        query = game.get_next_query()

        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryPlaceTile() as q:
                    bot_state.move_count += 1
                    return handle_place_tile(game, bot_state, q)

                case QueryPlaceMeeple() as q:
                    return handle_place_meeple(game, bot_state, q)

        game.send_move(choose_move(query))


def is_river_phase(game: Game) -> bool:
    """Detect if we're in river phase by checking if any tiles in hand have river structures."""
    for tile in game.state.my_tiles:
        for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
            if tile.internal_edges[edge] == StructureType.RIVER:
                return True
    return False


def find_opponent_structures(game: Game) -> list[OpponentStructure]:
    """Find incomplete structures with opponent meeples that can be sabotaged"""
    opponent_structures = []
    grid = game.state.map._grid
    
    for row in grid:
        for tile in row:
            if tile is None:
                continue
                
            # Check each edge for opponent meeples
            for edge in tile.get_edges():
                claims = game.state._get_claims(tile, edge)
                if claims and len(claims) > 0:
                    # Check if structure is incomplete
                    is_completed = game.state._check_completed_component(tile, edge)
                    if not is_completed:
                        structure_type = tile.internal_edges[edge]
                        # Focus on cities and roads for sabotage
                        if structure_type in [StructureType.CITY, StructureType.ROAD, StructureType.ROAD_START]:
                            meeple_count = len(claims)
                            opponent_structures.append(
                                OpponentStructure(tile, edge, structure_type, meeple_count)
                            )
    
    return opponent_structures


def find_monasteries_to_complete(game: Game) -> list[tuple[int, int]]:
    """Find monastery positions where we can place tiles to complete them"""
    completion_positions = []
    grid = game.state.map._grid
    
    for row_idx, row in enumerate(grid):
        for col_idx, tile in enumerate(row):
            if tile is None:
                continue
                
            # Check if this tile has a monastery
            if (hasattr(tile, "modifiers") and 
                any(mod.name == "MONESTERY" for mod in tile.modifiers)):
                
                # Check if monastery is incomplete
                is_completed = game.state._check_completed_component(tile, MONASTARY_IDENTIFIER)
                if not is_completed:
                    # Find empty adjacent positions where we can place tiles
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            x, y = col_idx + dx, row_idx + dy
                            if (0 <= x < len(grid[0]) and 0 <= y < len(grid) and 
                                grid[y][x] is None):
                                completion_positions.append((x, y))
    
    return completion_positions


def find_sabotage_positions(game: Game, opponent_structures: list[OpponentStructure], tile_in_hand: Tile) -> tuple[int, int] | None:
    """Find positions where we can place tiles to sabotage opponent structures"""
    
    for structure in opponent_structures:
        if not structure.position:
            continue
            
        # Try to place adjacent to opponent structure
        directions = {
            (1, 0): "left_edge",
            (0, 1): "top_edge", 
            (-1, 0): "right_edge",
            (0, -1): "bottom_edge",
        }
        
        for (dx, dy), edge in directions.items():
            target_x = structure.position[0] + dx
            target_y = structure.position[1] + dy
            
            # Check bounds
            if not (0 <= target_x < MAX_MAP_LENGTH and 0 <= target_y < MAX_MAP_LENGTH):
                continue
                
            # Check if position is empty
            grid = game.state.map._grid
            if grid[target_y][target_x] is not None:
                continue
            
            # Check if this tile can expand the structure and we can place a meeple
            if can_sabotage_structure(game, tile_in_hand, target_x, target_y, structure):
                return (target_x, target_y)
    
    return None


def can_sabotage_structure(game: Game, tile: Tile, x: int, y: int, structure: OpponentStructure) -> bool:
    """Check if placing a tile at (x,y) would allow us to sabotage the opponent structure"""
    
    # First check if any valid placement exists at this position
    if not game.can_place_tile_at(tile, x, y):
        return False
    
    # Check if any edge of this tile matches the structure type
    for edge in tile.get_edges():
        if tile.internal_edges[edge] == structure.structure_type:
            # Check if this edge would connect to the opponent's structure
            if would_connect_to_opponent_structure(game, tile, x, y, edge, structure):
                return True
    
    return False


def would_connect_to_opponent_structure(game: Game, tile: Tile, x: int, y: int, edge: str, structure: OpponentStructure) -> bool:
    """Check if placing this tile's edge would connect to the opponent's structure"""
    
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
    
    # Check if the adjacent position has the opponent's structure
    grid = game.state.map._grid
    if not (0 <= adj_y < len(grid) and 0 <= adj_x < len(grid[0])):
        return False
        
    adjacent_tile = grid[adj_y][adj_x]
    if adjacent_tile is None:
        return False
        
    # Check if this connects to the opponent's structure
    return (adjacent_tile.placed_pos == structure.position and 
            adjacent_tile.tile_type == structure.tile.tile_type)


def find_defensive_positions(game: Game, tile_in_hand: Tile) -> tuple[int, int] | None:
    """Find positions that don't expand opponent structures without meeples"""
    grid = game.state.map._grid
    placed_tiles = game.state.map.placed_tiles
    
    directions = {
        (1, 0): "left_edge",
        (0, 1): "top_edge", 
        (-1, 0): "right_edge",
        (0, -1): "bottom_edge",
    }
    
    # Start with most recent tiles and work backwards (traceback approach)
    for tile_index in range(len(placed_tiles) - 1, max(-1, len(placed_tiles) - 10), -1):
        reference_tile = placed_tiles[tile_index]
        if not reference_tile.placed_pos:
            continue
            
        reference_pos = reference_tile.placed_pos
        
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
            
            # Check if placement is valid
            if game.can_place_tile_at(tile_in_hand, target_x, target_y):
                # Check if this would expand an unoccupied structure
                if not would_expand_unoccupied_structure(game, tile_in_hand, target_x, target_y):
                    return (target_x, target_y)
    
    return None


def would_expand_unoccupied_structure(game: Game, tile: Tile, x: int, y: int) -> bool:
    """Check if placing this tile would expand a structure without any meeples"""
    
    directions = {
        "left_edge": (-1, 0),
        "right_edge": (1, 0), 
        "top_edge": (0, -1),
        "bottom_edge": (0, 1),
    }
    
    grid = game.state.map._grid
    
    for edge in tile.get_edges():
        structure_type = tile.internal_edges[edge]
        
        # Skip grass and river
        if structure_type in [StructureType.GRASS, StructureType.RIVER]:
            continue
            
        # Check adjacent tile
        if edge in directions:
            dx, dy = directions[edge]
            adj_x, adj_y = x + dx, y + dy
            
            if (0 <= adj_y < len(grid) and 0 <= adj_x < len(grid[0]) and 
                grid[adj_y][adj_x] is not None):
                
                adjacent_tile = grid[adj_y][adj_x]
                
                # Check if adjacent structure has same type and no meeples
                opposite_edge = get_opposite_edge(edge)
                if (adjacent_tile.internal_edges[opposite_edge] == structure_type and
                    not game.state._get_claims(adjacent_tile, opposite_edge)):
                    return True
    
    return False


def get_opposite_edge(edge: str) -> str:
    """Get the opposite edge"""
    opposites = {
        "left_edge": "right_edge",
        "right_edge": "left_edge",
        "top_edge": "bottom_edge",
        "bottom_edge": "top_edge"
    }
    return opposites.get(edge, edge)


def handle_place_tile(game: Game, bot_state: TrollBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Handle tile placement with troll sabotage strategy"""
    
    # Check if we're in river phase - use river logic from complex_rotate.py
    if is_river_phase(game):
        return handle_river_phase(game, bot_state, query)
    
    # Strategy 1: Prioritize completing monasteries
    monastery_positions = find_monasteries_to_complete(game)
    for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
        for pos in monastery_positions:
            target_x, target_y = pos
            if game.can_place_tile_at(tile_in_hand, target_x, target_y):
                bot_state.last_tile = tile_in_hand
                bot_state.last_tile.placed_pos = (target_x, target_y)
                print(f"Completing monastery at {target_x}, {target_y}")
                return game.move_place_tile(query, tile_in_hand._to_model(), tile_hand_index)
    
    # Strategy 2: Sabotage opponent structures
    opponent_structures = find_opponent_structures(game)
    for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
        sabotage_pos = find_sabotage_positions(game, opponent_structures, tile_in_hand)
        if sabotage_pos:
            target_x, target_y = sabotage_pos
            if game.can_place_tile_at(tile_in_hand, target_x, target_y):
                bot_state.last_tile = tile_in_hand
                bot_state.last_tile.placed_pos = (target_x, target_y)
                print(f"Sabotaging opponent structure at {target_x}, {target_y}")
                return game.move_place_tile(query, tile_in_hand._to_model(), tile_hand_index)
    
    # Strategy 3: Defensive placement (avoid expanding unoccupied structures)
    for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
        defensive_pos = find_defensive_positions(game, tile_in_hand)
        if defensive_pos:
            target_x, target_y = defensive_pos
            if game.can_place_tile_at(tile_in_hand, target_x, target_y):
                bot_state.last_tile = tile_in_hand
                bot_state.last_tile.placed_pos = (target_x, target_y)
                print(f"Defensive placement at {target_x}, {target_y}")
                return game.move_place_tile(query, tile_in_hand._to_model(), tile_hand_index)
    
    # Fallback: Use traceback approach like 2_bot_domainexpansion
    return fallback_tile_placement(game, bot_state, query)


def handle_river_phase(game: Game, bot_state: TrollBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Handle river tile placement using complex_rotate.py logic with U-turn prevention"""
    grid = game.state.map._grid
    
    # Direction mappings for river placement
    directions = {
        (1, 0): "left_edge",   # if we place on the right of the target tile
        (0, 1): "top_edge",    # if we place at the bottom of the target tile
        (-1, 0): "right_edge", # if we place on the left of the target tile
        (0, -1): "bottom_edge", # if we place at the top of the target tile
    }
    
    # Get the latest tile
    latest_tile = game.state.map.placed_tiles[-1]
    latest_pos = latest_tile.placed_pos
    
    print(f"River phase - Available tiles: {game.state.my_tiles}")
    assert latest_pos
    
    # Try to place a tile adjacent to the latest tile
    for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
        # Check if this tile has river structures
        river_flag = False
        for find_edge in directions.values():
            if tile_in_hand.internal_edges[find_edge] == StructureType.RIVER:
                river_flag = True
                print(f"River tile detected: {tile_in_hand.tile_type}")
                break
        
        # Try each direction around the latest tile
        for (dx, dy), edge in directions.items():
            target_x = latest_pos[0] + dx
            target_y = latest_pos[1] + dy
            
            # Check bounds
            if not (0 <= target_x < MAX_MAP_LENGTH and 0 <= target_y < MAX_MAP_LENGTH):
                continue
            
            # Check if position is empty
            if grid[target_y][target_x] is not None:
                continue
            
            # Check if tile can be placed at this position
            if game.can_place_tile_at(tile_in_hand, target_x, target_y):
                if river_flag:
                    # River tile - perform U-turn check
                    print(f"River tile edge check: {tile_in_hand.internal_edges[edge]}")
                    if tile_in_hand.internal_edges[edge] != StructureType.RIVER:
                        continue
                    
                    # Check for U-turns (adapted from complex_rotate.py)
                    uturn_check = False
                    
                    for tile_edge in tile_in_hand.get_edges():
                        if (tile_edge == edge or 
                            tile_in_hand.internal_edges[tile_edge] != StructureType.RIVER):
                            continue
                        
                        # Check for direct U-turn
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
                            if (checking_x != target_x or checking_y != target_y):
                                if (0 <= checking_y < len(grid) and 
                                    0 <= checking_x < len(grid[0]) and
                                    grid[checking_y][checking_x] is not None):
                                    print("Direct U-turn detected")
                                    uturn_check = True
                        
                        # Check for future U-turn
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
                            if (0 <= checking_y < len(grid) and 
                                0 <= checking_x < len(grid[0]) and
                                grid[checking_y][checking_x] is not None):
                                print("Future U-turn detected")
                                uturn_check = True
                    
                    # If U-turn detected, try to rotate to avoid it
                    if uturn_check:
                        print("Attempting to rotate to avoid U-turn")
                        tile_in_hand.rotate_clockwise(1)
                        if tile_in_hand.internal_edges[edge] != StructureType.RIVER:
                            tile_in_hand.rotate_clockwise(2)
                
                # Valid placement found
                tile_in_hand.placed_pos = (target_x, target_y)
                bot_state.last_tile = tile_in_hand
                
                print(f"River tile placed: {tile_in_hand.tile_type} at ({target_x}, {target_y}) rotation {tile_in_hand.rotation}")
                
                return game.move_place_tile(query, tile_in_hand._to_model(), tile_hand_index)
    
    # If no placement found with heuristic, fall back to brute force
    print("River phase: Could not find placement with heuristic, trying brute force")
    return brute_force_tile(game, bot_state, query)


def fallback_tile_placement(game: Game, bot_state: TrollBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Fallback strategy: iterate through recently placed tiles (traceback approach)"""
    directions = {
        (1, 0): "left_edge",
        (0, 1): "top_edge", 
        (-1, 0): "right_edge",
        (0, -1): "bottom_edge",
    }
    
    grid = game.state.map._grid
    placed_tiles = game.state.map.placed_tiles
    
    # Start with most recent tiles and work backwards (traceback approach)
    for tile_index in range(len(placed_tiles) - 1, -1, -1):
        reference_tile = placed_tiles[tile_index]
        if not reference_tile.placed_pos:
            continue
            
        reference_pos = reference_tile.placed_pos
        print(f"Trying to place adjacent to tile at {reference_pos}")
        
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
                
                # Use game's validation which handles rotation internally
                if game.can_place_tile_at(tile_in_hand, target_x, target_y):
                    bot_state.last_tile = tile_in_hand
                    bot_state.last_tile.placed_pos = (target_x, target_y)
                    print(f"Fallback placement at {target_x}, {target_y} next to {reference_pos}")
                    
                    return game.move_place_tile(
                        query, tile_in_hand._to_model(), tile_hand_index
                    )
    
    # Final fallback - brute force
    print("Using brute force as final fallback")
    return brute_force_tile(game, bot_state, query)


def brute_force_tile(game: Game, bot_state: TrollBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Brute force tile placement as last resort"""
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    directions = {
        (0, 1): "top",
        (1, 0): "right",
        (0, -1): "bottom",
        (-1, 0): "left",
    }

    print("Using brute force tile placement")
    print(f"Available tiles: {game.state.my_tiles}")

    for y in range(height):
        for x in range(width):
            if grid[y][x] is not None:
                for tile_index, tile in enumerate(game.state.my_tiles):
                    for direction in directions:
                        dx, dy = direction
                        x1, y1 = x + dx, y + dy

                        if game.can_place_tile_at(tile, x1, y1):
                            bot_state.last_tile = tile
                            bot_state.last_tile.placed_pos = (x1, y1)
                            print(f"Brute force placement at ({x1}, {y1})")
                            return game.move_place_tile(query, tile._to_model(), tile_index)

    # Should never reach here in a valid game
    print("ERROR: No valid tile placement found")
    first_tile = game.state.my_tiles[0]
    return game.move_place_tile(query, first_tile._to_model(), 0)


def handle_place_meeple(game: Game, bot_state: TrollBotState, query: QueryPlaceMeeple) -> MovePlaceMeeple | MovePlaceMeeplePass:
    """Aggressive meeple placement with monastery priority and sabotage focus"""
    recent_tile = bot_state.last_tile
    if not recent_tile:
        return game.move_place_meeple_pass(query)

    # Check if we have meeples available
    if bot_state.meeples_placed >= 7:
        print("No meeples available")
        return game.move_place_meeple_pass(query)

    # Priority 1: Monasteries (highest priority)
    if (hasattr(recent_tile, "modifiers") and 
        any(mod.name == "MONESTERY" for mod in recent_tile.modifiers)):
        
        # Check if monastery is completed
        is_monastery_completed = game.state._check_completed_component(recent_tile, MONASTARY_IDENTIFIER)
        if not is_monastery_completed:
            print("Placing meeple on monastery")
            bot_state.meeples_placed += 1
            return game.move_place_meeple(query, recent_tile._to_model(), MONASTARY_IDENTIFIER)

    # Priority 2: Cities (for sabotage)
    city_edges = []
    for edge in recent_tile.get_edges():
        structure_type = recent_tile.internal_edges[edge]
        if structure_type == StructureType.CITY:
            city_edges.append(edge)
    
    # Place meeple on cities if unclaimed and incomplete
    for edge in city_edges:
        if (not game.state._get_claims(recent_tile, edge) and
            not game.state._check_completed_component(recent_tile, edge)):
            print(f"Placing meeple on city edge: {edge}")
            bot_state.meeples_placed += 1
            return game.move_place_meeple(query, recent_tile._to_model(), edge)

    # Priority 3: Roads (for sabotage)
    road_edges = []
    for edge in recent_tile.get_edges():
        structure_type = recent_tile.internal_edges[edge]
        if structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
            road_edges.append(edge)
    
    # Place meeple on roads if unclaimed and incomplete
    for edge in road_edges:
        if (not game.state._get_claims(recent_tile, edge) and
            not game.state._check_completed_component(recent_tile, edge)):
            print(f"Placing meeple on road edge: {edge}")
            bot_state.meeples_placed += 1
            return game.move_place_meeple(query, recent_tile._to_model(), edge)

    # Priority 4: Any other valid structure (aggressive placement)
    for edge in recent_tile.get_edges():
        structure_type = recent_tile.internal_edges[edge]
        
        # Skip grass and river
        if structure_type in [StructureType.GRASS, StructureType.RIVER]:
            continue
            
        if (not game.state._get_claims(recent_tile, edge) and
            not game.state._check_completed_component(recent_tile, edge)):
            print(f"Aggressive meeple placement on {edge}: {structure_type}")
            bot_state.meeples_placed += 1
            return game.move_place_meeple(query, recent_tile._to_model(), edge)

    # No valid placement found
    print("No valid meeple placement found")
    return game.move_place_meeple_pass(query)


if __name__ == "__main__":
    main()