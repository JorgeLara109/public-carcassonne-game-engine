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

def find_expansion_opportunities(
    game: Game, 
    bot_state: DomainExpansionBotState, 
    tile_in_hand: Tile,
    structure_type: StructureType
) -> tuple[int, int] | None:
    """Find positions to expand specific structure types"""
    # Filter meeples by requested structure type
    relevant_meeples = [
        m for m in bot_state.placed_meeples 
        if m.structure_type == structure_type
    ]
    
    # Try most recent meeples first
    for meeple_info in sorted(relevant_meeples, key=lambda m: -m.move_number):
        if not meeple_info.position:
            continue
            
        # Try adjacent positions (same as before)
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

def is_river_phase(game: Game) -> bool:
    for tile in game.state.my_tiles:
        for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
            if tile.internal_edges[edge] == StructureType.RIVER:
                return True
    return False

def handle_place_tile(
    game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceTile
) -> MovePlaceTile:

    if is_river_phase(game):
        return handle_river_phase(game, bot_state, query)

    grid = game.state.map._grid
    
    structure_priority = [
        StructureType.MONASTARY,
        StructureType.CITY,
        StructureType.ROAD,
        StructureType.ROAD_START
    ]

    for priority in structure_priority:
        for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
            expansion_pos = find_expansion_opportunities(
                game, bot_state, tile_in_hand, priority
            )


