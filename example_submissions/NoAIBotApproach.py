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

def get_our_recent_tiles(game: Game, lookback: int = 4) -> list[Tile]:
    """Get our recent tile placements by analyzing move history"""
    our_tiles = []
    placed_tiles = game.state.map.placed_tiles
    
    # Look at the last few tiles, accounting for 4 players
    # Every 4th move should be ours (assuming 4 players)
    total_tiles = len(placed_tiles)
    
    for i in range(min(lookback, total_tiles)):
        tile_index = total_tiles - 1 - (i * 4)  # Go back by multiples of 4
        if tile_index >= 0:
            our_tiles.append(placed_tiles[tile_index])
    
    return our_tiles