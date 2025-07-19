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

import math
import random
from collections import defaultdict
from lib.interact.structure import StructureType

class MCTSNode:
    """Node for Monte Carlo Tree Search"""
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move  # (tile_index, x, y, rotation)
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = None

class MCTS:
    """Monte Carlo Tree Search implementation"""
    def __init__(self, game, bot_state, iteration_limit=50, exploration_param=1.4):
        self.game = game
        self.bot_state = bot_state
        self.iteration_limit = iteration_limit
        self.exploration_param = exploration_param
        
    def search(self, candidate_moves):
        """Run MCTS search from current state"""
        if not candidate_moves:
            return None
            
        root = MCTSNode(self._get_game_state())
        root.untried_moves = candidate_moves
        
        for _ in range(self.iteration_limit):
            node = self._select(root)
            if node.untried_moves:
                node = self._expand(node)
            if node:
                result = self._simulate(node)
                self._backpropagate(node, result)
        
        return self._get_best_move(root)
    
    def _select(self, node):
        """Selection phase: traverse tree until leaf node"""
        while node.untried_moves is None or not node.untried_moves:
            if not node.children:
                return node
            node = self._best_child(node)
        return node
    
    def _expand(self, node):
        """Expansion phase: add a new child node"""
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        
        # Create new child node with simulated state
        new_state = self._simulate_move(node.game_state, move)
        child_node = MCTSNode(new_state, parent=node, move=move)
        node.children.append(child_node)
        return child_node
    
    def _simulate(self, node):
        """Simulation phase: play random moves to terminal state"""
        state = node.game_state.copy()
        
        # Simulate a few moves ahead (not full game)
        for _ in range(5):  # Limited simulation depth
            moves = self._generate_candidate_moves(state)
            if moves:
                move = random.choice(moves)
                state = self._simulate_move(state, move)
        
        # Use virtual score difference as reward
        our_score = state.score
        max_opponent = max(
            state.point
            for p in state.players 
            if p != self.game.state.me.player_id
        )
        return our_score - max_opponent
    
    def _backpropagate(self, node, result):
        """Backpropagation phase: update statistics along path"""
        while node:
            node.visits += 1
            node.wins += result
            node = node.parent
    
    def _best_child(self, node):
        """Select best child using UCT formula"""
        choices_weights = [
            (child.wins / child.visits) + 
            self.exploration_param * math.sqrt(math.log(node.visits)) / child.visits
            for child in node.children
        ]
        return node.children[choices_weights.index(max(choices_weights))]
    
    def _get_best_move(self, node):
        """Select move with highest visit count"""
        return max(node.children, key=lambda c: c.visits).move if node.children else None
    
    def _get_game_state(self):
        """Get simplified game state representation"""
        # In a real implementation, this would capture key game state
        return {
            'board': self.game.state.map,
            'scores': self.game.state.points,
            'meeples': self.bot_state.placed_meeples.copy()
        }
    
    def _simulate_move(self, state, move):
        """Simulate a move on the game state"""
        # In a real implementation, this would apply the move to a state copy
        state = state.copy()
        # ... apply move logic ...
        return state
    
    def _generate_candidate_moves(self, state):
        """Generate candidate moves for simulation"""
        # Simplified version of your candidate move generation
        return []

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

def find_expansion_opportunities(
    game: Game, 
    bot_state: DomainExpansionBotState, 
    tile_in_hand: Tile,
    structure_type: StructureType
) -> tuple[int, int] | None:
    # Try most recent meeples first
    relevant_meeples = [
        m for m in bot_state.placed_meeples 
        if m.structure_type == structure_type
    ]
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

def can_expand_structure(game: Game, tile: Tile, x: int, y: int, meeple_info: MeepleInfo) -> bool:
    """Check if placing a tile at (x,y) would expand the structure where we have a meeple"""
    
    # First check if any valid placement exists at this position
    # This will set the tile to a valid rotation if one exists
    if not is_valid_placement(game, tile, x, y):
        return False
    
    # After validation, the tile is now in a valid rotation
    # Check if any edge of this tile matches the structure type of our meeple
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


def is_river_phase(game: Game) -> bool:
    """
    Detect if we're in river phase by checking if any tiles in hand have river structures.
    """
    for tile in game.state.my_tiles:
        for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
            if tile.internal_edges[edge] == StructureType.RIVER:
                return True
    return False

def handle_place_tile(
    game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceTile
) -> MovePlaceTile:
    # Check for river phase first
    if is_river_phase(game):
        return handle_river_phase(game, bot_state, query)
    
    # Generate candidate moves using domain expansion
    candidate_moves = []
    for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
        for priority in [StructureType.MONASTARY, StructureType.CITY, StructureType.ROAD]:
            expansion_pos = find_expansion_opportunities(
                game, bot_state, tile_in_hand, priority
            )
            if expansion_pos:
                target_x, target_y = expansion_pos
                if is_valid_placement(game, tile_in_hand, target_x, target_y):
                    candidate_moves.append((tile_hand_index, target_x, target_y))
    
    # If no expansion opportunities, use fallback
    if not candidate_moves:
        return fallback_tile_placement(game, bot_state, query)
    
    # Run MCTS to select best candidate
    mcts = MCTS(game, bot_state)
    best_move = mcts.search(candidate_moves)
    
    # If MCTS didn't select a move, use first candidate
    if best_move is None:
        tile_hand_index, target_x, target_y = candidate_moves[0]
    else:
        tile_hand_index, target_x, target_y = best_move
    
    tile_in_hand = game.state.my_tiles[tile_hand_index]
    if is_valid_placement(game, tile_in_hand, target_x, target_y):
        bot_state.last_tile = tile_in_hand
        bot_state.last_tile.placed_pos = (target_x, target_y)
        return game.move_place_tile(
            query, tile_in_hand._to_model(), tile_hand_index
        )
    
    return fallback_tile_placement(game, bot_state, query)

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


def fallback_tile_placement(
    game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceTile
) -> MovePlaceTile:
    """
    Fallback strategy: iterate through recently placed tiles until a legal placement is found
    """
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
                
                # Use can_place_tile_at which handles rotation validation internally
                if is_valid_placement(game, tile_in_hand, target_x, target_y):
                    bot_state.last_tile = tile_in_hand
                    bot_state.last_tile.placed_pos = (target_x, target_y)
                    print(f"Fallback placement at {target_x}, {target_y} next to {reference_pos}")
                    
                    return game.move_place_tile(
                        query, tile_in_hand._to_model(), tile_hand_index
                    )
    
    # Final fallback - brute force
    print("Using brute force as final fallback")
    return brute_force_tile(game, bot_state, query)


def is_valid_placement(game: Game, tile: Tile, x: int, y: int) -> bool:
    """
    Simple validation that uses the game's own can_place_tile_at function
    This function handles rotation internally and sets the tile to valid rotation
    """
    # Just use the game's validation function directly
    # It handles rotation internally and will set the tile to a valid rotation if one exists
    return game.can_place_tile_at(tile, x, y)



def handle_place_meeple(
    game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceMeeple
) -> MovePlaceMeeplePass | MovePlaceMeeple:
    """
    Copy complex.py meeple placement exactly but with our priority: Monastery > City > Road = Grass
    """
    recent_tile = bot_state.last_tile
    if not recent_tile:
        return game.move_place_meeple_pass(query)

    # Create custom priority order: Monastery > City > Road (no grass)
    def get_edge_priority(edge: str) -> int:
        if edge == MONASTARY_IDENTIFIER:
            return 0  # Highest priority
        structure_type = bot_state.last_tile.internal_edges[edge]
        if structure_type == StructureType.CITY:
            return 1  # City priority
        elif structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
            return 2  # Road priority
        else:
            return 3  # Other structures (grass, river, etc.)

    # Create placement priorities with our custom order
    base_edges = ["top_edge", "right_edge", "bottom_edge", "left_edge"]
    edges_with_priority = [(get_edge_priority(edge), edge) for edge in base_edges]
    edges_with_priority.sort(key=lambda x: x[0])  # Sort by priority
    
    placement_priorities = [MONASTARY_IDENTIFIER] + [edge for _, edge in edges_with_priority]

    if bot_state.meeples_placed == 7:
        print("no meeple :(")
        return game.move_place_meeple_pass(query)

    for edge in placement_priorities:
        # Check if this edge has a valid structure and is unclaimed
        if edge == MONASTARY_IDENTIFIER:
            print("looking for")
            # Check if tile has monastery and it's unclaimed
            if (
                hasattr(recent_tile, "modifiers")
                and any(mod.name == "MONESTERY" for mod in recent_tile.modifiers)
            ):
                # Check if monastery is completed - cannot place meeples on completed structures
                is_monastery_completed = game.state._check_completed_component(recent_tile, MONASTARY_IDENTIFIER)
                if is_monastery_completed:
                    print("monastery already completed, cannot place meeple")
                    continue
                print("found monastary")
                print(
                    "[ Placed meeple ] M ",
                    recent_tile,
                    edge,
                    "monastery",
                    flush=True,
                )
                bot_state.meeples_placed += 1
                
                # Track this meeple placement
                meeple_info = MeepleInfo(
                    recent_tile, 
                    StructureType.MONASTARY, 
                    MONASTARY_IDENTIFIER, 
                    bot_state.move_count
                )
                bot_state.placed_meeples.append(meeple_info)
                
                return game.move_place_meeple(
                    query, recent_tile._to_model(), MONASTARY_IDENTIFIER
                )
        else:
            # Check if edge has a claimable structure (copied exactly from complex.py)
            assert bot_state.last_tile
            structures = list(
                game.state.get_placeable_structures(
                    bot_state.last_tile._to_model()
                ).items()
            )
            print("structurees: ", structures)

            if recent_tile.internal_claims.get(edge) is None:
                print("Edge:", edge)
                # Check if the structure is actually unclaimed (not connected to claimed structures)
                print(game.state._get_claims(recent_tile, edge))
                
                # Check if structure is completed - cannot place meeples on completed structures
                structure_type = bot_state.last_tile.internal_edges[edge]
                is_completed = game.state._check_completed_component(recent_tile, edge)
                
                # Use complex.py's exact validation (exclude grass and river)
                if (
                    not game.state._get_claims(recent_tile, edge)
                    and structure_type != StructureType.RIVER
                    and structure_type != StructureType.GRASS
                    and not is_completed  # Don't place meeples on completed structures
                ):
                    print(
                        "[ Placed meeple ] ",
                        recent_tile,
                        edge,
                        bot_state.last_tile.internal_edges[edge],
                        flush=True,
                    )
                    bot_state.meeples_placed += 1
                    
                    # Track this meeple placement
                    meeple_info = MeepleInfo(
                        recent_tile, 
                        structure_type, 
                        edge, 
                        bot_state.move_count
                    )
                    bot_state.placed_meeples.append(meeple_info)
                    
                    return game.move_place_meeple(query, recent_tile._to_model(), edge)

    # No valid placement found, pass
    print("[ ERROR ] ", flush=True)
    return game.move_place_meeple_pass(query)


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


if __name__ == "__main__":
    main()