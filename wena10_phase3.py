"""
Enhanced Wena10 Bot - Phase 3: Advanced Optimization

World-Class Strategic Intelligence Features:
- Portfolio Theory Application: Optimize meeple investment risk/reward balance
- Information Theory: Minimize strategy leakage, maximize opponent intelligence
- Network Analysis: Advanced structure connection optimization
- MCTS Integration: Monte Carlo Tree Search for complex decisions

Based on competitive game AI research and financial portfolio theory:
- Diversified meeple investment strategies
- Information-theoretic opponent modeling
- Graph-theoretic board analysis
- Probabilistic decision trees with simulation

This represents the pinnacle of Carcassonne AI strategy.
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
    EARLY = "early"
    MID = "mid"
    LATE = "late"


class StructureNode:
    """Node in the structure network graph"""
    def __init__(self, position: Tuple[int, int], structure_type: StructureType):
        self.position = position
        self.structure_type = structure_type
        self.connections: List['StructureNode'] = []
        self.importance_score = 0.0
        self.centrality_score = 0.0
        self.completion_probability = 0.0
        self.meeple_owner = None  # None, 'us', or 'opponent'


class StructureNetwork:
    """Graph analysis of board structure connections"""
    def __init__(self):
        self.nodes: Dict[Tuple[int, int], StructureNode] = {}
        self.adjacency_matrix: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        
    def add_node(self, position: Tuple[int, int], structure_type: StructureType):
        """Add a structure node to the network"""
        if position not in self.nodes:
            self.nodes[position] = StructureNode(position, structure_type)
            
    def add_connection(self, pos1: Tuple[int, int], pos2: Tuple[int, int]):
        """Add connection between two structure nodes"""
        if pos1 in self.nodes and pos2 in self.nodes:
            self.nodes[pos1].connections.append(self.nodes[pos2])
            self.nodes[pos2].connections.append(self.nodes[pos1])
            self.adjacency_matrix[pos1].append(pos2)
            self.adjacency_matrix[pos2].append(pos1)
    
    def calculate_centrality(self, position: Tuple[int, int]) -> float:
        """Calculate betweenness centrality for a node"""
        if position not in self.nodes:
            return 0.0
            
        # Simplified centrality calculation
        node = self.nodes[position]
        connection_count = len(node.connections)
        
        # Weight by connection quality
        quality_score = 0.0
        for connected_node in node.connections:
            if connected_node.structure_type == StructureType.CITY:
                quality_score += 3.0
            elif connected_node.structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
                quality_score += 2.0
            elif connected_node.structure_type == StructureType.MONASTARY:
                quality_score += 4.0
            else:
                quality_score += 1.0
                
        return connection_count * quality_score
    
    def find_critical_positions(self) -> List[Tuple[int, int]]:
        """Find positions that are critical for network connectivity"""
        critical_positions = []
        
        for position, node in self.nodes.items():
            centrality = self.calculate_centrality(position)
            if centrality > 10.0:  # Threshold for critical positions
                critical_positions.append(position)
                
        return critical_positions


class PortfolioOptimizer:
    """Optimize meeple placement using portfolio theory"""
    
    def __init__(self):
        self.risk_tolerance = 0.5  # 0.0 = very conservative, 1.0 = very aggressive
        self.expected_returns = {
            StructureType.MONASTARY: 8.0,  # Guaranteed 6-9 points
            StructureType.CITY: 12.0,      # High potential but risky
            StructureType.ROAD: 6.0,       # Medium risk/reward
            StructureType.GRASS: 15.0,     # High reward but very risky
        }
        self.risk_levels = {
            StructureType.MONASTARY: 0.2,  # Low risk
            StructureType.CITY: 0.6,       # Medium-high risk
            StructureType.ROAD: 0.4,       # Medium risk
            StructureType.GRASS: 0.9,      # High risk
        }
        
    def optimize_portfolio(self, available_structures: List[StructureType], 
                          meeples_remaining: int, game_phase: str) -> Dict[StructureType, float]:
        """Calculate optimal meeple allocation using portfolio theory"""
        
        # Adjust risk tolerance based on game phase
        if game_phase == GamePhase.EARLY:
            self.risk_tolerance = 0.3  # Conservative early
        elif game_phase == GamePhase.MID:
            self.risk_tolerance = 0.6  # Balanced mid
        else:  # LATE
            self.risk_tolerance = 0.8  # Aggressive late
        
        # Calculate portfolio weights
        portfolio_weights = {}
        total_weight = 0.0
        
        for structure_type in available_structures:
            expected_return = self.expected_returns.get(structure_type, 5.0)
            risk_level = self.risk_levels.get(structure_type, 0.5)
            
            # Portfolio weight = (Expected Return / Risk) * Risk Tolerance
            if risk_level > 0:
                weight = (expected_return / risk_level) * self.risk_tolerance
                
                # Adjust for meeple scarcity
                if meeples_remaining <= 2:
                    # Prefer safer investments when meeples are scarce
                    weight *= (1.0 - risk_level)
                
                portfolio_weights[structure_type] = weight
                total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for structure_type in portfolio_weights:
                portfolio_weights[structure_type] /= total_weight
        
        return portfolio_weights
    
    def calculate_sharpe_ratio(self, structure_type: StructureType, 
                              completion_probability: float) -> float:
        """Calculate Sharpe ratio for a meeple investment"""
        expected_return = self.expected_returns.get(structure_type, 5.0)
        risk = self.risk_levels.get(structure_type, 0.5)
        
        # Adjust expected return by completion probability
        adjusted_return = expected_return * completion_probability
        
        # Sharpe ratio = (Expected Return - Risk-Free Rate) / Risk
        risk_free_rate = 3.0  # Baseline guaranteed points
        
        if risk > 0:
            return (adjusted_return - risk_free_rate) / risk
        else:
            return adjusted_return


class InformationTheory:
    """Information-theoretic opponent modeling and strategy concealment"""
    
    def __init__(self):
        self.information_entropy = {}  # Track information about opponents
        self.strategy_concealment_score = 0.0
        self.opponent_prediction_accuracy = {}
        
    def calculate_information_gain(self, game: Game, tile_placement: Tuple[int, int]) -> float:
        """Calculate information gain from a tile placement"""
        # How much does this placement reveal about our strategy?
        information_leak = 0.0
        
        x, y = tile_placement
        
        # Check if placement reveals strategic patterns
        nearby_our_meeples = self.count_nearby_our_meeples(game, x, y)
        if nearby_our_meeples > 2:
            information_leak += 5.0  # Reveals expansion strategy
            
        # Check if placement reveals monastery preference
        if self.is_monastery_focused_placement(game, x, y):
            information_leak += 3.0
            
        # Check if placement reveals blocking behavior
        if self.is_obvious_blocking_move(game, x, y):
            information_leak += 4.0
            
        return -information_leak  # Negative because we want to minimize leakage
    
    def count_nearby_our_meeples(self, game: Game, x: int, y: int) -> int:
        """Count our meeples within 2 tiles of position"""
        count = 0
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < MAX_MAP_LENGTH and 0 <= check_y < MAX_MAP_LENGTH):
                    tile = game.state.map._grid[check_y][check_x]
                    if tile:
                        # Check if we have meeples on this tile
                        for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
                            if tile.internal_claims.get(edge) is not None:
                                # This is our meeple (simplified assumption)
                                count += 1
        return count
    
    def is_monastery_focused_placement(self, game: Game, x: int, y: int) -> bool:
        """Check if placement suggests monastery-focused strategy"""
        # Look for monastery tiles in our hand
        for tile in game.state.my_tiles:
            if hasattr(tile, "modifiers") and any(mod.name == "MONESTARY" for mod in tile.modifiers):
                return True
        return False
    
    def is_obvious_blocking_move(self, game: Game, x: int, y: int) -> bool:
        """Check if placement is obviously a blocking move"""
        # Check if we're placing next to opponent structures without expanding our own
        adjacent_opponent_structures = 0
        adjacent_our_structures = 0
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            adj_x, adj_y = x + dx, y + dy
            if (0 <= adj_x < MAX_MAP_LENGTH and 0 <= adj_y < MAX_MAP_LENGTH):
                tile = game.state.map._grid[adj_y][adj_x]
                if tile:
                    # Simplified check for opponent vs our structures
                    if tile.internal_claims:
                        adjacent_opponent_structures += 1
                    else:
                        adjacent_our_structures += 1
        
        return adjacent_opponent_structures > adjacent_our_structures
    
    def evaluate_opponent_predictability(self, game: Game, bot_state) -> float:
        """Evaluate how predictable opponent behavior is"""
        predictability_score = 0.0
        
        # Check for patterns in opponent moves
        for opponent in bot_state.opponent_models:
            if opponent.moves_observed > 5:
                # High monastery preference = predictable
                if opponent.monastery_preference > 0.7:
                    predictability_score += 0.3
                
                # High expansion strategy = predictable
                if opponent.expansion_strategy > 0.8:
                    predictability_score += 0.4
                
                # Consistent blocking behavior = predictable
                if opponent.aggressive_blocking > 0.6:
                    predictability_score += 0.2
        
        return predictability_score


class MCTSNode:
    """Monte Carlo Tree Search node"""
    def __init__(self, game_state=None, parent=None, action=None):
        self.game_state = game_state
        self.parent = parent
        self.action = action  # (tile_placement, meeple_placement)
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []
        
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def is_terminal(self):
        # Simplified terminal check
        return self.game_state is None or self.visits > 50
    
    def calculate_uct_value(self, c=1.41):
        """Calculate UCT (Upper Confidence Bound) value"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


class MCTSSimulator:
    """Monte Carlo Tree Search for complex decision making"""
    
    def __init__(self, simulation_count=100):
        self.simulation_count = simulation_count
        
    def search(self, game: Game, bot_state, possible_actions: List[Tuple]) -> Tuple:
        """Run MCTS to find best action"""
        if len(possible_actions) <= 1:
            return possible_actions[0] if possible_actions else None
        
        # Create root node
        root = MCTSNode()
        root.untried_actions = possible_actions.copy()
        
        # Run simulations
        for _ in range(min(self.simulation_count, 20)):  # Limit for performance
            # Selection
            node = self.select(root)
            
            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self.expand(node)
            
            # Simulation
            reward = self.simulate(game, bot_state, node.action)
            
            # Backpropagation
            self.backpropagate(node, reward)
        
        # Return best action
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.action
    
    def select(self, node: MCTSNode) -> MCTSNode:
        """Select most promising node using UCT"""
        while not node.is_terminal() and node.is_fully_expanded():
            node = max(node.children, key=lambda child: child.calculate_uct_value())
        return node
    
    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node with a new child"""
        if node.untried_actions:
            action = node.untried_actions.pop()
            child = MCTSNode(parent=node, action=action)
            node.children.append(child)
            return child
        return node
    
    def simulate(self, game: Game, bot_state, action: Tuple) -> float:
        """Simulate random gameplay from this action"""
        # Simplified simulation - evaluate action quality
        if not action:
            return 0.0
        
        tile_placement, meeple_placement = action
        
        # Evaluate tile placement
        tile_score = 0.0
        if tile_placement:
            tile, x, y = tile_placement
            tile_score += self.evaluate_tile_placement_simulation(game, tile, x, y)
        
        # Evaluate meeple placement
        meeple_score = 0.0
        if meeple_placement:
            edge, structure_type = meeple_placement
            meeple_score += self.evaluate_meeple_placement_simulation(structure_type)
        
        return tile_score + meeple_score
    
    def evaluate_tile_placement_simulation(self, game: Game, tile: Tile, x: int, y: int) -> float:
        """Evaluate tile placement for simulation"""
        score = 0.0
        
        # Monastery bonus
        if hasattr(tile, "modifiers") and any(mod.name == "MONESTARY" for mod in tile.modifiers):
            score += 15.0
        
        # Adjacency bonus
        adjacent_count = 0
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            adj_x, adj_y = x + dx, y + dy
            if (0 <= adj_x < MAX_MAP_LENGTH and 0 <= adj_y < MAX_MAP_LENGTH):
                if game.state.map._grid[adj_y][adj_x] is not None:
                    adjacent_count += 1
        
        score += adjacent_count * 2.0
        
        # Random variation for simulation
        score += random.uniform(-3.0, 3.0)
        
        return score
    
    def evaluate_meeple_placement_simulation(self, structure_type: StructureType) -> float:
        """Evaluate meeple placement for simulation"""
        if structure_type == StructureType.MONASTARY:
            return 12.0
        elif structure_type == StructureType.CITY:
            return 10.0
        elif structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
            return 8.0
        elif structure_type == StructureType.GRASS:
            return 6.0
        else:
            return 0.0
    
    def backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent


class AdvancedOpponentModel:
    """Advanced opponent modeling with information theory"""
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.monastery_preference = 0.5
        self.aggressive_blocking = 0.5
        self.structure_completion_rate = 0.5
        self.meeple_efficiency = 0.5
        self.expansion_strategy = 0.5
        self.risk_tolerance = 0.5
        
        # Advanced tracking
        self.information_entropy = 0.0
        self.predictability_score = 0.0
        self.strategy_stability = 0.0
        self.adaptation_rate = 0.0
        
        # Tracking variables
        self.moves_observed = 0
        self.monastery_placements = 0
        self.blocking_moves = 0
        self.completed_structures = 0
        self.total_structures = 0
        self.meeples_placed = 0
        self.expansion_moves = 0
        self.risky_moves = 0
        
        # Pattern recognition
        self.move_patterns = []
        self.recent_moves = []
        
    def update_from_move(self, move_type: str, context: dict):
        """Advanced move analysis with pattern recognition"""
        self.moves_observed += 1
        self.recent_moves.append(move_type)
        
        # Keep only recent moves for pattern analysis
        if len(self.recent_moves) > 10:
            self.recent_moves.pop(0)
        
        # Basic updates
        if move_type == "monastery_placement":
            self.monastery_placements += 1
        elif move_type == "blocking_move":
            self.blocking_moves += 1
        elif move_type == "structure_completion":
            self.completed_structures += 1
        elif move_type == "meeple_placement":
            self.meeples_placed += 1
        elif move_type == "expansion_move":
            self.expansion_moves += 1
        elif move_type == "risky_move":
            self.risky_moves += 1
        
        # Update preferences
        if self.moves_observed > 0:
            self.monastery_preference = self.monastery_placements / self.moves_observed
            self.aggressive_blocking = self.blocking_moves / self.moves_observed
            self.expansion_strategy = self.expansion_moves / self.moves_observed
            self.risk_tolerance = self.risky_moves / self.moves_observed
        
        # Calculate information entropy
        self.calculate_information_entropy()
        
        # Calculate predictability
        self.calculate_predictability()
    
    def calculate_information_entropy(self):
        """Calculate information entropy of opponent moves"""
        if len(self.recent_moves) < 3:
            return
        
        # Count move type frequencies
        move_counts = {}
        for move in self.recent_moves:
            move_counts[move] = move_counts.get(move, 0) + 1
        
        # Calculate entropy
        total_moves = len(self.recent_moves)
        entropy = 0.0
        
        for count in move_counts.values():
            if count > 0:
                probability = count / total_moves
                entropy -= probability * math.log2(probability)
        
        self.information_entropy = entropy
    
    def calculate_predictability(self):
        """Calculate how predictable this opponent is"""
        if self.moves_observed < 5:
            return
        
        # High consistency in preferences = high predictability
        consistency_score = 0.0
        
        if self.monastery_preference > 0.7 or self.monastery_preference < 0.3:
            consistency_score += 0.3
        
        if self.expansion_strategy > 0.7 or self.expansion_strategy < 0.3:
            consistency_score += 0.3
        
        if self.aggressive_blocking > 0.7 or self.aggressive_blocking < 0.3:
            consistency_score += 0.4
        
        self.predictability_score = consistency_score
    
    def predict_next_move(self, context: dict) -> str:
        """Predict opponent's next move type"""
        if self.monastery_preference > 0.6:
            return "monastery_focused"
        elif self.expansion_strategy > 0.6:
            return "expansion_focused"
        elif self.aggressive_blocking > 0.6:
            return "blocking_focused"
        else:
            return "balanced"


class WorldClassBotState:
    """World-class bot state with Phase 3 advanced optimization"""

    def __init__(self):
        # Phase 1 & 2 state
        self.last_tile: Tile | None = None
        self.meeples_placed: int = 0
        self.placed_meeples: list[MeepleInfo] = []
        self.move_count = 0
        self.our_score = 0
        self.game_phase = GamePhase.EARLY
        self.opponent_meeples: list[MeepleInfo] = []
        self.blocking_opportunities: list[tuple[int, int]] = []
        self.completion_urgency_factor = 1.0
        self.opponent_models = [AdvancedOpponentModel(i) for i in range(4)]
        self.strategy_mode = "balanced"
        self.performance_history = []
        self.structure_completion_probabilities = {}
        
        # Phase 3 advanced features
        self.portfolio_optimizer = PortfolioOptimizer()
        self.information_theory = InformationTheory()
        self.structure_network = StructureNetwork()
        self.mcts_simulator = MCTSSimulator()
        
        # Advanced state tracking
        self.information_concealment_score = 0.0
        self.network_centrality_positions = []
        self.meeple_portfolio_balance = {}
        self.opponent_predictability_scores = {}
        
    def update_game_phase(self, game: Game):
        """Update game phase with advanced analysis"""
        tiles_played = len(game.state.map.placed_tiles)
        our_score = game.state.points
        
        # Update our score
        self.our_score = our_score
        
        # Advanced phase detection
        if tiles_played < 15:
            self.game_phase = GamePhase.EARLY
        elif our_score < 30 or tiles_played < 35:
            self.game_phase = GamePhase.MID
        else:
            self.game_phase = GamePhase.LATE
        
        # Update portfolio optimizer risk tolerance
        self.portfolio_optimizer.risk_tolerance = self.calculate_optimal_risk_tolerance()
        
        # Adjust completion urgency
        if self.game_phase == GamePhase.EARLY:
            self.completion_urgency_factor = 0.7
        elif self.game_phase == GamePhase.MID:
            self.completion_urgency_factor = 1.0
        else:  # LATE
            self.completion_urgency_factor = 1.8
    
    def calculate_optimal_risk_tolerance(self) -> float:
        """Calculate optimal risk tolerance based on game state"""
        base_tolerance = 0.5
        
        # Adjust based on our position
        if self.our_score < 25:
            base_tolerance += 0.3  # More aggressive when behind
        elif self.our_score > 45:
            base_tolerance -= 0.2  # More conservative when ahead
        
        # Adjust based on meeple availability
        meeples_remaining = 7 - self.meeples_placed
        if meeples_remaining <= 2:
            base_tolerance -= 0.3  # Conservative when few meeples
        
        # Adjust based on game phase
        if self.game_phase == GamePhase.LATE:
            base_tolerance += 0.2  # More aggressive in endgame
        
        return max(0.1, min(0.9, base_tolerance))
    
    def update_structure_network(self, game: Game):
        """Update structure network analysis"""
        self.structure_network = StructureNetwork()
        
        # Add all placed tiles as nodes
        for tile in game.state.map.placed_tiles:
            if tile.placed_pos:
                for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
                    structure_type = tile.internal_edges.get(edge)
                    if structure_type and structure_type != StructureType.RIVER:
                        self.structure_network.add_node(tile.placed_pos, structure_type)
        
        # Add connections between adjacent structures
        for tile in game.state.map.placed_tiles:
            if tile.placed_pos:
                x, y = tile.placed_pos
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    adj_x, adj_y = x + dx, y + dy
                    if (0 <= adj_x < MAX_MAP_LENGTH and 0 <= adj_y < MAX_MAP_LENGTH):
                        adj_tile = game.state.map._grid[adj_y][adj_x]
                        if adj_tile and adj_tile.placed_pos:
                            self.structure_network.add_connection(tile.placed_pos, adj_tile.placed_pos)
        
        # Update critical positions
        self.network_centrality_positions = self.structure_network.find_critical_positions()
    
    def select_world_class_strategy(self, game: Game) -> str:
        """Select strategy using advanced analysis"""
        # Get opponent predictability
        avg_predictability = sum(model.predictability_score for model in self.opponent_models) / len(self.opponent_models)
        
        # Portfolio analysis
        meeples_remaining = 7 - self.meeples_placed
        available_structures = [StructureType.MONASTARY, StructureType.CITY, StructureType.ROAD]
        portfolio_weights = self.portfolio_optimizer.optimize_portfolio(
            available_structures, meeples_remaining, self.game_phase
        )
        
        # Information theory analysis
        information_gain = self.information_theory.evaluate_opponent_predictability(game, self)
        
        # Strategic selection based on all factors
        if self.game_phase == GamePhase.EARLY:
            if avg_predictability > 0.6:
                return "counter_exploitation"  # Exploit predictable opponents
            elif portfolio_weights.get(StructureType.MONASTARY, 0) > 0.4:
                return "monastery_portfolio"
            else:
                return "balanced_network"
        
        elif self.game_phase == GamePhase.MID:
            if information_gain > 0.5:
                return "information_warfare"  # Focus on concealment and intelligence
            elif len(self.network_centrality_positions) > 3:
                return "network_control"  # Control key network positions
            else:
                return "portfolio_optimization"
        
        else:  # LATE
            if self.our_score > 40:
                return "defensive_network"  # Protect network positions
            else:
                return "aggressive_mcts"  # Use MCTS for complex endgame


def main():
    game = Game()
    bot_state = WorldClassBotState()

    while True:
        query = game.get_next_query()

        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryPlaceTile() as q:
                    print("placing tile")
                    bot_state.move_count += 1
                    bot_state.update_game_phase(game)
                    bot_state.update_structure_network(game)
                    return handle_place_tile_advanced(game, bot_state, q)

                case QueryPlaceMeeple() as q:
                    print("meeple")
                    return handle_place_meeple_advanced(game, bot_state, q)
                case _:
                    assert False

        print("sending move")
        game.send_move(choose_move(query))


def analyze_opponent_meeples(game: Game, bot_state: WorldClassBotState):
    """Advanced opponent analysis"""
    bot_state.opponent_meeples.clear()
    
    # Go through all placed tiles and find opponent meeples
    for tile in game.state.map.placed_tiles:
        if not tile.placed_pos:
            continue
            
        # Check for meeples on this tile
        for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
            if tile.internal_claims.get(edge) is not None:
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
    """Detect if we're in river phase"""
    for tile in game.state.my_tiles:
        for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
            if tile.internal_edges[edge] == StructureType.RIVER:
                return True
    return False


def handle_place_tile_advanced(game: Game, bot_state: WorldClassBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """World-class tile placement with Phase 3 optimization"""
    
    # Update opponent analysis
    analyze_opponent_meeples(game, bot_state)
    
    # Select world-class strategy
    bot_state.strategy_mode = bot_state.select_world_class_strategy(game)
    
    # Check for river phase first
    if is_river_phase(game):
        return handle_river_phase(game, bot_state, query)
    
    # Get possible actions for MCTS
    possible_actions = []
    best_actions = []
    
    for tile_index, tile_in_hand in enumerate(game.state.my_tiles):
        positions_to_check = get_adjacent_positions(game)
        
        for x, y in positions_to_check:
            if not game.can_place_tile_at(tile_in_hand, x, y):
                continue
            
            # Create action tuple
            action = ((tile_in_hand, x, y), None)  # (tile_placement, meeple_placement)
            possible_actions.append(action)
            
            # Evaluate with advanced scoring
            score = evaluate_placement_world_class(game, bot_state, tile_in_hand, x, y)
            best_actions.append((score, tile_index, tile_in_hand, x, y))
    
    # Use MCTS for complex decisions
    if len(possible_actions) > 5 and bot_state.strategy_mode in ["aggressive_mcts", "information_warfare"]:
        best_action = bot_state.mcts_simulator.search(game, bot_state, possible_actions)
        if best_action and best_action[0]:
            tile_in_hand, x, y = best_action[0]
            # Find tile index
            for tile_index, tile in enumerate(game.state.my_tiles):
                if tile == tile_in_hand:
                    if game.can_place_tile_at(tile_in_hand, x, y):
                        bot_state.last_tile = tile_in_hand
                        bot_state.last_tile.placed_pos = (x, y)
                        return game.move_place_tile(query, tile_in_hand._to_model(), tile_index)
    
    # Fallback to best scoring action
    if best_actions:
        best_actions.sort(key=lambda x: x[0], reverse=True)
        score, tile_index, tile_in_hand, x, y = best_actions[0]
        
        # Re-validate
        if game.can_place_tile_at(tile_in_hand, x, y):
            bot_state.last_tile = tile_in_hand
            bot_state.last_tile.placed_pos = (x, y)
            return game.move_place_tile(query, tile_in_hand._to_model(), tile_index)
    
    # Final fallback
    return fallback_tile_placement(game, bot_state, query)


def evaluate_placement_world_class(game: Game, bot_state: WorldClassBotState, tile: Tile, x: int, y: int) -> float:
    """World-class placement evaluation with Phase 3 optimization"""
    score = 0.0
    
    # Base Phase 1 & 2 factors
    score += evaluate_monastery_potential(game, tile, x, y) * 2.0
    score += evaluate_structure_expansion(game, bot_state, tile, x, y) * 1.5
    score += evaluate_quick_completion(game, tile, x, y) * 1.8
    score += evaluate_blocking_opportunities(game, bot_state, tile, x, y) * 1.0
    score += evaluate_structure_completion_urgency(game, tile, x, y) * bot_state.completion_urgency_factor
    
    # Phase 3 advanced factors
    score += evaluate_portfolio_optimization(game, bot_state, tile, x, y) * 2.0
    score += evaluate_information_theory(game, bot_state, tile, x, y) * 1.5
    score += evaluate_network_centrality(game, bot_state, tile, x, y) * 1.8
    
    # Strategy-specific bonuses
    if bot_state.strategy_mode == "monastery_portfolio":
        score += evaluate_monastery_potential(game, tile, x, y) * 1.5
    elif bot_state.strategy_mode == "network_control":
        score += evaluate_network_centrality(game, bot_state, tile, x, y) * 2.0
    elif bot_state.strategy_mode == "counter_exploitation":
        score += evaluate_opponent_exploitation(game, bot_state, tile, x, y) * 1.8
    elif bot_state.strategy_mode == "information_warfare":
        score += evaluate_information_theory(game, bot_state, tile, x, y) * 2.5
    
    return score


def evaluate_portfolio_optimization(game: Game, bot_state: WorldClassBotState, tile: Tile, x: int, y: int) -> float:
    """Evaluate placement using portfolio theory"""
    score = 0.0
    
    # Get available structures from this tile
    available_structures = []
    for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
        structure_type = tile.internal_edges.get(edge)
        if structure_type and structure_type != StructureType.RIVER:
            available_structures.append(structure_type)
    
    # Get portfolio weights
    meeples_remaining = 7 - bot_state.meeples_placed
    portfolio_weights = bot_state.portfolio_optimizer.optimize_portfolio(
        available_structures, meeples_remaining, bot_state.game_phase
    )
    
    # Calculate portfolio score
    for structure_type in available_structures:
        weight = portfolio_weights.get(structure_type, 0)
        completion_prob = calculate_structure_completion_probability(
            game, bot_state, structure_type, (x, y)
        )
        
        # Sharpe ratio for this structure
        sharpe_ratio = bot_state.portfolio_optimizer.calculate_sharpe_ratio(
            structure_type, completion_prob
        )
        
        score += weight * sharpe_ratio * 10.0
    
    return score


def evaluate_information_theory(game: Game, bot_state: WorldClassBotState, tile: Tile, x: int, y: int) -> float:
    """Evaluate placement using information theory"""
    # Calculate information gain/loss
    information_gain = bot_state.information_theory.calculate_information_gain(game, (x, y))
    
    # Bonus for moves that don't reveal strategy
    concealment_bonus = 0.0
    if information_gain > -2.0:  # Low information leakage
        concealment_bonus = 5.0
    
    # Bonus for positions that gather opponent information
    intelligence_bonus = 0.0
    if bot_state.information_theory.is_obvious_blocking_move(game, x, y):
        intelligence_bonus = 3.0  # Reveals opponent priorities
    
    return information_gain + concealment_bonus + intelligence_bonus


def evaluate_network_centrality(game: Game, bot_state: WorldClassBotState, tile: Tile, x: int, y: int) -> float:
    """Evaluate placement using network analysis"""
    score = 0.0
    
    # Bonus for critical network positions
    if (x, y) in bot_state.network_centrality_positions:
        score += 20.0
    
    # Calculate potential centrality if we place here
    potential_centrality = bot_state.structure_network.calculate_centrality((x, y))
    score += potential_centrality * 0.5
    
    # Bonus for connecting multiple structures
    connection_bonus = 0.0
    adjacent_structures = 0
    
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        adj_x, adj_y = x + dx, y + dy
        if (0 <= adj_x < MAX_MAP_LENGTH and 0 <= adj_y < MAX_MAP_LENGTH):
            if (adj_x, adj_y) in bot_state.structure_network.nodes:
                adjacent_structures += 1
    
    if adjacent_structures >= 3:
        connection_bonus = 15.0
    elif adjacent_structures >= 2:
        connection_bonus = 8.0
    
    score += connection_bonus
    
    return score


def evaluate_opponent_exploitation(game: Game, bot_state: WorldClassBotState, tile: Tile, x: int, y: int) -> float:
    """Evaluate how well this exploits predictable opponents"""
    score = 0.0
    
    for opponent in bot_state.opponent_models:
        if opponent.predictability_score > 0.6:
            # Exploit predictable behavior
            if opponent.monastery_preference > 0.7:
                # Block their monastery strategy
                for opponent_meeple in bot_state.opponent_meeples:
                    if (opponent_meeple.structure_type == StructureType.MONASTARY and
                        opponent_meeple.position):
                        monastery_x, monastery_y = opponent_meeple.position
                        if abs(x - monastery_x) <= 1 and abs(y - monastery_y) <= 1:
                            score += 20.0
            
            if opponent.expansion_strategy > 0.7:
                # Block their expansion
                for opponent_meeple in bot_state.opponent_meeples:
                    if opponent_meeple.position:
                        dist = math.sqrt((x - opponent_meeple.position[0])**2 + (y - opponent_meeple.position[1])**2)
                        if dist < 2:
                            score += 15.0
    
    return score


# Include Phase 1 & 2 evaluation functions
def evaluate_monastery_potential(game: Game, tile: Tile, x: int, y: int) -> float:
    """Evaluate monastery placement potential"""
    if not (hasattr(tile, "modifiers") and any(mod.name == "MONESTARY" for mod in tile.modifiers)):
        return 0.0
    
    surrounding_tiles = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            check_x, check_y = x + dx, y + dy
            if (0 <= check_x < MAX_MAP_LENGTH and 0 <= check_y < MAX_MAP_LENGTH 
                and game.state.map._grid[check_y][check_x] is not None):
                surrounding_tiles += 1
    
    completion_factor = surrounding_tiles / 9.0
    return 25.0 + (completion_factor * 10.0)


def evaluate_structure_expansion(game: Game, bot_state: WorldClassBotState, tile: Tile, x: int, y: int) -> float:
    """Evaluate structure expansion opportunities"""
    expansion_score = 0.0
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in directions:
        adj_x, adj_y = x + dx, y + dy
        if (0 <= adj_x < MAX_MAP_LENGTH and 0 <= adj_y < MAX_MAP_LENGTH):
            adjacent_tile = game.state.map._grid[adj_y][adj_x]
            if adjacent_tile:
                for our_meeple in bot_state.placed_meeples:
                    if our_meeple.position == (adj_x, adj_y):
                        if our_meeple.structure_type == StructureType.CITY:
                            expansion_score += 15.0
                        elif our_meeple.structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
                            expansion_score += 10.0
    
    return expansion_score


def evaluate_quick_completion(game: Game, tile: Tile, x: int, y: int) -> float:
    """Evaluate potential for quick structure completion"""
    completion_score = 0.0
    
    adjacent_count = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in directions:
        adj_x, adj_y = x + dx, y + dy
        if (0 <= adj_x < MAX_MAP_LENGTH and 0 <= adj_y < MAX_MAP_LENGTH):
            if game.state.map._grid[adj_y][adj_x] is not None:
                adjacent_count += 1
    
    completion_score += adjacent_count * 3.0
    return completion_score


def evaluate_blocking_opportunities(game: Game, bot_state: WorldClassBotState, tile: Tile, x: int, y: int) -> float:
    """Evaluate how much this placement blocks opponent progress"""
    blocking_score = 0.0
    
    for opponent_meeple in bot_state.opponent_meeples:
        if opponent_meeple.structure_type == StructureType.MONASTARY and opponent_meeple.position:
            monastery_x, monastery_y = opponent_meeple.position
            
            if abs(x - monastery_x) <= 1 and abs(y - monastery_y) <= 1:
                surrounding_tiles = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        check_x, check_y = monastery_x + dx, monastery_y + dy
                        if (0 <= check_x < MAX_MAP_LENGTH and 0 <= check_y < MAX_MAP_LENGTH 
                            and game.state.map._grid[check_y][check_x] is not None):
                            surrounding_tiles += 1
                
                completion_percentage = surrounding_tiles / 9.0
                blocking_score += completion_percentage * 15.0
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in directions:
        adj_x, adj_y = x + dx, y + dy
        if (0 <= adj_x < MAX_MAP_LENGTH and 0 <= adj_y < MAX_MAP_LENGTH):
            adjacent_tile = game.state.map._grid[adj_y][adj_x]
            if adjacent_tile:
                for opponent_meeple in bot_state.opponent_meeples:
                    if opponent_meeple.position == (adj_x, adj_y):
                        blocking_score += 8.0
    
    return blocking_score


def evaluate_structure_completion_urgency(game: Game, tile: Tile, x: int, y: int) -> float:
    """Evaluate how urgent it is to complete structures at this position"""
    urgency_score = 0.0
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in directions:
        adj_x, adj_y = x + dx, y + dy
        if (0 <= adj_x < MAX_MAP_LENGTH and 0 <= adj_y < MAX_MAP_LENGTH):
            adjacent_tile = game.state.map._grid[adj_y][adj_x]
            if adjacent_tile:
                for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
                    if adjacent_tile.internal_claims.get(edge) is not None:
                        structure_type = adjacent_tile.internal_edges[edge]
                        if structure_type == StructureType.CITY:
                            urgency_score += 20.0
                        elif structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
                            urgency_score += 10.0
    
    return urgency_score


def calculate_structure_completion_probability(game: Game, bot_state: WorldClassBotState, 
                                             structure_type: StructureType, tile_pos: tuple[int, int]) -> float:
    """Calculate probability of structure completion"""
    x, y = tile_pos
    
    cache_key = (structure_type, x, y, len(game.state.map.placed_tiles))
    if cache_key in bot_state.structure_completion_probabilities:
        return bot_state.structure_completion_probabilities[cache_key]
    
    probability = 0.0
    
    if structure_type == StructureType.MONASTARY:
        surrounding_tiles = 0
        total_surrounding = 0
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < MAX_MAP_LENGTH and 0 <= check_y < MAX_MAP_LENGTH):
                    total_surrounding += 1
                    if game.state.map._grid[check_y][check_x] is not None:
                        surrounding_tiles += 1
        
        if total_surrounding > 0:
            probability = (surrounding_tiles / total_surrounding) * 0.8 + 0.2
    
    elif structure_type == StructureType.CITY:
        connected_edges = 0
        total_edges = 0
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            check_x, check_y = x + dx, y + dy
            if (0 <= check_x < MAX_MAP_LENGTH and 0 <= check_y < MAX_MAP_LENGTH):
                total_edges += 1
                adjacent_tile = game.state.map._grid[check_y][check_x]
                if adjacent_tile and StructureType.CITY in adjacent_tile.internal_edges.values():
                    connected_edges += 1
        
        probability = 0.6 + (connected_edges / max(total_edges, 1)) * 0.3
    
    elif structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
        probability = 0.7
    
    bot_state.structure_completion_probabilities[cache_key] = probability
    return probability


def get_adjacent_positions(game: Game) -> list[tuple[int, int]]:
    """Get positions adjacent to existing tiles"""
    positions = set()
    grid = game.state.map._grid
    
    for tile in game.state.map.placed_tiles:
        if tile.placed_pos:
            x, y = tile.placed_pos
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < MAX_MAP_LENGTH and 0 <= new_y < MAX_MAP_LENGTH 
                    and grid[new_y][new_x] is None):
                    positions.add((new_x, new_y))
    
    return list(positions)


def handle_river_phase(game: Game, bot_state: WorldClassBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Handle river phase with proper U-turn detection"""
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
                                if (0 <= checking_x < MAX_MAP_LENGTH and 0 <= checking_y < MAX_MAP_LENGTH 
                                    and grid[checking_y][checking_x] is not None):
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
                            if (0 <= checking_x < MAX_MAP_LENGTH and 0 <= checking_y < MAX_MAP_LENGTH 
                                and grid[checking_y][checking_x] is not None):
                                uturn_check = True
                    
                    if uturn_check:
                        tile_in_hand.rotate_clockwise(1)
                        if tile_in_hand.internal_edges[edge] != StructureType.RIVER:
                            tile_in_hand.rotate_clockwise(2)
                
                bot_state.last_tile = tile_in_hand
                bot_state.last_tile.placed_pos = (target_x, target_y)
                return game.move_place_tile(query, tile_in_hand._to_model(), tile_hand_index)
    
    return fallback_tile_placement(game, bot_state, query)


def handle_place_meeple_advanced(game: Game, bot_state: WorldClassBotState, query: QueryPlaceMeeple) -> MovePlaceMeeplePass | MovePlaceMeeple:
    """World-class meeple placement with Phase 3 optimization"""
    
    if not bot_state.last_tile or bot_state.meeples_placed >= 7:
        return game.move_place_meeple_pass(query)
    
    recent_tile = bot_state.last_tile
    our_meeples_available = 7 - bot_state.meeples_placed
    
    # 1. MONASTERY PRIORITY (guaranteed points)
    if (hasattr(recent_tile, "modifiers") 
        and any(mod.name == "MONESTARY" for mod in recent_tile.modifiers)
        and not game.state._check_completed_component(recent_tile, MONASTARY_IDENTIFIER)):
        
        # Phase 3: Portfolio analysis for monastery
        portfolio_weights = bot_state.portfolio_optimizer.optimize_portfolio(
            [StructureType.MONASTARY], our_meeples_available, bot_state.game_phase
        )
        
        monastery_weight = portfolio_weights.get(StructureType.MONASTARY, 0)
        
        if monastery_weight > 0.3 or bot_state.strategy_mode == "monastery_portfolio":
            bot_state.meeples_placed += 1
            meeple_info = MeepleInfo(recent_tile, StructureType.MONASTARY, MONASTARY_IDENTIFIER, bot_state.move_count)
            bot_state.placed_meeples.append(meeple_info)
            return game.move_place_meeple(query, recent_tile._to_model(), MONASTARY_IDENTIFIER)
    
    # 2. ADVANCED STRUCTURE EVALUATION
    structures = list(game.state.get_placeable_structures(recent_tile._to_model()).items())
    
    best_meeple_placement = None
    best_meeple_score = 0.0
    
    for edge, structure in structures:
        structure_type = recent_tile.internal_edges.get(edge)
        if structure_type is None or structure_type == StructureType.RIVER:
            continue
        
        if (not game.state._get_claims(recent_tile, edge) 
            and not game.state._check_completed_component(recent_tile, edge)):
            
            # Phase 3: World-class meeple evaluation
            score = evaluate_meeple_placement_world_class(game, bot_state, recent_tile, edge, structure_type)
            
            if score > best_meeple_score:
                best_meeple_score = score
                best_meeple_placement = (edge, structure_type)
    
    # Place best meeple
    if best_meeple_placement:
        edge, structure_type = best_meeple_placement
        bot_state.meeples_placed += 1
        meeple_info = MeepleInfo(recent_tile, structure_type, edge, bot_state.move_count)
        bot_state.placed_meeples.append(meeple_info)
        return game.move_place_meeple(query, recent_tile._to_model(), edge)
    
    return game.move_place_meeple_pass(query)


def evaluate_meeple_placement_world_class(game: Game, bot_state: WorldClassBotState, tile: Tile, edge: str, structure_type: StructureType) -> float:
    """World-class meeple placement evaluation with Phase 3 optimization"""
    score = 0.0
    
    # Base structure value
    if structure_type == StructureType.CITY:
        score += 15.0
    elif structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
        score += 10.0
    elif structure_type == StructureType.GRASS:
        score += 5.0
    
    # Phase 3: Portfolio optimization
    available_structures = [structure_type]
    meeples_remaining = 7 - bot_state.meeples_placed
    portfolio_weights = bot_state.portfolio_optimizer.optimize_portfolio(
        available_structures, meeples_remaining, bot_state.game_phase
    )
    
    portfolio_weight = portfolio_weights.get(structure_type, 0)
    score *= (1.0 + portfolio_weight)
    
    # Phase 3: Sharpe ratio analysis
    if tile.placed_pos:
        completion_prob = calculate_structure_completion_probability(
            game, bot_state, structure_type, tile.placed_pos
        )
        sharpe_ratio = bot_state.portfolio_optimizer.calculate_sharpe_ratio(
            structure_type, completion_prob
        )
        score += sharpe_ratio * 8.0
    
    # Phase 3: Network analysis
    if tile.placed_pos in bot_state.network_centrality_positions:
        score += 12.0
    
    # Phase 3: Information theory
    information_gain = bot_state.information_theory.calculate_information_gain(game, tile.placed_pos)
    score += information_gain
    
    # Strategy-specific adjustments
    if bot_state.strategy_mode == "portfolio_optimization":
        score *= 1.5
    elif bot_state.strategy_mode == "network_control":
        score += 10.0
    elif bot_state.strategy_mode == "information_warfare":
        score += information_gain * 2.0
    
    # Completion urgency factor
    score *= bot_state.completion_urgency_factor
    
    return score


def fallback_tile_placement(game: Game, bot_state: WorldClassBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Fallback tile placement strategy"""
    grid = game.state.map._grid
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for tile_index, tile in enumerate(game.state.my_tiles):
        for y in range(MAX_MAP_LENGTH):
            for x in range(MAX_MAP_LENGTH):
                if grid[y][x] is not None:
                    for dx, dy in directions:
                        x1, y1 = x + dx, y + dy
                        if (0 <= x1 < MAX_MAP_LENGTH and 0 <= y1 < MAX_MAP_LENGTH and
                            grid[y1][x1] is None and
                            game.can_place_tile_at(tile, x1, y1)):
                            bot_state.last_tile = tile
                            bot_state.last_tile.placed_pos = (x1, y1)
                            return game.move_place_tile(query, tile._to_model(), tile_index)
    
    return game.move_place_tile(query, game.state.my_tiles[0]._to_model(), 0)


if __name__ == "__main__":
    main()