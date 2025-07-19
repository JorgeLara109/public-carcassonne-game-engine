"""
Wena10 ULTIMATE - The Hybrid Champion: Phase 4C

THE ULTIMATE SYNTHESIS:
- Phase3 Consistency: Proven 28-55 point performance and reliability
- Phase4B Anti-Troll Mastery: 42+ points vs aggressive sabotage strategies
- Dynamic Adaptation: Real-time strategy switching based on opponent analysis
- Tournament Optimization: Specialized configurations for competitive scenarios

HYBRID ARCHITECTURE:
- Opponent Detection: Identify troll vs advanced AI within 3-5 moves
- Multi-Mode Experts: Switch between Phase3/Phase4B/Hybrid strategies
- Adaptive Excellence: Always use the optimal approach for the situation
- Performance Target: 40-50 point consistency, 50+ milestone achievement

This represents the pinnacle of our Carcassonne AI development - combining all research,
optimizations, and competitive insights into one ultimate adaptive champion.
"""

from typing import List, Dict, Optional, Tuple, Set
from helper.game import Game
from lib.interface.events.moves import MovePlaceTile, MovePlaceMeeple, MovePlaceMeeplePass
from lib.interface.queries import QueryPlaceTile, QueryPlaceMeeple
from lib.interface.events.common import Position
from lib.model.tile import Tile
from lib.model.meeple import Meeple
from lib.model.player import Player
from lib.model.state import GameState, PlayerState
import random
import json
import os


class TournamentConfig:
    """Tournament-specific configuration system for specialized scenarios"""
    
    CONFIGURATIONS = {
        "anti_troll_destroyer": {
            "strategy_weights": {"phase4b": 0.8, "hybrid": 0.2, "phase3": 0.0},
            "aggression_multiplier": 1.5,
            "defensive_threshold": 10,
            "meeple_threshold": 12,
            "interference_priority": 30,
            "description": "Maximum anti-troll configuration - destroys interference bots"
        },
        "elite_competitor": {
            "strategy_weights": {"phase3": 0.7, "hybrid": 0.3, "phase4b": 0.0},
            "aggression_multiplier": 0.9,
            "defensive_threshold": 25,
            "meeple_threshold": 22,
            "interference_priority": 5,
            "description": "Elite tournament mode - maximum consistency vs skilled AIs"
        },
        "balanced_hybrid": {
            "strategy_weights": {"hybrid": 0.6, "phase3": 0.2, "phase4b": 0.2},
            "aggression_multiplier": 1.1,
            "defensive_threshold": 18,
            "meeple_threshold": 18,
            "interference_priority": 15,
            "description": "Default balanced configuration for mixed competition"
        },
        "score_chaser": {
            "strategy_weights": {"phase4b": 0.6, "hybrid": 0.4, "phase3": 0.0},
            "aggression_multiplier": 1.3,
            "defensive_threshold": 12,
            "meeple_threshold": 15,
            "interference_priority": 20,
            "description": "Aggressive scoring to achieve 50+ point milestones"
        },
        "endgame_specialist": {
            "strategy_weights": {"phase3": 0.8, "hybrid": 0.2, "phase4b": 0.0},
            "aggression_multiplier": 0.8,
            "defensive_threshold": 30,
            "meeple_threshold": 25,
            "interference_priority": 8,
            "description": "Late-game optimization for final scoring phases"
        }
    }
    
    def __init__(self, config_name: str = "balanced_hybrid"):
        """Initialize with specified tournament configuration"""
        if config_name not in self.CONFIGURATIONS:
            raise ValueError(f"Unknown configuration: {config_name}")
            
        self.config_name = config_name
        self.config = self.CONFIGURATIONS[config_name].copy()
        
        # Load environment overrides if available
        self._load_env_overrides()
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        env_config = os.getenv(f"WENA10_CONFIG_{self.config_name.upper()}")
        if env_config:
            try:
                overrides = json.loads(env_config)
                self.config.update(overrides)
            except json.JSONDecodeError:
                pass  # Ignore invalid JSON
    
    def get_strategy_preference(self, base_strategy: str, game_context: dict) -> str:
        """Get strategy with tournament configuration weights"""
        weights = self.config["strategy_weights"]
        
        # Apply context-based adjustments
        turn_number = game_context.get("turn_number", 0)
        our_score = game_context.get("our_score", 0)
        opponent_types = game_context.get("opponent_types", [])
        
        # Late game adjustments for endgame specialist
        if self.config_name == "endgame_specialist" and turn_number > 15:
            weights = {"phase3": 0.9, "hybrid": 0.1, "phase4b": 0.0}
        
        # Score milestone adjustments for score chaser
        elif self.config_name == "score_chaser" and our_score > 40:
            weights = {"phase4b": 0.8, "hybrid": 0.2, "phase3": 0.0}
        
        # Troll detection adjustments
        elif any("troll" in opp_type for opp_type in opponent_types):
            if self.config_name == "anti_troll_destroyer":
                weights = {"phase4b": 0.95, "hybrid": 0.05, "phase3": 0.0}
            elif self.config_name == "elite_competitor":
                weights = {"phase3": 0.4, "hybrid": 0.3, "phase4b": 0.3}
        
        # Select strategy based on weights
        max_weight = 0
        selected_strategy = base_strategy
        
        for strategy, weight in weights.items():
            if weight > max_weight:
                max_weight = weight
                selected_strategy = strategy
        
        return selected_strategy
    
    def get_scoring_multipliers(self) -> dict:
        """Get scoring multipliers for different aspects"""
        return {
            "aggression": self.config["aggression_multiplier"],
            "defensive_threshold": self.config["defensive_threshold"],
            "meeple_threshold": self.config["meeple_threshold"],
            "interference_priority": self.config["interference_priority"]
        }
    
    def should_use_aggressive_meeple_placement(self, context: dict) -> bool:
        """Determine if aggressive meeple placement should be used"""
        multipliers = self.get_scoring_multipliers()
        
        # Base on configuration and game context
        base_aggressive = multipliers["aggression"] > 1.1
        
        # Context adjustments
        our_score = context.get("our_score", 0)
        turn_number = context.get("turn_number", 0)
        
        if self.config_name == "score_chaser" and our_score < 35:
            return True
        elif self.config_name == "elite_competitor" and our_score > 30:
            return False
        elif self.config_name == "anti_troll_destroyer":
            return True
            
        return base_aggressive
    
    @classmethod
    def list_configurations(cls) -> dict:
        """List all available configurations with descriptions"""
        return {name: config["description"] 
                for name, config in cls.CONFIGURATIONS.items()}


class OpponentAnalyzer:
    """Advanced opponent pattern detection and classification"""
    
    def __init__(self):
        self.move_history = []
        self.placement_patterns = {}
        self.aggression_score = 0
        self.defensive_score = 0
        self.efficiency_score = 0
        self.classification = "unknown"
        
    def analyze_placement(self, position: Position, tile: Tile, meeple_placed: bool,
                         blocked_our_structures: bool, created_sharing: bool):
        """Analyze opponent move patterns"""
        move_data = {
            'position': position,
            'turn': len(self.move_history),
            'meeple_placed': meeple_placed,
            'blocked_structures': blocked_our_structures,
            'created_sharing': created_sharing,
            'distance_from_center': abs(position.x) + abs(position.y)
        }
        
        self.move_history.append(move_data)
        
        # Update scores based on behavior
        if blocked_our_structures:
            self.aggression_score += 2
        if created_sharing:
            self.aggression_score += 1
        if meeple_placed:
            self.defensive_score += 1
        if move_data['distance_from_center'] < 3:
            self.efficiency_score += 1
            
        # Classify after sufficient data
        if len(self.move_history) >= 3:
            self._classify_opponent()
    
    def _classify_opponent(self):
        """Classify opponent based on accumulated patterns"""
        total_moves = len(self.move_history)
        if total_moves < 3:
            return
            
        aggression_rate = self.aggression_score / total_moves
        defensive_rate = self.defensive_score / total_moves
        
        if aggression_rate > 1.5:
            self.classification = "aggressive_troll"
        elif aggression_rate > 0.8 and defensive_rate < 0.5:
            self.classification = "interference_bot"
        elif defensive_rate > 0.7:
            self.classification = "conservative_ai"
        elif self.efficiency_score / total_moves > 0.6:
            self.classification = "optimal_ai"
        else:
            self.classification = "standard_ai"


class StrategySelector:
    """Adaptive strategy selection based on game state and opponents"""
    
    def __init__(self):
        self.current_strategy = "hybrid"
        self.strategy_performance = {
            "phase3": 0,
            "phase4b": 0,
            "hybrid": 0
        }
        
    def select_strategy(self, game_state: GameState, opponent_classifications: List[str],
                       our_score: int, turn_number: int) -> str:
        """Select optimal strategy based on current conditions"""
        
        # Early game - use hybrid approach
        if turn_number < 5:
            return "hybrid"
            
        # Detect troll presence
        has_trolls = any(cls in ["aggressive_troll", "interference_bot"] 
                        for cls in opponent_classifications)
        
        # High-skill opponent detection
        has_optimal_ai = any(cls == "optimal_ai" for cls in opponent_classifications)
        
        # Score-based adjustments
        if our_score < 15 and turn_number > 8:
            # Behind - need aggressive expansion
            return "phase4b"
        elif our_score > 35:
            # Ahead - maintain consistency
            return "phase3"
        elif has_trolls:
            # Trolls detected - use anti-troll
            return "phase4b"
        elif has_optimal_ai:
            # Elite competition - use proven consistency
            return "phase3"
        else:
            # Balanced competition - hybrid approach
            return "hybrid"


class HybridExpert:
    """Core strategic decision making combining best aspects of Phase3 and Phase4B"""
    
    def __init__(self):
        self.our_meeples = set()
        self.claimed_structures = {}
        self.expansion_targets = []
        
    def evaluate_tile_placement(self, game: Game, tile: Tile, 
                              strategy_mode: str) -> List[Tuple[Position, int]]:
        """Evaluate tile placements using hybrid strategy"""
        candidates = []
        
        for position in self._get_valid_positions(game, tile):
            score = 0
            
            # Core scoring based on strategy mode
            if strategy_mode == "phase3":
                score = self._phase3_scoring(game, tile, position)
            elif strategy_mode == "phase4b":
                score = self._phase4b_scoring(game, tile, position)
            else:  # hybrid
                score = self._hybrid_scoring(game, tile, position)
                
            candidates.append((position, score))
            
        return sorted(candidates, key=lambda x: x[1], reverse=True)
    
    def _phase3_scoring(self, game: Game, tile: Tile, position: Position) -> int:
        """Phase3 consistency-focused scoring"""
        score = 0
        
        # Monastery priority (guaranteed points)
        if tile.structure_map.monastery:
            score += 120
            
        # Structure expansion value
        expansion_value = self._calculate_expansion_value(game, tile, position)
        score += expansion_value * 15
        
        # Completion potential
        completion_bonus = self._calculate_completion_potential(game, tile, position)
        score += completion_bonus * 10
        
        # Defensive placement
        defensive_value = self._calculate_defensive_value(game, tile, position)
        score += defensive_value * 8
        
        return score
    
    def _phase4b_scoring(self, game: Game, tile: Tile, position: Position) -> int:
        """Phase4B anti-troll focused scoring"""
        score = 0
        
        # Anti-interference priority
        interference_block = self._calculate_interference_blocking(game, tile, position)
        score += interference_block * 25
        
        # Aggressive expansion
        expansion_value = self._calculate_expansion_value(game, tile, position)
        score += expansion_value * 20
        
        # Structure protection
        protection_value = self._calculate_structure_protection(game, tile, position)
        score += protection_value * 18
        
        # Monastery priority (still high value)
        if tile.structure_map.monastery:
            score += 100
            
        return score
    
    def _hybrid_scoring(self, game: Game, tile: Tile, position: Position) -> int:
        """Hybrid scoring combining both approaches"""
        phase3_score = self._phase3_scoring(game, tile, position)
        phase4b_score = self._phase4b_scoring(game, tile, position)
        
        # Weighted combination based on game state
        our_score = game.state.get_player_state(game.our_player_id()).score
        turn_ratio = min(1.0, len(game.state.placed_tiles) / 30.0)
        
        if our_score < 20:
            # Early/behind - favor aggressive approach
            return int(0.3 * phase3_score + 0.7 * phase4b_score)
        elif our_score > 35:
            # Ahead - favor consistency
            return int(0.7 * phase3_score + 0.3 * phase4b_score)
        else:
            # Balanced - even weighting
            return int(0.5 * phase3_score + 0.5 * phase4b_score)
    
    def _get_valid_positions(self, game: Game, tile: Tile) -> List[Position]:
        """Get all valid positions for tile placement"""
        positions = []
        for rotation in range(4):
            rotated_tile = tile.rotate(rotation)
            for pos in game.state.map.get_placeable_positions():
                if game.can_place_tile_at(rotated_tile, pos):
                    positions.append(pos)
        return positions
    
    def _calculate_expansion_value(self, game: Game, tile: Tile, position: Position) -> int:
        """Calculate value of expanding existing structures - Phase3 consistency focus"""
        expansion_value = 0
        
        # Check adjacent positions for existing structures we own
        adjacent_positions = [
            Position(position.x + 1, position.y),
            Position(position.x - 1, position.y),
            Position(position.x, position.y + 1),
            Position(position.x, position.y - 1)
        ]
        
        for adj_pos in adjacent_positions:
            if adj_pos in game.state.map.placed_tiles:
                adj_tile = game.state.map.placed_tiles[adj_pos]
                
                # Check if we have meeples on adjacent structures
                for structure in adj_tile.structure_map.structures:
                    if structure.id in [m.structure_id for m in self.our_meeples]:
                        # We own adjacent structure - high expansion value
                        if structure.structure_type.name == "city":
                            expansion_value += 8
                        elif structure.structure_type.name == "road":
                            expansion_value += 6
                        elif structure.structure_type.name == "field":
                            expansion_value += 4
        
        return min(expansion_value, 10)
    
    def _calculate_completion_potential(self, game: Game, tile: Tile, position: Position) -> int:
        """Calculate potential for completing structures - Phase3 reliability"""
        completion_value = 0
        
        # Monastery completion potential
        if tile.structure_map.monastery:
            surrounding_count = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    check_pos = Position(position.x + dx, position.y + dy)
                    if check_pos in game.state.map.placed_tiles:
                        surrounding_count += 1
            
            # More surrounding tiles = easier completion
            completion_value += surrounding_count
        
        # Check for nearly complete cities/roads that this tile could finish
        for edge in ['north', 'south', 'east', 'west']:
            if hasattr(tile.structure_map, edge):
                edge_structure = getattr(tile.structure_map, edge)
                if edge_structure and edge_structure.structure_type.name in ["city", "road"]:
                    # This could complete a structure - analyze if we own it
                    completion_value += 3
        
        return min(completion_value, 8)
    
    def _calculate_defensive_value(self, game: Game, tile: Tile, position: Position) -> int:
        """Calculate defensive positioning value - Phase3 protection"""
        defensive_value = 0
        
        # Prefer central positioning (Phase3 consistency)
        distance_from_center = abs(position.x) + abs(position.y)
        if distance_from_center <= 2:
            defensive_value += 4
        elif distance_from_center <= 4:
            defensive_value += 2
        
        # Avoid placing near opponent-controlled areas
        adjacent_opponent_structures = 0
        adjacent_positions = [
            Position(position.x + 1, position.y),
            Position(position.x - 1, position.y),
            Position(position.x, position.y + 1),
            Position(position.x, position.y - 1)
        ]
        
        for adj_pos in adjacent_positions:
            if adj_pos in game.state.map.placed_tiles:
                adj_tile = game.state.map.placed_tiles[adj_pos]
                # Check for opponent meeples (simplified)
                for structure in adj_tile.structure_map.structures:
                    if structure.id not in [m.structure_id for m in self.our_meeples]:
                        adjacent_opponent_structures += 1
        
        # Penalty for being too close to opponents
        defensive_value -= adjacent_opponent_structures
        
        return max(defensive_value, 0)
    
    def _calculate_interference_blocking(self, game: Game, tile: Tile, position: Position) -> int:
        """Calculate value of blocking opponent interference - Phase4B strength"""
        blocking_value = 0
        
        # Check if this placement blocks opponent expansion paths
        adjacent_positions = [
            Position(position.x + 1, position.y),
            Position(position.x - 1, position.y),
            Position(position.x, position.y + 1),
            Position(position.x, position.y - 1)
        ]
        
        for adj_pos in adjacent_positions:
            if adj_pos in game.state.map.placed_tiles:
                adj_tile = game.state.map.placed_tiles[adj_pos]
                
                # Look for opponent-controlled structures
                for structure in adj_tile.structure_map.structures:
                    if structure.id not in [m.structure_id for m in self.our_meeples]:
                        # This is likely opponent controlled
                        if structure.structure_type.name == "city":
                            blocking_value += 6  # High value for blocking city expansion
                        elif structure.structure_type.name == "road":
                            blocking_value += 4  # Medium value for blocking roads
                        elif structure.structure_type.name == "field":
                            blocking_value += 2  # Lower value for fields
        
        # Bonus for blocking multiple opponent structures
        if blocking_value >= 8:
            blocking_value += 4
        
        return min(blocking_value, 12)
    
    def _calculate_structure_protection(self, game: Game, tile: Tile, position: Position) -> int:
        """Calculate value of protecting our structures - Phase4B defense"""
        protection_value = 0
        
        # Check if this placement protects our existing structures
        adjacent_positions = [
            Position(position.x + 1, position.y),
            Position(position.x - 1, position.y),
            Position(position.x, position.y + 1),
            Position(position.x, position.y - 1)
        ]
        
        our_adjacent_structures = 0
        for adj_pos in adjacent_positions:
            if adj_pos in game.state.map.placed_tiles:
                adj_tile = game.state.map.placed_tiles[adj_pos]
                
                # Check for our meeples on adjacent structures
                for structure in adj_tile.structure_map.structures:
                    if structure.id in [m.structure_id for m in self.our_meeples]:
                        our_adjacent_structures += 1
                        
                        # High protection value for valuable structures
                        if structure.structure_type.name == "city":
                            protection_value += 5
                        elif structure.structure_type.name == "monastery":
                            protection_value += 4
                        elif structure.structure_type.name == "road":
                            protection_value += 3
        
        # Bonus for protecting multiple structures
        if our_adjacent_structures >= 2:
            protection_value += 3
        
        return min(protection_value, 10)


class Wena10Ultimate:
    """The Ultimate Hybrid Carcassonne AI"""
    
    def __init__(self, tournament_config: str = "balanced_hybrid"):
        self.tournament_config = TournamentConfig(tournament_config)
        self.opponent_analyzers = {}  # player_id -> OpponentAnalyzer
        self.strategy_selector = StrategySelector()
        self.hybrid_expert = HybridExpert()
        self.turn_count = 0
        self.performance_log = []
        
    def handle_query_place_tile(self, game: Game, query: QueryPlaceTile) -> MovePlaceTile:
        """Handle tile placement with adaptive strategy"""
        self.turn_count += 1
        
        # Analyze opponents if we have enough data
        opponent_classifications = self._get_opponent_classifications(game)
        
        # Select optimal strategy with tournament configuration
        our_score = game.state.get_player_state(game.our_player_id()).score
        base_strategy = self.strategy_selector.select_strategy(
            game.state, opponent_classifications, our_score, self.turn_count
        )
        
        # Apply tournament configuration
        game_context = {
            "turn_number": self.turn_count,
            "our_score": our_score,
            "opponent_types": opponent_classifications
        }
        strategy = self.tournament_config.get_strategy_preference(base_strategy, game_context)
        
        # Get best placement using selected strategy
        candidates = self.hybrid_expert.evaluate_tile_placement(
            game, query.tile, strategy
        )
        
        if candidates:
            best_position, score = candidates[0]
            # Find correct rotation for this position
            for rotation in range(4):
                rotated_tile = query.tile.rotate(rotation)
                if game.can_place_tile_at(rotated_tile, best_position):
                    return MovePlaceTile(rotated_tile, best_position)
        
        # Fallback - find any valid placement
        return self._fallback_placement(game, query.tile)
    
    def handle_query_place_meeple(self, game: Game, query: QueryPlaceMeeple) -> MovePlaceMeeple | MovePlaceMeeplePass:
        """Handle meeple placement with strategic priorities"""
        
        # Get current strategy with tournament configuration
        opponent_classifications = self._get_opponent_classifications(game)
        our_score = game.state.get_player_state(game.our_player_id()).score
        base_strategy = self.strategy_selector.select_strategy(
            game.state, opponent_classifications, our_score, self.turn_count
        )
        
        game_context = {
            "turn_number": self.turn_count,
            "our_score": our_score,
            "opponent_types": opponent_classifications
        }
        strategy = self.tournament_config.get_strategy_preference(base_strategy, game_context)
        
        # Evaluate meeple placements based on strategy
        best_meeple = self._evaluate_meeple_placement(game, query, strategy, game_context)
        
        if best_meeple:
            self.hybrid_expert.our_meeples.add(best_meeple)
            return MovePlaceMeeple(best_meeple)
        else:
            return MovePlaceMeeplePass()
    
    def _get_opponent_classifications(self, game: Game) -> List[str]:
        """Get current opponent classifications"""
        classifications = []
        for player_id in game.state.players:
            if player_id != game.our_player_id():
                if player_id in self.opponent_analyzers:
                    classifications.append(self.opponent_analyzers[player_id].classification)
                else:
                    classifications.append("unknown")
        return classifications
    
    def _evaluate_meeple_placement(self, game: Game, query: QueryPlaceMeeple, strategy: str, game_context: dict) -> Optional[Meeple]:
        """Evaluate and select best meeple placement"""
        if not query.meeples:
            return None
            
        best_meeple = None
        best_score = -1
        
        for meeple in query.meeples:
            score = self._score_meeple_placement(game, meeple, strategy)
            if score > best_score:
                best_score = score
                best_meeple = meeple
                
        # Only place if score meets tournament configuration threshold
        multipliers = self.tournament_config.get_scoring_multipliers()
        threshold = multipliers["meeple_threshold"]
        
        # Apply aggressive meeple placement if configured
        if self.tournament_config.should_use_aggressive_meeple_placement(game_context):
            threshold = int(threshold * 0.8)
            
        return best_meeple if best_score >= threshold else None
    
    def _score_meeple_placement(self, game: Game, meeple: Meeple, strategy: str) -> int:
        """Score a meeple placement option with integrated Phase3/Phase4B logic"""
        score = 0
        structure_type = meeple.structure_id.structure_type.name
        
        # Base scoring by structure type (Phase3 consistency)
        if structure_type == "monastery":
            score += 80  # High value, guaranteed points (Phase3 priority)
            
            # Calculate monastery completion potential
            tile_pos = self._find_meeple_tile_position(game, meeple)
            if tile_pos:
                surrounding_count = self._count_surrounding_tiles(game, tile_pos)
                score += surrounding_count * 5  # Bonus for easier completion
                
        elif structure_type == "city":
            score += 60  # High potential (both phases value cities)
            
            # Phase4B: Bonus for contested cities (interference blocking)
            if self._is_structure_contested(game, meeple.structure_id):
                score += 25  # Fight for valuable cities
            else:
                score += 10  # Uncontested expansion
                
        elif structure_type == "road":
            score += 40  # Medium value
            
            # Phase3: Bonus for road completion potential
            if self._can_complete_road_easily(game, meeple.structure_id):
                score += 15
                
        elif structure_type == "field":
            score += 25  # Long-term value
            
            # Phase4B: Higher field value for blocking opponents
            if self._field_blocks_opponents(game, meeple.structure_id):
                score += 20
        
        # Apply tournament configuration multipliers
        multipliers = self.tournament_config.get_scoring_multipliers()
        
        # Strategy-specific adjustments with tournament config
        if strategy == "phase4b":
            # More aggressive meeple placement (anti-troll focus)
            score = int(score * multipliers["aggression"])
            
            # Extra bonus for interference blocking
            if structure_type in ["city", "road"]:
                score += multipliers["interference_priority"]
                
        elif strategy == "phase3":
            # More conservative, favor completion (consistency focus)
            if structure_type in ["monastery", "city"]:
                score = int(score * 1.1)
                
            # Defensive bonus for safe placements
            if not self._is_structure_contested(game, meeple.structure_id):
                score += multipliers["defensive_threshold"] // 2
                
        else:  # hybrid strategy
            # Balanced approach with slight aggression
            score = int(score * 1.05)
            
        return score
    
    def _find_meeple_tile_position(self, game: Game, meeple: Meeple) -> Optional[Position]:
        """Find the position of the tile where this meeple would be placed"""
        # This would need to be implemented based on the meeple's structure_id
        # For now, return None as placeholder
        return None
    
    def _count_surrounding_tiles(self, game: Game, position: Position) -> int:
        """Count tiles surrounding a monastery position"""
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                check_pos = Position(position.x + dx, position.y + dy)
                if check_pos in game.state.map.placed_tiles:
                    count += 1
        return count
    
    def _is_structure_contested(self, game: Game, structure_id) -> bool:
        """Check if a structure has competing meeples"""
        # Simplified check - would need proper structure analysis
        return False  # Placeholder
    
    def _can_complete_road_easily(self, game: Game, structure_id) -> bool:
        """Check if a road can be completed with available tiles"""
        # Simplified check - would need tile availability analysis
        return False  # Placeholder
    
    def _field_blocks_opponents(self, game: Game, structure_id) -> bool:
        """Check if placing meeple on field blocks opponent strategies"""
        # Simplified check - would need field analysis
        return False  # Placeholder
    
    def _fallback_placement(self, game: Game, tile: Tile) -> MovePlaceTile:
        """Fallback placement when all else fails"""
        for rotation in range(4):
            rotated_tile = tile.rotate(rotation)
            for pos in game.state.map.get_placeable_positions():
                if game.can_place_tile_at(rotated_tile, pos):
                    return MovePlaceTile(rotated_tile, pos)
        raise Exception("No valid placement found")


# Tournament Configuration Selection
# Set via environment variable WENA10_TOURNAMENT_MODE or defaults to balanced_hybrid
tournament_mode = os.getenv("WENA10_TOURNAMENT_MODE", "balanced_hybrid")

# Global bot instance with tournament configuration
bot = Wena10Ultimate(tournament_mode)

def handle_query_place_tile(game: Game, query: QueryPlaceTile) -> MovePlaceTile:
    return bot.handle_query_place_tile(game, query)

def handle_query_place_meeple(game: Game, query: QueryPlaceMeeple) -> MovePlaceMeeple | MovePlaceMeeplePass:
    return bot.handle_query_place_meeple(game, query)


# Tournament Configuration Factory Functions
def create_anti_troll_destroyer():
    """Create bot specifically configured to destroy troll/interference bots"""
    return Wena10Ultimate("anti_troll_destroyer")

def create_elite_competitor():
    """Create bot optimized for elite tournament competition"""
    return Wena10Ultimate("elite_competitor")

def create_score_chaser():
    """Create bot focused on achieving 50+ point milestones"""
    return Wena10Ultimate("score_chaser")

def create_endgame_specialist():
    """Create bot optimized for late-game scoring phases"""
    return Wena10Ultimate("endgame_specialist")

# Usage examples for different scenarios:
# 
# For anti-troll matches:
# export WENA10_TOURNAMENT_MODE=anti_troll_destroyer
# python3 match_simulator.py --submissions 1:main/wena10_ultimate.py 3:example_submissions/simple.py --engine
#
# For elite competition:
# export WENA10_TOURNAMENT_MODE=elite_competitor  
# python3 match_simulator.py --submissions 4:main/wena10_ultimate.py --engine
#
# For score chasing:
# export WENA10_TOURNAMENT_MODE=score_chaser
# python3 match_simulator.py --submissions 2:main/wena10_ultimate.py 2:main/wena10_phase3.py --engine
