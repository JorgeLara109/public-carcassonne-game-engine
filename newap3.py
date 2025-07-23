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
    """Track information about placed meeples with reward estimation"""
    def __init__(self, tile: Tile, structure_type: StructureType, edge: str, move_number: int):
        self.tile = tile
        self.structure_type = structure_type
        self.edge = edge
        self.move_number = move_number
        self.position = tile.placed_pos
        self.estimated_reward = self._calculate_estimated_reward()
    
    def _calculate_estimated_reward(self) -> float:
        """Calculate estimated reward potential for this meeple placement"""
        if self.structure_type == StructureType.CITY:
            return 8.0  # Cities: 2 points per tile + potential banners
        elif self.structure_type == StructureType.ROAD:
            return 4.0  # Roads: 1 point per tile
        elif self.structure_type == StructureType.MONASTARY:
            return 9.0  # Monasteries: fixed 9 points when completed
        elif self.structure_type == StructureType.GRASS:
            return 3.0  # Fields: 3 points per completed adjacent city
        else:
            return 1.0  # Default value

class DomainExpansionBotState:
    """Enhanced bot state focused on meeple reward optimization"""

    def __init__(self):
        self.last_tile: Tile | None = None
        self.meeples_placed: int = 0
        self.placed_meeples: list[MeepleInfo] = []
        self.move_count = 0
        self.opponent_meeples: dict[int, list[MeepleInfo]] = {1: [], 2: [], 3: []}
        self.completed_structures: set[str] = set()
        self.structure_ownership: dict[str, int] = {}
        self.total_meeple_reward_potential = 0.0
    
    def track_opponent_meeple(self, player_id: int, tile: Tile, structure_type: StructureType, edge: str, move_number: int):
        """Track an opponent's meeple placement and reward potential"""
        if player_id != 0 and player_id in self.opponent_meeples:
            meeple_info = MeepleInfo(tile, structure_type, edge, move_number)
            self.opponent_meeples[player_id].append(meeple_info)
            
            structure_key = f"{structure_type}_{tile.placed_pos}_{edge}"
            self.structure_ownership[structure_key] = player_id
    
    def add_our_meeple(self, meeple_info: MeepleInfo):
        """Add our meeple and update reward potential"""
        self.placed_meeples.append(meeple_info)
        self.meeples_placed += 1
        self.total_meeple_reward_potential += meeple_info.estimated_reward
        
        # Update structure ownership
        if meeple_info.tile.placed_pos:
            structure_key = f"{meeple_info.structure_type}_{meeple_info.tile.placed_pos}_{meeple_info.edge}"
            self.structure_ownership[structure_key] = 0
    
    def get_total_reward_potential(self) -> float:
        """Get total estimated reward from all our meeples"""
        return sum(meeple.estimated_reward for meeple in self.placed_meeples)
    
    def get_opponent_reward_potential(self, player_id: int) -> float:
        """Get estimated reward potential for specific opponent"""
        if player_id not in self.opponent_meeples:
            return 0.0
        return sum(meeple.estimated_reward for meeple in self.opponent_meeples[player_id])

class MeepleRewardGameState:
    """Game state representation optimized for meeple reward calculation"""
    
    def __init__(self, game: Game):
        self.map_grid = [[None for _ in range(MAX_MAP_LENGTH)] for _ in range(MAX_MAP_LENGTH)]
        self.placed_tiles = []
        self.my_tiles = []
        self.player_scores = [0.0, 0.0, 0.0, 0.0]  # Use float for precise reward calculation
        self.player_meeples = [[], [], [], []]
        self.meeple_rewards = [0.0, 0.0, 0.0, 0.0]  # Track estimated rewards per player
        self.completed_structures = set()
        self.current_player = 0
        
        # Initialize from game state
        self._extract_game_state(game)
    
    def _extract_game_state(self, game: Game):
        """Extract game state with focus on meeple reward potential"""
        # Copy placed tiles with meeple reward information
        for tile in game.state.map.placed_tiles:
            if hasattr(tile, 'placed_pos') and tile.placed_pos:
                x, y = tile.placed_pos
                tile_data = {
                    'tile_type': tile.tile_type,
                    'rotation': tile.rotation,
                    'internal_edges': tile.internal_edges.copy(),
                    'position': (x, y),
                    'meeples': {},  # {edge: player_id}
                    'meeple_rewards': {}  # {edge: reward_value}
                }
                
                # Extract meeples and calculate rewards
                for edge in tile.get_edges():
                    claims = game.state._get_claims(tile, edge)
                    if claims:
                        player_id = 0  # Simplified - in real game, determine actual player
                        tile_data['meeples'][edge] = player_id
                        
                        # Calculate reward for this meeple placement
                        structure_type = tile.internal_edges[edge]
                        reward = self._calculate_structure_reward(structure_type, tile, edge)
                        tile_data['meeple_rewards'][edge] = reward
                        self.meeple_rewards[player_id] += reward
                
                self.map_grid[y][x] = tile_data
                self.placed_tiles.append(tile_data)
        
        # Copy available tiles
        for tile in game.state.my_tiles:
            self.my_tiles.append({
                'tile_type': tile.tile_type,
                'rotation': tile.rotation,
                'internal_edges': tile.internal_edges.copy()
            })
    
    def _calculate_structure_reward(self, structure_type: StructureType, tile: Tile, edge: str) -> float:
        """Calculate reward potential for placing meeple on this structure"""
        if structure_type == StructureType.CITY:
            # Cities: 2 points per tile, +2 if has banner
            base_reward = 4.0  # Assume 2 tiles minimum
            if self._has_banner(tile):
                base_reward += 2.0
            return base_reward
        elif structure_type == StructureType.ROAD:
            # Roads: 1 point per tile
            return 3.0  # Assume 3 tiles average
        elif structure_type == StructureType.MONASTARY:
            # Monasteries: 9 points when completed
            return 9.0
        elif structure_type == StructureType.GRASS:
            # Fields: 3 points per completed adjacent city
            return 6.0  # Assume 2 adjacent cities
        else:
            return 1.0
    
    def _has_banner(self, tile: Tile) -> bool:
        """Check if tile has banner (simplified)"""
        # This is a simplified check - in real game, check tile properties
        return random.random() < 0.3  # 30% chance for banner
    
    def can_place_meeple(self, tile_position, edge):
        """Enhanced meeple placement check considering rewards"""
        x, y = tile_position
        if self.map_grid[y][x] is None:
            return False
        
        tile_data = self.map_grid[y][x]
        
        # Check if edge already has a meeple
        if edge in tile_data['meeples']:
            return False
        
        # Check if structure would give good reward
        structure_type = tile_data['internal_edges'].get(edge)
        if not structure_type:
            return False
        
        # Don't place on low-value structures
        potential_reward = self._calculate_structure_reward(structure_type, None, edge)
        return potential_reward >= 2.0  # Minimum reward threshold
    
    def simulate_meeple_placement(self, tile_position, edge, player_id):
        """Simulate meeple placement and calculate reward impact"""
        x, y = tile_position
        if self.map_grid[y][x] is None:
            return 0.0
        
        tile_data = self.map_grid[y][x]
        structure_type = tile_data['internal_edges'].get(edge)
        
        if not structure_type:
            return 0.0
        
        # Calculate reward for this placement
        reward = self._calculate_structure_reward(structure_type, None, edge)
        
        # Simulate placement
        tile_data['meeples'][edge] = player_id
        tile_data['meeple_rewards'][edge] = reward
        self.meeple_rewards[player_id] += reward
        
        return reward

class MeepleRewardNode:
    """MCTS node optimized for meeple reward-based decisions"""
    
    def __init__(self, game_state: MeepleRewardGameState, current_player: int = 0, parent_move=None, move_type="tile"):
        self.state = game_state
        self.current_player = current_player
        self.parent_move = parent_move
        self.move_type = move_type
        self._children = None
        self._is_terminal = None
        self.reward_cache = None
    
    def __hash__(self):
        # Hash based on meeple placements and rewards
        state_str = f"{self.current_player}_{len(self.state.placed_tiles)}_{sum(self.state.meeple_rewards)}"
        return hash(state_str)
    
    def __eq__(self, other):
        return isinstance(other, MeepleRewardNode) and hash(self) == hash(other)
    
    def is_terminal(self):
        """Check if terminal state based on tiles or score threshold"""
        if self._is_terminal is None:
            self._is_terminal = (
                len(self.state.my_tiles) == 0 or
                any(reward >= 50 for reward in self.state.meeple_rewards) or
                self.state.current_player > 80  # Move limit
            )
        return self._is_terminal
    
    def get_meeple_reward_scores(self):
        """Get scores based on meeple reward potential"""
        if self.reward_cache is not None:
            return self.reward_cache
        
        scores = [0.0, 0.0, 0.0, 0.0]
        
        # Base score from current meeple rewards
        for player_id in range(4):
            scores[player_id] = self.state.meeple_rewards[player_id]
        
        # Add potential future rewards based on tile placement opportunities
        for tile_data in self.state.placed_tiles:
            for edge, structure_type in tile_data['internal_edges'].items():
                if edge not in tile_data['meeples']:  # Unclaimed structure
                    potential_reward = self.state._calculate_structure_reward(structure_type, None, edge)
                    scores[0] += potential_reward * 0.3  # 30% probability of claiming
        
        # Bonus for completing structures
        completion_bonus = self._calculate_completion_bonus()
        scores[0] += completion_bonus
        
        # Add randomness for opponents
        for i in range(1, 4):
            scores[i] += random.uniform(0, 10)
        
        self.reward_cache = scores
        return scores
    
    def _calculate_completion_bonus(self) -> float:
        """Calculate bonus for potentially completing structures"""
        bonus = 0.0
        
        # Check for near-completion structures
        for tile_data in self.state.placed_tiles:
            for edge, player_id in tile_data['meeples'].items():
                if player_id == 0:  # Our meeple
                    structure_type = tile_data['internal_edges'][edge]
                    completion_likelihood = self._estimate_completion_likelihood(tile_data, edge, structure_type)
                    reward_value = tile_data.get('meeple_rewards', {}).get(edge, 0)
                    bonus += reward_value * completion_likelihood
        
        return bonus
    
    def _estimate_completion_likelihood(self, tile_data, edge, structure_type) -> float:
        """Estimate likelihood of completing a structure (0.0 to 1.0)"""
        if structure_type == StructureType.MONASTARY:
            # Count surrounding tiles
            x, y = tile_data['position']
            surrounding_count = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    if (0 <= x+dx < MAX_MAP_LENGTH and 0 <= y+dy < MAX_MAP_LENGTH and
                        self.state.map_grid[y+dy][x+dx] is not None):
                        surrounding_count += 1
            return min(1.0, surrounding_count / 8.0)
        
        elif structure_type in [StructureType.CITY, StructureType.ROAD]:
            # Simplified: assume 70% completion likelihood for connected structures
            return 0.7
        
        else:
            return 0.5  # Default likelihood
    
    def find_children(self):
        """Generate children focusing on meeple reward optimization"""
        if self._children is not None:
            return self._children
        
        children = set()
        
        if self.is_terminal():
            self._children = children
            return children
        
        if self.move_type == "tile":
            children = self._generate_tile_placement_children()
        elif self.move_type == "meeple":
            children = self._generate_meeple_reward_children()
        
        self._children = children
        return children
    
    def _generate_tile_placement_children(self):
        """Generate tile placement children optimized for meeple opportunities"""
        children = set()
        max_children = 16  # More exploration for better rewards
        
        # Try multiple tiles and rotations
        for tile_idx in range(min(4, len(self.state.my_tiles))):
            for rotation in [0, 1, 2, 3]:
                child_state = copy.deepcopy(self.state)
                
                if child_state.my_tiles and len(child_state.my_tiles) > tile_idx:
                    placed_tile = child_state.my_tiles.pop(tile_idx)
                    
                    # Evaluate meeple potential for this tile placement
                    meeple_potential = self._evaluate_tile_meeple_potential(placed_tile, rotation)
                    
                    move_info = {
                        'tile_idx': tile_idx,
                        'rotation': rotation,
                        'meeple_potential': meeple_potential,
                        'simplified': True
                    }
                    
                    # Next decision is meeple placement
                    child_node = MeepleRewardNode(child_state, self.current_player, move_info, "meeple")
                    children.add(child_node)
                    
                    if len(children) >= max_children:
                        break
            
            if len(children) >= max_children:
                break
        
        return children
    
    def _evaluate_tile_meeple_potential(self, tile_info, rotation) -> float:
        """Evaluate meeple placement potential for a tile"""
        potential = 0.0
        
        for edge, structure_type in tile_info['internal_edges'].items():
            reward = self.state._calculate_structure_reward(structure_type, None, edge)
            potential += reward
        
        return potential
    
    def _generate_meeple_reward_children(self):
        """Generate meeple placement children optimized for reward maximization"""
        children = set()
        
        if not self.state.placed_tiles:
            return children
        
        last_tile = self.state.placed_tiles[-1]
        if not last_tile:
            return children
        
        tile_position = last_tile['position']
        next_player = (self.current_player + 1) % 4
        
        # Option 1: Pass on meeple placement
        child_state_pass = copy.deepcopy(self.state)
        pass_move = {'action': 'pass', 'meeple': False, 'reward': 0.0}
        child_pass = MeepleRewardNode(child_state_pass, next_player, pass_move, "tile")
        children.add(child_pass)
        
        # Options 2-5: Place meeple on edges with highest reward potential
        edge_rewards = []
        for edge in ['top_edge', 'right_edge', 'bottom_edge', 'left_edge']:
            if self.state.can_place_meeple(tile_position, edge):
                structure_type = last_tile['internal_edges'].get(edge)
                if structure_type:
                    reward = self.state._calculate_structure_reward(structure_type, None, edge)
                    edge_rewards.append((edge, reward))
        
        # Sort by reward potential and create children for top options
        edge_rewards.sort(key=lambda x: x[1], reverse=True)
        for edge, reward in edge_rewards[:3]:  # Top 3 reward options
            child_state = copy.deepcopy(self.state)
            
            # Simulate meeple placement with reward calculation
            actual_reward = child_state.simulate_meeple_placement(tile_position, edge, self.current_player)
            
            move_info = {
                'action': 'place_meeple',
                'edge': edge,
                'tile_position': tile_position,
                'reward': actual_reward
            }
            
            child_node = MeepleRewardNode(child_state, next_player, move_info, "tile")
            children.add(child_node)
        
        return children
    
    def find_random_child(self):
        """Find random child with bias towards higher rewards"""
        children = list(self.find_children())
        if not children:
            return None
        
        # Weight children by their potential reward
        weights = []
        for child in children:
            if hasattr(child.parent_move, 'get') and child.parent_move.get('reward'):
                weights.append(child.parent_move['reward'] + 1.0)  # +1 to avoid zero weights
            else:
                weights.append(1.0)
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(children)
        
        rand_val = random.uniform(0, total_weight)
        cumsum = 0
        for i, weight in enumerate(weights):
            cumsum += weight
            if rand_val <= cumsum:
                return children[i]
        
        return children[-1]  # Fallback
    
    def reward(self):
        """Calculate reward based on meeple reward differential"""
        scores = self.get_meeple_reward_scores()
        our_reward = scores[0]
        opponent_max_reward = max(scores[1], scores[2], scores[3])
        
        reward_differential = our_reward - opponent_max_reward
        
        # Normalize to [0, 1] with sigmoid, emphasizing meeple rewards
        normalized_reward = 1.0 / (1.0 + math.exp(-reward_differential / 5.0))
        
        return normalized_reward

class MeepleRewardMCTS:
    """MCTS optimized for meeple reward-based decisions"""
    
    def __init__(self, exploration_weight=1.4, max_simulation_depth=15):
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self.children = dict()
        self.exploration_weight = exploration_weight
        self.max_simulation_depth = max_simulation_depth
        self.reward_history = []  # Track rewards for analysis
    
    def _select(self, node):
        """Selection with bias towards meeple reward potential"""
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path
            
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                path.append(unexplored.pop())
                return path
            
            node = self._uct_select_with_reward_bias(node)
    
    def _uct_select_with_reward_bias(self, node):
        """UCT selection with meeple reward bias"""
        log_n_vertex = math.log(self.N[node])
        
        def enhanced_uct_value(child):
            if self.N[child] == 0:
                return float('inf')
            
            # Standard UCT
            exploitation = self.Q[child] / self.N[child]
            exploration = self.exploration_weight * math.sqrt(log_n_vertex / self.N[child])
            
            # Meeple reward bias
            reward_bias = 0.0
            if hasattr(child.parent_move, 'get') and child.parent_move.get('reward'):
                reward_bias = child.parent_move['reward'] / 20.0  # Scale factor
            
            return exploitation + exploration + reward_bias
        
        return max(self.children[node], key=enhanced_uct_value)
    
    def _expand(self, node):
        """Expansion with meeple opportunity prioritization"""
        if node in self.children:
            return
        self.children[node] = node.find_children()
    
    def _simulate(self, node):
        """Simulation focused on meeple reward accumulation"""
        current_player = node.current_player
        depth = 0
        
        while depth < self.max_simulation_depth:
            if node.is_terminal():
                scores = node.get_meeple_reward_scores()
                our_reward = scores[0]
                opponent_max = max(scores[1], scores[2], scores[3])
                
                raw_reward = our_reward - opponent_max
                reward = 1.0 / (1.0 + math.exp(-raw_reward / 8.0))
                
                # Track reward for analysis
                self.reward_history.append(our_reward)
                
                return reward if current_player == 0 else 1.0 - reward
            
            next_node = node.find_random_child()
            if next_node is None:
                break
            
            node = next_node
            depth += 1
        
        # Evaluate final position based on meeple rewards
        scores = node.get_meeple_reward_scores()
        our_reward = scores[0]
        opponent_max = max(scores[1], scores[2], scores[3])
        reward_diff = our_reward - opponent_max
        
        reward = 1.0 / (1.0 + math.exp(-reward_diff / 8.0))
        return reward if current_player == 0 else 1.0 - reward
    
    def _backpropagate(self, path, reward):
        """Backpropagation with meeple reward weighting"""
        for i, node in enumerate(reversed(path)):
            self.N[node] += 1
            
            # Weight updates by meeple reward potential
            weight = 1.0
            if hasattr(node.parent_move, 'get') and node.parent_move.get('reward'):
                weight = 1.0 + (node.parent_move['reward'] / 10.0)  # Boost high-reward moves
            
            if i % 4 == 0:  # Same player
                self.Q[node] += reward * weight
            else:  # Opponents
                self.Q[node] += (1.0 - reward) * weight
    
    def do_rollout(self, node, num_rollouts=100):
        """Execute rollouts optimized for meeple rewards"""
        self.reward_history.clear()
        
        for _ in range(num_rollouts):
            path = self._select(node)
            leaf = path[-1]
            self._expand(leaf)
            reward = self._simulate(leaf)
            self._backpropagate(path, reward)
    
    def best_child(self, node, exploration_weight=0):
        """Select best child considering meeple reward potential"""
        def enhanced_score(child):
            if self.N[child] == 0:
                return float('-inf')
            
            base_score = self.Q[child] / self.N[child]
            exploration_bonus = exploration_weight * math.sqrt(math.log(self.N[node]) / self.N[child])
            
            # Meeple reward bonus
            reward_bonus = 0.0
            if hasattr(child.parent_move, 'get') and child.parent_move.get('reward'):
                reward_bonus = child.parent_move['reward'] / 50.0  # Scaled bonus
            
            return base_score + exploration_bonus + reward_bonus
        
        return max(self.children[node], key=enhanced_score)
    
    def get_reward_statistics(self):
        """Get statistics about rewards found during search"""
        if not self.reward_history:
            return {'avg': 0, 'max': 0, 'min': 0}
        
        return {
            'avg': sum(self.reward_history) / len(self.reward_history),
            'max': max(self.reward_history),
            'min': min(self.reward_history)
        }

# Main game functions
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
                    return handle_place_tile_with_mcts(game, bot_state, q)

                case QueryPlaceMeeple() as q:
                    print("meeple decision")
                    return handle_place_meeple_with_mcts(game, bot_state, q)
                case _:
                    assert False

        print("sending move")
        game.send_move(choose_move(query))

def is_river_phase(game: Game) -> bool:
    """Check if we're still in the river phase"""
    # River phase typically ends when all river tiles are placed
    for tile in game.state.map.placed_tiles:
        for edge in tile.get_edges():
            if tile.internal_edges[edge] == StructureType.RIVER:
                return True
    return len(game.state.map.placed_tiles) < 12 

def handle_place_tile_with_mcts(game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Handle tile placement with MCTS optimized for meeple rewards"""
    
    # Handle river phase with simple heuristic
    if is_river_phase(game):
        return handle_river_phase(game, bot_state, query)
    
    # Use MCTS for strategic tile placement focusing on meeple opportunities
    try:
        reward_state = MeepleRewardGameState(game)
        mcts = MeepleRewardMCTS(exploration_weight=1.4, max_simulation_depth=15)
        root_node = MeepleRewardNode(reward_state, current_player=0, move_type="tile")
        
        print("Running MCTS tile placement optimized for meeple rewards...")
        
        # Execute MCTS search
        mcts.do_rollout(root_node, num_rollouts=80)
        
        # Get statistics
        reward_stats = mcts.get_reward_statistics()
        print(f"MCTS reward analysis: avg={reward_stats['avg']:.2f}, max={reward_stats['max']:.2f}")
        
        # Get best move
        if root_node in mcts.children and mcts.children[root_node]:
            best_child = mcts.best_child(root_node)
            move_info = best_child.parent_move
            
            if move_info and 'tile_idx' in move_info:
                tile_idx = move_info['tile_idx']
                rotation = move_info.get('rotation', 0)
                
                print(f"MCTS selected tile {tile_idx} with rotation {rotation} (meeple potential: {move_info.get('meeple_potential', 0):.2f})")
                
                return brute_force_tile_with_rotation(game, bot_state, query, tile_idx, rotation)
    
    except Exception as e:
        print(f"MCTS tile placement error: {e}")
    
    # Fallback
    return brute_force_tile(game, bot_state, query)

def handle_place_meeple_with_mcts(game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceMeeple) -> MovePlaceMeeple | MovePlaceMeeplePass:
    """Handle meeple placement with MCTS optimized for reward maximization"""
    
    if not bot_state.last_tile:
        return game.move_place_meeple_pass(query)
    
    # Meeple limit check
    if bot_state.meeples_placed >= 7:
        return game.move_place_meeple_pass(query)
    
    try:
        reward_state = MeepleRewardGameState(game)
        mcts = MeepleRewardMCTS(exploration_weight=1.2, max_simulation_depth=10)
        root_node = MeepleRewardNode(reward_state, current_player=0, move_type="meeple")
        
        print("Running MCTS meeple placement for reward optimization...")
        
        # Execute MCTS search focusing on meeple rewards
        mcts.do_rollout(root_node, num_rollouts=60)
        
        # Get reward statistics
        reward_stats = mcts.get_reward_statistics()
        print(f"Meeple MCTS rewards: avg={reward_stats['avg']:.2f}, best={reward_stats['max']:.2f}")
        
        # Get best meeple decision
        if root_node in mcts.children and mcts.children[root_node]:
            best_child = mcts.best_child(root_node)
            move_info = best_child.parent_move
            
            if move_info:
                if move_info.get('action') == 'place_meeple':
                    edge = move_info.get('edge')
                    expected_reward = move_info.get('reward', 0)
                    
                    if edge and edge in bot_state.last_tile.get_edges():
                        # Verify move validity
                        if not game.state._get_claims(bot_state.last_tile, edge):
                            is_completed = game.state._check_completed_component(bot_state.last_tile, edge)
                            
                            if not is_completed:
                                print(f"MCTS meeple placement on {edge} (expected reward: {expected_reward:.2f})")
                                
                                # Create meeple info with reward calculation
                                structure_type = bot_state.last_tile.internal_edges[edge]
                                meeple_info = MeepleInfo(bot_state.last_tile, structure_type, edge, bot_state.move_count)
                                bot_state.add_our_meeple(meeple_info)
                                
                                print(f"Total reward potential now: {bot_state.get_total_reward_potential():.2f}")
                                
                                return game.move_place_meeple(query, bot_state.last_tile._to_model(), placed_on=edge)
                
                elif move_info.get('action') == 'pass':
                    print("MCTS recommends meeple pass for reward optimization")
                    return game.move_place_meeple_pass(query)
    
    except Exception as e:
        print(f"MCTS meeple placement error: {e}")
    
    # Fallback to heuristic placement
    return fallback_meeple_placement(game, bot_state, query)

# Helper functions (simplified versions of complex functions)
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

def brute_force_tile(game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceTile) -> MovePlaceTile:
    """Brute force tile placement fallback"""
    grid = game.state.map._grid
    
    for y in range(MAX_MAP_LENGTH):
        for x in range(MAX_MAP_LENGTH):
            if grid[y][x] is not None:
                for tile_index, tile in enumerate(game.state.my_tiles):
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        x1, y1 = x + dx, y + dy
                        if game.can_place_tile_at(tile, x1, y1):
                            bot_state.last_tile = tile
                            bot_state.last_tile.placed_pos = (x1, y1)
                            return game.move_place_tile(query, tile._to_model(), tile_index)

def brute_force_tile_with_rotation(game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceTile, tile_idx: int, preferred_rotation: int) -> MovePlaceTile:
    """Brute force with specific tile and rotation preference"""
    if tile_idx >= len(game.state.my_tiles):
        return brute_force_tile(game, bot_state, query)
    
    tile = game.state.my_tiles[tile_idx]
    tile.rotation = preferred_rotation
    
    grid = game.state.map._grid
    for y in range(MAX_MAP_LENGTH):
        for x in range(MAX_MAP_LENGTH):
            if grid[y][x] is None and game.can_place_tile_at(tile, x, y):
                tile.placed_pos = (x, y)
                bot_state.last_tile = tile
                return game.move_place_tile(query, tile._to_model(), tile_idx)
    
    # Fallback if preferred doesn't work
    return brute_force_tile(game, bot_state, query)

def fallback_meeple_placement(game: Game, bot_state: DomainExpansionBotState, query: QueryPlaceMeeple) -> MovePlaceMeeple | MovePlaceMeeplePass:
    """Simple heuristic meeple placement fallback"""
    last_tile = bot_state.last_tile
    
    if bot_state.meeples_placed >= 7:
        return game.move_place_meeple_pass(query)
    
    # Priority: Monastery > City > Road
    for edge in [MONASTARY_IDENTIFIER, "top_edge", "right_edge", "bottom_edge", "left_edge"]:
        if edge == MONASTARY_IDENTIFIER:
            if (hasattr(last_tile, "modifiers") and 
                any(mod.name == "MONESTERY" for mod in last_tile.modifiers)):
                
                is_completed = game.state._check_completed_component(last_tile, MONASTARY_IDENTIFIER)
                if not is_completed:
                    meeple_info = MeepleInfo(last_tile, StructureType.MONASTARY, MONASTARY_IDENTIFIER, bot_state.move_count)
                    bot_state.add_our_meeple(meeple_info)
                    return game.move_place_meeple(query, last_tile._to_model(), placed_on=MONASTARY_IDENTIFIER)
        else:
            structure_type = last_tile.internal_edges[edge]
            is_completed = game.state._check_completed_component(last_tile, edge)
            
            if (not game.state._get_claims(last_tile, edge) and
                structure_type in [StructureType.CITY, StructureType.ROAD] and
                not is_completed):
                
                meeple_info = MeepleInfo(last_tile, structure_type, edge, bot_state.move_count)
                bot_state.add_our_meeple(meeple_info)
                return game.move_place_meeple(query, last_tile._to_model(), placed_on=edge)
    
    return game.move_place_meeple_pass(query)

if __name__ == "__main__":
    main()