class MCTS:
    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)      # total reward of each node
        self.N = defaultdict(int)      # visit count of each node
        self.children = dict()         # children of each node
        self.exploration_weight = exploration_weight

    def _select(self, node):
        "Selection: descend until a leaf or unexpanded node is found"
        path = []
        while True:
            path.append(node)
            # if node is not expanded or is terminal, stop
            if node not in self.children or not self.children[node]:
                return path
            # choose any unexplored child if available
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                path.append(unexplored.pop())
                return path
            # otherwise use UCT to select among children
            node = self._uct_select(node)

    def _expand(self, node):
        "Expansion: add all legal children of node to the tree"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Simulation: play random moves to the end of the game"
        invert = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return (1 - reward) if invert else reward
            node = node.find_random_child()
            invert = not invert

    def _backpropagate(self, path, reward):
        "Backpropagation: update statistics along the path"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            # alternate the reward for the other player
            reward = 1 - reward

class TicTacToeBoard(Node):
    def find_children(self):
        if self.terminal:
            return set()
        # return all boards after a legal move
        return {self.make_move(i) for i, v in enumerate(self.tup) if v is None}

    def make_move(self, index):
        tup = self.tup[:index] + (self.turn,) + self.tup[index+1:]
        turn = not self.turn
        winner = _find_winner(tup)
        is_terminal = (winner is not None) or not any(v is None for v in tup)
        return TicTacToeBoard(tup, turn, winner, is_terminal)

tree = MCTS()
board = new_tic_tac_toe_board()  # initial state
for _ in range(1000):
    tree.do_rollout(board)
best_node = tree.choose(board)  # select child with best average reward