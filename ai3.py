import time
import math
import random
import numpy as np

class MCTSNode:
    def __init__(self, state, player_number, parent=None):
        """
        Initialize an MCTSNode.
        
        :param state: The current state of the game (board)
        :param player_number: The player number (1 or 2) who made the move leading to this node
        :param parent: The parent node (None for root)
        """
        self.state = state
        self.player_number = player_number  # The player who made the move leading to this node
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.get_legal_moves())

    def best_child(self, exploration_weight=1.414):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            if child.visits == 0:
                uct_score = float('inf')
            else:
                uct_score = child.wins / child.visits + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child

    def get_legal_moves(self):
        # Return all empty positions (state == 0)
        return list(zip(*np.where(self.state == 0)))

    def expand(self):
        legal_moves = self.get_legal_moves()
        for move in legal_moves:
            new_state = self.state.copy()
            new_state[move] = 3 - self.player_number  # Switch to the next player (1 -> 2, 2 -> 1)
            child_node = MCTSNode(new_state, 3 - self.player_number, parent=self)  # Alternate player
            self.children.append(child_node)

    def simulate(self):
        """
        Simulate a random game from the current node to a terminal state.
        """
        current_state = self.state.copy()
        current_player = self.player_number  # Continue from the current player
        legal_moves = list(zip(*np.where(current_state == 0)))

        while len(legal_moves) > 0:
            move = random.choice(legal_moves)
            current_state[move] = current_player
            current_player = 3 - current_player  # Switch player
            legal_moves = list(zip(*np.where(current_state == 0)))

        return 1 if np.sum(current_state) % 2 == 1 else 2

    def backpropagate(self, result):
        """
        Backpropagate the result of the simulation through the tree.
        :param result: The result of the simulation (winning player)
        """
        self.visits += 1
        if result == self.player_number:
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(result)


class AIPlayer:
    def __init__(self, player_number: int, timer):
        """
        Initialize the AIPlayer Agent.

        Parameters:
        player_number (int): Current player number, num==1 starts the game.
        timer: Timer object to fetch remaining time.
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer

    def get_move(self, state: np.array) -> tuple[int, int]:
        """
        Given the current state of the board, return the next move.

        :param state: A numpy array containing the state of the board (0 = empty, 1 = player 1, 2 = player 2)
        :return: Tuple[int, int] representing the coordinates of the chosen move
        """
        if self.should_use_minimax(state):
            print("minimax start")
            return self.minimax_move(state, self.player_number, depth=4)
        else:
            root = MCTSNode(state, self.player_number)
            best_state = self.uct_search(root, time_limit=5)
            return tuple(zip(*np.where(state != best_state)))[0]

    def uct_search(self, root: MCTSNode, time_limit=5):
        """
        Perform UCT (Upper Confidence Tree) Search to find the best move.
        
        :param root: The root MCTSNode representing the current game state
        :param time_limit: Time limit in seconds for the search
        :return: The best state after searching
        """
        end_time = time.time() + time_limit
        while time.time() < end_time:
            node = self.tree_policy(root)
            result = node.simulate()
            node.backpropagate(result)

        return root.best_child(exploration_weight=0).state

    def tree_policy(self, node: MCTSNode):
        """
        Apply the tree policy to select a node for expansion.
        
        :param node: The current node
        :return: The selected node for expansion
        """
        while not self.is_terminal(node):
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child()
        return node

    def expand(self, node: MCTSNode):
        """
        Expand a node by adding its children (legal moves).
        
        :param node: The node to expand
        :return: A new child node
        """
        legal_moves = node.get_legal_moves()
        for move in legal_moves:
            new_state = node.state.copy()
            new_state[move] = 3 - node.player_number  # Alternate player
            child_node = MCTSNode(new_state, 3 - node.player_number, parent=node)
            node.children.append(child_node)
        return random.choice(node.children)

    def is_terminal(self, node: MCTSNode):
        """
        Check if the node represents a terminal state (game over).
        
        :param node: The node to check
        :return: True if the game is over, otherwise False
        """
        return np.sum(node.state == 0) == 0  # Full board

    def should_use_minimax(self, state):
        """
        Check if we should switch to Minimax based on the state space being half-filled.
        """
        empty_cells = np.sum(state == 0)
        total_cells = state.size
        return empty_cells <= total_cells // 2

    def minimax_move(self, state, player, depth):
        """
        Perform the minimax algorithm to find the best move.
        
        :param state: The current game state (board).
        :param player: The current player (1 or 2).
        :param depth: The depth to search.
        :return: The best move for the current player.
        """
        best_move, best_score = self.minimax(state, player, depth, -float('inf'), float('inf'), True)
        return best_move

    def minimax(self, state, player, depth, alpha, beta, is_maximizing):
        """
        Minimax algorithm with alpha-beta pruning.

        :param state: The current game state (board).
        :param player: The player number (1 or 2).
        :param depth: The depth limit for the recursion.
        :param alpha: Alpha value for pruning.
        :param beta: Beta value for pruning.
        :param is_maximizing: Whether the current player is maximizing or minimizing.
        :return: The best move and its score.
        """
        legal_moves = list(zip(*np.where(state == 0)))

        # Base case: If depth is 0 or the game is over, evaluate the score
        if depth == 0 or self.is_terminal_state(state):
            return None, self.evaluate(state, player)

        if is_maximizing:
            best_score = -float('inf')
            best_move = None
            for move in legal_moves:
                new_state = state.copy()
                new_state[move] = player  # Player makes a move
                _, score = self.minimax(new_state, 3 - player, depth - 1, alpha, beta, False)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return best_move, best_score
        else:
            best_score = float('inf')
            best_move = None
            for move in legal_moves:
                new_state = state.copy()
                new_state[move] = player  # Player makes a move
                _, score = self.minimax(new_state, 3 - player, depth - 1, alpha, beta, True)
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, best_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return best_move, best_score

    def is_terminal_state(self, state):
        """
        Check if the game has reached a terminal state (win/loss/draw).
        This method will need to be updated based on Havannah's winning conditions.
        """
        legal_moves = np.sum(state == 0)
        return legal_moves == 0  # The game ends when no more moves are possible

    def evaluate(self, state, player):
        """
        Evaluate the current game state for a player.
        (This is a heuristic that needs to be defined for Havannah.)
        
        :param state: The game state to evaluate.
        :param player: The current player (1 or 2).
        :return: The heuristic score for the current player.
        """
        return np.sum(state == player) - np.sum(state == (3 - player))

