import time
import math
import random
import numpy as np
from helper import *

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
        return list(zip(*np.where(self.state == 0)))

    def expand(self):
        legal_moves = self.get_legal_moves()
        for move in legal_moves:
            new_state = self.state.copy()
            new_state[move] = 3 - self.player_number  # Switch to the next player
            child_node = MCTSNode(new_state, 3 - self.player_number, parent=self)
            self.children.append(child_node)

    def simulate(self):
        current_state = self.state.copy()
        current_player = self.player_number
        legal_moves = list(zip(*np.where(current_state == 0)))

        while len(legal_moves) > 0:
            move = random.choice(legal_moves)
            current_state[move] = current_player
            current_player = 3 - current_player  # Switch player
            legal_moves = list(zip(*np.where(current_state == 0)))

        # Simplified simulation check
        return 1 if np.sum(current_state) % 2 == 1 else 2

    def backpropagate(self, result):
        self.visits += 1
        if result == self.player_number:
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(result)


class AIPlayer:
    def __init__(self, player_number: int, timer):
        """
        Initialize the AIPlayer.

        # Parameters
        player_number (int): The current player number.
        timer (Timer): Timer object to fetch remaining time.
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = f'Player {player_number}: ai'
        self.timer = timer

    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Given the current state of the board, return the next move.

        :param state: A numpy array containing the state of the board (0 = empty, 1 = player 1, 2 = player 2)
        :return: Tuple[int, int] representing the coordinates of the chosen move
        """
        # Step 1: Check if the AI can win with its current move
        legal_moves = list(zip(*np.where(state == 0)))
        for move in legal_moves:
            # Simulate the AI making the move
            state[move] = self.player_number
            is_win, win_type = check_win(state, move, self.player_number)
            state[move] = 0  # Undo the move
            if is_win:
                # If this move results in a win, play it immediately
                return move

        # Step 2: Check if the opponent can win with their next move, and block it
        opponent = 3 - self.player_number
        for move in legal_moves:
            # Simulate the opponent making the move
            state[move] = opponent
            is_win, win_type = check_win(state, move, opponent)
            state[move] = 0  # Undo the move
            if is_win:
                # If this move allows the opponent to win, block it
                return move

        # Step 3: If no immediate win or block, proceed with MCTS
        root = MCTSNode(state, self.player_number)
        best_state = self.uct_search(root, num_simulations=10000)
        return tuple(zip(*np.where(state != best_state)))[0]

    def uct_search(self, root: MCTSNode, num_simulations=10000):
        """
        Perform UCT (Upper Confidence Tree) Search to find the best move using a fixed number of simulations.

        :param root: The root MCTSNode representing the current game state.
        :param num_simulations: Number of simulations to perform.
        :return: The best state after searching.
        """
        for _ in range(num_simulations):
            node = self.tree_policy(root)
            result = node.simulate()
            node.backpropagate(result)

        return root.best_child(exploration_weight=0).state

    def tree_policy(self, node: MCTSNode):
        """
        Apply the tree policy to select a node for expansion.

        :param node: The current node.
        :return: The selected node for expansion.
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

        :param node: The node to expand.
        :return: A new child node.
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

        :param node: The node to check.
        :return: True if the game is over, otherwise False.
        """
        return np.sum(node.state == 0) == 0  # Full board
