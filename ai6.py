import time
import math
import random
import numpy as np
from helper import *
import sys

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
            print(uct_score)
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

    def simulate(self, player):
        current_state = self.state.copy()
        current_player = self.player_number
        
        legal_moves = list(zip(*np.where(current_state == 0)))

        while len(legal_moves) > 0:
            move = random.choice(legal_moves)
            current_state[move] = current_player
            win,win_type = check_win(current_state,move,current_player,[])
            if(win):
                if(win_type == 'ring'):
                    path = find_ring(current_state,move)
                elif(win_type == 'fork'):
                    path = find_fork(current_state,move)
                else:
                    path = find_bridge(current_state,move)
                return current_player,len(path)
            current_player = 3 - current_player  # Switch player
            legal_moves = list(zip(*np.where(current_state == 0)))

        # Simplified simulation check
        return (1,0) if np.sum(current_state) % 2 == 1 else (2,0)
    
    def backpropagate(self, result,val):
        self.visits += 1
        if result == self.player_number:
            self.wins += val
        if self.parent:
            self.parent.backpropagate(result,val)


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
        self.move = 0
        self.dim = 0
        self.limit = 0
        self.simulations = 0
        self.corners = []
        self.selected_corner = 0
        self.second_corner = 0
        self.prev_state = None
        self.player_vc = []
        self.opponent_vc = []
    
    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Given the current state of the board, return the next move.

        :param state: A numpy array containing the state of the board (0 = empty, 1 = player 1, 2 = player 2)
        :return: Tuple[int, int] representing the coordinates of the chosen move
        """
        # Step 1: Check if the AI can win with its current move
        self.move += 1
        if(self.move == 1):
            self.dim = (len(state)+1)//2
            self.limit = len(list(zip(*np.where(state != 3))))
            self.limit -= 20
            self.limit //= 2
            print(self.limit)
            self.simulations = 500 * self.dim
            if(self.dim >= 8):
                self.simulations *= 2
            self.corners = [(0,0),(0,self.dim-1),(0,2*(self.dim-1)),(self.dim-1,2*(self.dim-1)),(2*(self.dim-1),self.dim-1),(self.dim-1,0)]
            print(self.corners)
            for i in range(6):
                corner = self.corners[i]
                if state[corner] == 1:  # Check if the corner is empty
                    self.selected_corner = (i+1)%6
                    return self.corners[(i+1)%6]
            self.selected_corner = 0
            return self.corners[0]
        if(self.move == 2):
            corner = self.corners[(self.selected_corner+2)%6]
            if(state[corner] == 0):
                state[corner] = self.player_number
                self.prev_state = state
                self.second_corner = (self.selected_corner+2)%6
                return corner
            else:
                corner = self.corners[(self.selected_corner-2)%6]
                state[corner] = self.player_number
                self.prev_state = state
                self.second_corner = (self.selected_corner-2)%6
                return corner
        
        if(self.move <= self.dim-1):
            opponent_move = list(zip(*np.where(state != self.prev_state)))[0]
            neighbors = get_neighbours(len(state),opponent_move)
            val = 0
            selected_move = None
            for move in neighbors:
                if(state[move] != 0):
                    continue
                neighbor_move = get_neighbours(len(state),move)
                ans = 0
                for near in neighbor_move:
                    if(state[near] == self.player_number):
                        ans += 2
                    elif(state[near] == 3 - self.player_number):
                        ans += 3
                    print(move,end = " ")
                    print(ans)
                if(ans > val):
                    val = ans
                    selected_move = move
            if(selected_move == None):
                pass
            else:
              state[selected_move] = self.player_number
              self.prev_state = state
              return selected_move
        
        legal_moves = list(zip(*np.where(state == 0)))
        for move in legal_moves:
            state[move] = self.player_number
            is_win, win_type = check_win(state, move, self.player_number)
            state[move] = 0  # Undo the move
            if is_win:
                return move

        # Step 2: Check if the opponent can win with their next move, and block it
        opponent = 3 - self.player_number
        for move in legal_moves:
            state[move] = opponent
            is_win, win_type = check_win(state, move, opponent)
            state[move] = 0  # Undo the move
            if is_win:
                return move

        # Step 3: AI's two-move win detection
        for move in legal_moves:
            state[move] = self.player_number
            second_moves = list(zip(*np.where(state == 0)))  # Remaining legal moves
            for second_move in second_moves:
                state[second_move] = self.player_number
                is_second_win, _ = check_win(state, second_move, self.player_number)
                state[second_move] = 0  # Undo second move
                if is_second_win:
                    state[move] = 0  # Undo first move
                    return move
            state[move] = 0  # Undo first move

        # Step 4: Check if the opponent can win in two moves and block it
        for move in legal_moves:
            state[move] = opponent
            second_moves = list(zip(*np.where(state == 0)))  # Remaining legal moves
            for second_move in second_moves:
                state[second_move] = opponent
                is_second_win, _ = check_win(state, second_move, opponent)
                state[second_move] = 0  # Undo second move
                if is_second_win:
                    state[move] = 0  # Undo first move
                    return move
            state[move] = 0  # Undo first move


        # Step 6: AI's three-move win-loss detection
        if(self.move > self.limit):
            for move in legal_moves:
                state[move] = self.player_number
                second_moves = list(zip(*np.where(state == 0)))  # Remaining legal moves
                for second_move in second_moves:
                    state[second_move] = self.player_number
                    third_moves = list(zip(*np.where(state == 0)))  # Remaining legal moves after second move
                    for third_move in third_moves:
                        state[third_move] = self.player_number
                        is_third_win, _ = check_win(state, third_move, self.player_number)
                        state[third_move] = 0  # Undo third move
                        if is_third_win:
                            state[second_move] = 0  # Undo second move
                            state[move] = 0  # Undo first move
                            return move
                    state[second_move] = 0  # Undo second move
                state[move] = 0  # Undo first move

            # Step 7: Check if the opponent can win in three moves and block it
            for move in legal_moves:
                state[move] = opponent
                second_moves = list(zip(*np.where(state == 0)))  # Remaining legal moves
                for second_move in second_moves:
                    state[second_move] = opponent
                    third_moves = list(zip(*np.where(state == 0)))  # Remaining legal moves after second move
                    for third_move in third_moves:
                        state[third_move] = opponent
                        is_third_win, _ = check_win(state, third_move, opponent)
                        state[third_move] = 0  # Undo third move
                        if is_third_win:
                            state[second_move] = 0  # Undo second move
                            state[move] = 0  # Undo first move
                            return move
                    state[second_move] = 0  # Undo second move
                state[move] = 0  # Undo first move
                


        # Step 5: If no immediate win or block, proceed with MCTS
        root = MCTSNode(state, self.player_number)
        best_state = self.uct_search(root)
        return tuple(zip(*np.where(state != best_state)))[0]

    # def virtual_connection(self,state,move):
    #     x,y = move
    #     dim = len(state)
    #     if(y < dim - 1):
            

    def uct_search(self, root: MCTSNode):
        """
        Perform UCT (Upper Confidence Tree) Search to find the best move using a fixed number of simulations.

        :param root: The root MCTSNode representing the current game state.
        :param num_simulations: Number of simulations to perform.
        :return: The best state after searching.
        """
        for _ in range(self.simulations):
            print(_)
            node,player = self.selection(root)
            result,val = node.simulate(player)
            l,b = (node.state).shape
            if(val == 0):
                node.backpropagate(result,0)
            else:
                node.backpropagate(result,(l*b//2) - val)

        return root.best_child(exploration_weight=0.9).state

    def selection(self, node: MCTSNode):
        """
        Apply the tree policy to select a node for expansion.

        :param node: The current node.
        :return: The selected node for expansion.
        """
        player = self.player_number
        while not self.is_terminal(node):
            if not node.is_fully_expanded():
                return (self.expand(node),(3-player))
            else:
                node = node.best_child()
                player = 3 - player
        return node,player

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
