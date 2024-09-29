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
        current_player = player
        
        while True:
            legal_moves = list(zip(*np.where(current_state == 0)))
            if not legal_moves:
                break
            
            # Apply heuristics to score moves
            move_scores = []
            for move in legal_moves:
                score = 0
                
                # Proximity to existing pieces
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = move[0] + dx, move[1] + dy
                        if 0 <= nx < current_state.shape[0] and 0 <= ny < current_state.shape[1]:
                            if current_state[nx, ny] == current_player:
                                score += 3  # Higher score for own pieces
                            elif current_state[nx, ny] != 0:
                                score += 3  # Lower score for opponent pieces
                
                # Check for potential connections
                current_state[move] = current_player
                connected_sides = self.count_connected_sides(current_state, move)
                score += connected_sides * 5  # Boost score for moves that connect to sides
                
                # Simple check for potential bridges, rings, or forks
                is_win, win_type = check_win(current_state, move, current_player)
                if is_win:
                    score += 100
                current_state[move] = 0
                
                move_scores.append((move, score))
            
            # Choose move with highest score, with some randomness
            total_score = sum(score for _, score in move_scores)
            if total_score == 0:
                chosen_move = random.choice(legal_moves)
            else:
                chosen_move = random.choices(
                    [move for move, _ in move_scores],
                    weights=[score / total_score for _, score in move_scores],
                    k=1
                )[0]
            
            current_state[chosen_move] = current_player
            
            is_win, win_type = check_win(current_state, chosen_move, current_player)
            if is_win:
                if win_type == 'ring':
                    path = find_ring(current_state, chosen_move)
                elif win_type == 'fork':
                    path = find_fork(current_state, chosen_move)
                else:
                    path = find_bridge(current_state, chosen_move)
                return (current_player, len(path))
            
            current_player = 3 - current_player  # Switch player

        # If no winner, use the simplified check
        return (1, 0) if np.sum(current_state) % 2 == 1 else (2, 0)

    def count_connected_sides(self, state, move):
        board_size = (state.shape[0] + 1) // 2
        sides_connected = set()

        def is_edge(x, y):
            if x == 0 or y == 0 or x + y == 2 * board_size - 2:
                return 0
            elif x == board_size - 1:
                return 1
            elif y == 2 * board_size - 2:
                return 2
            elif x + y == board_size - 1:
                return 3
            elif y == board_size - 1:
                return 4
            elif x == 2 * board_size - 2:
                return 5
            else:
                return -1

        def dfs(x, y, player):
            if x < 0 or y < 0 or x >= state.shape[0] or y >= state.shape[1] or state[x, y] != player:
                return
            
            edge = is_edge(x, y)
            if edge != -1:
                sides_connected.add(edge)
            
            state[x, y] = -1  # Mark as visited
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
                dfs(x + dx, y + dy, player)

        player = state[move]
        dfs(move[0], move[1], player)
        return len(sides_connected)
    
    
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
            self.limit = (self.dim * (self.dim - 1))//2
            self.simulations = 100 * self.dim
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
        
        if(self.move <= self.dim):
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


            # Step 6: AI's three-move win detection
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

        return root.best_child(exploration_weight=0).state

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
