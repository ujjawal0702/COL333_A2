from re import S
from shutil import which
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

    def best_child(self, exploration_weight=1.414, last_opponent_move=None):
        """
        Select the best child node, incorporating proximity-based heuristics.

        :param exploration_weight: The exploration parameter for UCT (default sqrt(2))
        :param last_opponent_move: The coordinates of the opponent's last move, used to boost nearby moves.
        :return: The child node with the highest score.
        """
        best_score = -float('inf')
        best_child = None

        for child in self.children:
            if child.visits == 0:
                uct_score = float('inf')
            else:
                # Standard UCT formula: exploit + explore
                uct_score = (child.wins / child.visits) + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)

                # Add heuristic for proximity to filled cells
                proximity_bonus = self.calculate_proximity_bonus(child.state, last_opponent_move)
                uct_score += proximity_bonus  # Increase the score based on proximity to filled cells

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child
    
    def best_selection_child(self, exploration_weight=1.414):
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

    def calculate_proximity_bonus(self, state, last_opponent_move):
        """
        Calculate a bonus score for moves based on proximity to filled cells and the opponent's last move.

        :param state: The current state of the board (2D numpy array).
        :param last_opponent_move: The coordinates of the opponent's last move, to further boost nearby cells.
        :return: A proximity-based bonus score.
        """
        proximity_bonus = 0
        filled_positions = list(zip(*np.where(state != 0)))  # Get positions of filled cells (non-zero entries)

        for pos in filled_positions:
            neighbors = get_neighbours(len(state), pos)  # Get neighboring cells for each filled position
            for neighbor in neighbors:
                if state[neighbor] == 0:  # Only consider empty neighboring cells
                    proximity_bonus += 1  # Add a general proximity bonus for neighbors of filled cells
                    
                    # Additional bonus for neighbors of the opponent's last move
                    if last_opponent_move and self.is_neighbor(last_opponent_move, neighbor):
                        proximity_bonus += 2  # Add a higher bonus for cells near the opponent's last move

        return proximity_bonus

    def is_neighbor(self, move1, move2):
        """
        Check if two moves are neighbors (within one cell in any direction).

        :param move1: Coordinates of the first move.
        :param move2: Coordinates of the second move.
        :return: True if the moves are neighbors, False otherwise.
        """
        return abs(move1[0] - move2[0]) <= 1 and abs(move1[1] - move2[1]) <= 1


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



class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.available = 0

    def find(self, x):
        """Finds the representative of the set containing x."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """Unions the sets containing x and y."""
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            # Union by rank
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

    def add(self, x):
        """Initializes the set for element x."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = self.available
            self.available += 1

uf =UnionFind()
opponent_uf = UnionFind()

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
        self.last_move = None
        self.union_find = uf
        self.opp_uf = opponent_uf
        self.ai_disjoint = {}
        self.opp_disjoint = {}
        
    
    def threat(self,opp_move):
        pass
    
    def power(self,move):
        pass
    
    def virtual_connection(self, state, move):
        """
        Returns the cells having virtual connection with a block.
        """
        x, y = move
        dim = self.dim
        vc = []
        if y < dim - 1:
            vc = [(x+2, y+1), (x+1, y-1), (x-1, y-2), (x-2, y-1), (x-1, y+1)]
            if y < dim - 2:
                vc.append((x+1, y+2))
            else:
                vc.append((x, y+2))
        elif y > dim - 1:
            vc = [(x+2, y-1), (x+1, y+1), (x-1, y+2), (x-2, y+1), (x-1, y-1)]
            if y < dim - 2:
                vc.append((x+1, y-2))
            else:
                vc.append((x, y-2))
        else:
            vc = [(x-1, y-2), (x-2, y-1), (x-2, y+1), (x-1, y+2), (x+1, y+1), (x+1, y-1)]
        
        vc_out = []
        for i in range(6):
            a, b = vc[i]
            if is_valid(a, b, len(state)):
                vc_out.append(vc[i])
        return vc_out
    
    def add_move_to_union_find(self, move):
        """
        Add the AI's move to the Union-Find structure and connect it with its neighbors.
        """
        self.union_find.add(move)
        neighbors = get_neighbours((self.dim * 2) - 1, move)
        
        # Connect with neighboring AI player's cells
        for neighbor in neighbors:
            if neighbor in self.union_find.parent and self.union_find.find(neighbor) != self.union_find.find(move):
                self.union_find.union(move, neighbor)
                
    def add_move_to_union_find2(self, move,uf_object):
        """
        Add the AI's move to the Union-Find structure and connect it with its neighbors.
        """
        uf_object.add(move)
        neighbors = get_neighbours((self.dim * 2) -1, move)
        
        # Connect with neighboring AI player's cells
        for neighbor in neighbors:
            if neighbor in uf_object.parent and uf_object.find(neighbor) != uf_object.find(move):
                uf_object.union(move, neighbor)

    def get_connected_component_moves(self,state):
        """
        Retrieve all the moves in the connected component of the AI's most recent move.
        """
        # Find the root of the most recent move and gather all connected moves
        connected_moves = set()
        print(self.union_find.parent)
        if(self.last_move == None):
            return
        for move in self.union_find.parent:
            if self.union_find.find(move) == self.union_find.find(self.last_move):
                connected_moves.add(move)
        print(connected_moves)
        # return connected_moves
        
    def update_adjoint_set(self, state, last_move, player, disjoint_set_dict, uf_object):
        neighbors = get_neighbours(len(state), last_move)
        corner_last_move = get_corner(last_move, len(state))
        edge_last_move = get_edge(last_move, len(state))
        corner_set = set()
        edge_set = set()

        # Add the corner and edge of the last move to the respective sets
        if corner_last_move != -1:
            corner_set.add(corner_last_move)
        if edge_last_move != -1:
            edge_set.add(edge_last_move)

        # Iterate over the neighbors
        for neigh in neighbors:
            if state[neigh] == player:
                par = uf_object.find(neigh)
                
                # Ensure that the parent exists in the disjoint set dictionary
                if par in disjoint_set_dict:
                    # Merge the parent's corner and edge sets into the current sets
                    corner_par = disjoint_set_dict[par]['corner']
                    edge_par = disjoint_set_dict[par]['edge']
                    
                    # Remove the parent from the dictionary as we are merging it
                    del disjoint_set_dict[par]
                    
                    # Update the corner and edge sets with the parent's sets
                    corner_set.update(corner_par)
                    edge_set.update(edge_par)
        
        # Add or update the entry for the last move in the disjoint set dictionary
        disjoint_set_dict[last_move] = {'corner': corner_set, 'edge': edge_set}
        
        return

    
    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Given the current state of the board, return the next move.

        :param state: A numpy array containing the state of the board (0 = empty, 1 = player 1, 2 = player 2)
        :return: Tuple[int, int] representing the coordinates of the chosen move
    
        """
        self.move += 1
        if(self.move == 1):
            #Initialize vars
            self.dim = (len(state)+1)//2
            self.limit = (self.dim * (self.dim - 1))//2
            self.simulations = 500 * self.dim
            # if(self.dim >= 8):
            #     self.simulations *= 2
            self.corners = [(0,0),(0,self.dim-1),(0,2*(self.dim-1)),(self.dim-1,2*(self.dim-1)),(2*(self.dim-1),self.dim-1),(self.dim-1,0)]
            opp = list(zip(*np.where(state == (3-self.player_number))))
            if(len(opp) != 0):
                self.opp_uf.add(opp[0])
                self.opp_disjoint[opp[0]] = {"corner": [],"edge": []}
                opp_corn = get_corner(opp[0],len(state))
                opp_edge = get_edge(opp[0],len(state))
                if(opp_corn != -1):
                  self.opp_disjoint[opp[0]]['corner'] = [opp_corn]
                if(opp_edge != -1):
                  self.opp_disjoint[opp[0]]['edge'] = [opp_edge]    
            move = (0,0)
            self.selected_corner = 0
            for i in range(6):
                corner = self.corners[i]
                if state[corner] == (3-self.player_number):  # Check if the corner is occupied by opponent, if it is then select the next corner
                    self.selected_corner = (i+1)%6
                    move = self.corners[(i+1)%6]
                    break
            self.add_move_to_union_find(move)
            self.last_move = move
            state[move] = self.player_number
            self.prev_state = state
            which_corner = get_corner(move,len(state))
            self.ai_disjoint[move] = {"corner": [which_corner], "edge":[]}
            return move
        
        opponent_last_move = list(zip(*np.where(state != self.prev_state)))[0]
        print(opponent_last_move)
        
        self.update_adjoint_set(state,self.last_move,self.player_number,self.ai_disjoint,self.union_find)
        self.update_adjoint_set(state,opponent_last_move,3 - self.player_number,self.opp_disjoint,self.opp_uf)
        print(self.ai_disjoint)
        print(self.opp_disjoint)
        
        self.add_move_to_union_find2(self.last_move,self.union_find)     
        
        self.add_move_to_union_find2(opponent_last_move,self.opp_uf)
        
        
        
        
        if(self.move == 2):
            corner = self.corners[(self.selected_corner+2)%6]
            if(state[corner] == 0):
                state[corner] = self.player_number
                self.prev_state = state
                self.second_corner = (self.selected_corner+2)%6
                self.last_move = corner
                return corner
            else:
                corner = self.corners[(self.selected_corner-2)%6]
                state[corner] = self.player_number
                self.prev_state = state
                self.second_corner = (self.selected_corner-2)%6
                self.last_move = corner
                return corner
        
        # Step 1: Check if the AI can win with its current move
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
                state[move] = self.player_number
                self.prev_state = state
                self.last_move = move
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
                    state[move] = self.player_number  
                    self.prev_state = state
                    self.last_move = move
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
                    state[move] = self.player_number  # Undo first move
                    self.prev_state = state
                    self.last_move = move
                    return move
            state[move] = 0  # Undo first move

        # Step 2: Focus on neighbors of currently filled cells
        filled_positions = list(zip(*np.where((state == 1) | (state == 2))))  # Get filled positions
        neighbor_moves = set()
        # Get neighbors of all filled positions
        for pos in filled_positions:
            neighbors = get_neighbours(len(state), pos)  # Get neighbors of filled blocks
            for neighbor in neighbors:
                if state[neighbor] == 0:  # Only consider empty neighbors
                    neighbor_moves.add(neighbor)  # Add to the set to avoid duplicates

        decision = random.choices([0, 1, 2], [0.5, 0.5, 0.5], k=1)[0]
        # decision =0
        if decision == 0:  # Virtual Connection Move (AI or opponent)
            print("selcted VC")
            selected_move = None
            if random.choice([True, False]):  # Choose between AI or opponent virtual connection
                # Virtual connection for AI player
                for pos in filled_positions:
                    if state[pos] == self.player_number:
                        vc_moves = self.virtual_connection(state, pos)
                        for vc_move in vc_moves:
                            if state[vc_move] == 0:
                                selected_move = vc_move
                                break
                if selected_move:
                    self.last_move = selected_move
                    state[selected_move] = self.player_number
                    self.prev_state = state
                    return selected_move
            else:
                # Virtual connection for opponent 
                for pos in filled_positions:
                    if state[pos] == 3 - self.player_number:
                        vc_moves = self.virtual_connection(state, pos)
                        for vc_move in vc_moves:
                            if state[vc_move] == 0:
                                selected_move = vc_move
                                break
                if selected_move:
                    self.last_move = selected_move
                    state[selected_move] = self.player_number
                    self.prev_state = state
                    return selected_move

        elif decision == 1:  # Three-move win/loss detection
            opponent = 3 - self.player_number
            # Step 7: Check if the opponent can win in three moves and block it
            for move in neighbor_moves:
                state[move] = opponent
                second_moves = list(zip(*np.where(state == 0)))  # Remaining legal moves
                for second_move in second_moves:
                    state[second_move] = opponent
                    third_moves = list(zip(*np.where(state == 0)))
                    for third_move in third_moves:
                        state[third_move] = opponent
                        is_third_win, _ = check_win(state, third_move, opponent)
                        state[third_move] = 0  # Undo third move
                        if is_third_win:
                            state[second_move] = 0
                            state[move] = self.player_number
                            self.prev_state = state
                            self.last_move = move
                            self.last_move = move
                            return move
                    state[second_move] = 0
                state[move] = 0

            # AI's three-move win detection
            for move in neighbor_moves:
                state[move] = self.player_number
                second_moves = list(zip(*np.where(state == 0)))
                for second_move in second_moves:
                    state[second_move] = self.player_number
                    third_moves = list(zip(*np.where(state == 0)))
                    for third_move in third_moves:
                        state[third_move] = self.player_number
                        is_third_win, _ = check_win(state, third_move, self.player_number)
                        state[third_move] = 0
                        if is_third_win:
                            state[second_move] = 0
                            state[move] = self.player_number
                            self.prev_state = state
                            self.last_move = move
                            self.last_move = move
                            return move
                    state[second_move] = 0
                state[move] = 0

        # Step 5: If no immediate win or block, proceed with MCTS (40% chance)
        root = MCTSNode(state, self.player_number)
        best_state = self.uct_search(root)

        move_=tuple(zip(*np.where(state != best_state)))[0]
        state[move_] = self.player_number
        self.prev_state = state
        self.last_move = move_
        return move_

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
                node = node.best_selection_child()
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
