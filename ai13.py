from re import S
from shutil import which
import time
import math
import random
import numpy as np
from helper import *
import sys
from queue import Queue

class MCTSNode:
    def __init__(self, state, player_number, parent=None, k_value=0.07, tn=0.01, tw_prime=0.25, p=0.8):
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
        self.rave_visits = 0
        self.rave_wins = 0
        self.k_value=k_value
         # PPR-related parameters
        self.tn = tn
        self.tw_prime = tw_prime
        self.p = p

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
                # RAVE Formula
                if child.rave_visits > 0:
                    rave_score = child.rave_wins / child.rave_visits
                    β = child.visits / (child.visits + child.rave_visits + 1e-6)  # Small epsilon to avoid division by zero
                    uct_score = (1 - β) * uct_score + β * rave_score
                
                # if self.visits > 0:
                #     knowledge_term = (child.visits * self.k_value) / math.sqrt(child.visits * self.visits)
                #         print("buzzinga")  uct_score += knowledge_term
                # Add heuristic for proximity to filled cells
                if (len(self.state)+1)//2 >=5:
              
                    
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
        filled_positions = list(zip(*np.where(state ==self.player_number)))  # Get positions of filled cells (non-zero entries)

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

    def prune_moves_based_on_rave(self, node):
        """
        Prune moves based on RAVE statistics. Only keep moves with a RAVE winning rate higher than the threshold.
        :param node: The current MCTS node.
        :return: A pruned list of moves.
        """
   
        pruned_moves = []
        
        # Move up the tree to find a suitable parent node with sufficient playouts
        while node.visits < self.tn * sum(child.visits for child in node.children):
            if node.parent is None:
                break
            node = node.parent

        # Prune moves based on the RAVE winning rate
        for child in node.children:
            rave_win_rate = child.rave_wins / (child.rave_visits + 1e-6)  # Avoid division by zero
            if rave_win_rate > self.tw_prime:
                pruned_moves.append(child)
        
        return pruned_moves


    def simulate(self, player, num_rollouts=1):
        """
        Simulate multiple rollouts for the current player, using Playout Pruning with RAVE (PPR).
        
        :param player: The current player to simulate for.
        :param num_rollouts: Number of rollouts to run.
        :return: Aggregated result from rollouts: (player who wins, average path length, combined rave actions).
        """
        aggregated_result = {1: 0, 2: 0}  # To track the number of wins for each player
        total_path_length = 0  # To track the total path lengths
        aggregated_rave_actions = set()  # To track the rave actions from all rollouts
        
        for _ in range(num_rollouts):
            current_state = self.state.copy()
            current_player = self.player_number
            rave_actions = set()  # Track actions taken in this rollout

            legal_moves = list(zip(*np.where(current_state == 0)))
            
            # While the game is not at a terminal state
            while len(legal_moves) > 0:
                # Prune moves based on RAVE statistics
                pruned_moves = self.prune_moves_based_on_rave(self)

                # Choose a move either from the pruned list or randomly based on probability 'p'
                if random.random() <= self.p and pruned_moves:
                    move = random.choice(pruned_moves).state  # Choose from pruned moves
                else:
                    move = random.choice(legal_moves)  # Choose randomly

                rave_actions.add(move)
                current_state[move] = current_player

                # Check for a win
                win, win_type = check_win(current_state, move, current_player, [])
                if win:
                    if win_type == 'ring':
                        path = find_ring(current_state, move)
                    elif win_type == 'fork':
                        path = find_fork(current_state, move)
                    else:
                        path = find_bridge(current_state, move)

                    # Add result to the aggregate
                    aggregated_result[current_player] += 1
                    total_path_length += len(path)
                    aggregated_rave_actions.update(rave_actions)
                    break

                current_player = 3 - current_player  # Switch player
                legal_moves = list(zip(*np.where(current_state == 0)))

            # If no one won (end of simulation), increment based on the remaining state
            if len(legal_moves) == 0:
                winner = 1 if np.sum(current_state) % 2 == 1 else 2
                aggregated_result[winner] += 1

        # Calculate average result
        final_winner = 1 if aggregated_result[1] > aggregated_result[2] else 2
        average_path_length = total_path_length // num_rollouts
        
        return final_winner, average_path_length, aggregated_rave_actions


    
    def backpropagate(self, result,val, rave_actions):
        self.visits += 1
        if result == self.player_number:
            self.wins += val

        for move in rave_actions:
            if move in self.children:
                child = self.children[move]
                child.rave_visits += 1
                if result == self.player_number:
                    child.rave_wins += val
        if self.parent:
            self.parent.backpropagate(result,val,rave_actions)

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
        self.total = 0
        

        pass
    
    def is_valid_this(self,a,b,len_state):
        if(a < 0 or a >= len_state or b < 0 or b >= len_state):
            return False
        else:
            if(a <= len_state//2):
                return True
            else:
                x = a - len_state//2
                if(b < x or b >= len_state - x):
                    return False
                else:
                    return True
    
    def threat_VC(self,state,legal_moves,player,disjoint_set_dict,union_find):
        
        threat = []
        
        for move_ in legal_moves:
            corn_set = set()
            edge_set = set()
            q = Queue()
            q.put(move_)
            visited = set([move_]) 
            near = set()
            while(not(q.empty())):
                move = q.get()
                vcs = self.virtual_connection(state,move)
                for m in vcs:
                    if(m in visited):
                        continue
                    mv1 = None
                    mv2 = None
                    if(state[m] == (player)):
                        if(abs(m[0] - move[0]) ==2):
                            mv1 = ((m[0] + move[0])//2,m[1])
                            mv2 = ((m[0] + move[0])//2,move[1])
                        elif(abs(m[1] - move[1]) ==2):
                            if(m[0] == move[0]):
                                mv1 = (m[0],(m[1] + move[1])//2)
                                mv2 = (m[0]+1,(m[1] + move[1])//2)
                            else:
                                mv1 = (m[0],(m[1] + move[1])//2)
                                mv2 = (move[0],(m[1] + move[1])//2)  
                        else:
                            mv1 = (m[0],move[1])
                            mv2 = (move[0],m[1])
                        if(state[mv1] == 0 and state[mv2] ==0):
                            visited.add(m)
                            q.put(m)
                            near.add(union_find.find(m))
                ed1 = None
                for neigh in get_neighbours(len(state),move):
                    x = get_edge(neigh,len(state))
                    if(x != -1):
                        if(state[neigh] == 0):
                            if(ed1 != None):
                                ed2 = x
                                edge_set.add(x)
                            else:
                                ed1 = x
                    if(neigh in visited):
                        continue
                    if(state[neigh] == (player)):
                        visited.add(neigh)
                        q.put(neigh)
                        near.add(union_find.find(neigh))
                    

            
            
            for elem in near:
                corn_set.update(disjoint_set_dict[elem]['corner'])
                edge_set.update(disjoint_set_dict[elem]['edge'])
            corn = get_corner(move_,len(state))
            edge = get_edge(move_,len(state))
            if(corn != -1):
                corn_set.add(corn)
            if(edge != -1):
                edge_set.add(edge)
            if(len(corn_set) >= 2 or len(edge_set) >= 3):
                threat_ = math.inf
                threat.append(threat_)
            elif(len(corn_set) ==0 and len(edge_set) == 0):
                threat_ = -(math.inf)
                threat.append(threat_)
            else:
                threat_ = 33 * len(edge_set) + 50 * len(corn_set)
                threat.append(threat_)
            state[move_] = 0
        return threat
   
    def virtual_connection(self,state,move):
        x,y = move
        dim = (len(state)+1)//2
        vc = []
        if(y < dim - 1):
            vc = [(x+2,y+1),(x+1,y-1),(x-1,y-2),(x-2,y-1),(x-1,y+1)] 
            if(y < dim - 2):
                vc.append((x+1,y+2))
            else:
                vc.append((x,y+2))
        elif(y > dim - 1):
            vc = [(x+2,y-1),(x+1,y+1),(x-1,y+2),(x-2,y+1),(x-1,y-1)] 
            if(y > dim):
                vc.append((x+1,y-2))
            else:
                vc.append((x,y-2))
        else:
            vc = [(x-1,y-2),(x-2,y-1),(x-2,y+1),(x-1,y+2),(x+1,y+1),(x+1,y-1)]
        vc_out = []
        for i in range(6):
            a,b = vc[i]
            if(self.is_valid_this(a,b,len(state))):
                vc_out.append(vc[i])
        return vc_out
   
                
    def add_move_to_union_find2(self, move,uf_object):
        """
        Add the AI's move to the Union-Find structure and connect it with its neighbors.
        """
        if(move == None):
            return
        uf_object.add(move)
        neighbors = get_neighbours((2*self.dim) -1, move)
        
        # Connect with neighboring AI player's cells
        for neighbor in neighbors:
            if neighbor in uf_object.parent and uf_object.find(neighbor) != uf_object.find(move):
                uf_object.union(move, neighbor)

    def update_adjoint_set(self,state,last_move,player,disjoint_set_dict,uf_object):
        neighbors = get_neighbours(len(state),last_move)
        corner_last_move = get_corner(last_move,len(state))
        edge_last_move = get_edge(last_move,len(state))
        corner_set = set()
        edge_set = set()
        if(corner_last_move != -1):
            corner_set.add(corner_last_move)
        if(edge_last_move != -1):
            edge_set.add(edge_last_move)
            
        for neigh in neighbors:
            if(state[neigh] == player):
                par = uf_object.find(neigh)
                if par in disjoint_set_dict:
                    corner_par = disjoint_set_dict[par]['corner']
                    edge_par = disjoint_set_dict[par]['edge']
                    del disjoint_set_dict[par]
                    corner_set.update(corner_par)
                    edge_set.update(edge_par)
        disjoint_set_dict[last_move] = {'corner' : corner_set, 'edge' : edge_set}
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
            self.simulations = 600* self.dim
            self.total = len(list(zip(*np.where(state != 3))))
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
                self.opp_uf.add(opp[0]) 
            move = (0,0)
            self.selected_corner = 0
            for i in range(6):
                corner = self.corners[i]
                if state[corner] == (3-self.player_number):  # Check if the corner is occupied by opponent, if it is then select the next corner
                    self.selected_corner = (i+1)%6
                    move = self.corners[(i+1)%6]
                    break
            self.last_move = move
            state[move] = self.player_number
            self.prev_state = state
            return move
        
        opponent_last_move = list(zip(*np.where(state != self.prev_state)))[0]
        self.update_adjoint_set(state,self.last_move,self.player_number,self.ai_disjoint,self.union_find)
        self.update_adjoint_set(state,opponent_last_move,3 - self.player_number,self.opp_disjoint,self.opp_uf)
        self.add_move_to_union_find2(self.last_move,self.union_find)     
        self.add_move_to_union_find2(opponent_last_move,self.opp_uf)
        
        if(self.move == 2 and self.dim >=5):
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

        print(f"starting threat function {self.player_number}")
        threat = self.threat_VC(state,legal_moves,opponent,self.opp_disjoint,self.opp_uf)
        my_threat = self.threat_VC(state,legal_moves,self.player_number,self.ai_disjoint,self.union_find)
        
        max_value = max(threat)
        max_value_mine = max(my_threat)
        if(max_value > 83):
            print(f"played against opponent most threat {max_value}")
            max_index = threat.index(max_value)
            move_ = legal_moves[max_index]
            state[move_] = self.player_number
            self.prev_state = state
            self.last_move = move_
            return move_
        
        if(max_value_mine == math.inf):
            print(f"played my mine most threat {max_value_mine}")
            max_index = my_threat.index(max_value_mine)
            move_ = legal_moves[max_index]
            state[move_] = self.player_number
            self.prev_state = state
            self.last_move = move_
            return move_

        
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

        decision = random.choices([0, 1, 2], [0.0, 0.0, 1], k=1)[0]
        

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

        elif self.move >= self.total//2: # Three-move win/loss detection
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

           
           

        # Step 5: If no immediate win or block, proceed with MCTS (40% chance)
        root = MCTSNode(state, self.player_number, k_value=0.07)
        best_state = self.uct_search(root, num_rollouts=1)

        move_=tuple(zip(*np.where(state != best_state)))[0]
        state[move_] = self.player_number
        self.prev_state = state
        self.last_move = move_
        return move_

    def uct_search(self, root: MCTSNode, num_rollouts=1):
        """
        Perform UCT (Upper Confidence Tree) Search to find the best move using a fixed number of simulations.

        :param root: The root MCTSNode representing the current game state.
        :param num_rollouts: Number of rollouts to run in each simulation.
        :return: The best state after searching.
        """
        for _ in range(self.simulations):
            print(_)
            node, player = self.selection(root)
            result, val, rave_actions = node.simulate(player, num_rollouts)  # Simulate with multiple rollouts
            l, b = (node.state).shape

            if val == 0:
                node.backpropagate(result, 0, rave_actions)  # Pass rave_actions to backpropagate
            else:
                node.backpropagate(result, (l * b // 2) - val, rave_actions)  # Pass rave_actions to backpropagate

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
