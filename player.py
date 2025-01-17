import torch
import os
import torch.nn.functional as func
import numpy as np
import collections
from environment import Environment
from memory import ReplayMemory, Transition
from model import DQN


def get_newest_model(models_path):
    files = os.listdir(models_path)
    paths = [os.path.join(models_path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

class Player:
    def __init__(self, device, board, player_num=1, memory_size=10000, batch_size=32, gamma=0.999, max_moves=100, load_trained_model=False):
        self.device=device
        self.board = board
        self.pnum=player_num
        self.reward=0.0
        self.max_moves=max_moves
        self.gamma = gamma
        self.episode_reward=0.0
        self.batch_size=batch_size

        self.env = Environment(device, board.board_fen(), player_num=player_num)
        self.memory = ReplayMemory(memory_size)

        self.policy_net = DQN(max_moves).to(self.device)
        self.target_net = DQN(max_moves).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters())

        if(load_trained_model):
            newest_model = get_newest_model("./model_progress/player"+str(player_num)+"/")
            print("Loading Model "+str(newest_model))
            self.policy_net.load_state_dict(torch.load(newest_model))

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.pieces_score = self.get_pieces_score()
        self.letter_number_dict = {}
        self.setup_dict()

    def reset(self):
        self.reward=0.0
        self.episode_reward=0.0
        self.update_board()

    def setup_dict(self):
        cnt=0
        for i in range(1, 9):
            for l in ['a','b', 'c','d', 'e', 'f', 'g', 'h']:
                self.letter_number_dict[l+str(i)]=cnt
                cnt+=1

    def get_pieces_score(self):
        sum = 0
        for i in range(1, 7):
            for x in range(self.env.state.shape[0]):
                for y in range(self.env.state.shape[1]):
                    #player one has positive pieces and player two has negative pieces
                    if(self.pnum==1):
                        if(self.env.state[x,y]==i):
                            sum+=i
                    elif(self.pnum==2):
                        if(self.env.state[x,y]==(-1.0*i)):
                            sum+=i
        return sum

    #method to convert move string to a number for network
    def convert_move_to_num(self, move_str):
        move_encoding = torch.zeros((2, 64))

        from_index = self.letter_number_dict[move_str[:2]]

        #get index of number within string
        to_string = move_str[2:]
        num_index=0
        for i in range(len(to_string)):
            if(to_string[i].isdigit()):
                num_index=i
        #to move will always be where number is in string and one before it
        to_index = self.letter_number_dict[to_string[num_index-1:num_index+1]]

        move_encoding[0][from_index]=1.0
        move_encoding[1][to_index]=1.0
        return move_encoding

    def convert_action_to_string(self, action):   
        #find total number of possible moves (added padding to make same size)      
        possible_move_count=self.board.legal_moves.count()

        #get move indices predicted by network
        action_indices = torch.topk(action, self.max_moves, dim=1).indices[0].tolist()

        #make sure move is valid (included in possible move count)
        while(action_indices[0]>=possible_move_count):
            action_indices.pop(0)

        #now have valid chosen move
        chosen_index = action_indices[0]

        action = torch.tensor([[chosen_index]], device=self.device)
        chosen_move_str = ""
        cnt=0
        for move in self.board.legal_moves:
            #find which of the legal moves the network wanted
            if(cnt==chosen_index):
                #now found move, convert to right notation
                from_int = self.letter_number_dict[str(move)[:2]]

                piece_value = self.env.state[int(from_int/8), from_int%8]

                #if piece is not a pawn (piece value != 1), need to add piece info to string
                if(abs(piece_value)!=1):
                    chosen_key = ""
                    for key, value in self.env.dict.items():
                        if(value==abs(piece_value)):
                            chosen_key = key
                            break
                    chosen_move_str= chosen_key+str(move)[2:]
                
                #if just moving a pawn, just have to specify end location
                else:
                    chosen_move_str=str(move)[2:]
                break
            cnt+=1
        return action, chosen_move_str

    def get_selected_action(self, action):
        #find total number of possible moves (added padding to make same size)      
        possible_move_count=self.board.legal_moves.count()

        #get move indices predicted by network
        action_indices = torch.topk(action, self.max_moves, dim=1).indices[0].tolist()

        #make sure move is valid (included in possible move count)
        while(action_indices[0]>=possible_move_count):
            action_indices.pop(0)

        #now have valid chosen move
        chosen_index = action_indices[0]

        action = torch.tensor([[chosen_index]], device=self.device)
        chosen_move_str = ""
        cnt=0
        for move in self.board.legal_moves:
            #find which of the legal moves the network wanted
            if(cnt==chosen_index):
                chosen_move_str=str(move)
                break
            cnt+=1
        return action, chosen_move_str


    def select_action(self):
        #get all possible moves, padding with zeros to make sure input is same size
        possible_moves = torch.zeros((self.max_moves, 2, 64))

        i=0
        for move in self.board.legal_moves:
            move_vector = self.convert_move_to_num(str(move))
            if(i>=self.max_moves):
                break
            possible_moves[i, :, :]=move_vector
            i+=1

        possible_moves = possible_moves.view(1, -1)

        #make model totallly deterministic for now
        action = self.policy_net(self.env.get_view(), possible_moves)

        # action_t, action_str = self.convert_action_to_string(action)
        action_t, action_str = self.get_selected_action(action)

        return action_t, action_str, possible_moves
        

    def update_board(self):
        self.env.setup_board(self.board.board_fen())

    #this function returns the value of all of the pieces lost on the previous turn
    def compute_losses(self):
        #get total value of pieces left
        new_pieces_score = self.get_pieces_score()

        #get difference in score
        loss = new_pieces_score-self.pieces_score

        #set new total value of pieces left
        self.pieces_score=new_pieces_score
        return loss

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        #use all next states for now
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                         batch.next_state)), device=self.device, dtype=torch.uint8)
        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        possible_moves_batch = torch.cat(batch.possible_moves)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        state_action_values = self.policy_net(state_batch, possible_moves_batch).gather(1, action_batch)

        # next_state_values = torch.zeros(self.batch_size, device=self.device)
        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        next_state_action_values = self.target_net(next_state_batch, possible_moves_batch).max(1)[0].detach()

        expected_state_action_values = (next_state_action_values * self.gamma) + reward_batch

        loss = func.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
