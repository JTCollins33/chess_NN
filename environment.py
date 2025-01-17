import torch
import numpy as np

class Environment:
    def __init__(self, device, starting_fen, player_num=1):
        self.device=device
        self.state=np.zeros((8,8))
        self.dict={'p': -1, 'r': -2, 'n':-3, 'b':-4, 'q':-5, 'k':-6, 'P':1, 'R':2, 'N':3, 'B':4, 'Q':5, 'K':6}
        self.starting_fen=starting_fen
        self.pnum = player_num
        self.setup_board(starting_fen)

    def setup_board(self, board_fen):
        self.state=np.zeros((8,8))
        lines = board_fen.split('/')
        for i in range(len(lines)):
            line = lines[len(lines)-(i+1)]
            if(line!='8'):
                offset=0
                for j in range(len(line)):
                    if(line[j].isdigit()):
                        offset+=int(line[j])
                    else:
                        self.state[i,offset]=self.dict[line[j]]
                        offset+=1

        # #have player1 see a mirror image of board
        # if(self.pnum==2):
        #     self.state*=-1.0
        #     self.state = np.flip(self.state)

    def get_view(self):
        view = self.state.copy()
        view_t = torch.tensor([view], device=self.device)
        return view_t.unsqueeze(0)
