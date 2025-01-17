import pygame as p
import sys

WIDTH=HEIGHT=512


class Display:
    def __init__(self, dims=(8,8), width=512, height=512, max_fps=15):
        # p.init()
        self.dims=dims
        self.width=width
        self.height=height
        self.max_fps=max_fps
        self.square_size = int(width/dims[0])
        self.image_dir = "./images/"
        self.images={}
        self.screen=None
        # self.screen = p.display.set_mode((self.width, self.height))
        # self.screen.fill(p.Color("white"))
        self.clock = p.time.Clock()


        self.load_images()

    #load all images in at beginning so don't have to do it each time
    def load_images(self):
        self.images['p'] = p.transform.scale(p.image.load("./images/bp.png"), (self.square_size, self.square_size))
        self.images['r'] = p.transform.scale(p.image.load("./images/br.png"), (self.square_size, self.square_size))
        self.images['n'] = p.transform.scale(p.image.load("./images/bn.png"), (self.square_size, self.square_size))
        self.images['b'] = p.transform.scale(p.image.load("./images/bb.png"), (self.square_size, self.square_size))
        self.images['q'] = p.transform.scale(p.image.load("./images/bq.png"), (self.square_size, self.square_size))
        self.images['k'] = p.transform.scale(p.image.load("./images/bk.png"), (self.square_size, self.square_size))
        self.images['P'] = p.transform.scale(p.image.load("./images/wp.png"), (self.square_size, self.square_size))
        self.images['R'] = p.transform.scale(p.image.load("./images/wr.png"), (self.square_size, self.square_size))
        self.images['N'] = p.transform.scale(p.image.load("./images/wn.png"), (self.square_size, self.square_size))
        self.images['B'] = p.transform.scale(p.image.load("./images/wb.png"), (self.square_size, self.square_size))
        self.images['Q'] = p.transform.scale(p.image.load("./images/wq.png"), (self.square_size, self.square_size))
        self.images['K'] = p.transform.scale(p.image.load("./images/wk.png"), (self.square_size, self.square_size))
        self.images['ds'] = p.transform.scale(p.image.load("./images/square_dark.jpg"), (self.square_size, self.square_size))
        self.images['ls'] = p.transform.scale(p.image.load("./images/square_light.jpg"), (self.square_size, self.square_size))

    def check_picks(self, move_locations, board):
        valid = False
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        for move in board.legal_moves:
                if(letters[move_locations[0][1]] in str(move)[:2] and str(8-move_locations[0][0]) in str(move)[:2] and str(8-move_locations[1][0]) in str(move)[2:] and letters[move_locations[1][1]] in str(move)[2:]):
                    valid=True
                    break
        return valid


    def show_board(self, board_fen, env, board):
        p.init()
        self.screen = p.display.set_mode((self.width, self.height))
        self.screen.fill(p.Color("white"))

        p.event.clear()

        square_selected=() # keeps track of last click by user
        player_clicks = [] #keeps track fo player clicks
        while True:
            self.draw_board()
            self.draw_pieces(board_fen)
            if(len(player_clicks)==1):
                self.highlight_squares(player_clicks[0], env)

            event = p.event.wait()
            if(event.type==p.QUIT):
                p.quit()
                sys.exit()

            #if user clicks on piece to move it
            elif(event.type == p.MOUSEBUTTONDOWN):
                #get x and y location of mouse
                location = p.mouse.get_pos()
                col = location[0]//self.square_size
                row = location[1]//self.square_size

                #make sure user doesn't click on same square twice
                if(square_selected==(row, col)):
                    square_selected=() #unselect square
                    player_clicks = [] #clear player clicks

                else:
                    square_selected = (row, col)

                    if(env.state[7-row, col]<=0 and len(player_clicks)==0):
                        print("Error: Must click on own piece for first click.\n")
                    else:
                        player_clicks.append(square_selected)
                
                #if user clicked twice, make move and return selected squares
                if(len(player_clicks)==2):
                    valid_picks = self.check_picks(player_clicks, board)
                    #if pick is valid, return it
                    if(valid_picks):
                        return player_clicks
                    #else tell user not valid and reset picks
                    else:
                        print("That is not a valid move. Please make a valid move.")
                        square_selected=()
                        player_clicks=[]

            self.clock.tick(self.max_fps)
            p.display.flip()
        
        return [(-1,-1), (-1,-1)]
    
    #draw square on board
    def draw_board(self):
        colors = [p.Color("white"), p.Color("gray")]
        for row in range(self.dims[0]):
            for col in range(self.dims[1]):
                color = colors[(row+col)%2]
                p.draw.rect(self.screen, color, p.Rect(col*self.square_size, row*self.square_size, self.square_size, self.square_size))


    #draw pieces on board
    def draw_pieces(self, board_fen):
        lines = board_fen.split("/")
        for i in range(len(lines)):
            line = lines[i]
            offset=0
            for j in range(len(line)):
                if(line[j].isdigit()):
                    offset+=int(line[j])
                else:
                    piece = line[j]
                    self.screen.blit(self.images[piece], p.Rect(offset*self.square_size, i*self.square_size, self.square_size, self.square_size))
                    offset+=1


        # for row in range(self.dims[0]):
        #     for col in range(self.dims[1]):
        #         piece = board[row][col]

        #         #if piece is not an empty square, draw it
        #         if(piece != "."):
        #             self.screen.blit(self.images[piece], p.Rect(col*self.square_size, row*self.square_size, self.square_size, self.square_size))

    #highlight square selected and moves for piece selected
    def highlight_squares(self, square_selected, env):
        row, col = square_selected

        #higlight square selected first
        s = p.Surface((self.square_size, self.square_size))
        s.set_alpha(100)  #transparency value --> 0 = transparent and 255 = solid
        s.fill(p.Color("blue"))
        self.screen.blit(s, (col*self.square_size, row*self.square_size))



if __name__=="__main__":
    display = Display()
    display.show_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")