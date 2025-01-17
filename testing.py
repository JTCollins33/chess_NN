import torch
import numpy as np
import pygame as p
import chess
import sys
import time
import matplotlib.pyplot as plt
from player import Player
from display import Display
from environment import Environment


class TestGame:
    def __init__(self, device, board, cpu, display):
        self.device=device
        self.board = board
        self.p_turn=True
        self.cpu = cpu
        self.display=display
        self.env = Environment(device, board.board_fen())

    # def display_board(self):
    #     lines = self.board.board_fen().split('/')
    #     print("\n\n")
    #     print("     a  b  c  d  e  f  g  h")
    #     print("    ________________________")

    #     for i in range(1, len(lines)+1):
    #         print(str(9-i)+" | ", end="")
    #         for l in lines[i-1]:
    #             if(l.isdigit()):
    #                 for j in range(int(l)):
    #                     print(" . ", end="")
    #             else:
    #                 print(" "+l+" ", end="")
    #         print(" | "+str(9-i))

    #     print("    ________________________")
    #     print("     a  b  c  d  e  f  g  h")
    #     print("\n\n")

    def convert_locations_to_move(self, move_locations):
        if(move_locations[0]!=(-1,-1)):
            letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


            for move in self.board.legal_moves:
                if(letters[move_locations[0][1]] in str(move)[:2] and str(8-move_locations[0][0]) in str(move)[:2] and str(8-move_locations[1][0]) in str(move)[2:] and letters[move_locations[1][1]] in str(move)[2:]):
                    return str(move)

        print("\nThe selected move was not valid. Exiting now.")
        sys.exit()
        return ""


    def get_player_action(self):
        # self.display_board()
        move_locations = self.display.show_board(self.board.board_fen(), self.env, self.board)   

        move_str = self.convert_locations_to_move(move_locations)     

        # waiting=True
        # while(waiting):
        #     move_str = input("What is your move?  ")

        #     valid_move = False
        #     print("\n\n")
        #     for move in self.board.legal_moves:
        #         if(move_str==str(move)):
        #             valid_move=True
        #             waiting=False
        #             break

        return move_str
        

    def play(self):
        #reset board and have players reset
        self.board.reset_board()
        self.cpu.reset()

        while True:           
            """
            Have Player Go First
            """
            self.p_turn=True
            action_str = self.get_player_action()

            #make move on the board
            self.board.push_uci(action_str)

            #update board for CPU and player
            self.env.setup_board(self.board.board_fen())
            self.cpu.update_board()

            done = self.board.is_checkmate() or self.board.is_stalemate() or self.board.is_fivefold_repetition() or self.board.is_seventyfive_moves()

            """
            CPU's Turn
            """
            if(not(done)):
                self.p_turn = False
                #have player 2 make move
                cpu_action, action_str, cpu_possible_moves = self.cpu.select_action()

                #make move on board and update board
                self.board.push_uci(action_str)

                self.env.setup_board(self.board.board_fen())
                self.cpu.update_board()


            """
            Deciding Rewards and if Game is Over
            """
            done = self.board.is_checkmate() or self.board.is_stalemate() or self.board.is_fivefold_repetition() or self.board.is_seventyfive_moves()

            if(done):
                # self.display_board()
                _ = self.display.show_board(self.board.board_fen(), self.env)

                #if one player won, get good feedback to that player and bad feedback to the other
                if(self.board.is_checkmate()):
                    if(self.p_turn):
                        print("\n\nYou Won!!")
                    else:
                        print("\n\nCPU Won :(")


                #if game ended in stalemate, give each player negative reward
                elif(self.board.is_stalemate()):
                    print("\n\nGame Ended in Stalemate :(")

                #if five repeated moves, game ends and one player gets negative rewards
                elif(self.board.is_fivefold_repetition()):
                    print("\n\nGame Ended Because of Too Many Repetitions :(")
                
                #if there are 75 moves without a capture, penalize both players
                elif(self.board.is_seventyfive_moves()):
                    print("\n\nGame Ended Because 75 moves without a piece taken.")

            if(done):
                break


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cpu_num = int(input("Do you want to go first (0) or second(1)? "))
    # if cpu_num==0 : cpu_num=2
    cpu_num=2
    
    board = chess.Board()

    cpu = Player(device, board, player_num=cpu_num, load_trained_model=True)

    board_display = Display(dims=(8,8))

    game = TestGame(device, board, cpu, board_display)
    game.play()