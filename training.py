import torch
import numpy as np
import chess
import matplotlib.pyplot as plt
from player import Player

MAX_MOVES=100

class Game:
    def __init__(self, device, board, p1, p2, episodes=10000, save_freq=1000, update_freq=10):
        self.device=device
        self.board = board
        self.episodes=episodes
        self.p1_turn=True
        self.player1 = p1
        self.player2 = p2
        self.reward_scale=0.1
        self.save_freq=save_freq
        self.update_freq=update_freq

    def plot_episode_rewards(self, rewards, player):
        plt.clf()
        plt.plot(rewards)
        plt.title("Episode Rewards Over "+str(len(rewards))+" Episodes for "+player.capitalize())
        plt.xlabel("Episode Number")
        plt.ylabel("Reward")
        plt.savefig("./reward_plots/"+player+"/episode_rewards_plot_"+str(len(rewards))+"_epochs.png")

    def play(self):
        p1_wins=0
        p1_rewards=[]
        p2_wins=0
        p2_rewards=[]

        for i_episode in range(12400+1, 12400+self.episodes+1):
            num_moves=0
            #reset board and have players reset
            self.board.reset_board()
            self.player1.reset()
            self.player2.reset()

            while True:
                print("Game "+str(i_episode)+"/"+str(self.episodes+12400)+"\tMoves: "+str(num_moves), end='\r')
                self.player1.reward=0.0
                self.player2.reward=0.0
                """
                Player 1's Turn
                """
                self.p1_turn=True

                before_p1_state = self.player1.env.get_view()

                #have player select move
                p1_action, action_str, p1_possible_moves = self.player1.select_action()

                #make move on the board
                self.board.push_uci(action_str)

                if(self.board.is_check()):
                    self.player1.reward+=1.0

                num_moves+=1

                #have both players update board
                self.player1.update_board()
                self.player2.update_board()

                after_p1_state = self.player1.env.get_view()

                done = self.board.is_checkmate() or self.board.is_stalemate() or self.board.is_fivefold_repetition() or self.board.is_seventyfive_moves()


                """
                Player 2's Turn
                """
                if(not(done)):
                    self.p1_turn = False

                    before_p2_state = self.player2.env.get_view()
                    #have player 2 make move
                    p2_action, action_str, p2_possible_moves = self.player2.select_action()

                    #make move on board and update board
                    self.board.push_uci(action_str)
                    num_moves+=1

                    if(self.board.is_check()):
                        self.player2.reward+=1.0

                    self.player1.update_board()
                    self.player2.update_board()

                    after_p2_state= self.player2.env.get_view()




                """
                Deciding Rewards and if Game is Over
                """
                done = self.board.is_checkmate() or self.board.is_stalemate() or self.board.is_fivefold_repetition() or self.board.is_seventyfive_moves()

                #if game not over, compute reward for each player based on how many pieces won/lost
                if(not(done)):
                    p1_loss = self.player1.compute_losses()
                    p2_loss = self.player2.compute_losses()

                    #now define reward for each player
                    self.player1.reward+= (-1.0*p2_loss)+p1_loss
                    self.player2.reward+= (-1.0*p1_loss)+p2_loss

                else:
                    #if one player won, get good feedback to that player and bad feedback to the other
                    if(self.board.is_checkmate()):
                        if(self.p1_turn):
                            self.player1.reward+=100.0
                            self.player2.reward+=-100.0
                            p1_wins+=1
                        else:
                            self.player1.reward+=-100.0
                            self.player2.reward+=100.0
                            p2_wins+=1


                    #if game ended in stalemate, give each player negative reward
                    elif(self.board.is_stalemate()):
                        self.player1.reward+=-50.0
                        self.player2.reward+=-50.0

                    #if five repeated moves, game ends and one player gets negative rewards
                    elif(self.board.is_fivefold_repetition()):
                        #penalize player who keeps repeating move, giving nothing to other player
                        if(self.p1_turn):
                            self.player1.reward+=-50.0
                            self.player2.reward+=0.0
                        else:
                            self.player1.reward+=0.0
                            self.player2.reward+=-50.0
                    
                    #if there are 75 moves without a capture, penalize both players
                    elif(self.board.is_seventyfive_moves()):
                        self.player1.reward+=-50.0
                        self.player2.reward+=-50.0



                #scale player rewards
                self.player1.reward*=self.reward_scale
                self.player2.reward*=self.reward_scale

                self.player1.episode_reward+=self.player1.reward
                self.player2.episode_reward+=self.player2.reward


                """
                Adding to Memory and optimizing
                """

                #add moves to each players' memories
                self.player1.memory.push(before_p1_state, p1_possible_moves, p1_action, after_p1_state, torch.tensor([self.player1.reward], device=self.device))
                self.player2.memory.push(before_p2_state, p2_possible_moves, p2_action, after_p2_state, torch.tensor([self.player2.reward], device=self.device))

                #optimize each player's networks
                self.player1.optimize()
                self.player2.optimize()

                if(done):
                    break


                
            p1_rewards.append(self.player1.reward)
            p2_rewards.append(self.player2.reward)


            #transfer policy models over to target models when updating
            if(i_episode%self.update_freq==0):
                self.player1.target_net.load_state_dict(self.player1.policy_net.state_dict())
                self.player2.target_net.load_state_dict(self.player2.policy_net.state_dict())

                
            #save models and create plots ever save_freq_epochs
            if(i_episode%self.save_freq==0):
                torch.save(self.player1.policy_net.state_dict(), "./model_progress/player1/policy_model_"+str(i_episode)+"_episodes.pt")
                torch.save(self.player2.policy_net.state_dict(), "./model_progress/player2/policy_model_"+str(i_episode)+"_episodes.pt")

                self.plot_episode_rewards(p1_rewards, "player1")
                self.plot_episode_rewards(p2_rewards, "player2")



if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board = chess.Board()

    cpu1 = Player(device, board, player_num=1, memory_size=10000, batch_size=32, max_moves=MAX_MOVES, load_trained_model=True)
    cpu2 = Player(device, board, player_num=2, memory_size=10000, batch_size=32, max_moves=MAX_MOVES, load_trained_model=True)

    game = Game(device, board, cpu1, cpu2, episodes=10000, save_freq=100)
    game.play()