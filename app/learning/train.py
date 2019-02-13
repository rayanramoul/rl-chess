import chess
import random
import json
import operator
import numpy as np
import pickle 
from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
import tensorflow as tf
import argparse
import os

# Create network. Input is two consecutive game states, output is Q-values of the possible moves.

model = Sequential()
model.add(Dense(20, input_shape=(65,) , init='uniform', activation='relu'))
model.add(Dense(18, init='uniform', activation='relu'))
model.add(Dense(18, init='uniform', activation='relu'))
model.add(Dense(18, init='uniform', activation='relu'))
model.add(Dense(18, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='relu'))    # Same number of outputs as possible actions
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
np.set_printoptions(threshold=np.inf)
state_board=np.zeros((1,65)) # The array representing the board of 8*8 and also a selected move from the possible ones
# Value of every piece
switch={
            'p':10,
            'P':-10,
            'q':90,
            'Q':-90,
            'n':30,
            'N':-30,
            'r':50,
            'R':-50,
            'b':30,
            'B':-30,
            'k':900,
            'K':-900,
            'None':0
        }
parser = argparse.ArgumentParser()
parser.add_argument('--number_of_games',type=float, default=100)
parser.add_argument('--winner_reward',type=float, default=1)
parser.add_argument('--loser_malus',type=float, default=-1)
parser.add_argument('--epsilon',type=float, default=1)
parser.add_argument('--decremental_epsilon',type=float, default=0.0001)
parser.add_argument('--gamma',type=float, default=0.05)
args = parser.parse_args()
arguments = {'training_games': args.number_of_games, 'winner_reward': args.winner_reward,'loser_malus': args.loser_malus, 'epsilon': args.epsilon,'decremental_epsilon': args.decremental_epsilon, 'gamma': args.gamma}
general_moves={}


steps=1000
training_games=int(arguments['training_games']) if (arguments['training_games'] is not None) else 100
winner_reward=int(arguments['winner_reward']) if (arguments['winner_reward'] is not None) else 1
loser_malus=int(arguments['loser_malus']) if (arguments['loser_malus'] is not None) else -1
epsilon = float(arguments['epsilon']) if (arguments['epsilon'] is not None) else 1                            # Probability of doing a random move
decremental_epsilon=float(arguments['decremental_epsilon']) if (arguments['decremental_epsilon'] is not None) else 1/training_games    # Each game we play we want to decrease the probability of random move
gamma = float(arguments['gamma']) if (arguments['gamma'] is not None) else 0.05                                # Discounted future reward. How much we care about steps further in time

print("Training the Deep-Q-Network with parameters : ")
print("Number of training games : "+str(training_games))
print("Winner Reward : "+str(winner_reward))
print("Loser Malus : "+str(loser_malus))
print("Epsilon : "+str(epsilon))
print("Decremental Epsilon : "+str(decremental_epsilon))
print("Gamma : "+str(gamma))

def evaluate_board(turn): # Evaluate the board following the value of each piece
    l=chess.SQUARES
    total=0
    if turn:
        mult=1
    else:
        mult=-1
    a=0
    b=0
    for i in l:
        total=total+(mult*switch[str(board.piece_at(i))])
        state_board[0][a]=mult*switch[str(board.piece_at(i))] # Update the state_board variable used for predictions
        a+=1
    return total
 
def get_int(move):  # Give the int representation(maping) of the move from the dictionary to give it as input for the deep neural network
    try:
        return general_moves[str(move)]
    except:
        general_moves[str(move)]=len(general_moves)
        return general_moves[str(move)]

def reward(fen_history, moves, lose_fen, lose_moves): # the final reward at the end of the game that reward each (state,move) of winner (and decrease the ones of loser).
    maxi=len(fen_history)
    i=0
    inputs=[]
    targets=[]
    while i<maxi:
        gamma=1/len(fen_history)
        fen_history[i][0][64]=get_int(moves[i])
        inputs.append(fen_history[i][0])
        model.train_on_batch(np.array(fen_history[i]),model.predict(np.array(fen_history[i]))+winner_reward*(gamma*i))
        i=i+1
    maxi=len(lose_fen)
    i=0
    while i<maxi:
        gamma=1/len(lose_fen)
        lose_fen[i][0][64]=get_int(lose_moves[i])
        model.train_on_batch(np.array(lose_fen[i]),model.predict(np.array(lose_fen[i]))+loser_malus*(gamma*i))
        i=i+1



winners={}    # Variable for counting number of wins of each player
for joum in range(0, steps):
    i=0
    evaluation_history=[]
    all_number_of_moves=[]            
    board=chess.Board()
    epsilon=1
    decremental_epsilon=1/training_games
    while i<training_games:
        os.system('clear')
        print("/------------------ Training -----------------/")
        print("Step ("+str(joum)+"/"+str(steps)+")")
        print("Game N°"+str(i))
        print("WINNERS COUNT : \n"+str(winners))
        print("Number of remaining training games : "+str(training_games-i))
        print("Winner Reward : "+str(winner_reward))
        print("Loser Malus : "+str(loser_malus))
        print("Epsilon : "+str(epsilon))
        print("Decremental Epsilon : "+str(decremental_epsilon))
        print("Gamma : "+str(gamma))
        fen_history=[]
        black_moves=[]
        white_moves=[]
        black_fen_history=[]
        white_fen_history=[]
        all_states=[]
        all_moves=[]
        number_of_moves=0
        evaluation_history=[]
        while not board.is_game_over():
            number_of_moves=number_of_moves+1
            if np.random.rand() <= epsilon:
                nmov=random.randint(0,board.legal_moves.count())
                cnt=0
                for k in board.legal_moves:
                    if cnt==nmov: 
                        god_damn_move = str(k)
                    cnt+=1
                        
            else:
                #print("q move")
                evaluate_board(True)
                Q={}
                for kr in board.legal_moves:
                    br=get_int(kr)
                    state_board[0][64]=br
                    #print(str([state_board]))
                    Q[kr]=model.predict(state_board)          # Q-values predictions for every action possible with the actual state
                god_damn_move = max(Q.items(), key=operator.itemgetter(1))[0] # Get the movest with the highest Q-value
            base_evaluation=evaluate_board(board.turn)
            fen=str(board.fen())
            all_states.append(np.array(state_board,copy=True))
            all_moves.append(np.array(god_damn_move,copy=True))
            evaluation_history.append(base_evaluation)
            if board.turn:
                white_moves.append(god_damn_move)
                white_fen_history.append(np.array(state_board,copy=True))
            else:
                black_moves.append(god_damn_move)
                black_fen_history.append(np.array(state_board,copy=True))
            board.push(chess.Move.from_uci(str(god_damn_move)))
        all_number_of_moves.append(number_of_moves)
        
        i=i+1
        if board.result()=="1-0":
            reward(white_fen_history, white_moves, black_fen_history, black_moves)
        elif board.result()=="0-1":
            reward(black_fen_history, black_moves, white_fen_history, white_moves)
        try:
            winners[str(board.result())]=winners[str(board.result())]+1
        except:
            winners[str(board.result())]=1
        board.reset()
        epsilon-=decremental_epsilon 
#        print("White moves : ")
#        print(white_moves)
#        print(" White states ")
#        print(white_fen_history)

print("WINNERS COUNT : \n"+str(winners))
#tf.clear_session()
with open('generalized_moves.json', 'w') as fp:   # Save the mapping Move/Index to be used on developement
    json.dump(general_moves, fp)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
graph = tf.get_default_graph()
with graph.as_default():
    model.save_weights("model.h5")

