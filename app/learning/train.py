import chess
import random
import json
import operator
import numpy as np
import pickle 
from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs

# Create network. Input is two consecutive game states, output is Q-values of the possible moves.

model = Sequential()
model.add(Dense(20, input_shape=(65,) , init='uniform', activation='relu'))
model.add(Dense(18, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='relu'))    # Same number of outputs as possible actions
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

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

general_moves={}
training_games=1000000
winner_reward=1
loser_malus=-1
epsilon = 0.98                              # Probability of doing a random move
gamma = 0.05                                # Discounted future reward. How much we care about steps further in time

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
        state_board[0][a]=switch[str(board.piece_at(i))] # Update the state_board variable used for predictions
        a+=1
    return total
 
def get_int(move):  # Give the int representation(maping) of the move from the dictionary to give it as input for the deep neural network
    try:
        return general_moves[str(move)]
    except:
        general_moves[str(move)]=len(general_moves)
        return general_moves[str(move)]

def maj_winner(fen_history, moves, lose_fen, lose_moves): # the final reward at the end of the game that reward each (state,move) of winner (and decrease the ones of loser).
    maxi=len(fen_history)
    i=0
    inputs=[]
    targets=[]
    while i<maxi:
        fen_history[i][0][64]=get_int(moves[i])
        inputs.append(fen_history[i][0])
        model.train_on_batch(np.array(fen_history[i]),model.predict(np.array(fen_history[i]))+winner_reward*(gamma*i))
        i=i+1
    maxi=len(lose_fen)
    i=0
    while i<maxi:
        lose_fen[i][0][64]=get_int(lose_moves[i])
        model.train_on_batch(np.array(lose_fen[i]),model.predict(np.array(lose_fen[i]))+loser_malus*(gamma*i))
        i=i+1

i=0
evaluation_history=[]
all_number_of_moves=[]
winners={}                                 # Variable for counting number of wins of each player
board=chess.Board()
while i<training_games:
    print("Game N°"+str(i))
    fen_history=[]
    black_moves=[]
    white_moves=[]
    black_fen_history=[]
    white_fen_history=[]
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
            print("q move")
            evaluate_board(True)
            Q={}
            for kr in board.legal_moves:
                br=get_int(kr)
                state_board[0][64]=br
                print(str([state_board]))
                Q[kr]=model.predict(state_board)          # Q-values predictions for every action possible with the actual state
            god_damn_move = max(Q.items(), key=operator.itemgetter(1))[0] # Get the movest with the highest Q-value
        base_evaluation=evaluate_board(board.turn)
        fen=str(board.fen())
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
        maj_winner(white_fen_history, white_moves, black_fen_history, black_moves)
    elif board.result()=="0-1":
        maj_winner(black_fen_history, black_moves, white_fen_history, white_moves)
    try:
        winners[str(board.result())]=winners[str(board.result())]+1
    except:
        winners[str(board.result())]=1
    board.reset()
    

print("WINNERS COUNT : \n"+str(winners))
with open('../index/q_table.json', 'w') as fp:   # Save the mapping Move/Index to be used on developement
    json.dump(general_moves, fp)