import chess
import json
#print(str(csv.head()))
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


q_table={}
training_games=1000
winner_reward=100
loser_malus=0
better_evaluation_reward=10
intermediate_evaluation_reward=2
bad_evaluation_malus=-10

def evaluate_board(turn):
    l=chess.SQUARES
    total=0
    if turn:
        mult=1
    else:
        mult=-1
    for i in l:
        total=total+(mult*switch[str(board.piece_at(i))])
    return total
 
def get_best_move(state, moves):
    for i in moves:
        best_key=str(i)
        break
    best_reward=0
#    print("All possible moves : "+str(moves))
    for i in moves:
#        print("MOVE : "+str(i))
        try:
            q_table[state]
        except:
            q_table[state]={}
        try:
            if q_table[state][str(i)]>=best_reward:
                best_key=str(i)
                best_reward=q_table[state][str(i)]
        except:
            q_table[state][str(i)]=0
            if q_table[state][str(i)]>=best_reward:
                best_key=str(i)
                best_reward=q_table[state][str(i)]

#    print("Best move : "+str(best_key)+" with a reward of "+str(q_table[state][str(i)]))
#    print("All moves : "+str(q_table[state]))
    return str(best_key)

def maj_winner(fen_history, moves, lose_fen, lose_moves):
    maxi=len(fen_history)
    print("Maj end : ")
    i=0
    while i<maxi:
        q_table[str(fen_history[i])][str(moves[i])]=q_table[str(fen_history[i])][str(moves[i])]+winner_reward
        i=i+1
    maxi=len(lose_fen)
    i=0
    while i<maxi:
        q_table[str(lose_fen[i])][str(lose_moves[i])]=q_table[str(lose_fen[i])][str(lose_moves[i])]+loser_malus
        i=i+1



i=0
evaluation_history=[]
all_number_of_moves=[]
winners={}
board=chess.Board()
while i<training_games:
    print("Game "+str(i))
    fen_history=[]
    black_moves=[]
    white_moves=[]
    black_fen_history=[]
    white_fen_history=[]
    number_of_moves=0
    evaluation_history=[]
    while not board.is_game_over():
        number_of_moves=number_of_moves+1
        god_damn_move=get_best_move(str(board.fen()), board.legal_moves)
        base_evaluation=evaluate_board(board.turn)
        fen=str(board.fen())
        evaluation_history.append(base_evaluation)
        if board.turn:
            white_moves.append(god_damn_move)
            white_fen_history.append(board.fen())
        else:
            black_moves.append(god_damn_move)
            black_fen_history.append(board.fen())
        board.push(chess.Move.from_uci(god_damn_move))
        end_evaluation=evaluate_board(not board.turn)
        if end_evaluation>base_evaluation:
#            print("eat maj god damn move : "+god_damn_move)
            q_table[str(fen)][god_damn_move]=q_table[str(fen)][str(god_damn_move)]+better_evaluation_reward

        elif end_evaluation<base_evaluation:
#            print("got beat god damn move : "+god_damn_move)
            q_table[str(fen)][god_damn_move]=q_table[str(fen)][str(god_damn_move)]+bad_evaluation_malus

        else:
#            print("it's okay maj god damn move : "+god_damn_move)
            q_table[str(fen)][god_damn_move]=q_table[str(fen)][str(god_damn_move)]+intermediate_evaluation_reward


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

print("All number of moves : \n"+str(all_number_of_moves))
print("WINNERS COUNT : \n"+str(winners))
with open('../index/q_table.json', 'w') as fp:
    json.dump(q_table, fp)