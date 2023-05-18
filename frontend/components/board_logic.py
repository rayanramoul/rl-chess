import os
import chess 

from pprint import pprint
import time
from dataclasses import dataclass
import flet as ft
import settings

TOP_POSITION_BOARD = 60
LEFT_POSITION_BOARD = 60
ACTUAL_PATH = os.getcwd()

closest_slot = None
closest_arrival_slot = None

promotion_gv = None


def drag(e: ft.DragUpdateEvent):
    e.control.top = max(0, e.control.top + e.delta_y)
    e.control.left = max(0, e.control.left + e.delta_x)
    
    e.control.update()


def update_turn_visual(player_box1, player_box2, white_turn):
    if white_turn:
        player_box1.content.controls[-1].color = "green"
        player_box2.content.controls[-1].color = "red"
        
    else:
        player_box2.content.controls[-1].color = "green"
        player_box1.content.controls[-1].color = "red"
    player_box1.content.controls[-1].update()
    player_box2.content.controls[-1].update()

    
def place(card, slot, page):
    """place card to the slot"""
    card.top = slot.top + TOP_POSITION_BOARD
    card.left = slot.left + LEFT_POSITION_BOARD
    page.update()
    
def create_piece_image(piece, piece_name, i, white_turn):
    return ft.GestureDetector(
        mouse_cursor=ft.MouseCursor.MOVE,
        drag_interval=5,
        on_pan_start=begin_drag,
        on_pan_update=drag,
        on_pan_end=lambda x: drop(x, white_turn),
        left=i*70+LEFT_POSITION_BOARD,
        top=420+TOP_POSITION_BOARD,
        content=ft.Image(
            src=os.path.join(ACTUAL_PATH, f"assets/{piece_name}.png"),
            width=70,
            height=70,
            fit=ft.ImageFit.CONTAIN,
        ))

def bounce_back(card, top, left, page):
    """return card to its original position"""
    card.top = top
    card.left = left
    page.update()


def drop(e: ft.DragEndEvent, white_turn, page, slots, chess_board_map, board, pawns, sound_panels, player_box1, player_box2, previous_moves_list):
    global closest_arrival_slot, closest_slot
    for slot in slots:
        if (
            abs(e.control.top - slot.top+(70/2) - TOP_POSITION_BOARD)  < 50
        and abs(e.control.left - slot.left+(70/2) - LEFT_POSITION_BOARD)  < 50
        ):
            place(e.control, slot, page)
            e.control.update()
            closest_arrival_slot = slots.index(slot)
            start_tile = None
            end_tile = None
            for tile in chess_board_map:
                if chess_board_map[tile][1] == closest_arrival_slot:
                    end_tile = tile
                if chess_board_map[tile][1] == closest_slot:
                    start_tile = tile
            break
    
    print("\n\n VISIBLE \n\n")
    promotion_gv.visible = True
    # promotion_gv.left = 200 # e.control.left
    # promotion_gv.top = 200# e.control.top
    #promotion_gv.visible = True
    #promotion_gv.update()
    # promotion_gv.left = e.control.left
    # promotion_gv.top = e.control.top
    promotion_gv.update()
    
    print("Moving from ", start_tile, " to ", end_tile)
    move = chess.Move.from_uci(f"{start_tile}{end_tile}")
    move_promotion = chess.Move.from_uci(f"{start_tile}{end_tile}{settings.next_promotion.lower()}")
    if move in board.legal_moves or move_promotion in board.legal_moves:
        # Remove the piece from the board
        for pawn in pawns.controls:
            if pawn.top == slots[closest_arrival_slot].top + TOP_POSITION_BOARD and pawn.left == slots[closest_arrival_slot].left + LEFT_POSITION_BOARD and pawn != e.control:
                pawn.top = 400
                pawn.left = 600+settings.number_last_death*20
                pawn.width=40
                pawn.height=40
                # pawn.visible = False
                pawn.update()
                break
        
        print("\nNext Promotion : ", settings.next_promotion)
        if move_promotion.promotion is not None and move_promotion in board.legal_moves:
            move = move_promotion
            e.control.content.src = os.path.join(ACTUAL_PATH, f"assets/{'w' if white_turn else 'b'}{settings.next_promotion}.png")
        
        previous_moves_list.controls.append(ft.Text(str(move)))
        previous_moves_list.update()
        
        sound_panels['move_sound'].play()
        board.push(move)
        white_turn = not white_turn
        update_turn_visual(player_box1, player_box2, white_turn)
        turn_text = f"Turn: {'White' if white_turn else 'Black'}"
        if board.is_game_over():
            turn_text += "\n - Game over!"

        # check if the game is a draw
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            turn_text += "\n - Game drawn by " + ("stalemate" if board.is_stalemate() else "insufficient material" if board.is_insufficient_material() else "seventy-five moves" if board.is_seventyfive_moves() else "five-fold repetition")
        # check if the current player is in check
        if board.is_check():
            turn_text += "\n - Current player is in check!"
        
        # turn_text.update()
    else:
        sound_panels['wrong_move_sound'].play()
        original_color = slots[closest_slot].bgcolor
        slots[closest_slot].bgcolor=ft.colors.RED
        bounce_back(e.control, slots[closest_slot].top+TOP_POSITION_BOARD, slots[closest_slot].left+LEFT_POSITION_BOARD, page)
        slots[closest_slot].update()
        time.sleep(2)
        slots[closest_slot].bgcolor=original_color
        slots[closest_slot].update()
        
    print(board)
    closest_arrival_slot = None
    closest_arrival_slot = None
    e.control.update()
    
    
def begin_drag(e: ft.DragStartEvent, slots):
    print("BEGIN DRAG")
    global closest_slot
    for slot in slots:
        if (
            abs(e.control.top - slot.top+(70/2) - TOP_POSITION_BOARD)  < 50
        and abs(e.control.left - slot.left+(70/2) - LEFT_POSITION_BOARD)  < 50
        ):
            print("CLOSEST ???")
            closest_slot = slots.index(slot)
            return


def generate_grid(page, chess_board_map, board, sound_panels,  player_box1, player_box2, white_turn, promotion_gv_page, previous_moves_list):
    global promotion_gv
    promotion_gv = promotion_gv_page
    slots = []
    the_grid = ft.Stack(width=1920, height=1080, top=TOP_POSITION_BOARD, left=LEFT_POSITION_BOARD)
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    slots = []
    idx = 0
    
    for l1 in range(8):
        for l2 in range(8):
            square = ft.Container(width=70, height=70, top=l1*70, left=l2*70, border=ft.border.all(1))
            slots.append(square)
            the_grid.controls.append(square)
            chess_board_map[f"{letters[l2]}{8-l1}"] = (square, idx)
            
            if l1%2==0 and l2%2==0 or l1%2==1 and l2%2==1:
                the_grid.controls[-1].bgcolor=ft.colors.WHITE
            else:
                the_grid.controls[-1].bgcolor=ft.colors.BROWN
            idx += 1
    pawns = []
    


    black_parts = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    white_parts = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    for i in range(8):
        card = ft.GestureDetector(
        mouse_cursor=ft.MouseCursor.MOVE,
        drag_interval=5,
        on_pan_start= lambda x: begin_drag(x, slots),
        on_pan_update=drag,
        on_pan_end=lambda x: drop(x, white_turn, page, slots, chess_board_map, board, pawns, sound_panels, player_box1, player_box2, previous_moves_list),
        left=i*70+LEFT_POSITION_BOARD,
        top=70+TOP_POSITION_BOARD,
        content=ft.Image(
            src=os.path.join(ACTUAL_PATH, f"assets/bP.png"),
            width=70,
            height=70,
            fit=ft.ImageFit.CONTAIN,
        ))
    
        pawns.append(card)
        
        card = ft.GestureDetector(
        mouse_cursor=ft.MouseCursor.MOVE,
        drag_interval=5,
        on_pan_start=lambda x: begin_drag(x, slots),
        on_pan_update=drag,
        on_pan_end=lambda x: drop(x, white_turn, page, slots, chess_board_map, board, pawns, sound_panels, player_box1, player_box2, previous_moves_list),
        left=i*70+LEFT_POSITION_BOARD,
        top=0+ TOP_POSITION_BOARD,
        content=ft.Image(
            src=os.path.join(ACTUAL_PATH, f"assets/b{black_parts[i]}.png"),
            width=70,
            height=70,
            fit=ft.ImageFit.CONTAIN,
        ))
        pawns.append(card)
        
        card = ft.GestureDetector(
        mouse_cursor=ft.MouseCursor.MOVE,
        drag_interval=5,
        on_pan_start=lambda x: begin_drag(x, slots),
        on_pan_update=drag,
        on_pan_end=lambda x: drop(x, white_turn, page, slots, chess_board_map, board, pawns, sound_panels, player_box1, player_box2, previous_moves_list),
        left=i*70+LEFT_POSITION_BOARD,
        top=420+TOP_POSITION_BOARD,
        content=ft.Image(
            src=os.path.join(ACTUAL_PATH, f"assets/wP.png"),
            width=70,
            height=70,
            fit=ft.ImageFit.CONTAIN,
        ))
        pawns.append(card)
        
        card = ft.GestureDetector(
        mouse_cursor=ft.MouseCursor.MOVE,
        drag_interval=5,
        on_pan_start=lambda x: begin_drag(x, slots),
        on_pan_update=drag,
        on_pan_end=lambda x: drop(x, white_turn, page, slots, chess_board_map, board, pawns, sound_panels, player_box1, player_box2, previous_moves_list),
        left=i*70+LEFT_POSITION_BOARD,
        top=490+TOP_POSITION_BOARD,
        content=ft.Image(
            src=os.path.join(ACTUAL_PATH, f"assets/w{white_parts[i]}.png"),
            width=70,
            height=70,
            fit=ft.ImageFit.CONTAIN,
        ))
        pawns.append(card)
        
    pawns = ft.Stack(controls=pawns, width=1920, height=1080)  
    return the_grid, pawns
