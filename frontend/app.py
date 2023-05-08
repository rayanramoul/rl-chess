from dataclasses import dataclass
import os
import time

import chess

import flet as ft
from pprint import pprint
# Use of GestureDetector for with on_pan_update event for dragging card
# Absolute positioning of controls within stack


page = None
board = chess.Board()
ACTUAL_PATH = os.getcwd()
chess_board_map = {}
closest_slot = None
closest_arrival_slot = None
turn_text = None
white_turn = True
number_last_death = 0

move_sound = ft.Audio(src=os.path.join(ACTUAL_PATH, "sounds/move.wav"))
wrong_move_sound = ft.Audio(src=os.path.join(ACTUAL_PATH, "sounds/wrong_move.mp3"))
background_music = ft.Audio(src=os.path.join(ACTUAL_PATH, "sounds/background_music.mp3"), autoplay=True)

def drag(e: ft.DragUpdateEvent):
    e.control.top = max(0, e.control.top + e.delta_y)
    e.control.left = max(0, e.control.left + e.delta_x)
    e.control.update()

    
def place(card, slot):
    global page
    """place card to the slot"""
    card.top = slot.top
    card.left = slot.left
    page.update()



def generate_grid(page):
    global chess_board_map
    slots = []
    the_grid = ft.Stack(width=1920, height=1080)
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
    
    def bounce_back(card, top, left):
        """return card to its original position"""
        card.top = top
        card.left = left
        page.update()

    
    def drop(e: ft.DragEndEvent):
        global closest_arrival_slot, closest_slot, white_turn, number_last_death
        for slot in slots:
            if (
                abs(e.control.top - slot.top+(70/2)) < 50
            and abs(e.control.left - slot.left+(70/2)) < 50
            ):
                place(e.control, slot)
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
        # print("Moving from ", start_tile, " to ", end_tile)
        move = chess.Move.from_uci(f"{start_tile}{end_tile}")
        if move in board.legal_moves:
            print("ACCEPTED")
            # Remove the piece from the board
            for pawn in pawns.controls:
                if pawn.top == slots[closest_arrival_slot].top and pawn.left == slots[closest_arrival_slot].left and pawn != e.control:
                    pawn.top = 400
                    pawn.left = 600+number_last_death*20
                    pawn.width=40
                    pawn.height=40
                    # pawn.visible = False
                    pawn.update()
                    break
            move_sound.play()
            result = board.push(move)
            white_turn = not white_turn
            turn_text.value = f"Turn: {'White' if white_turn else 'Black'}"
            if board.is_game_over():
                print("Game over!")
                turn_text.value += "\n - Game over!"

            # check if the game is a draw
            if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
                print("Game drawn by " + ("stalemate" if board.is_stalemate() else "insufficient material" if board.is_insufficient_material() else "seventy-five moves" if board.is_seventyfive_moves() else "five-fold repetition"))
                turn_text.value += "\n - Game drawn by " + ("stalemate" if board.is_stalemate() else "insufficient material" if board.is_insufficient_material() else "seventy-five moves" if board.is_seventyfive_moves() else "five-fold repetition")
            # check if the current player is in check
            if board.is_check():
                print("Current player is in check!")
                turn_text.value += "\n - Current player is in check!"
            
            turn_text.update()
            print("turn text updated")
        else:
            print("REFUSED")
            wrong_move_sound.play()
            original_color = slots[closest_slot].bgcolor
            slots[closest_slot].bgcolor=ft.colors.RED
            bounce_back(e.control, slots[closest_slot].top, slots[closest_slot].left)
            slots[closest_slot].update()
            time.sleep(2)
            slots[closest_slot].bgcolor=original_color
            slots[closest_slot].update()
            
        print(board)
        # 
        closest_arrival_slot = None
        closest_arrival_slot = None
        
        e.control.update()
        
    def begin_drag(e: ft.DragStartEvent):
        print("BEGIN DRAG")
        global closest_slot
        for slot in slots:
            if (
                abs(e.control.top - slot.top+(70/2)) < 50
            and abs(e.control.left - slot.left+(70/2)) < 50
            ):
                closest_slot = slots.index(slot)
                return

    """"""
    black_parts = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    white_parts = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    for i in range(8):
        card = ft.GestureDetector(
        mouse_cursor=ft.MouseCursor.MOVE,
        drag_interval=5,
        on_pan_start=begin_drag,
        on_pan_update=drag,
        on_pan_end=drop,
        left=i*70,
        top=70,
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
        on_pan_start=begin_drag,
        on_pan_update=drag,
        on_pan_end=drop,
        left=i*70,
        top=0,
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
        on_pan_start=begin_drag,
        on_pan_update=drag,
        on_pan_end=drop,
        left=i*70,
        top=420,
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
        on_pan_start=begin_drag,
        on_pan_update=drag,
        on_pan_end=drop,
        left=i*70,
        top=490,
        content=ft.Image(
            src=os.path.join(ACTUAL_PATH, f"assets/w{white_parts[i]}.png"),
            width=70,
            height=70,
            fit=ft.ImageFit.CONTAIN,
        ))
        pawns.append(card)
        
    pawns = ft.Stack(controls=pawns, width=1920, height=1080)  
    return the_grid, pawns
    
    
def main(the_page: ft.Page): 
    global turn_text
    global page 
    turn_text = ft.Text(value="TURN : WHITE", color="green", left=650, size=25)
    # death_area = ft.Stack(width=200, height=200, ft.Container(bgcolor=ft.colors.RED, top=400, left=600)
    page = the_page     
    the_grid, pawns = generate_grid(the_page)
    the_page.add(ft.Stack(controls=[the_grid, pawns, turn_text, move_sound, wrong_move_sound, background_music], width=1920, height=1100))
    the_page.add(turn_text)
    

# Run flet app in browser
ft.app(target=main)