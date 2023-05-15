import os

import chess

import flet as ft
from components.board_logic import generate_grid
# Use of GestureDetector for with on_pan_update event for dragging card
# Absolute positioning of controls within stack


page = None
board = chess.Board()
ACTUAL_PATH = os.getcwd()
chess_board_map = {}

turn_text = None
white_turn = True
number_last_death = 0
TOP_POSITION_BOARD = 60
LEFT_POSITION_BOARD = 60

sound_panels  = {
'move_sound' : ft.Audio(src=os.path.join(ACTUAL_PATH, "sounds/move.wav")),
'wrong_move_sound' : ft.Audio(src=os.path.join(ACTUAL_PATH, "sounds/wrong_move.mp3"))
}

player_box1 = None
player_box2 = None


def player_box(player_name, player_color):
    player_icon = ft.Image(
            src=os.path.join(ACTUAL_PATH, f"assets/wB.png"),
            width=70,
            height=70,
            fit=ft.ImageFit.CONTAIN,
            color=player_color,
             
    )
    player_column = ft.Column(
        controls=[
            ft.Text(value=player_name, color=player_color),
            player_icon,
        ],
        width=600,
        height=200,
    )
    player_container = ft.Container(
        content=player_column,
        bgcolor=ft.colors.WHITE,
    )
    return player_container 




    
def main(the_page: ft.Page): 
    global turn_text
    global page 
    global player_box1, player_box2
    
    turn_text = ft.Text(value="TURN : WHITE", color="green") #, left=250, size=50)
    player_box1 = player_box("Player 1", "blue")
    player_box2 = player_box("Player 2", "red")
    
    previous_moves_list = ft.ListView(expand=True, spacing=10, item_extent=50, width=500, height=100)
    for i in range(5000):
        previous_moves_list.controls.append(ft.Text(f"Move {i}"))
    middle_right_row = ft.Row(width=500, height=200, controls=[previous_moves_list])
    
    controls_row = ft.Row(
            [
                ft.IconButton(
                    icon=ft.icons.ARROW_BACK_IOS_NEW_OUTLINED,
                    icon_color="blue400",
                    icon_size=40,
                    tooltip="Pause record",
                ),
                ft.IconButton(
                    icon=ft.icons.ARROW_FORWARD_IOS_OUTLINED,
                    icon_color="pink600",
                    icon_size=40,
                    tooltip="Delete record",
                ),
            ]
        )
    
    right_stack = ft.Column(width=500, height=1080, controls=[turn_text, player_box1, middle_right_row, controls_row, player_box2])
    
    
    # death_area = ft.Stack(width=200, height=200, ft.Container(bgcolor=ft.colors.RED, top=400, left=600)
    page = the_page
    
    
    
    the_grid, pawns = generate_grid(the_page, chess_board_map, board, sound_panels, player_box1, player_box2, white_turn)
    game_stack = ft.Stack(controls=[the_grid, pawns, sound_panels['move_sound'], sound_panels['wrong_move_sound']],  width=800, height=1100)
    main_row = ft.Row(controls=[game_stack, right_stack], width=1920, height=1080)
    
    the_page.add(main_row)
    # the_page.add(turn_text)
    

# Run flet app in browser
ft.app(target=main)