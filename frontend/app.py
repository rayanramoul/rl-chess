from dataclasses import dataclass
import chess

import flet as ft
from pprint import pprint
# Use of GestureDetector for with on_pan_update event for dragging card
# Absolute positioning of controls within stack


page = None
board = chess.Board()

@dataclass
class Square():
    top: int
    left: int


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
    slots = []
    '''
    the_grid = ft.GridView(
        runs_count=5,
        max_extent=65,
        child_aspect_ratio=1.0,
        spacing=5,
        run_spacing=5,
        
    )
    '''
    the_grid = ft.Stack(width=1920, height=1080)
    slots = []
    for l1 in range(8):
        for l2 in range(8):
            square = ft.Container(bgcolor=ft.colors.BLUE, width=70, height=70, top=l1*70, left=l2*70, border=ft.border.all(1))
            slots.append(square)
            the_grid.controls.append(square)
            if l1%2==0 and l2%2==0 or l1%2==1 and l2%2==1:
                the_grid.controls[-1].bgcolor=ft.colors.WHITE
            else:
                the_grid.controls[-1].bgcolor=ft.colors.BROWN
    pawns = []
    
    def bounce_back(game, top, left):
        """return card to its original position"""
        card.top = top
        card.left = left
        page.update()

    
    def drop(e: ft.DragEndEvent):
        print("DROP EVENT")
        for slot in slots:
            print(f"slot : {slot.top}, {slot.left}")
            print(f"card : {e.control.top}, {e.control.left}")
            if (
                abs(e.control.top - slot.top+(70/2)) < 50
            and abs(e.control.left - slot.left+(70/2)) < 50
            ):
                place(e.control, slot)
                e.control.update()
                return None
        bounce_back(solitaire, e.control)
        # e.control.update()

    """"""
    black_parts = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    white_parts = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    for i in range(8):
        card = ft.GestureDetector(
        mouse_cursor=ft.MouseCursor.MOVE,
        drag_interval=5,
        on_pan_update=drag,
        on_pan_end=drop,
        left=i*70,
        top=70,
        content=ft.Image(
            src=f"./assets/bP.png",
            width=70,
            height=70,
            fit=ft.ImageFit.CONTAIN,
        ))
    
        pawns.append(card)
        
        card = ft.GestureDetector(
        mouse_cursor=ft.MouseCursor.MOVE,
        drag_interval=5,
        on_pan_update=drag,
        on_pan_end=drop,
        left=i*70,
        top=0,
        content=ft.Image(
            src=f"./assets/b{black_parts[i]}.png",
            width=70,
            height=70,
            fit=ft.ImageFit.CONTAIN,
        ))
        pawns.append(card)
        
        card = ft.GestureDetector(
        mouse_cursor=ft.MouseCursor.MOVE,
        drag_interval=5,
        on_pan_update=drag,
        on_pan_end=drop,
        left=i*70,
        top=420,
        content=ft.Image(
            src=f"./assets/wP.png",
            width=70,
            height=70,
            fit=ft.ImageFit.CONTAIN,
        ))
        pawns.append(card)
        
        card = ft.GestureDetector(
        mouse_cursor=ft.MouseCursor.MOVE,
        drag_interval=5,
        on_pan_update=drag,
        on_pan_end=drop,
        left=i*70,
        top=500,
        content=ft.Image(
            src=f"./assets/w{white_parts[i]}.png",
            width=70,
            height=70,
            fit=ft.ImageFit.CONTAIN,
        ))
        pawns.append(card)
        
    pawns = ft.Stack(controls=pawns, width=1920, height=1080)    
    return the_grid, pawns
    
    
def main(the_page: ft.Page): 
    global page 
    page = the_page     
    the_grid, pawns = generate_grid(the_page)
    the_page.add(ft.Stack(controls=[the_grid, pawns], width=1920, height=1080))


# Run flet app in browser
ft.app(target=main)