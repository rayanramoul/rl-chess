import flet as ft
 
# Use of GestureDetector for with on_pan_update event for dragging card
# Absolute positioning of controls within stack


def generate_grid(page):
    for l1 in range(8):
        controls = []
        for l2 in range(8):
            controls.append(ft.Container(bgcolor=ft.colors.GREEN, width=70, height=100))
        stack = ft.Stack(controls = controls, width=1000, height=500)
        page.add(stack)
        
    
def main(page: ft.Page):
 
    def drag(e: ft.DragUpdateEvent):
       e.control.top = max(0, e.control.top + e.delta_y)
       e.control.left = max(0, e.control.left + e.delta_x)
       e.control.update()
    
    def drop(e: ft.DragEndEvent):
        if (
            abs(e.control.top - slot.top) < 20
            and abs(e.control.left - slot.left) < 20
        ):
            place(e.control, slot)
        e.control.update()

    def place(card, slot):
        """place card to the slot"""
        card.top = slot.top
        card.left = slot.left
        page.update()
 
    card = ft.GestureDetector(
       mouse_cursor=ft.MouseCursor.MOVE,
       drag_interval=5,
       on_pan_update=drag,
       on_pan_end=drop,
       left=0,
       top=0,
       content=ft.Image(
        src=f"./frontend/assets/bB.png",
        width=100,
        height=100,
        fit=ft.ImageFit.CONTAIN,
    ),
    )   
    slot = ft.Container(
    width=70, height=100, left=200, top=0, border=ft.border.all(1)
    )
    page.add(ft.Stack(controls = [slot, card], width=1000, height=500))
    generate_grid(page)
 

ft.app(target=main)