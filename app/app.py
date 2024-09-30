import pygame
import pickle

from src.board import Board


# Add command line argument to read an agent pickle file from
# the command line
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--agent",
    type=str,
    default="random",
    help="The agent to play against (path to pickle file)",
)
# choose which color the agentt plays
parser.add_argument(
    "--agent_color", type=str, default="black", help="The color of the agent"
)
args = parser.parse_args()

# Initialize the agent
agent = "random"
if args.agent != "random":
    agent = pickle.load(open(args.agent, "rb"))


pygame.init()
pygame.display.set_caption("Arcane Chess")
WINDOW_SIZE = (600, 600)
screen = pygame.display.set_mode(WINDOW_SIZE)

board = Board(WINDOW_SIZE[0], WINDOW_SIZE[1], agent=agent, agent_color=args.agent_color)


def draw(display):
    display.fill("white")
    board.draw(display)
    pygame.display.update()


if __name__ == "__main__":
    running = True
    while running:
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            # Quit the game if the user presses the close button
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # If the mouse is clicked
                if event.button == 1:
                    board.handle_click(mx, my)
        if board.turn == args.agent_color:
            board.agent_move()
        if board.is_in_checkmate("black"):  # If black is in checkmate
            print("White wins!")
            running = False
        elif board.is_in_checkmate("white"):  # If white is in checkmate
            print("Black wins!")
            running = False

        # Draw the board
        draw(screen)
