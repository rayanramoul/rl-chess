import pygame
import pickle
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from src.board import Board


# Add command line argument to read an agent pickle file from
# the command line
import argparse

console = Console()


def print_game_config(agent, agent_color):
    """Print game configuration"""
    rprint("[bold green]Starting Chess Game[/bold green]")

    game_table = Table(title="Game Configuration")
    game_table.add_column("Setting", style="cyan")
    game_table.add_column("Value", style="magenta")

    game_table.add_row("Player Color", "White" if agent_color == "black" else "Black")
    game_table.add_row("Agent Color", agent_color.title())
    game_table.add_row(
        "Agent Type", "Random" if agent == "random" else "Deep Q-Network"
    )

    console.print(game_table)

    # If it's a trained agent, print its configuration
    if agent != "random" and hasattr(agent, "print_config"):
        agent.print_config("Agent Configuration")


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

# Print game configuration
print_game_config(agent, args.agent_color)

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
    game_over = False
    
    while running:
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            # Quit the game if the user presses the close button
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                # If the mouse is clicked and game is not over
                if event.button == 1:
                    board.handle_click(mx, my)
        
        # Agent move (only if game is not over and it's agent's turn)
        if board.turn == args.agent_color and not game_over:
            board.agent_move()
        
        # Check for game over conditions
        if not game_over:
            if board.is_game_over():
                game_result = board.get_game_result()
                print(game_result)
                
                # Display game result
                rprint(f"[bold red]{game_result}[/bold red]")
                
                # Show additional game state information
                if board.is_in_check("white"):
                    rprint("[yellow]White is in check![/yellow]")
                elif board.is_in_check("black"):
                    rprint("[yellow]Black is in check![/yellow]")
                
                game_over = True

        # Draw the board
        draw(screen)
