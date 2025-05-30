.PHONY: help train play play-white play-black clean install test lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies using uv"
	@echo "  make train       - Train the chess agent"
	@echo "  make train-resume - Resume training from latest checkpoint"
	@echo "  make play        - Play against latest checkpoint (you play white, agent plays black)"
	@echo "  make play-white  - Play as white against the agent (agent plays black)"
	@echo "  make play-black  - Play as black against the agent (agent plays white)"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linting"
	@echo "  make format      - Format code"
	@echo "  make clean       - Clean up generated files"
	@echo "  make tensorboard - Start tensorboard server"
	@echo "  make wandb       - Open wandb dashboard"

# Install dependencies
install:
	uv sync
	@echo "Dependencies installed successfully!"

# Train the chess agent from scratch
train:
	@echo "Starting training from scratch..."
	uv run src/rl_chess/train.py

# Resume training from latest checkpoint
train-resume:
	@echo "Resuming training from latest checkpoint..."
	uv run src/rl_chess/train.py agent.continue_from_checkpoint_path=checkpoints/last.ckpt

# Play against the latest checkpoint (default: you play white, agent plays black)
play: play-white

# Play as white against the agent
play-white:
	@echo "Starting game: You (White) vs Agent (Black)"
	@if [ ! -f "checkpoints/last.ckpt/deep_q_agent.pickle" ]; then \
		echo "No trained model found. Please run 'make train' first."; \
		exit 1; \
	fi
	uv run app/app.py --agent checkpoints/last.ckpt/deep_q_agent.pickle --agent_color black

# Play as black against the agent
play-black:
	@echo "Starting game: Agent (White) vs You (Black)"
	@if [ ! -f "checkpoints/last.ckpt/deep_q_agent.pickle" ]; then \
		echo "No trained model found. Please run 'make train' first."; \
		exit 1; \
	fi
	uv run app/app.py --agent checkpoints/last.ckpt/deep_q_agent.pickle --agent_color white

# Play against a specific checkpoint
play-checkpoint:
	@read -p "Enter checkpoint episode number: " episode; \
	if [ ! -f "checkpoints/episode_$$episode.ckpt/deep_q_agent.pickle" ]; then \
		echo "Checkpoint episode_$$episode.ckpt not found."; \
		exit 1; \
	fi; \
	echo "Starting game against episode $$episode checkpoint..."; \
	uv run app/app.py --agent checkpoints/episode_$$episode.ckpt/deep_q_agent.pickle --agent_color black

# Run tests
test:
	uv run pytest tests/ -v

# Run linting
lint:
	uv run ruff check src/ app/
	uv run mypy src/ app/

# Format code
format:
	uv run ruff format src/ app/
	uv run ruff check --fix src/ app/

# Clean up generated files
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf src/**/__pycache__/
	rm -rf app/**/__pycache__/
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/
	@echo "Cleanup completed!"

# Start tensorboard server
tensorboard:
	@echo "Starting TensorBoard server..."
	@echo "Open http://localhost:6006 in your browser"
	uv run tensorboard --logdir=runs

# Open wandb dashboard
wandb:
	@echo "Opening Weights & Biases dashboard..."
	uv run wandb dashboard

# Show training progress
status:
	@echo "=== Training Status ==="
	@if [ -d "checkpoints" ]; then \
		echo "Available checkpoints:"; \
		ls -la checkpoints/ | grep -E "episode_|last"; \
	else \
		echo "No checkpoints found. Training hasn't started yet."; \
	fi
	@echo ""
	@if [ -d "wandb" ]; then \
		echo "Wandb logs available. Run 'make wandb' to view dashboard."; \
	fi
	@if [ -d "runs" ]; then \
		echo "TensorBoard logs available. Run 'make tensorboard' to view."; \
	fi

# Quick setup for new users
setup: install
	@echo "=== Chess RL Setup Complete ==="
	@echo "Next steps:"
	@echo "1. Run 'make train' to start training"
	@echo "2. Run 'make play' to play against the trained agent"
	@echo "3. Run 'make status' to check training progress"

# Development commands
dev-train:
	@echo "Starting development training (shorter episodes)..."
	uv run src/rl_chess/train.py agent.number_episodes=100 agent.save_every_n_episodes=10

# Benchmark the agent
benchmark:
	@echo "Running benchmark against random player..."
	# This would need a separate benchmark script
	@echo "Benchmark feature coming soon!"
