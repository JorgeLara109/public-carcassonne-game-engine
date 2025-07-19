# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Carcassonne Bot Battle 2025** game engine for SYNCS programming competition. The project implements a multiplayer Carcassonne board game engine where students create AI bots to compete against each other. The game follows modified Carcassonne rules with river expansion, roads, cities, monasteries, farmers, and towers.

## Development Environment

This project uses **UV** (Astral's package manager) for dependency management:

```bash
# Initial setup
uv venv
source .venv/bin/activate  # Linux/macOS
uv sync

# Alternative setup
uv pip install -e .
```

The project requires **Python 3.12+** and uses a monorepo workspace structure with three main packages.

## Architecture

The codebase is organized as a UV workspace with three main packages:

### Core Packages
- **`src/lib/`** - Shared game logic library containing:
  - Game rules and logic (`lib/game/game_logic.py`)
  - Map, tile, and meeple interactions (`lib/interact/`)
  - Event system and query interfaces (`lib/interface/`)
  - Configuration for expansions and scoring (`lib/config/`)

- **`src/engine/`** - Game engine that manages matches:
  - Main game engine (`engine/game_engine.py`)
  - State management (`engine/state/`)
  - Player I/O and validation (`engine/interface/`)
  - Game configuration (`engine/config/`)

- **`src/helper/`** - Client helper library for bot development:
  - Connection handling (`helper/interface.py`)
  - Game state utilities (`helper/client_state.py`)
  - Bot development utilities (`helper/utils.py`)

### Communication Architecture
- Uses **named pipes** for engine-bot communication
- Bots communicate via `./io/to_engine.pipe` and `./io/from_engine.pipe`
- JSON-based message protocol with size prefixes
- Event-driven architecture with queries and moves

### Key Components
- **Tile System**: Tiles contain roads, cities, monasteries, and fields
- **Meeple Management**: Player pieces for claiming territories
- **Scoring System**: Points for completed roads (1/tile), cities (2/tile + banners), monasteries (9 total)
- **Expansion Support**: River tiles, farmer fields, and tower mechanics

## Common Development Tasks

### Running the Game Engine
```bash
# Start match simulator with bot submissions
python match_simulator.py --submissions 1:example_submissions/simple.py 1:example_submissions/complex.py 1:example_submissions/claude.py 1:bot_file.py --engine

# Run engine separately
python -m engine
```

### Code Quality
```bash
# Format code (follows Black style)
ruff format

# Lint code
ruff check

# Type checking
mypy .
```

### Testing
Check individual package README files for specific test commands. The project uses the standard Python testing approach.

## Game Rules Implementation

### Core Mechanics
- **River Phase**: Game starts with river tile placement (no U-turns allowed)
- **Land Phase**: Players draw tiles, place them, and optionally claim territories
- **Scoring**: Roads (1pt/tile), Cities (2pt/tile + banners), Monasteries (9pt total)
- **Win Condition**: First to 50 points, but highest score wins after final scoring

### Expansions
- **Farmers**: Claim fields for end-game scoring (3pts per completed adjacent city)
- **Towers**: Capture opponent meeples, ransom system (3pts to release)

## Bot Development

### Bot Structure
Bots must implement the connection interface and respond to queries:
- `QueryPlaceTile`: Choose tile placement and rotation
- `QueryPlaceMeeple`: Choose meeple placement on territories

### Example Bot Location
See `example_submissions/` for reference implementations:
- `simple.py` - Basic bot implementation
- `complex.py` - Advanced strategy example
- `claude.py` - Claude-generated bot example

### Development Tips
- Use `src/helper/` utilities for state management
- Implement proper error handling for invalid moves
- Test against multiple bot strategies
- Consider scoring optimization vs. opponent blocking

## File Structure Notes

- Bot files can be placed in root directory or `example_submissions/`
- Engine logs to `output/` directory
- Configuration files use JSON format in `input/`
- Game rules documentation in `game_rules.md`
- Assets and rule images in `assets/game_rules/`