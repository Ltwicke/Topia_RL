# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains two interconnected projects:

**Project 1: Game simulator for a turn-based, checkerboard-composed and incomplete information settlers-like strategy game** The goal of this game simulator is to run as fast and efficient as possible, so that it may be used to simulate a lot of games in parallel
**Project 2: Training RL agents with PPO and its derivatives to play this game in a self-play scenario** The goal is to train agents in an on-policy setting and to reach superhuman level in this game

## Project 1: Game simulator

### Components
- `game/enums.py`: this file includes python Enum classes that hold the different features of the game. These classes will be used throughout the codebase to control the game simulator. It also includes slices that are important for generating the partial graph for each player, which is the main observation input for the RL agents. 
- `game/components`: this folder contains individual game object classes that have their own behaviour and can be created as an object during the game simulations.
- `game/components/tile.py`: tile objects hold the universal information of the game board. It also includes transform_to_node_features function which featurizes the nodes for downstream tasks.
- `game/components/player.py`: the player class holds all the information present to each individual player in the game. The players partial graph attribute is the basis of the observation module in the RL downstream task.
- `game/game.py`: This file contains the game class. It includes the apply_action function, which defines logic for each action included in the ActionTypes IntEnum class. It is also here, where movement and unit_turn_state is calculated.

### Core Concepts
- **2-player game**: Although this game can be played with more than 2 players per game, for now, it should be kept as a two player game. 
- **Game flow**: The game object is created with specification of the board size and type, number of players, which currently is always 2, and some other specificities. Then, an external source sequentially decides for actions and the game class modifies its board, players and other classes based on the action applied. 
- **Validity assertion**: The game class currently does not check for validity of chosen action, this is handled later in downstream tasks.


## Project 2: RL agents to play the game

### Components
- `env/wrapper.py`: This wraps the game simulator to be used as the environment for the RL agent. It is designed to satisfy the GYM setup for RL. Crucially, it uses get_action_mask to construct the possible actions accessible to the agent in every situation and stores it in the valid_actions variable. 
- `RL/models`: This folder contains the modules and the model-builder policy.py for the deep network to be trained. 
- `RL/ppo`: This folder contains all the necessary logic for the ppo training algorithm to work, including game_manager.py to manage the creations of the environment and batch_processing.py to manage the batch creation. The actual training is run with a script in the RL folder.

### Training and logging

Not important for now.

## Development Commands

### Testing game simulations

### Git Workflows
- Always run all required tests before staging a new commit. 
- Document any behaviour changes in commit messages.


## Technical Guidelines

### Python
- Use efficient parallelization when applicable
- Avoid loops when possible and make use of list comprehensions and lambda functions
- Use scientificly-relevant libraries and pay focus on geometric features of the game
- Bundle code features in their respective modules and avoid cross-referencing between game objects

### C++
- Construct efficient, directed objects to be called and used from pythons workflow
- Pay special attention to memory management and garbage collection

## Refactoring guidelines

The game simulator is build step by step in incremental changes. Always run some test games when a new feature was added and catch any potential errors that may arise. Fix any errors before moving to the next task (Use test-driven development).

### Performance language utilization

- The game simulations will run on multiple CPU cores in parallel. If applicable, make use of the sparsity of the action_mask and run performance oriented code in a separate c++ compiled object that is linked to python via ctypes. 
- Only use c++ code for performance oriented tasks
- Make proper use of garbage collection, because memory is very valuable in this project. Ideally, the game simulations are as memory-efficient as possible.

### Current To-Dos and open design choices

- Read `todos.md` from the projects root folder


## Hard Rules

These are non-negotiable rules during any work on this project.

- 

