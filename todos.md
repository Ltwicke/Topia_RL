# Current To-Dos and open design choices

This file lays down some important design choices for the todos that should be existentialized.

## Game simulator refactorization

### Tile featurization

- Any game-features that get introduced can cause an increase in the dimensionality of the tile featurization. Use the enum.py file to have a central controller to define the starting and ending dimensions of all features of the tile objects. 
- Opt for one-hot-encoding for the featurizers.

### City class

- A city (of either player) or a village is a feature of a tile, it does not move. Currently, they are instantiated as objects which can hold a tile reference, unit reference etc. This causes a lot of cross-referencing which is highly undesired. Find a new design of the city object that alleviates the need for endless cross-referencing

### Board.py

- Currently, NODE_FEAT_DIM is hardcoded here, which is a problem. Board just wraps the tiles that make up the board. 
- The movement_topology_graph must be prepared to give values to the edges between tiles to be used for the road mechanic. Standard value is 1.

### Player.py

- In construct_partial_graph_2players function, the tile freature dimensions are hard-coded, which is a problem. Use the enum.py class to have this be dynamically coded to switch the correct dimensions

### Game.py

- The function calc_movement_target_and_shortest_path also hard-codes a lot, which is a problem. The dimension values for checking movement options should also be controlled from the enums.py file.
- Improve the calc_movement_target_and_shortest_path for new features to come, such as enemy zone of control, better way to remove target nodes and not path nodes.
- advance_unit_turn_state function needs to be a bit more structured. This is very game specific and should be done by a human programmer.


## Environment wrapper

### unit identifier

- Currently, there is a design error with identifying opponent units eligible for attacking; In get_action_mask, the length of the list of opponent units is shown to the agent although he should only get information of the enemy units revealed to the player via its partial graph. Also, for human play, the unit needs to have an identifier other than being at a specific position in the list of units_under_control. This calls for the necessity of a light-weight id for units to make sure that the mask does not give away hidden information. This will also change a lot of the unit vs unit interaction code as it is currently written.

### Moving the renderer to its own python file

- Currently the render function is build into the EnvWrapper, but it is probably better to isolate it into its own module. 

