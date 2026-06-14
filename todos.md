# Current To-Dos and open design choices

This file lays down some important design choices for the todos that should be existentialized.

## Game simulator overhaul and additions

### Giant Houdini error
DONE
- When a giant is created on a city due to a city upgrade, it can happen that it deleted the current occupant of the city due to a lack of possible surrounding tiles to push the unit to
- In this case, there is a key error trying to delete the unit id of the occupant.
- Find the source of the error and try to resolve it.

## Environment wrapper


### Rendering function visual changes

- Currently, there is no visualization for bridges. When roads are rendered, check if the underlying tile is a water tile. If its a water tile, render a bridge with the _bridge_axis direction of the bridge.


### More complex UI on the horizon
IGNORE FOR NOW!
- In the long term, the goal is that the user can play against a trained agent. For this, a more refined UI will be needed. But this is something for the future.


## RL agent model and PPO pipeline

### MultiscaleConvs usage is suboptimal for CreateUnitTypeHead
- Currently, the power of convolutions is not really used in the current implementation for the unittype selection in CreateUnitTypeHead class in "Stage 1 — MULTI-SCALE CONVOLUTIONS". The convolutions are centered on the queried tiles (cities), but this needs bigger convolutions in order to see further which makes the model too big.
- Change the code as follows: For the queried tiles, also use the context_bias hyperparameter to slide the convolutions in a squeare of context_bias around the city tile (contex_bias in each direction making a square of size 2* context_bias + 1). 
- Then max_pool the result from each layer onto the queried tile and continue like before. 
- This way, the convolutions can see much further

### sel_n_heads does not influence the policy parameter size
- If I change the sel_n_heads hyperparameter of the network architecture, I do not see any changes in the size of the network when using "model_summary()" from  policy.py; This is not expected and I expect an error there.
- Check SequenceSelectionHead class if everything is implemented correctly.

### Recompute GAE after every epoch
- It is generally recommended to recompute the GAE value after every epoch, because the value network has changed and it generally leads to a performance increase during training. 
- I could honestly recalculate it for every single minibatch given the time each minibatch takes for me...

### Entropy fusing strategy

Not important for now!

- Currently, entropy is just fused with a linear input projection using nn.Linear onto all of the necessary embeddings.
- Given the objective being to push the network into choosing clear decisions (those with low entropy, essentially telling, this is a must do move), is there a better way to include these low level entropy values?




