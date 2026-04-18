import numpy as np

from game.enums import BoardType, Tribes, ActionTypes
from env.wrapper import EnvWrapper


def select_random_matrix_element(mat):
    nonzero_rows = np.where(mat.any(axis=1))[0]
    row_idx = np.random.choice(nonzero_rows)
    col_idx = np.random.choice(np.flatnonzero(mat[row_idx]))
    return row_idx, col_idx

def select_random_array_element(arr):
    return np.random.choice(np.flatnonzero(arr))

def _create_random_action_from_action_mask(action_mask):
    
    action = []
    choose_action_type = np.random.choice(np.flatnonzero(action_mask[0]))

    if ActionTypes(choose_action_type) == ActionTypes.MoveUnit:
        action.append(ActionTypes.MoveUnit.value)
        unit_id, loc_id = select_random_matrix_element(action_mask[1])
        action.append(unit_id)
        action.append(loc_id)

    elif ActionTypes(choose_action_type) == ActionTypes.Attack:
        action.append(ActionTypes.Attack.value)
        unit_id, defender_id = select_random_matrix_element(action_mask[2])
        action.append(unit_id)
        action.append(defender_id)

    elif ActionTypes(choose_action_type) == ActionTypes.CreateUnit:
        action.append(ActionTypes.CreateUnit.value)
        city_id, unit_type = select_random_matrix_element(action_mask[3])
        action.append(city_id)
        action.append(unit_type)

    elif ActionTypes(choose_action_type) == ActionTypes.CaptureCity:
        action.append(ActionTypes.CaptureCity.value)
        action.append(select_random_array_element(action_mask[4])) 

    elif ActionTypes(choose_action_type) == ActionTypes.EndTurn:
        action.append(ActionTypes.EndTurn.value)

    return action
    

for i in range(10):
    env = EnvWrapper({'board_size': (9,9), 'board_type': BoardType.Dummy, 'n_players': 2}, [Tribes.Omaji, Tribes.Yaddak])
    obs = env.reset()
    
    
    mask = env.get_action_mask()
    
    # Play 5 EndTurn actions (always valid)
    for turn in range(250):
        action = _create_random_action_from_action_mask(mask)
        obs, reward, done, info = env.step(action)
        print(f'Turn {turn+1}: reward={reward}, done={done}')
    
    print('Simulation OK ', i)
