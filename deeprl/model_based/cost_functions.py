import numpy as np


#========================================================
#
# Environment-specific cost functions:
#

#NOTE: RoboschoolHalfCheetah observation is
#      [z_rel_init, sin_to_target, cos_to_target,
#       vx, vy, vz,
#       roll, pitch,
#       bthigh_pos, bthigh_vel,
#       bshin_pos, bshin_vel,
#       bfoot_pos, bfoot_vel,
#       fthigh_pos, fthigh_vel,
#       fshin_pos, fshin_vel,
#       ffoot_pos, ffoot_vel]
front_thigh_pos_i = 14 #Front leg.
front_shin_pos_i = 16
front_foot_pos_i = 18
#
cost_i = 0

def cheetah_cost_fn(state, action, next_state):
    if len(state.shape) > 1:
        #print("state[:4,17]:", state[:4,17])
        #print("next_state[:4,17]:", next_state[:4,17])
        #print("action[0,:]:", action[0,:])

        heading_penalty_factor=10
        scores=np.zeros((state.shape[0],))

        #dont move front shin back so far that you tilt forward
        #front_leg = state[:,5]
        front_leg = state[:,front_thigh_pos_i]
        my_range = 0.2
        scores[front_leg>=my_range] += heading_penalty_factor

        #front_shin = state[:,6]
        front_shin = state[:,front_shin_pos_i]
        my_range = 0
        scores[front_shin>=my_range] += heading_penalty_factor

        #front_foot = state[:,7]
        front_foot = state[:,front_foot_pos_i]
        my_range = 0
        scores[front_foot>=my_range] += heading_penalty_factor

        #NOTE: The mujoco observation is [state_pos, state_vel, joint_tree_pos (flattened)].
        #      It is seriously a pain to figure out what state_pos and state_vel are.
        #      So I'm really guessing what state index 17 is. Perhaps it's the longitudinal position?
        #print("next_state-state[:10,17]:", (next_state[:,17]-state[:,17])[:10])
        #print("scores[:10]:", scores[:10])
        #scores-= (next_state[:,17] - state[:,17]) / 0.01 #+ 0.1 * (np.sum(action**2, axis=1))
        scores-= (next_state[:,cost_i] - state[:,cost_i]) / 0.01 #+ 0.1 * (np.sum(action**2, axis=1))
        #print("scores[:4]:", scores[:4])

        #NOTE: Added to handle nans in next_state.
        #scores += np.isnan(next_state).any(axis=1) * heading_penalty_factor * 3
        #scores[np.isnan(scores)] = heading_penalty_factor * 3 * 2

        return scores
    #else:
    #    print("state:", state)
    #    print("action:", action)
    #    print("next_state:", next_state)

    heading_penalty_factor=10
    score = 0

    #dont move front shin back so far that you tilt forward
    #front_leg = state[5]
    front_leg = state[front_thigh_pos_i]
    my_range = 0.2
    if front_leg>=my_range:
        score += heading_penalty_factor

    #front_shin = state[6]
    front_shin = state[front_shin_pos_i]
    my_range = 0
    if front_shin>=my_range:
        score += heading_penalty_factor

    #front_foot = state[7]
    front_foot = state[front_foot_pos_i]
    my_range = 0
    if front_foot>=my_range:
        score += heading_penalty_factor

    #score -= (next_state[17] - state[17]) / 0.01 #+ 0.1 * (np.sum(action**2))
    score -= (next_state[cost_i] - state[cost_i]) / 0.01 #+ 0.1 * (np.sum(action**2))

    #NOTE: Added to handle nan in next_state.
    #if np.isnan(score): score = heading_penalty_factor * 3

    return score

#========================================================
#
# Cost function for a whole trajectory:
#

def trajectory_cost_fn(cost_fn, states, actions, next_states):
    #print("trajectory_cost_fn:")
    trajectory_cost = 0
    for i in range(len(actions)):
        #trajectory_cost += cost_fn(states[i], actions[i], next_states[i])
        tmp_cost = cost_fn(states[i], actions[i], next_states[i])
        #print(tmp_cost)
        trajectory_cost += tmp_cost
    return trajectory_cost
