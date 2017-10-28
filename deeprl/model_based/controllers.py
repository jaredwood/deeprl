import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		""" YOUR CODE HERE """
		self.action_dim = env.action_space.shape[0] #Assuming continuous
		self.action_low = env.action_space.low
		self.action_high = env.action_space.high

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Your code should randomly sample an action uniformly from the action space """
		#NOTE: Initial state is ignored.
		actions = np.random.rand(self.action_dim) * (self.action_high - self.action_low) + self.action_low
		return actions


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self,
				 env,
				 dyn_model,
				 horizon=5,
				 cost_fn=None,
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """
		action_dims = self.env.action_space.shape[0]
		state_dims = self.env.observation_space.shape[0]

		action_low = self.env.action_space.low
		#print("action_low:", action_low)
		action_high = self.env.action_space.high
		#print("action_high:", action_high)
		action_delta = action_high - action_low
		#print("action_delta:", action_delta)

		# Form batch initial state.
		state = np.ones((self.num_simulated_paths, state_dims)) * state

		horizon_penalty = 0
		states = []; actions = []; next_states = []
		for h in range(self.horizon):
			#states.append(state)

			# Randomly sample (num_simulated_paths, action_dims) actions.
			#action = np.random.rand(self.num_simulated_paths, action_dims) * action_delta + action_low
			action = np.random.randn(self.num_simulated_paths, action_dims) * action_delta/4
			#actions.append(action)

			# Compute approximated next state.
			next_state = self.dyn_model.predict(states=state, actions=action)
			#next_states.append(state)

			#if np.sum(np.isnan(next_state)) > 0:
			#	# Not able to continue...some states are NAN.
			#	horizon_penalty = 30 * (self.horizon - (h+1)) #TODO: Add max_cost method to cost_fn (this is based on cheetah_cost_fn).
			#	break

			#print("state[:4,17]:     ", state[:4,17])
			#print("next_state[:4,17]:", next_state[:4,17])

			states.append(state)
			actions.append(action)
			next_states.append(next_state)
			#
			state = next_state

		# Compute each trajectory's cost.
		costs = horizon_penalty + trajectory_cost_fn(cost_fn=self.cost_fn, states=states, actions=actions, next_states=next_states)
		#print(costs.shape)
		#print(costs[0])
		#print("MPC path costs range: [", np.min(costs), ",", np.max(costs), "]")

		# Get the min-cost path index.
		ind = np.argmin(costs)
		#print("MPC path cost:", costs[ind])

		best_action = actions[0][ind]
		#print("MPC best action:", best_action)
		return best_action
