import sys
import importlib
from os.path import expanduser

from deeprl.sys_util.util import find_dir
sys.path.append(find_dir(expanduser("~"), "agent_zoo"))

import numpy as np
import tensorflow as tf
import gym
import roboschool #Register roboschool environments

class PolicyRoboschool(object):#should inherit from Policy
    def __init__(self, policy_str, env, sess=None):
        #NOTE: An interactive TF session is required to be up.

        if not sess:
            sess = tf.InteractiveSession()

        # Dynamically load desired policy.
        Mod = importlib.import_module(policy_str)
        AgentPolicy = getattr(Mod, "ZooPolicyTensorflow")

        self.env = env
        self.zoo_policy = AgentPolicy("agent_model", env.observation_space, env.action_space)

    def policy(self, obs):
        return self.zoo_policy.act(obs, self.env)


def run_agent(env_str, roboschool_policy_str, render, max_timesteps, num_rollouts):
    print("Rendering?", render)
    print("num_rollouts:", num_rollouts)

    #config = tf.ConfigProto(inter_op_parallelism_threads=1,
    #                        intra_op_parallelism_threads=1,
    #                        device_count={"GPU": 0})
    #sess = tf.InteractiveSession(config=config)

    ## Dynamically load desired policy.
    #Mod = importlib.import_module(roboschool_policy_str)
    #AgentPolicy = getattr(Mod, "ZooPolicyTensorflow")

    env = gym.make(env_str)

    robo_policy = PolicyRoboschool(roboschool_policy_str, env)
    policy = lambda x : robo_policy.policy(x)

    max_steps = max_timesteps or env.spec.timestep_limit
    print("max_timesteps:", max_steps)

    #policy = AgentPolicy("agent_model", env.observation_space, env.action_space)

    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        print("episode", i)
        steps = 0
        totalr = 0
        obs = env.reset()
        while True:
            #action = policy.act(obs, env)
            action = policy(obs)
            #print("action =", action)
            #print("action: (%d, %d) =" % (action.shape[0], action.shape[1]), action)
            observations.append(obs)
            actions.append(action)

            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print("returns", returns)
    print("mean(return)", np.mean(returns))
    print("std(return)", np.std(returns))

    agent_data = {"observations": np.array(observations),
                  "actions": np.array(actions)}
    #TODO: Maybe want to write to file.
    return agent_data
