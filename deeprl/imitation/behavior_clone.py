import sys
import argparse
import pickle
import importlib
from os.path import expanduser

from deeprl.sys_util.util import find_dir
sys.path.append(find_dir(expanduser("~"), "agent_zoo"))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import roboschool #Register roboschool environments

from deeprl.model.fully_connected import FullyConnected
import deeprl.data_util.dataset as ds

def run_clone(env, policy, num_rollouts, max_steps, render):
    returns = []
    observations = []
    actions = []

    #TODO: Need a steps list to index obs/acts starts corresponding to returns.

    for i in range(num_rollouts):
        print("episode", i)
        steps = 0
        totalr = 0
        obs = env.reset()
        while True:
            action = np.squeeze(policy(obs))

            observations.append(obs)
            actions.append(action)

            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0: print("rollout step %i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print("returns", returns)
    print("mean(return)", np.mean(returns))
    print("std(return)", np.std(returns))

    return returns, observations, actions

def run_behavioral_cloning(model, X_train, Y_train, X_dev, Y_dev, num_epochs, render):

    # Train the policy.
    parameters, costs_train, costs_dev = model.train(X_train, Y_train, X_dev, Y_dev,
                                                     num_epochs=num_epochs)

    # Plot training performance.
    fig, ax = plt.subplots()
    ax.plot(costs_train)
    ax.plot(costs_dev)
    plt.ylabel("cost")
    plt.xlabel("epoch (by 100)")
    plt.show()

    policy = lambda x : model.predict(x)

    # Return policy, model weights, and performance metrics.
    return policy, parameters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name")
    parser.add_argument("--expert_name")
    parser.add_argument("--obs_file")
    parser.add_argument("--act_file")
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Number of training epochs.")
    parser.add_argument("--max_timesteps", type=int, default=200,
                        help="Max number of time steps per rollout in testing.")
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of rollouts in testing.')
    args = parser.parse_args()

    # Load the expert data.
    with open(args.obs_file, "rb") as f:
        observations = pickle.load(f)
    with open(args.act_file, "rb") as f:
        actions = pickle.load(f)

    X_train, Y_train, X_dev, Y_dev = ds.construct_train_dev(observations, actions)
    print("X_train.shape =", X_train.shape)
    print("Y_train.shape =", Y_train.shape)
    print("X_dev.shape =", X_dev.shape)
    print("Y_dev.shape =", Y_dev.shape)

    # Define the clone model.
    layer_dims = [X_train.shape[1], 100, 100, Y_train.shape[1]]
    #with tf.Session() as sess:
    # Build the clone model.
    model = FullyConnected(layer_dims)#, sess)

    policy, parameters = run_behavioral_cloning(model, X_train, Y_train, X_dev, Y_dev, args.num_epochs, args.render)

    print("Writing model parameters to file.")
    with open("datasets/parameters-" + args.env_name + "-steps" + str(args.max_timesteps) + "-rollouts" + str(args.num_rollouts) + "-epochs" + str(args.num_epochs) + ".pickle", "wb") as f:
        pickle.dump(parameters, f)

    #TODO: Test the learned policy against the expert policy.
    env = gym.make(args.env_name)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    ##
    print("Running trained clone:")
    clone_returns, _, _ = run_clone(env, policy, args.num_rollouts, max_steps, args.render)
    #run_agent(...)

    return 0

if __name__ == "__main__":
    main()
