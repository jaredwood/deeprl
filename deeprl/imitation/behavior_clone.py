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
from deeprl.imitation.run_agent import run_agent
import deeprl.data_util.dataset as ds

def run_behavioral_cloning(model, X_train, Y_train, X_dev, Y_dev, num_epochs):

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
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of rollouts in testing.')
    parser.add_argument("--max_timesteps", type=int, default=200,
                        help="Max number of time steps per rollout in testing.")
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

    policy, parameters = run_behavioral_cloning(model, X_train, Y_train, X_dev, Y_dev, args.num_epochs)

    print("Writing model parameters to file.")
    with open("datasets/parameters-bc-" + args.env_name + "-epochs" + str(args.num_epochs) + ".pickle", "wb") as f:
        pickle.dump(parameters, f)

    #TODO: Test the learned policy against the expert policy.
    env = gym.make(args.env_name)
    ##
    print("Running trained clone:")
    clone_returns, _, _ = run_agent(env, policy, args.num_rollouts, args.max_timesteps, args.render)
    #run_agent(...)

    return 0

if __name__ == "__main__":
    main()
