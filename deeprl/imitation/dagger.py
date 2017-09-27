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
from deeprl.roboschool.run_agent import PolicyRoboschool
from deeprl.imitation.run_agent import run_agent
import deeprl.data_util.dataset as ds

def run_dagger(env, model, expert_policy, X_train, Y_train, X_dev, Y_dev, num_iterations, num_epochs, num_sim_rollouts, num_sim_steps):
    final_costs_train = []
    final_costs_dev = []
    returns_mean = []
    returns_std = []
    for i in range(num_iterations):
        # Train the policy.
        print("Iteration %d: Training new clone..." % i)
        parameters, costs_train, costs_dev = model.train(X_train, Y_train, X_dev, Y_dev,
                                                         num_epochs=num_epochs)
        policy = lambda x : model.predict(x)

        # Plot training performance.
        #fig, ax = plt.subplots()
        #ax.plot(costs_train)
        #ax.plot(costs_dev)
        #plt.ylabel("cost")
        #plt.xlabel("epoch (by 100)")
        #plt.show()

        #print("len(costs_train):", len(costs_train))
        #print("costs_train.shape:", np.array(costs_train).shape)
        final_costs_train.append(costs_train[-1])
        final_costs_dev.append(costs_dev[-1])

        # Simulate clone observations.
        print("Collecting trained-clone observations...")
        returns, observations, _ = run_agent(env, policy, num_sim_rollouts, num_sim_steps, render=False)

        returns_mean.append(np.mean(np.array(returns)))
        returns_std.append(np.std(np.array(returns)))
        print("Iteration %d: mean(returns) = %f, std(returns) = %f" % (i, returns_mean[i], returns_std[i]))

        if i >= num_iterations-1: break

        # Expert-label clone observations.
        print("Expert-labeling clone-generated observations...")
        actions = []
        for obs in observations:
            actions.append(expert_policy(obs))

        # Add simulated data to training set.
        print("Joining new training data...")
        ##
        obs = np.array(observations)
        X_train = np.concatenate((X_train, obs), axis=0)
        ##
        act = np.array(actions)
        Y_train = np.concatenate((Y_train, act), axis=0)

    # Plot training performance.
    fig, ax = plt.subplots()
    ax.plot(np.array(final_costs_train))
    ax.plot(np.array(final_costs_dev))
    plt.ylabel("final training cost per dagger iteration")
    plt.xlabel("dagger iteration")
    plt.legend(("train", "dev"))

    # Plot mean returns.
    fig, ax = plt.subplots()
    ax.plot(np.array(returns_mean))
    plt.ylabel("return")
    plt.xlabel("dagger iteration")

    plt.show()

    # Return policy, model weights, and performance metrics.
    return policy, parameters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name")
    parser.add_argument("--expert_name")
    parser.add_argument("--obs_file")
    parser.add_argument("--act_file")
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--iterations", type=int, default=10,
                        help="number of dagger iterations.")
    parser.add_argument("--num_epochs", type=int, default=500,
                        help="Number of training epochs.")
    parser.add_argument("--train_sim_rollouts", type=int, default=1)
    parser.add_argument("--train_sim_steps", type=int, default=200)
    parser.add_argument('--test_rollouts', type=int, default=20,
                        help='Number of rollouts in testing.')
    parser.add_argument("--test_steps", type=int, default=200,
                        help="Max number of time steps per rollout in testing.")
    args = parser.parse_args()

    # Create the environment.
    env = gym.make(args.env_name)

    # Build expert policy.
    zoo_policy = PolicyRoboschool(args.expert_name, env)
    expert_policy = lambda x : zoo_policy.policy(x)

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
    model = FullyConnected(layer_dims)

    # Train the model using DAgger.
    policy, parameters = run_dagger(env, model, expert_policy, X_train, Y_train, X_dev, Y_dev, args.iterations, args.num_epochs, args.train_sim_rollouts, args.train_sim_steps)

    print("Writing model parameters to file.")
    with open("datasets/parameters-dagger-" + args.env_name + "-iters" + str(args.iterations) + "-epochs" + str(args.num_epochs) + "-rollouts" + str(args.train_sim_rollouts) + "-steps" + str(args.train_sim_steps) + ".pickle", "wb") as f:
        pickle.dump(parameters, f)

    #TODO: Test the learned policy against the expert policy.
    print("Running trained clone:")
    clone_returns, _, _ = run_agent(env, policy, args.test_rollouts, args.test_steps, args.render)
    #run_agent(...)

    return 0

if __name__ == "__main__":
    main()
