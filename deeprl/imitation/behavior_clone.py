import sys
import argparse
import pickle
import importlib
from os.path import expanduser
sys.path.append(expanduser("~/installs/roboschool/agent_zoo"))

import numpy as np
import tensorflow as tf
import gym
import roboschool #Register roboschool environments

def create_placeholders(n_x, n_y):
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_x])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, n_y])
    return X, Y

def initialize_parameters(layer_dims):
    parameters = {}
    for l in range(1, len(layer_dims)):
        W_str = "W"+str(l)
        parameters[W_str] = tf.get_variable(W_str, shape=[layer_dims[l-1], layer_dims[l]], initializer=tf.contrib.layers.xavier_initializer())
        b_str = "b"+str(l)
        parameters[b_str] = tf.get_variable(b_str, shape=[1, layer_dims[l]], initializer=tf.zeros_initializer())
    return parameters

def forward_propagation(X, parameters, layer_dims):
    #NOTE: Follow the TF convention of X.shape = [num_samples, ...].

    for l in range(1, len(layer_dims)):
        if l == 1:
            out = tf.matmul(X, parameters["W"+str(l)]) + parameters["b"+str(l)]
        else:
            out = tf.matmul(out, parameters["W"+str(l)]) + parameters["b"+str(l)]
        if l < len(layer_dims)-1:
            out = tf.nn.tanh(out)
    return out

def model(X, parameters, layer_dims):
    # Create place holders (input/labels).
    #X, Y = create_placeholders(layer_dims[0], layer_dims[-1])
    #X = tf.placeholder(dtype=tf.float32, shape=[None, layer_dims[0]])

    # Network parameters.
    #parameters = initialize_parameters(layer_dims)

    #
    logits = forward_propagation(X, parameters, layer_dims)
    return logits

def compute_cost(logits, Y):
    #Y = tf.placeholder(dtype=tf.float32, shape=[None, n_y])
    cost = tf.reduce_mean(tf.reduce_sum(tf.pow(logits - Y, 2.), axis=1))
    return cost

def minibatch_shuffle(X, Y, minibatch_size=64):
    # X.shape = [num_samples, ...]
    # Y.shape = [num_samples, ...]
    num_samples = X.shape[0]

    ind = np.random.permutation(num_samples)
    X_shuffle = X[ind, :]
    Y_shuffle = Y[ind, :]

    minibatches = []

    num_full_size = int(num_samples / minibatch_size)
    for i in range(num_full_size):
        X_mini = X_shuffle[i*minibatch_size : (i+1)*minibatch_size, :]
        Y_mini = Y_shuffle[i*minibatch_size : (i+1)*minibatch_size, :]
        minibatches.append((X_mini, Y_mini))
    if num_samples % minibatch_size != 0:
        X_mini = X_shuffle[num_full_size*minibatch_size : num_samples, :]
        Y_mini = Y_shuffle[num_full_size*minibatch_size : num_samples, :]

    return minibatches

#TODO: Need to better structure model...into class.
#      Model should have predict method.
#      Create policy with learned model.

def train(X_train, Y_train, X_dev, Y_dev, layer_dims, learning_rate=.001, minibatch_size=64, num_epochs=1000):
    #layer_dims = [n_x, 100, 100, n_y]
    X, Y = create_placeholders(layer_dims[0], layer_dims[-1])
    parameters = initialize_parameters(layer_dims)
    logits = model(X, parameters, layer_dims) #actions
    cost = compute_cost(logits, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    #NOTE: This is shuffling the data so it loses its sequential nature.
    #      Technically loses information doing this.
    #      Maybe try a recurrent model and sequential rollouts for training?

    with tf.Session() as sess:
        sess.run(init) # Iniitialize the variables.

        costs_train = []
        costs_dev = []
        for epoch in range(num_epochs):

            # Get the minibatches.
            minibatches = minibatch_shuffle(X_train, Y_train, minibatch_size)
            num_minibatches = len(minibatches)

            epoch_cost = 0.
            for minibatch in minibatches:
                # Fetch minibatch data.
                X_mini, Y_mini = minibatch
                feed_dict = {X: X_mini, Y: Y_mini}
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict=feed_dict)
                epoch_cost += minibatch_cost / num_minibatches

            if epoch % 100 == 0 or epoch == num_epochs-1:
                # Check performance on training and dev sets.

                # Dev cost.
                dev_cost = sess.run(cost, feed_dict={X: X_dev, Y: Y_dev})

                costs_train.append(epoch_cost)
                costs_dev.append(dev_cost)

                print("Epoch %d: cost (train) = %f" % (epoch, epoch_cost))
                print("          cost (dev)   = %f" %(dev_cost))

        # Get learned parameters.
        params = sess.run(parameters)
        return params, costs_train, costs_dev

#TODO: Need a better approach to modularity with policy model/precition.

def predictor(x_sample, parameters, layer_dims):
    x_sample = x_sample.reshape(1, -1)
    #print("x_sample.shape =", x_sample.shape)
    X = tf.placeholder(dtype=tf.float32, shape=[1, layer_dims[0]])

    # Convert parameters to tensors.
    parameter_tensors = {}
    for l in range(1, len(layer_dims)):
        W_str = "W" + str(l)
        parameter_tensors[W_str] = tf.convert_to_tensor(parameters[W_str])
        b_str = "b" + str(l)
        parameter_tensors[b_str] = tf.convert_to_tensor(parameters[b_str])

    logits = forward_propagation(X, parameter_tensors, layer_dims)

    #x_sample.shape = [1, ...] (one sample)
    with tf.Session() as sess:
        p = sess.run(logits, feed_dict={X: x_sample})
        return p

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
            #print("action:", action)
            #print("action: (%d, %d)" % (action.shape[0], action.shape[1]), action)

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

    return returns, observations, actions

def run_behavioral_cloning(env_name, expert_name, layer_dims, X_train, Y_train, X_dev, Y_dev, num_epochs, max_timesteps, num_rollouts, render):
    env = gym.make(env_name)

    # Train the policy.
    parameters, costs_train, costs_dev = train(X_train, Y_train, X_dev, Y_dev,
                                               layer_dims=layer_dims, num_epochs=num_epochs)
    #policy = predictor(parameters, layer_dims)
    policy = lambda x : predictor(x, parameters, layer_dims)

    #TODO: Compare the trained policy with the expert policy.
    max_steps = max_timesteps or env.spec.timestep_limit
    clone_returns, _, _ = run_clone(env, policy, num_rollouts, max_steps, render)
    #run_agent(...)

def construct_train_dev(X, Y, train_frac=.9):
    # shape = [num_samples, ...].
    num_samples = X.shape[0]
    num_train = int(num_samples * train_frac)
    ind = np.random.permutation(num_samples)
    X_train = X[ind[:num_train], :]
    Y_train = Y[ind[:num_train], :]
    X_dev = X[ind[num_train:], :]
    Y_dev = Y[ind[num_train:], :]
    return X_train, Y_train, X_dev, Y_dev

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name")
    parser.add_argument("--expert_name")
    parser.add_argument("--obs_file")
    parser.add_argument("--act_file")
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Number of training epochs.")
    parser.add_argument("--max_timesteps", type=int, default=100,
                        help="Max number of time steps per rollout in testing.")
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of rollouts in testing.')
    args = parser.parse_args()

    # Load the expert data.
    with open(args.obs_file, "rb") as f:
        observations = pickle.load(f)
    with open(args.act_file, "rb") as f:
        actions = pickle.load(f)

    X_train, Y_train, X_dev, Y_dev = construct_train_dev(observations, actions)
    print("X_train.shape =", X_train.shape)
    print("Y_train.shape =", Y_train.shape)
    print("X_dev.shape =", X_dev.shape)
    print("Y_dev.shape =", Y_dev.shape)

    layer_dims = [X_train.shape[1], 100, 100, Y_train.shape[1]]
    run_behavioral_cloning(args.env_name, args.expert_name, layer_dims, X_train, Y_train, X_dev, Y_dev, args.num_epochs, args.max_timesteps, args.num_rollouts, args.render)

if __name__ == "__main__":
    main()
