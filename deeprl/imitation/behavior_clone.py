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

class FullyConnected(object):
    def __init__(self, layer_dims, sess=None):
        #layer_dims: [] layer[0] = input

        self.close_sess = False
        if not sess:
            sess = tf.Session()
            self.close_sess = True
        self.sess = sess

        self.layer_dims = layer_dims

        # Specify data slots.
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, layer_dims[0]])
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, layer_dims[-1]])

        # Initialize the parameters.
        self.parameters = self.initialize_parameters()

        # Define the computation graph.
        self.output = self.forward_propagation(self.X)

        # Define the objective.
        self.cost = self.compute_cost()
    def __del__(self):
        if self.close_sess: self.sess.close()

    def initialize_parameters(self):
        parameters = {}
        for l in range(1, len(self.layer_dims)):
            W_str = "W"+str(l)
            parameters[W_str] = tf.get_variable(W_str, shape=[self.layer_dims[l-1], self.layer_dims[l]], initializer=tf.contrib.layers.xavier_initializer())
            b_str = "b"+str(l)
            parameters[b_str] = tf.get_variable(b_str, shape=[1, self.layer_dims[l]], initializer=tf.zeros_initializer())
        return parameters

    def forward_propagation(self, X):
        #NOTE: Follow the TF convention of X.shape = [num_samples, ...].

        for l in range(1, len(self.layer_dims)):
            if l == 1:
                out = tf.matmul(X, self.parameters["W"+str(l)]) + self.parameters["b"+str(l)]
            else:
                out = tf.matmul(out, self.parameters["W"+str(l)]) + self.parameters["b"+str(l)]
            if l < len(self.layer_dims)-1:
                out = tf.nn.tanh(out)
        return out

    def compute_cost(self):
        #Y = tf.placeholder(dtype=tf.float32, shape=[None, n_y])
        cost = tf.reduce_mean(tf.reduce_sum(tf.pow(self.output - self.Y, 2.), axis=1))
        return cost

    def train(self, X_train, Y_train, X_dev, Y_dev, learning_rate=.001, minibatch_size=64, num_epochs=1000):
        #layer_dims = [n_x, 100, 100, n_y]

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        init = tf.global_variables_initializer()

        #NOTE: This is shuffling the data so it loses its sequential nature.
        #      Technically loses information doing this.
        #      Maybe try a recurrent model and sequential rollouts for training?

        self.sess.run(init) # Iniitialize the variables.

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
                feed_dict = {self.X: X_mini, self.Y: Y_mini}
                _, minibatch_cost = self.sess.run([optimizer, self.cost], feed_dict=feed_dict)
                epoch_cost += minibatch_cost / num_minibatches

            if epoch % 100 == 0 or epoch == num_epochs-1:
                # Check performance on training and dev sets.

                # Dev cost.
                dev_cost = self.sess.run(self.cost, feed_dict={self.X: X_dev, self.Y: Y_dev})

                costs_train.append(epoch_cost)
                costs_dev.append(dev_cost)

                print("Epoch %d: cost (train) = %f" % (epoch, epoch_cost))
                print("          cost (dev)   = %f" %(dev_cost))

        # Get learned parameters.
        params = self.sess.run(self.parameters)
        return params, costs_train, costs_dev

    def predict(self, x_sample):
        x_sample = x_sample.reshape(1, -1)

        p = self.sess.run(self.output, feed_dict={self.X: x_sample})

        #NOTE: This is currently a regression so output is prediction.
        return p

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
