import numpy as np
import tensorflow as tf

import deeprl.data_util.dataset as ds

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
            minibatches = ds.minibatch_shuffle(X_train, Y_train, minibatch_size)
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
