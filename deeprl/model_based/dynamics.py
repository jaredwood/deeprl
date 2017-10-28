import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        # Function approximator.
        input_size = env.observation_space.shape[0] + env.action_space.shape[0]
        output_size = env.observation_space.shape[0]
        self.X = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
        self.dynamics_norm = build_mlp(input_placeholder=self.X,
                                       output_size=output_size,
                                       scope="dynamics",
                                       n_layers=n_layers,
                                       size=size,
                                       activation=activation,
                                       output_activation=output_activation)

        # Training cost and optimizer.
        self.Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)
        self.cost = tf.reduce_mean(tf.nn.l2_loss(self.dynamics_norm - self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Needed properties.
        self.batch_size = batch_size
        self.iterations = iterations
        self.sess = sess
        self.mean_obs, self.std_obs, self.mean_deltas, self.std_deltas, self.mean_action, self.std_action = normalization

    def _normalize(self, x, x_mean, x_std):
        x_norm = x - x_mean
        zero_std_cols = x_std == 0
        if len(x.shape) > 0:
            # Batch.
            #NOTE: Don't normalize zero-std columns.
            x_norm[:, ~zero_std_cols] = x_norm[:, ~zero_std_cols] / x_std[~zero_std_cols]
        else:
            # Single sample.
            x_norm[~zero_std_cols] = x_norm[~zero_std_cols] / x_std[~zero_std_cols]
        return x_norm
    def _denormalize(self, x_norm, x_mean, x_std):
        x = x_norm
        zero_std_cols = x_std == 0
        if len(x.shape) > 0:
            # Batch.
            x[:, ~zero_std_cols] = x[:, ~zero_std_cols] * x_std[~zero_std_cols]
        else:
            # Single sample.
            x[~zero_std_cols] = x[~zero_std_cols] * x_std[~zero_std_cols]
        x = x + x_mean
        return x

    def _build_input(self, observations, actions):
        #obs_norm = (observations - self.mean_obs) / (self.std_obs + 1e-12)
        obs_norm = self._normalize(observations, self.mean_obs, self.std_obs)
        #action_norm = (actions - self.mean_action) / (self.std_action + 1e-12)
        action_norm = self._normalize(actions, self.mean_action, self.std_action)
        input_norm = np.concatenate((obs_norm, action_norm), axis=1)
        return input_norm

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
        #print("Fitting NN to data...")
        #print("mean_obs:", self.mean_obs)
        #print("std_obs:", self.std_obs)
        #print("mean_deltas:", self.mean_deltas)
        #print("std_deltas:", self.std_deltas)
        #print("mean_action:", self.mean_action)
        #print("std_action:", self.std_action)

        num_data = data["observations"].shape[0]
        obs_dim = data["observations"].shape[1]
        #act_dim = data["actions"].shape[1]

        # Randomly shuffle data indices for mini-batch training.
        ind = np.arange(num_data)
        np.random.shuffle(ind)

        for index in range(self.iterations):
            # Get the mini batch.
            start = (index * self.batch_size) % num_data
            end = min(start + self.batch_size, num_data)
            batch_ind = ind[start:end]
            #print("batch: start:", start, ", end:", end)

            # Get the data.
            observations = data["observations"][batch_ind, :]
            actions = data["actions"][batch_ind, :]
            next_observations = data["next_observations"][batch_ind, :]

            #print("observations[0]:", observations[0,:])
            #print("next_observations[0]:", next_observations[0,:])
            #print("actions[0]:", actions[0,:])

            # Construct training input.
            input_norm = self._build_input(observations, actions)

            #print("obs_norm range: [", np.min(input_norm[:, :obs_dim]), ",", np.max(input_norm[:, :obs_dim]), "]")
            #print("obs_norm range: [", np.min(input_norm[:, obs_dim:]), ",", np.max(input_norm[:, obs_dim:]), "]")
            #print("obs_norm mean:", np.mean(input_norm[:,:obs_dim], axis=0))
            #print("obs_norm std :", np.std(input_norm[:,:obs_dim], axis=0))
            #print("obs_norm[0]:", input_norm[0,:obs_dim])
            #print("act_norm[0]:", input_norm[0,obs_dim:])

            # Construct training labels.
            deltas = next_observations - observations
            #output_norm = (deltas - self.mean_deltas) / (self.std_deltas + 1e-12)
            output_norm = self._normalize(deltas, self.mean_deltas, self.std_deltas)

            #print("output range: [", np.min(deltas), ",", np.max(deltas), "]")
            #print("output_norm range: [", np.min(output_norm), ",", np.max(output_norm), "]")
            #print("deltas[0]:", deltas[0,:])
            #print("deltas_norm[0]:", output_norm[0,:])
            #print("deltas_norm mean:", np.mean(output_norm, axis=0))
            #print("deltas_norm std :", np.std(output_norm, axis=0))

            # Run optimizer for this mini batch.
            feed_dict = {self.X: input_norm, self.Y: output_norm}
            _, cost = self.sess.run([self.optimizer, self.cost], feed_dict=feed_dict)
            #print("cost:", cost)
        print("Final training cost:", self.sess.run(self.cost, feed_dict=feed_dict))


    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        input_norm = self._build_input(observations=states, actions=actions)
        deltas_norm = self.sess.run(self.dynamics_norm, feed_dict={self.X: input_norm})
        #deltas = deltas_norm * (self.std_deltas + 1e-12) + self.mean_deltas
        deltas = self._denormalize(deltas_norm, self.mean_deltas, self.std_deltas)
        next_observations = deltas + states
        return next_observations
