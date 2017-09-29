import os
import time
import inspect
from multiprocessing import Process

import numpy as np
import tensorflow as tf
import gym
import roboschool
#import scipy.signal

import deeprl.policy_gradient.logz as logz #TODO: Why do I have to do this???

#============================================================================================#
# Utilities
#============================================================================================#

def gaussian_log_prob(x_sample, means, log_std):
    #NOTE: Multivariate Gaussian with independent dimensions with same std dev. As such we have
    # N(mu, std) = (2*pi*std^2)^-n/2 * exp(-1/(2*std^2) * sum(xj - muj)^2)
    # logN(mu, std) = -n/2 * log(2*pi*std^2) - 1/(2*std^2) * sum(xj - muj)^2
    #  = -n/2 * log(2*pi) - n/2 * log(std^2) - 1/(2*std^2) * sum((xj-muj)^2)
    #  = -n/2 * log(2*pi) - n * log_std - 1/(2*exp(2*log_std)) * sum((xj-muj)^2)
    assert(x_sample.shape[1] == means.shape[1])
    log_probs = -.5 * tf.log(tf.constant(2*np.pi)) - tf.square(x_sample - means) / tf.exp(2*log_std)
    # size = (n, a)
    # log_prob = sum(log_probs) ==> log_prob.size = (n,)
    return tf.reduce_sum(log_probs, axis=1)

def build_mlp(
        input_placeholder,
        output_size,
        scope,
        #n_layers=2,
        #size=64,
        hidden_layers=[64, 64, 64], #default 3 hidden layers of size 64
        activation=tf.tanh, #TODO: Make this an array
        output_activation=None #default is regression
        ):
    #========================================================================================#
    #                           ----------SECTION 3----------
    # Network building
    #
    # Your code should make a feedforward neural network (also called a multilayer perceptron)
    # with 'n_layers' hidden layers of size 'size' units.
    #
    # The output layer should have size 'output_size' and activation 'output_activation'.
    #
    # Hint: use tf.layers.dense
    #========================================================================================#

    with tf.variable_scope(scope):
        output = input_placeholder
        for l in hidden_layers:
            output = tf.layers.dense(inputs=output,
                                     units=l,
                                     activation=activation,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
        output = tf.layers.dense(inputs=output,
                                 units=output_size,
                                 activation=output_activation,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        return output

def pathlength(path):
    return len(path["reward"])



#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100,
             gamma=1.0,
             min_timesteps_per_batch=1000,
             max_path_length=None,
             learning_rate=5e-3,
             reward_to_go=True,
             animate=True,
             logdir=None,
             normalize_advantages=True,
             nn_baseline=False,
             seed=0,
             # network arguments
             #n_layers=1,
             #size=32
             hidden_layers=[32, 32, 32]
             #TODO: Add argument for activations.
             ):

    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)

    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    #========================================================================================#
    # Notes on notation:
    #
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    #
    # Prefixes and suffixes:
    # ob - observation
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    #
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    #========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Placeholders
    #
    # Need these for batch observations / actions / advantages in policy gradient loss function.
    #========================================================================================#

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    if discrete: #NOTE: If discrete then output is single int within num_output.
        sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) #TODO: doesn't use ac_dim here?
    else: #NOTE: If continuous then output is multiple continuous controls.
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)

    # Define a placeholder for advantages.
    #NOTE: Avdantage is the log likelihood weighting function in the objective.
    #      Recall the objective:
    #      J = E_{tau} [(sum_{t=0}^T log pi(a_t|o_t)) *  R(tau)]
    #        = E_{tau} [(sum_{t=0}^T log pi(a_t|o_t)) *  (sum_{t=0}^T R_t)]
    #        = E_{tau} [(sum_{t=0}^T log pi(a_t|o_t)) *  A(tau)]
    #      Here the advantage (weighting funtion) is the total return (reward) of the trajectory tau.
    #      An alternative formulation of the objective spreads the weighting is perhaps bettern over the trajectory:
    #      For this formulation, bring out the sum on the returns.
    #      The objective becomes
    #      J = E_{tau} [sum_{t=0}^T log pi(a_t|o_t) *  sum_{t'=t}^T R_t']
    #        = E_{tau} [sum_{t=0}^T log pi(a_t|o_t) *  Qhat_t]
    #      Here the advantage (weighting function) is the return (reward) to go following the trajectory tau.
    #      The reward to go (Qhat_t) is an estimate of Q_t.
    #      For either formulation a baseline can be subtracted from the weighting function.
    #      The objective is then
    #      J = E_{tau} [sum_{t=0}^T log pi(a_t|o_t) * A_t]
    #        = E_{tau} [sum_{t=0}^T log pi(a_t|o_t) * (Qhat_t - b_t)]
    #        = E_{tau} [sum_{t=0}^T] log pi(a_t|o_t) * (R(tau) - b(tau))] <-- when summing over entire trajectory
    #NOTE: In any case note that R_t is a scalar so A_t is size (num_batch,)
    # YOUR_CODE_HERE
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)

    #NOTE: Technically need the number of trajectories in the batch.
    sy_num_paths = tf.placeholder(shape=[1,], dtype=tf.float32)


    #========================================================================================#
    #                           ----------SECTION 4----------
    # Networks
    #
    # Make symbolic operations for
    #   1. Policy network outputs which describe the policy distribution.
    #       a. For the discrete case, just logits for each action.
    #
    #       b. For the continuous case, the mean / log std of a Gaussian distribution over
    #          actions.
    #
    #      Hint: use the 'build_mlp' function you defined in utilities.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ob_no'
    #
    #   2. Producing samples stochastically from the policy distribution.
    #       a. For the discrete case, an op that takes in logits and produces actions.
    #
    #          Should have shape [None]
    #
    #       b. For the continuous case, use the reparameterization trick:
    #          The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
    #
    #               mu + sigma * z,         z ~ N(0, I)
    #
    #          This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
    #
    #          Should have shape [None, ac_dim]
    #
    #      Note: these ops should be functions of the policy network output ops.
    #
    #   3. Computing the log probability of a set of actions that were actually taken,
    #      according to the policy.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ac_na', and the
    #      policy network output ops.
    #
    #========================================================================================#

    #NOTE: The probability of a given trajectory (tau) is given by p_theta(tau).
    #      A trajectory is chosen by sampling from this distribution. Hence the
    #      policy is pi_theta(tau) = p_theta(tau).
    #      The objective function is then
    #      J(theta) = E_tau~p_theta(tau) [sum_t^T r(s_t, a_t)]
    #               = E_tau~pi_theta(tau) [sum_t^T r(s_t, a_t)]
    #               = E_tau~pi_theta(tau) [r(tau)] <-- r(tau) = sum_t^T r(s_t, a_t)
    #               = int_tau pi_theta(tau) * r(tau) dtau
    #      To maximize this objective take the derivative wrt theta:
    #      grad_theta J(theta) = int_tau grad_theta pi_theta(tau) * r(tau) dtau
    #                          = int_tau (grad_theta pi_theta(tau)) * r(tau) dtau
    #                          = int_tau (pi_theta(tau) * grad_theta log pi_theta(tau)) * r(tau) dtau <-- identity: p_theta(x) grad_theta log p_theta(x) = p_theta(x) * grad_theta p_theta(x) / p_theta(x) = grad_theta p_theta(x)
    #                          = E_tau~pi_theta(tau) [(grad_theta log pi_theta(tau)) * r(tau)]
    #      Now consider grad_theta log pi_theta(tau). pi_theta(tau) is
    #      pi_theta(tau) = p(s_1) * prod_t^T pi_theta(a_t|s_t) * p(s_t+1 | s_t, a_t)
    #      log pi_theta(tau) = log p(s_1) + sum_t^T log pi_theta(a_t|s_t) + log p(s_t+1 | s_t, a_t)
    #      The gradient of this is then
    #      grad_theta log pi_theta(tau) = sum_t^T log pi_theta(a_t|s_t)
    #      So we get the derivative of the objective to be
    #      grad_theta J(theta) = E_tau~pi_theta(tau) [(sum_t^T grad_theta log pi_theta(a_t|s_t)) * r(tau)]
    #NOTE: Notice that this gradient shows that we are equivalently evaluating the objective
    #      J(theta) = E_tau~pi_theta(tau) [(sum_t^T log pi_theta(a_t|s_t)) * r(tau)]
    #      The log pi sum is the log likelihood of actions in the trajectory.
    #      The objective is then a weighted log likelihood of trajectory actions.
    #      Without the weights (rewards) the policy distribution would be shaped to increase the log prob of actions.
    #      This would work if the input actions were desireable actions (such as imitation learning).
    #      However, because the actions are sampled from the policy itself, then a weighting is required.
    #      The policy distribution will be shaped to favor actions that result in high weights (rewards).
    #NOTE: Now consider the value of log pi_theta(a_t|s_t). What is this?
    #      Case: Discrete action space.
    #        In this case the action is one of a set of categories (such as in classification).
    #        Let the possible actions be c in {1,...,C}. This is equivalent to a one-hot-encoding
    #        a_c = 1(a=c) <-- only 1 if c = a
    #        Then we can write
    #        pi_theta(a|s) = prod_c^C p_c^a_c
    #        The log pi_theta(a|s) can be written
    #        log pi_theta(a|s) = log prod_c^C p_c^a_c
    #                          = sum_c^C a_c * log p_c <-- negative cross entropy of one-hot action
    #                          = a_c * log p_c , where a = c
    #        So the objective becomes
    #        J(theta) = E_tau~pi_theta(tau) [(sum_t^T neg_cross_entropy(a_t, pi_theta(a_t|s_t))) * r(tau)]
    #        Maximizing this objective is then the same as minimizing
    #        J(theta) = E_tau~pi_theta(tau) [(sum_t^T cross_entropy(a_t, pi_theta(a_t|s_t))) * r(tau)]
    #NOTE: The expectation is approximated over a batch of trajectories. It is then
    #      J(theta) ~= 1/N * sum_n^N (sum_t^Tn neg_cross_entropy(a_t, pi_theta(a_t|s_t))) * r(tau)]
    #      But notice that each of the cross entropies are independent samples.
    #      As such the trajectories can be concatenated into one big batch.
    #      J(theta) ~= 1/N * sum_n^N*T neg_cross_entropy(a_n, pi_theta(a_n|s_n)) * r_n

    if discrete:
        # YOUR_CODE_HERE
        #NOTE: Discrete actions.
        #      The neural network is a classifier and outputs expected action.
        #      The policy is a distribution over actions.
        #      So to complete the model we need this distribution.
        #      For the discrete case this is taken as a multinomial distribution.
        #NOTE: We will be learning the policy distribution. This policy is multinomial.
        ## Define the policy distribution.
        sy_logits_na = build_mlp(input_placeholder=sy_ob_no,
                                 output_size=ac_dim,
                                 scope="sy_logits",
                                 hidden_layers=hidden_layers,
                                 activation=tf.tanh, #TODO: What is a good activationfor this?
                                 output_activation=None)
        ## Operation for sampling actions from the policy distribution.
        sy_sampled_ac = tf.multinomial(sy_logits_na, num_samples=1)#[0] #TODO: Remove this zero index. # Hint: Use the tf.multinomial op
        ## Operation for log likelihood of input actions.
        #NOTE: For a discrete action space where only one action can be chosen, the probability distribution is
        #      p(a) = prod p_i^
        #the negative log likelihood is
        #      -log p()
        sy_logprob_n = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(sy_ac_na, ac_dim, 1., 0.), logits=sy_logits_na) #NOTE: negative log likelihoods
        #print("logits:", sy_logits_na)
        #print("labels:", sy_ac_na)

    else:
        # YOUR_CODE_HERE
        #NOTE: Continuous actions.
        #      The neural network is a regression and outputs expected action.
        #      The policy is a distribution over actions.
        #      So to complete the model we need this distribution.
        #      If we assume independent Gaussian distributions over each action dimension,
        #        we can define this distribution as the product of each single-variate Gaussian.
        #NOTE: We will be learning the policy distribution. The policy distribution is
        #        modeled as multi-variate Gaussian. So we need to learn the Gaussian
        #        mean and std dev. The mean is the output of a neural network so we need
        #        to learn the network parameters. The std dev is taken as an additional
        #        parameter outside of the network.
        ## Define the policy distribution.
        ###Compute the mean from the input observations.
        sy_mean = build_mlp(input_placeholder=sy_ob_no,
                            output_size=ac_dim,
                            scope="sy_mean",
                            hidden_layers=hidden_layers,
                            activation=tf.tanh, #TODO: What is a good activation for this?
                            output_activation=None)
        ###The std dev is a learned parameter!
        sy_logstd = tf.get_variable("sy_logstd", [ac_dim], initializer=tf.zeros_initializer()) # logstd should just be a trainable variable, not a network output.
        ## Define an operation for sampling from this policy distribution.
        sy_sampled_ac = tf.random_normal(sy_mean.shape) * tf.exp(sy_logstd) + sy_mean
        ## Define an operation for computing the log_prob of an input set of actions.
        sy_logprob_n = - gaussian_log_prob(sy_ac_na, sy_mean, sy_logstd)  # Hint: Use the log probability under a multivariate gaussian.



    #========================================================================================#
    #                           ----------SECTION 4----------
    # Loss Function and Training Operation
    #========================================================================================#
    #NOTE: The objective is to sum over the sample run the products of
    #        each log_prob_t * advantage_t <-- A = (Q - b)
    #loss = tf.reduce_mean(sy_logprob_n * sy_adv_n) # Loss function that we'll differentiate to get the policy gradient.
    loss = tf.reduce_sum(sy_logprob_n * sy_adv_n) / sy_num_paths[0] # Loss function that we'll differentiate to get the policy gradient.
    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


    #========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline
    #========================================================================================#

    if nn_baseline:
        baseline_prediction = tf.squeeze(build_mlp(
                                sy_ob_no,
                                1,
                                "nn_baseline",
                                #n_layers=n_layers,
                                #size=size
                                hidden_layers=hidden_layers))
        # Define placeholders for targets, a loss function and an update op for fitting a
        # neural network baseline. These will be used to fit the neural network baseline.
        # YOUR_CODE_HERE
        baseline_update_op = TODO


    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101



    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)

        # Collect data (generate trajectories) for this batch.

        # Generate trajectories for this batch.
        # Add paths until the cummulative timesteps > min_timesteps_per_batch.
        #NOTE: One batch might have multiple trajectories.
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]}) #NOTE: Single observation, single action.
                ac = np.squeeze(ac)
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {"observation" : np.array(obs),
                    "reward" : np.array(rewards),
                    "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch


        # Concatenate trajectories on top of each other into one long batch.
        #NOTE: A batch is then
        #      [path1_o1,
        #       ...,
        #       path1_oT1,
        #       path2_o1,
        #       ...,
        #       path2_oT2,
        #       ...]

        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Computing Q-values
        #
        # Your code should construct numpy arrays for Q-values which will be used to compute
        # advantages (which will in turn be fed to the placeholder you defined above).
        #
        # Recall that the expression for the policy gradient PG is
        #
        #       PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
        #
        # where
        #
        #       tau=(s_0, a_0, ...) is a trajectory,
        #       Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
        #       and b_t is a baseline which may depend on s_t.
        #
        # You will write code for two cases, controlled by the flag 'reward_to_go':
        #
        #   Case 1: trajectory-based PG
        #
        #       (reward_to_go = False)
        #
        #       Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over
        #       entire trajectory (regardless of which time step the Q-value should be for).
        #
        #       For this case, the policy gradient estimator is
        #
        #           E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
        #
        #       where
        #
        #           Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
        #
        #       Thus, you should compute
        #
        #           Q_t = Ret(tau)
        #
        #   Case 2: reward-to-go PG
        #
        #       (reward_to_go = True)
        #
        #       Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
        #       from time step t. Thus, you should compute
        #
        #           Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        #
        #
        # Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
        # like the 'ob_no' and 'ac_na' above.
        #
        #====================================================================================#

        # YOUR_CODE_HERE
        # Get the reward to go for each trajectory and each step within a trajectory.
        q_paths = []
        for path in paths:
            num_path = len(path["reward"])
            q_tmp = np.zeros((num_path,))
            tmp = 0
            for qi in reversed(range(num_path)):
                tmp = gamma * tmp + path["reward"][qi]
                #q_tmp[qi] = gamma * tmp + path["reward"][qi]
                q_tmp[qi] = tmp
            q_paths.append(q_tmp)
        # Concatenate into batch.
        if reward_to_go:
            q_n = np.concatenate(q_paths)
        else:
            # For each timestep use the full trajectory's return.
            q_n = np.concatenate([np.ones(q_path.shape) * q_path[0] for q_path in q_paths])
        #print("q_paths:")
        #print(q_paths)
        #print("q_n:")
        #print(q_n)

        #====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        #====================================================================================#

        if nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
            # #bl2 below.)

            b_n = TODO
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            # YOUR_CODE_HERE
            mean_adv = np.mean(adv_n)
            std_adv = np.std(adv_n)
            #print("std_adv:", std_adv)
            adv_n = (adv_n - mean_adv) / std_adv


        #====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if nn_baseline:
            # ----------SECTION 5----------
            # If a neural network baseline is used, set up the targets and the inputs for the
            # baseline.
            #
            # Fit it to the current batch in order to use for the next iteration. Use the
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)

            # YOUR_CODE_HERE
            pass

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on
        # the current batch of rollouts.
        #
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below.

        # YOUR_CODE_HERE
        feed_dict = {sy_ac_na: ac_na,
                     sy_ob_no: ob_no,
                     sy_adv_n: q_n,
                     sy_num_paths: [len(paths)]}
        print("num_batch:", sess.run(tf.shape(sy_ac_na)[0], feed_dict=feed_dict))
        print("num_paths:", sess.run(sy_num_paths[0], feed_dict=feed_dict))
        print("one-hot actions:")
        print(sess.run(tf.one_hot(sy_ac_na, ac_dim, 1., 0.), feed_dict=feed_dict))
        print("logits:")
        print(sess.run(sy_logits_na, feed_dict=feed_dict))
        print("softmax(policy logits):")
        print(sess.run(tf.nn.softmax(sy_logits_na), feed_dict=feed_dict))
        print("cross entropies:")
        print(sess.run(sy_logprob_n, feed_dict=feed_dict))
        print("test action samples:")
        print(sess.run(tf.multinomial(sy_logits_na, num_samples=1), feed_dict=feed_dict))
        #loss_val_prev = sess.run(loss, feed_dict=feed_dict)
        _, loss_val = sess.run([update_op, loss], feed_dict=feed_dict)


        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    #parser.add_argument('--n_layers', '-l', type=int, default=1)
    #parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument("--hidden", "-hl", type=str, default="[32,32,32]",
                        help="hidden layer dimensions [l1,l2,...,lL-1]")
    parser.add_argument("--logdir", type=str, default="logs/policy_gradient")
    args = parser.parse_args()

    #if not(os.path.exists('data')):
    #    os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    #logdir = os.path.join('data', logdir)
    logdir = os.path.join(args.logdir, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    # Parse the hidden layers into array.
    hidden_layers = args.hidden.replace(" ", "").replace("[", "").replace("]", "").split(",")
    print("NN hidden layer dimensions:", hidden_layers)

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline,
                seed=seed,
                #n_layers=args.n_layers,
                #size=args.size
                hidden_layers=hidden_layers
                )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()


if __name__ == "__main__":
    main()
