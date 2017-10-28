import numpy as np
import tensorflow as tf
import gym
from dynamics import NNDynamicsModel
from controllers import MPCcontroller, RandomController
from cost_functions import cheetah_cost_fn, trajectory_cost_fn
import time
import logz
import os
import copy
import matplotlib.pyplot as plt
#from cheetah_env import HalfCheetahEnvNew
#from roboschool.gym_mujoco_walkers import RoboschoolHalfCheetah
import roboschool
#from roboschool.gym_mujoco_walker import RoboschoolHalfCheetah

def sample(env,
           controller,
           num_paths=10,
           horizon=1000,
           render=False,
           verbose=False):
    """
        Write a sampler function which takes in an environment, a controller (either random or the MPC controller),
        and returns rollouts by running on the env.
        Each path can have elements for observations, next_observations, rewards, returns, actions, etc.
    """
    paths = []
    """ YOUR CODE HERE """
    for path_i in range(num_paths):
        obs = env.reset()
        done = False
        observations = []; actions = []; next_observations = []; rewards = []
        for _ in range(horizon):
            #if done: break
            #observations.append(obs)
            act = controller.get_action(obs)
            #print("***** ACTION:", act, "*****")
            #actions.append(act)
            next_obs, rew, done, _ = env.step(act)
            if done: break #NOTE: Don't let a done state be added.
            #next_observations.append(obs)
            #rewards.append([rew])
            if np.sum(np.isnan(next_obs)) > 0 or np.sum(np.isinf(next_obs)) > 0:
                # Infinite state so abort.
                print("****** Aborting path due to nan/inf *********")
                break
            #print("obs:", obs)
            #print("nxt:", next_obs)
            observations.append(obs)
            actions.append(act)
            next_observations.append(next_obs)
            rewards.append([rew])
            obs = next_obs
            if render and path_i == num_paths-1: env.render()
        path = {"observations": np.array(observations),
                "actions": np.array(actions),
                "next_observations": np.array(next_observations),
                "rewards": np.array(rewards)}
        paths.append(path)

    return paths

# Utility to compute cost a path for a given cost function
def path_cost(cost_fn, path):
    return trajectory_cost_fn(cost_fn, path['observations'], path['actions'], path['next_observations'])

def compute_normalization(data):
    """
    Write a function to take in a dataset and compute the means, and stds.
    Return 6 elements: mean of s_t, std of s_t, mean of (s_t+1 - s_t), std of (s_t+1 - s_t), mean of actions, std of actions
    """

    """ YOUR CODE HERE """
    mean_obs = np.mean(data["observations"], axis=0)
    std_obs = np.std(data["observations"], axis=0)
    mean_deltas = np.mean(data["next_observations"] - data["observations"], axis=0)
    std_deltas = np.std(data["next_observations"] - data["observations"], axis=0)
    mean_action = np.mean(data["actions"], axis=0)
    std_action = np.std(data["actions"], axis=0)

    return mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action


def plot_comparison(env, dyn_model):
    """
    Write a function to generate plots comparing the behavior of the model predictions for each element of the state to the actual ground truth, using randomly sampled actions.
    """
    """ YOUR CODE HERE """
    pass

def train(env,
         cost_fn,
         logdir=None,
         render=False,
         learning_rate=1e-3,
         onpol_iters=10,
         dynamics_iters=60,
         batch_size=512,
         num_paths_random=10,
         num_paths_onpol=10,
         num_simulated_paths=10000,
         env_horizon=1000,
         mpc_horizon=15,
         n_layers=2,
         size=500,
         activation=tf.nn.relu,
         output_activation=None
         ):

    """

    Arguments:

    onpol_iters                 Number of iterations of onpolicy aggregation for the loop to run.

    dynamics_iters              Number of iterations of training for the dynamics model
    |_                          which happen per iteration of the aggregation loop.

    batch_size                  Batch size for dynamics training.

    num_paths_random            Number of paths/trajectories/rollouts generated
    |                           by a random agent. We use these to train our
    |_                          initial dynamics model.

    num_paths_onpol             Number of paths to collect at each iteration of
    |_                          aggregation, using the Model Predictive Control policy.

    num_simulated_paths         How many fictitious rollouts the MPC policy
    |                           should generate each time it is asked for an
    |_                          action.

    env_horizon                 Number of timesteps in each path.

    mpc_horizon                 The MPC policy generates actions by imagining
    |                           fictitious rollouts, and picking the first action
    |                           of the best fictitious rollout. This argument is
    |                           how many timesteps should be in each fictitious
    |_                          rollout.

    n_layers/size/activations   Neural network architecture arguments.

    """

    logz.configure_output_dir(logdir)

    #========================================================
    #
    # First, we need a lot of data generated by a random
    # agent, with which we'll begin to train our dynamics
    # model.

    random_controller = RandomController(env)

    """ YOUR CODE HERE """
    # Get rollouts.
    rand_paths = sample(env, random_controller, num_paths_random, env_horizon, render=False)
    print("rand_paths initial states:")
    for path_i in range(5):
        print(rand_paths[path_i]["observations"][0,:])
    # Concatenate rollouts.
    data = {}
    data["observations"] = np.concatenate([path["observations"] for path in rand_paths])
    data["actions"] = np.concatenate([path["actions"] for path in rand_paths])
    data["next_observations"] = np.concatenate([path["next_observations"] for path in rand_paths])
    data["rewards"] = np.concatenate([path["rewards"] for path in rand_paths])

    print("observations:", data["observations"].shape)
    print("std:", np.std(data["observations"], axis=0))
    print("next_observations:", data["next_observations"].shape)
    print("std:", np.std(data["next_observations"], axis=0))


    #========================================================
    #
    # The random data will be used to get statistics (mean
    # and std) for the observations, actions, and deltas
    # (where deltas are o_{t+1} - o_t). These will be used
    # for normalizing inputs and denormalizing outputs
    # from the dynamics network.
    #
    """ YOUR CODE HERE """
    normalization = compute_normalization(data)


    #========================================================
    #
    # Build dynamics model and MPC controllers.
    #
    sess = tf.Session()

    dyn_model = NNDynamicsModel(env=env,
                                n_layers=n_layers,
                                size=size,
                                activation=activation,
                                output_activation=output_activation,
                                normalization=normalization,
                                batch_size=batch_size,
                                iterations=dynamics_iters,
                                learning_rate=learning_rate,
                                sess=sess)

    mpc_controller = MPCcontroller(env=env,
                                   dyn_model=dyn_model,
                                   horizon=mpc_horizon,
                                   cost_fn=cost_fn,
                                   num_simulated_paths=num_simulated_paths)


    #========================================================
    #
    # Tensorflow session building.
    #
    sess.__enter__()
    tf.global_variables_initializer().run()

    #========================================================
    #
    # Take multiple iterations of onpolicy aggregation at each iteration refitting the dynamics model to current dataset and then taking onpolicy samples and aggregating to the dataset.
    # Note: You don't need to use a mixing ratio in this assignment for new and old data as described in https://arxiv.org/abs/1708.02596
    #
    for itr in range(onpol_iters):
        """ YOUR CODE HERE """

        # Train the dynamics model.
        print("Fitting dynamics model...")
        dyn_model.fit(data) #TODO: Seems there may be a problem with this model...next_states are exploding.
        #TODO: Plot training error.

        # Collect multiple rollouts using model-based planning for actions.
        print("Sampling set of on-policy paths...")
        paths = sample(env, mpc_controller, num_paths=num_paths_onpol, horizon=env_horizon, render=render)

        # Concatenate on-policy paths.
        print("Concatenating new data...")
        observations = np.concatenate([path["observations"] for path in rand_paths])
        actions = np.concatenate([path["actions"] for path in rand_paths])
        next_observations = np.concatenate([path["next_observations"] for path in rand_paths])
        rewards = np.concatenate([path["rewards"] for path in rand_paths])

        # Append transition to data set.
        print("Appending new transitions to dataset...")
        data["observations"] = np.concatenate((data["observations"], observations))
        data["actions"] = np.concatenate((data["actions"], actions))
        data["next_observations"] = np.concatenate((data["next_observations"], next_observations))
        data["rewards"] = np.concatenate((data["rewards"], rewards))

        # Recompute normalization with addition of data.
        print("Recomputing normalization...")
        normalization = compute_normalization(data)
        print("normalization (std devs):")
        for norm_i in range(len(normalization)):
            if norm_i % 2 > 0:
                print(" ", norm_i, ":", normalization[norm_i])

        # Progress metrics.
        costs = np.array([path_cost(cost_fn, path) for path in paths])
        returns = np.array([np.sum(path["rewards"]) for path in paths])

        # LOGGING
        # Statistics for performance of MPC policy using
        # our learned dynamics model
        logz.log_tabular('Iteration', itr)
        # In terms of cost function which your MPC controller uses to plan
        logz.log_tabular('AverageCost', np.mean(costs))#TODO
        logz.log_tabular('StdCost', np.std(costs))#TODO
        logz.log_tabular('MinimumCost', np.min(costs))#TODO
        logz.log_tabular('MaximumCost', np.max(costs))#TODO
        # In terms of true environment reward of your rolled out trajectory using the MPC controller
        logz.log_tabular('AverageReturn', np.mean(returns))#TODO
        logz.log_tabular('StdReturn', np.std(returns))#TODO
        logz.log_tabular('MinimumReturn', np.min(returns))#TODO
        logz.log_tabular('MaximumReturn', np.max(returns))#TODO

        logz.dump_tabular()

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='RoboschoolHalfCheetahNew-v0')
    # Experiment meta-params
    parser.add_argument('--exp_name', type=str, default='mb_mpc')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--render', action='store_true')
    # Training args
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--onpol_iters', '-n', type=int, default=1)
    parser.add_argument('--dyn_iters', '-nd', type=int, default=60)
    parser.add_argument('--batch_size', '-b', type=int, default=512)
    # Data collection
    parser.add_argument('--random_paths', '-r', type=int, default=10)
    parser.add_argument('--onpol_paths', '-d', type=int, default=10)
    parser.add_argument('--simulated_paths', '-sp', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=int, default=1000)
    # Neural network architecture args
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=500)
    # MPC Controller
    parser.add_argument('--mpc_horizon', '-m', type=int, default=15)
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Make data directory if it does not already exist
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    # Make env
    if args.env_name in ("RoboschoolHalfCheetah-v1", "RoboschoolHalfCheetahNew-v0"):
        #env = HalfCheetahEnvNew()
        env = gym.make(args.env_name)
        #env = RoboschoolHalfCheetah()#frame_skip=1)
        cost_fn = cheetah_cost_fn
    else:
        raise NotImplementedError("ATTEMPTING TO RUN ON NON-CHEETAH ENVIRONMENT...")
    train(env=env,
          cost_fn=cost_fn,
          logdir=logdir,
          render=args.render,
          learning_rate=args.learning_rate,
          onpol_iters=args.onpol_iters,
          dynamics_iters=args.dyn_iters,
          batch_size=args.batch_size,
          num_paths_random=args.random_paths,
          num_paths_onpol=args.onpol_paths,
          num_simulated_paths=args.simulated_paths,
          env_horizon=args.ep_len,
          mpc_horizon=args.mpc_horizon,
          n_layers = args.n_layers,
          size=args.size,
          activation=tf.nn.relu,
          output_activation=None,
          )

if __name__ == "__main__":
    main()
