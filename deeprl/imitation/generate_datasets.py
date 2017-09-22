import argparse
import pickle

import numpy as np

from deeprl.roboschool.run_agent import run_agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name")
    parser.add_argument("--expert_name")
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument("--num_dev", type=int)
    args = parser.parse_args()

    #expert_data = run_agent("RoboschoolAnt-v1", "RoboschoolAnt_v1_2017jul", args.render, args.max_timesteps, args.num_rollouts)
    expert_data = run_agent(args.env_name, args.expert_name, args.render, args.max_timesteps, args.num_rollouts)
    print("observations", expert_data["observations"].shape, ":")
    print(expert_data["observations"])
    print("actions", expert_data["actions"].shape, ":")
    print(expert_data["actions"])

    num_samples = expert_data["observations"].shape[0]

    # Write data to file.
    with open("datasets/observations-imitation-"+args.expert_name+".pickle", "wb") as f:
        pickle.dump(expert_data["observations"], f)
    with open("datasets/actions-imitation-"+args.expert_name+".pickle", "wb") as f:
        pickle.dump(expert_data["actions"], f)

    """
    # Shuffle data.
    shuffle_ind = np.random.permutation(num_samples)
    train_ind = shuffle_ind[:num_samples-args.num_dev]
    dev_ind = shuffle_ind[args.num_dev:]
    ##training data.
    train_obs = expert_data["observations"][train_ind][:]
    train_act = expert_data["actions"][train_ind][:]
    ##dev data.
    dev_obs = expert_data["observations"][dev_ind][:]
    dev_act = expert_data["actions"][dev_ind][:]

    # Write data to file.
    ##
    with open("train-obs-imitation-"+args.expert_name+".pickle", "wb") as f:
        pickle.dump(train_obs, f)
    with open("train-act-imitation-"+args.expert_name+".pickle", "wb") as f:
        pickle.dump(train_act, f)
    ##
    with open("dev-obs-imitation-"+args.expert_name+".pickle", "wb") as f:
        pickle.dump(dev_obs, f)
    with open("dev-act-imitation-"+args.expert_name+".pickle", "wb") as f:
        pickle.dump(dev_act, f)
    """

if __name__ == "__main__":
    main()
