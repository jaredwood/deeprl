import argparse

from deeprl.roboschool.run_agent import run_agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    agent_data = run_agent("RoboschoolAnt-v1", "RoboschoolAnt_v1_2017jul", args.render, args.max_timesteps, args.num_rollouts)
    print("observations", agent_data["observations"].shape, ":")
    print(agent_data["observations"])
    print("actions", agent_data["actions"].shape, ":")
    print(agent_data["actions"])


if __name__ == "__main__":
    main()
