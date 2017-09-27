import numpy as np
#import gym
#import roboschool #Register roboschool environments

def run_agent(env, policy, num_rollouts, max_steps, render=True):
    returns = []
    observations = []
    actions = []

    #TODO: Need a steps list to index obs/acts starts corresponding to returns.

    max_steps = max_steps or env.spec.timestep_limit

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

    #print("returns", returns)
    print("episode %d: mean(return)" % i, np.mean(returns))
    print("            std(return)", np.std(returns))

    return returns, observations, actions
