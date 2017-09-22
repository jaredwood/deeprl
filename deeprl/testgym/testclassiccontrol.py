
import gym

env = gym.make("CartPole-v0")
print("Action space:")
print(env.action_space)
print("Observation space:")
print(env.observation_space)

env.reset()
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print("Observation:", observation)
        action = env.action_space.sample()
        print("Action:", action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after %d time steps" % (t+1))
            break
