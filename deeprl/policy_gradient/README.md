# Imitation Learning

Inspired by the Berkeley Deep RL course.

Instead of using MuJoCo I'm using bullet3. So the MuJoCo environments and experts are replaced by bullet3 environments and Roboschool bullet3 experts.

Dependencies: TensorFlow, OpenAI Gym, bullet3, Roboschool.

# Policy Gradient

Policy gradient is a direct whack at unsupervised learning to learn a policy distribution for making decisions. It is a reinforcement learning approach. Generally speaking it loops through the following steps.
1. Collect data executing decisions from policy $pi$.
2. Compute the objective gradient for the data.
3. Update the policy.
