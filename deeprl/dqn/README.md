# Q learning

This project is basically one of the homeworks from Berkeley's deep reinforcement learning course. It is an implementation of DQN to learn to play Atari games using convolutional networks for image processing...so using the same type of input a person would use to play the games.

Q learning is an approach toward reinforcement learning that assumes a greedy policy. The greedy policy is
pi(a|o) = 1 if a == argmax_a' A^pi(o, a)
          0 otherwise
And in particular A^pi(o, a) == Q(o, a), where the pi is dropped because the greedy policy is assumed.

With the policy assumed, it doesn't need to be learned (as in policy gradient or actor critic). Instead, the state-action value function is directly learned. And actions can be determined directly from it.

There are many ways to learn Q. However, most approximators will not be in the exact action space as the optimal Q function. The sad story here is that Q learning with Q-function approximators (deep Q learning in general) will generally not converge to the true Q function. With this said there are ways to help deep Q learning still achieve its real purpose.

The general algorithm consists of many components.
1. Agent experimenting in environment and generating transitions that are stored in a buffer.
2. Training Q function approximator (NN) on random batch of transitions.
3. Copying Q function approximator parameters to target approximator parameters.

Each of these components operate on a different frequency. Component 3 is the slowest so as to improve stability of learning...keeping fixed for a while ensures a non-changing target for some iterations.

A general view of Q learning can then be written as
1. Collect M transitions {(s, a, s', r)} using some policy...add to buffer.
   Agent experimenting in environment.
2. Sample a minibatch {(s, a, s', r)} from buffer.
3. Train Q-function approximator on target Q function from minibatch.
   target(s,a) = rew_batch + gamma * max_a' Q_target(s', a')
   objective = l2_loss(Q(s,a) - target(s,a)) / batch_size
4. Copy Q function approximator to target Q-function approximator.

Here 4 is the slowest. The target Q-function approximator is updated occasionally. The faster it is updated the faster the Q function is learned. However, the slower it is updated the more stable the learning...and it is difficult to stabilize.

1 is the next slowest frequency. This is the data collection step.

2 and 3 train the Q function...they can happen much faster. Note that for DQN 4 operates at the same frequency and right after 2 and 3.
