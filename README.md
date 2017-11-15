# Deep Reinforcement Learning Sampler

This repository consists of many small projects that test out various deep reinforcement learning methods.

Machine learning is finding/learning an approximation to some system in question. There are many types of machine learning. The most common type of learning is supervised learning. In supervised learning a dataset has been collected of the system's inputs and outputs. The objective is then straight forward...find an approximation that, when injected with the same input, minimizes the error between the approximation output and the system output.

It's not always possible to get an input-output dataset of the system of interest. For example, a self-driving car. With input sensor data, what should the output be to navigate the car? That output is actually unknown. So it is not straight forward to apply supervised learning (although it is possible to try and mimic human drivers). A self-driving car has some input (from sensors) and has some underlying state. Actions it takes affect future states and inputs. With this sequential dependency in mind perhaps we can reward/penalize transitions taken. Then we can choose an approximation of the system that maximizes this reward over the car's trajectory. This is reinforcement learning.

Reinforcement learning approximates a system by maximizing rewards collected by exploring transitions. A lot more is going on in this framework when compared to supervised learning. As such there are many very different methods for reinforcement learning. The different methods are derived based on which aspects of the system are approximated, which are ignored, etc. For example, maybe you want to approximate the physical dynamics or kinematics of a self-driving car. Or maybe you want to abstract away any such physics.

The word "deep" comes into play with how the various components are approximated. "Deep" implies the usage of deep neural network models for function approximation.

Topics so far:
* Imitation learning (from Berkeley Deep RL course).
  * Technically a supervised-learning approach to a reinforcement-learning problem.
* Policy gradient deep RL (from Berkeley Deep RL course).
  * Approximates a distribution over actions.
* Deep Q Networks (DQN) RL (from Berkeley Deep RL course).
  * Approximates rewards to go from states.
* Model-based RL.
  * Approximates dynamics and then plans with that.

Some of the work is derived from the Berkeley Deep RL course. In that course Mujoco is used for physics simulation. I have modified any such usage to instead use bullet3 and Roboschool environments.

This is a work in progress of course since I do work.
