# Imitation Learning

Inspired by the Berkeley Deep RL course.

Instead of using MuJoCo I'm using bullet3. So the MuJoCo environments and experts are replaced by bullet3 environments and Roboschool bullet3 experts.

Dependencies: TensorFlow, OpenAI Gym, bullet3, Roboschool.

# Policy Gradient

Policy gradient is a direct whack at unsupervised learning to learn a policy distribution for making decisions. It is a reinforcement learning approach. Generally speaking it loops through the following steps.
1. Collect data executing decisions from policy $pi$.
2. Compute the objective gradient for the data.
3. Update the policy.

Two types of action spaces are considered.
1. Action is selected from a set of possible actions (discrete).
2. Action is multi-dimensional vector of continuous values.

In both cases a neural network models the policy function. The input to the neural network is the observations. In the discrete case the number of outputs is the number of categories a single discrete action can take. Note that if there are multiple discrete actions, their categories can be permuted to form a single discrete variable. So this model can in fact take multiple discrete variables as actions. But the number of categories is large. In the continuous case the number of outputs is the number of continuous-valued actions. For the continuous case, if multiple actions are required then multiple variables are also required.

For the discrete case the output of the neural network is category unnormalized log probabilities (or logits). These logits are fed to a softmax to get the category probabilities. The likelihood of a given action is then the category probability associated with that action. So for a discrete action space (only one category chosen at a time) the probability of an action (action likelihood) is
  p(a=i) = prod p_i
This is generalized by one-hot encoding an action (a_i = 1 if a==i else 0). Then the action probability (likelihood) is then
  prod_i p_i^a_i <-- product = p_i
And the negative log likelihood is
  -log p(a) = - sum_i a_i * log p_i
            = H(a, p) <-- cross entropy of p(a) with one-hot-encoding actions
This cross entropy (negative log likelihood) is used for training the policy.

For the continuous case the output of the neural network is regressions in the action space. This action space is considered disjoint and independent. So each output is taken as the estimated mean value for corresponding action variables. Assuming Gaussian distributions, additional standard deviation parameters are added to the learned model and the action likelihoods are then their Gaussian deviation from the mean. And the neural network is trained with the associated negative log likelihood.

# What is the objective?

In supervised learning each sample has a label. As such each sample specifies a desired outcome. So you can directly optimize the label likelihoods.

The reinforcement learning problem is different. There are no labels. Instead there are actions. Actions can be viewed as a generalization of labels. Whereas all labels are desired outcomes, actions may have varying degree of desirability. So you cannot directly optimize the action likelihoods. Instead, action likelihoods need to be weighted by their degree of desirability. But an action alone is insufficient to determine its desirability. You need to look at the trajectory from which the action was taken. The desirability of trajectory is more easily determined...it either accomplishes the goal or it doesn't. So the trajectory is then used to decide the weighting for a given action. And the weighted action likelihood can then be optimized. This is in fact the objective of policy gradient methods.

Let's formalize this objective. The probability of a given trajectory (tau) is given by p_theta(tau). A trajectory is chosen by sampling from this distribution. Hence the policy is pi_theta(tau) = p_theta(tau). The objective function is then
  J(theta) = E_tau~p_theta(tau) [sum_t^T r(s_t, a_t)]
           = E_tau~pi_theta(tau) [sum_t^T r(s_t, a_t)]
           = E_tau~pi_theta(tau) [r(tau)] <-- r(tau) = sum_t^T r(s_t, a_t)
           = int_tau pi_theta(tau) * r(tau) dtau
To maximize this objective take the derivative wrt theta:
  grad_theta J(theta) = int_tau grad_theta pi_theta(tau) * r(tau) dtau
                      = int_tau (grad_theta pi_theta(tau)) * r(tau) dtau
                      = int_tau (pi_theta(tau) * grad_theta log pi_theta(tau)) * r(tau) dtau <-- identity: p_theta(x) grad_theta log p_theta(x) = p_theta(x) * grad_theta p_theta(x) / p_theta(x) = grad_theta p_theta(x)
                      = E_tau~pi_theta(tau) [(grad_theta log pi_theta(tau)) * r(tau)]
Now consider grad_theta log pi_theta(tau). pi_theta(tau) is
  pi_theta(tau) = p(s_1) * prod_t^T pi_theta(a_t|s_t) * p(s_t+1 | s_t, a_t)
  log pi_theta(tau) = log p(s_1) + sum_t^T log pi_theta(a_t|s_t) + log p(s_t+1 | s_t, a_t)
The gradient of this is then
  grad_theta log pi_theta(tau) = sum_t^T log pi_theta(a_t|s_t)
So we get the derivative of the objective to be
  grad_theta J(theta) = E_tau~pi_theta(tau) [(sum_t^T grad_theta log pi_theta(a_t|s_t)) * r(tau)]
Notice that this gradient shows that we are equivalently evaluating the objective
  J(theta) = E_tau~pi_theta(tau) [(sum_t^T log pi_theta(a_t|s_t)) * r(tau)]
The log pi sum is the log likelihood of actions in the trajectory. The objective is then a weighted log likelihood of trajectory actions. Without the weights (rewards) the policy distribution would be shaped to increase the log prob of actions. This would work if the input actions were desirable actions (such as imitation learning). However, because the actions are sampled from the policy itself, then a weighting is required. The policy distribution will be shaped to favor actions that result in high weights (rewards).

Now consider the value of log pi_theta(a_t|s_t). What is this?
Case: Discrete action space.
In this case the action is one of a set of categories (such as in classification). Let the possible actions be c in {1,...,C}. This is equivalent to a one-hot-encoding
  a_c = 1(a=c) <-- only 1 if c = a
Then we can write
  pi_theta(a|s) = prod_c^C p_c^a_c
The log pi_theta(a|s) can be written
  log pi_theta(a|s) = log prod_c^C p_c^a_c
                    = sum_c^C a_c * log p_c <-- negative cross entropy of one-hot action
                    = a_c * log p_c , where a = c
So the objective becomes
  J(theta) = E_tau~pi_theta(tau) [(sum_t^T neg_cross_entropy(a_t, pi_theta(a_t|s_t))) * r(tau)]
Maximizing this objective is then the same as minimizing
  J(theta) = E_tau~pi_theta(tau) [(sum_t^T cross_entropy(a_t, pi_theta(a_t|s_t))) * r(tau)]

The expectation is approximated over a batch of N trajectories. It is then
  J(theta) ~= 1/N * sum_n^N (sum_t^Tn neg_cross_entropy(a_t, pi_theta(a_t|s_t))) * r(tau)]
But notice that each of the cross entropies are independent samples. As such the trajectories can be concatenated into one big batch.
  J(theta) ~= 1/N * sum_n^N*T neg_cross_entropy(a_n, pi_theta(a_n|s_n)) * r_n
To achieve this the trajectories (paths) are concatenated on top of each other into one long batch. A batch is then
  [path1_o1,
   ...,
   path1_oT1,
   path2_o1,
   ...,
   path2_oT2,
   ...,
   pathN_oTN]

# Advantage?

Above we have used the trajectory return as the direct action likelihood weighting function. Although this is the theoretical goal, we can generalize the weighting function and in the process improve performance. So let's rename the weighting to "advantage."

Advantage is the log likelihood weighting function in the objective. Recall the objective:
  J = E_{tau} [(sum_{t=0}^T log pi(a_t|o_t)) *  R(tau)]
    = E_{tau} [(sum_{t=0}^T log pi(a_t|o_t)) *  (sum_{t=0}^T R_t)]
    = E_{tau} [(sum_{t=0}^T log pi(a_t|o_t)) *  A(tau)]
Here the advantage (weighting function) is the total return (reward) of the trajectory tau. Alternatively consider bringing out the sum on the returns. The objective becomes
  J = E_{tau} [sum_{t=0}^T log pi(a_t|o_t) *  sum_{t'=t}^T R_t']
    = E_{tau} [sum_{t=0}^T log pi(a_t|o_t) *  Qhat_t]
Here the advantage (weighting function) is then the return (reward) to go following the trajectory tau. The reward to go (Qhat_t) is an estimate of Q_t. For either formulation a baseline can be subtracted from the weighting function. The objective is then
J = E_{tau} [sum_{t=0}^T log pi(a_t|o_t) * A_t]
  = E_{tau} [sum_{t=0}^T log pi(a_t|o_t) * (Qhat_t - b_t)]
  = E_{tau} [sum_{t=0}^T] log pi(a_t|o_t) * (R(tau) - b(tau))] <-- when summing over entire trajectory

#TODO

* Parallelize trajectory sampling.
* Implement GAE-lambda for advantage estimation. Does it speed up training?
* Explore multiple policy gradient update steps in a single iteration and compare with standard single step. Will need to use importance sampling or similar.
