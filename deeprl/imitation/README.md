# Imitation Learning

Inspired by the Berkeley Deep RL course.

Instead of using MuJoCo I'm using bullet3. So the MuJoCo environments and experts are replaced by bullet3 environments and Roboschool bullet3 experts.

Dependencies: TensorFlow, OpenAI Gym, bullet3, Roboschool.

# Behaviorial Cloning

Behaviorial cloning is the process of learning a model for the policy by training on expert-supplied observation-action pairs. So, in its simplest form, you could (for example) train a neural network given observations (inputs) and actions (outputs). An implementation of this can be seen in behavior_clone.py.

At first glance you are tempted to think this will work, and it can kind of appear to work sometimes. If the expert's dataset distribution were to completely cover the observation space then it would in fact work. However expert data tends to be a sparse covering of the observation space. As such, any small error in the learned policy will cause the clone to experience different observations than the expert at testing. And these observations will end up being different from the ones the expert experiences. So there is a mismatch is training and test dataset distributions.

One way to address this problem is to iteratively add observations the clone experiences and have the expert label (choose actions) for the clone-generated observations. Then the policy model is retrained with this additional data. And this loop repeats through training. This method is call DAgger (dataset aggregation). An implementation of this can be seen in dagger.py.

Notice that DAgger requires a potentially much slower training loop. First, clone-generated observations have to be generated. These have to be simulated. Second, the expert has to label the clone-generated observations. This either requires the expert as a function or (in reality) requires additional labeling. So the training loop is not generally a typical machine learning iteration.
