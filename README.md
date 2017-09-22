# neuralnet
a bunch of attempts to get computers to think like people

This is where I'm putting my forays into neural networks and, more generally, machine learning. Most of these are tested with and made for OpenAI Gym, a set of environments made for reinforced learning in Python. These are all my work, and don't use any external ML libraries like tensorflow.

# contains:

- PIDevolve.py was my first foray into machine learning, and doesn't actually involve neural nets, but a PID controller. It was built for, and only works on, problems where the goal is to keep some quantity as constant as possible, e.g. the 'cartpole' environment in OpenAI Gym (https://gym.openai.com/envs/CartPole-v1/). It tunes the coefficients in the PID function through a genetic algorithm.

- neuralnetevolve.py is a general-purpose neural net, with weights adjusted by genetic algorithm. Works in any environment, although results may vary for the more complex ones, and evolution may take quite some time...

- backpropgen1.py teaches a neural network through backwards propagation, a different method of machine learning which actively encourages good behaviour and discourages bad, through a technique known as gradient descent.
