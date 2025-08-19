# Implementing Backpropagation from scratch

per [Rumelhart et al. 1986](https://gwern.net/doc/ai/nn/1986-rumelhart-2.pdf)

Core intuition: find out which neurons contributed to the error between the networks guess and the correct answer.

We start at the end and work backwards, scaling the blame to each neuron according to its weight and how hard it fired (during the activation function). If a neuron is saturated (fired hard or not at all), it can't move much, so it gets less blame.

# LeNet 1998

per [LeCun 1998](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

Core intuition: add convolutional layers and pooling layers to reduce parameter count while retaining accuracy. Error is backprop'd through the fully connected, pooled and convolutional layers and blame is distributed accordingly
