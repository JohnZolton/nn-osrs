# Implementing Backpropagation from scratch

per [Rumelhart et al. 1986](https://gwern.net/doc/ai/nn/1986-rumelhart-2.pdf)

Core intuition: find out which neurons contributed to the error between the networks guess and the correct answer.

We start at the end and work backwards, scaling the blame to each neuron according to its weight and how hard it fired (during the activation function). If a neuron is saturated (fired hard or not at all), it can't move much, so it gets less blame.
