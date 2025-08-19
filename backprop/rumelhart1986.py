"""
https://gwern.net/doc/ai/nn/1986-rumelhart-2.pdf

Backpropagation
find out how much each neuron of each layer contributed to the output error
We find this by taking the partial derivative of the error WRT the neuron's output
(dE/dy) and the partial WRT to the input (dE/dx)

Equations:
    E = 1/2(sum over training cases(sum over units of the output layer(y-d)^2))
    dE/dy = (output) - (desired)
    dE/dx = dE/dy * dy/dx
    chain rule:
    dE/dx = dE/dy * y * (1-y)

"""

# XOR example
# [0,1] = True
# [1,0] = True
# [1,1] = False
# [0,0] = False

import random
import math

weights_1 = [[random.random() for _ in range(2)] for _ in range(2)] # shape (2x2)
bias_1 = [random.random() for _ in range(2)] # shape (2,)
weights_2 = [random.random() for _ in range(2)] # shape (1x2)
bias_2 = random.random() # scalar bias

def sigmoid_activation(x):
    return 1/(1+math.exp(-x))

def forward_pass(x):
    # hidden pre-activation
    z1 = [sum(weights_1[i][j] * x[j] for j in range(2)) + bias_1[i] for i in range(2)]
    # hidden activation
    h = [sigmoid_activation(z) for z in z1]
    
    # output preactivation
    z2 = sum(weights_2[j] * h[j] for j in range(2)) + bias_2
    # output activation
    y = sigmoid_activation(z2)
    return y, h, z1, z2

def backward_pass(x, y, t, h):
    delta_output = (y-t)*y*(1-y) # dE/dx at the output
    
    output_gradients = [delta_output * h_i for h_i in h]
    output_bias_gradient = delta_output
    
    hidden_deltas = [delta_output * w * h_i * (1-h_i) for w, h_i in zip(weights_2, h)]
    
    hidden_gradients = [
        [delta_h * x_j for x_j in x]
        for delta_h in hidden_deltas
    ]
    
    hidden_bias_gradients = hidden_deltas
    
    return hidden_gradients, hidden_bias_gradients, output_gradients, output_bias_gradient

    
# XOR dataset
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
targets = [0, 1, 1, 0]

learning_rate = 0.1
epochs = 20000

for epoch in range(epochs):
    print(f"\n--- Epoch {epoch+1} ---")
    total_loss = 0

    for x, t in zip(inputs,targets):
        y, h, _, _ = forward_pass(x)
        loss = 0.5 * (y-t)**2
        total_loss += loss

        hidden_grads, hidden_bias_grads, output_grads, output_bias_grad = backward_pass(x, y, t, h)
        
        # update hidden layer weights
        for i in range(2):
            for j in range(2):
                weights_1[i][j] -= learning_rate * hidden_grads[i][j]
        # update hidden bias
        for i in range(2):
            bias_1[i] -= learning_rate * hidden_bias_grads[i]
        
        for i in range(2):
            weights_2[i] -= learning_rate * output_grads[i]

        bias_2 -= learning_rate * output_bias_grad
        print(f"Input: {x}, Target: {t}, Output: {y:.4f}, Loss: {loss:.4f}")
    print(f"Total loss this epoch: {total_loss:.4f}")
    
