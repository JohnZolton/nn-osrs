import minimal_phosani
import numpy as np
import random
from tinygrad import Tensor, TinyJit, nn
from tinygrad.helpers import trange
import time
import os

# *** hyperparameters ***
BATCH_SIZE = 64
ENTROPY_SCALE = 0.01
REPLAY_BUFFER_SIZE = 1000
PPO_EPSILON = 0.2
HIDDEN_UNITS = 32
LEARNING_RATE = 1e-3
TRAIN_STEPS = 3
EPISODES = 300
DISCOUNT_FACTOR = 0.99

class ActorCritic:
    def __init__(self, in_features, out_features, hidden_state=HIDDEN_UNITS):
        self.l1 = nn.Linear(in_features, hidden_state)
        self.l2 = nn.Linear(hidden_state, out_features)
        
        self.c1 = nn.Linear(in_features, hidden_state)
        self.c2 = nn.Linear(hidden_state, 1)
    def __call__(self, obs: Tensor):
        x = self.l1(obs).tanh()
        act = self.l2(x).log_softmax()
        x = self.c1(obs).relu()
        return act, self.c2(x)

def evaluate(model, test_env):
    obs = test_env.reset()
    total_rew = 0.0
    for _ in range(100):
        act = model(Tensor(obs))[0].argmax().item()
        obs, rew = test_env.step(act)
        total_rew += float(rew)
    return total_rew


if __name__=="__main__":
    env = minimal_phosani.MinimalPhosaniEnv()
    model = ActorCritic(24,12)
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr=LEARNING_RATE)
    
    @TinyJit
    def train_step(x: Tensor, selected_action: Tensor, reward: Tensor, old_log_dist: Tensor):
        


