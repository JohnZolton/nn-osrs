class Environment:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.attack_cycle = 4 # 4 ticks
        self.time_since_attack = 0
        self.attack_type = 1 # 1, 2, 3 for later
        self.active_prayer = 0 # 0 = inactive, then matches with attack_type
        self.tick = 0
        self.max_ticks=40
    
    def _get_obs(self):
        return (
            float(self.tick),
            float(self.time_since_attack),
            float(self.attack_type),
            float(self.active_prayer),
        )
    
    def step(self, action):
        self.tick += 1
        self.active_prayer = action
        self.time_since_attack += 1
        attacked = self.time_since_attack % self.attack_cycle == 0
        
        reward = 0
        if attacked:
            # boss attack
            self.time_since_attack = 0
            if self.active_prayer == self.attack_type:
                # prayed correctly
                reward += 1
            else:
                reward -= 1
        done = self.tick >= self.max_ticks
        info = {
            'tick':self.tick, 
            'time_since_attack':self.time_since_attack, 
            'active_prayer':self.active_prayer, 
            'attacked': attacked
            }
        observation = self._get_obs()

        return (observation, reward, done, info)


ACTION_SPACE = {
    0: 'no_prayer',
    1: 'protect_from_melee'
}

from tinygrad import Tensor, nn
from tinygrad.nn.optim import optim
import numpy as np
import random

class Model:
    def __init__(self, input_dim):
        self.w1 = Tensor.randn(input_dim, 32) * 0.1
        self.b1 = Tensor.zeros(32)
        self.w2 = Tensor.randn(32, 16) * 0.1
        self.b2 = Tensor.zeros(16)
        self.w3 = Tensor.randn(16, len(ACTION_SPACE.items())) * 0.1
        self.b3 = Tensor.zeros(len(ACTION_SPACE.items()))

        # ensure grads
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        for p in self.params:
            p.requires_grad_(True)

    
    def __call__(self, x):
        h = x.dot(self.w1) + self.b1
        h = h.relu()
        h = h.dot(self.w2) + self.b2
        h = h.relu()
        logits = h.dot(self.w3) + self.b3
        return logits

    def parameters(self):
        return self.params

class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.state_dim = state_dim
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = numpy.zeros((capacity,), dtype=np.float32)

    def push(self, s, a, r, ns, d):
        ids = self.ptr % self.capacity
        self.states[idx] = s
        self.next_states[idx] = ns
        self.actions[idx] = a
        self.rewards[idx] = r
        self.dones[idx] = 1.0 if d else 0.0
        self.ptr += 1
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.dones[idxs])
    
    def __len__(self):
        return self.size


replay_buffer = ReplayBuffer(capactiy=1000, state_dim=4)

