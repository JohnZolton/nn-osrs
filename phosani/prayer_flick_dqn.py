"""
Tinygrad DQN agent for the prayer-flick toy.

Environment:
- 4-tick boss cycle. Attack happens when timer == 1.
- Attack type (0=magic,1=ranged,2=melee) chosen randomly each attack.
- Observation: timer one-hot (4) + previous action one-hot (4) => 8-dim.
- Actions: 0=no prayer, 1=protect_magic, 2=protect_ranged, 3=protect_melee.

Rewards:
- If timer == 1:
    - action == correct protection -> +1.0
    - action == 0 (no prayer) -> -1.0
    - action wrong protection -> -0.5
- If timer != 1:
    - action == 0 -> 0.0
    - action != 0 -> -0.01 (small penalty for unnecessary prayer)
Episodes are fixed-length.

This file implements:
- FlickEnv (toy env)
- QNet (tinygrad MLP)
- ReplayBuffer
- DQN training loop with target network, epsilon-greedy, periodic evaluation

Run:
    python3 phosani/prayer_flick_dqn.py

The script prints per-episode rewards and periodic greedy evaluation.
"""
import random
import time
import numpy as np
from typing import Tuple, Dict, Any
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
import tinygrad.nn.optim as optim

# reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
Tensor.manual_seed(SEED)

TIMER_CYCLE = 4
NUM_ACTIONS = 4  # 0: none, 1: magic, 2: ranged, 3: melee

def one_hot_np(indices, num_classes):
    b = indices.shape[0]
    oh = np.zeros((b, num_classes), dtype=np.float32)
    oh[np.arange(b), indices] = 1.0
    return oh

class FlickEnv:
    def __init__(self, seq_len:int=200, seed:int=None):
        self.seq_len = seq_len
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.reset()

    def reset(self):
        self.timer = TIMER_CYCLE
        self.prev_action = 0
        # pre-generate attack types for the whole episode
        n_attacks = (self.seq_len + TIMER_CYCLE - 1) // TIMER_CYCLE + 1
        self.attack_types = np.random.randint(0, 3, size=(n_attacks,))
        self.attack_idx = 0
        self.t = 0
        return self._get_obs()

    def _get_obs(self):
        t_oh = one_hot_np(np.array([self.timer - 1]), TIMER_CYCLE).reshape(-1)
        prev_oh = one_hot_np(np.array([self.prev_action]), NUM_ACTIONS).reshape(-1)
        return np.concatenate([t_oh, prev_oh]).astype(np.float32)

    def step(self, action:int) -> Tuple[np.ndarray, float, bool, Dict[str,Any]]:
        # compute reward
        reward = 0.0
        done = False
        info = {}
        if self.timer == 1:
            true_attack = int(self.attack_types[self.attack_idx])
            correct_action = true_attack + 1
            if action == correct_action:
                reward = 1.0
            elif action == 0:
                reward = -1.0
            else:
                reward = -0.5
            self.attack_idx += 1
        else:
            if action == 0:
                reward = 0.0
            else:
                reward = -0.01

        self.prev_action = int(action)
        self.timer -= 1
        if self.timer == 0:
            self.timer = TIMER_CYCLE
        self.t += 1
        if self.t >= self.seq_len:
            done = True
        return self._get_obs(), float(reward), done, info

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity:int, state_dim:int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def push(self, s, a, r, ns, d):
        idx = self.ptr % self.capacity
        self.states[idx] = s
        self.next_states[idx] = ns
        self.actions[idx] = a
        self.rewards[idx] = r
        self.dones[idx] = 1.0 if d else 0.0
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size:int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.dones[idxs])

    def __len__(self):
        return self.size

# Q-network (tinygrad)
class QNet:
    def __init__(self, state_dim:int, action_dim:int):
        self.w1 = Tensor.randn(state_dim, 64) * 0.1
        self.b1 = Tensor.zeros(64)
        self.w2 = Tensor.randn(64, 64) * 0.1
        self.b2 = Tensor.zeros(64)
        self.w3 = Tensor.randn(64, action_dim) * 0.1
        self.b3 = Tensor.zeros(action_dim)
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        for p in self.params:
            p.requires_grad_(True)

    def __call__(self, x:Tensor) -> Tensor:
        # x can be (state_dim,) or (B, state_dim)
        if len(x.shape) == 1:
            x = x.reshape(1, x.shape[0])
        h = x.dot(self.w1) + self.b1
        h = h.relu()
        h = h.dot(self.w2) + self.b2
        h = h.relu()
        q = h.dot(self.w3) + self.b3
        return q

    def parameters(self):
        return self.params

# helper: soft copy params
def copy_params(src:QNet, dst:QNet):
    for s,d in zip(src.parameters(), dst.parameters()):
        d.replace(s)

# select action
def select_action(qnet:QNet, state_np:np.ndarray, epsilon:float):
    if np.random.rand() < epsilon:
        return int(np.random.randint(0, NUM_ACTIONS))
    # use direct qnet call for action selection (avoids reliance on a jitted wrapper defined inside train)
    qvals = qnet(Tensor(state_np.astype(np.float32))).numpy()
    qvals = qvals.reshape(-1, NUM_ACTIONS)
    return int(np.argmax(qvals[0]))

# training loop
def train():
    env = FlickEnv(seq_len=200)
    eval_env = FlickEnv(seq_len=200)
    state_dim = len(env._get_obs())
    action_dim = NUM_ACTIONS

    qnet = QNet(state_dim, action_dim)
    qnet_target = QNet(state_dim, action_dim)
    copy_params(qnet, qnet_target)

    # JIT wrappers for faster inference (TinyJit)
    jitted_q = TinyJit(qnet.__call__)
    jitted_q_target = TinyJit(qnet_target.__call__)

    # JIT'd batched train step to speed up inner loop
    def train_step_batch(state_batch_t: Tensor, action_onehot_t: Tensor, reward_t: Tensor, next_state_batch_t: Tensor, done_t: Tensor):
        # q(s, a)
        q_vals = qnet(state_batch_t)  # (B, A)
        selected_q = (q_vals * action_onehot_t).sum(axis=1).reshape(-1, 1)

        # target: use target network max (stable)
        ns_q = qnet_target(next_state_batch_t)  # (B, A)
        max_next = ns_q.max(axis=1).reshape(-1, 1)
        target = reward_t + (1.0 - done_t) * (gamma * max_next)

        diff = selected_q - target
        loss = (diff * diff).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss

    # tinygrad jit wrapper
    jitted_train_step = TinyJit(train_step_batch)

    Tensor.training = True
    opt = optim.Adam(qnet.parameters(), lr=5e-4)

    replay = ReplayBuffer(10000, state_dim)

    episodes = 600
    batch_size = 256
    min_replay = 500
    gamma = 0.99
    target_update = 500  # update target every N training steps
    train_updates_per_step = 4

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    total_steps = 0
    train_steps = 0

    start = time.time()
    for ep in range(episodes):
        s = env.reset()
        ep_reward = 0.0
        done = False
        steps = 0
        while not done:
            a = select_action(qnet, s, epsilon)
            ns, r, done, _ = env.step(a)
            replay.push(s, a, r, ns, done)
            s = ns
            ep_reward += r
            steps += 1
            total_steps += 1

            if len(replay) >= min_replay:
                for _ in range(train_updates_per_step):
                    s_b, a_b, r_b, ns_b, d_b = replay.sample(batch_size)
                    # convert to tensors and prep batch inputs for jitted train step
                    s_t = Tensor(s_b)
                    ns_t = Tensor(ns_b)
                    a_oh = one_hot_np(a_b.astype(np.int32), action_dim)
                    a_t = Tensor(a_oh)
                    r_t = Tensor(r_b.reshape(-1,1))
                    d_t = Tensor(d_b.reshape(-1,1))
                    # run jitted batch train (does forward, loss, backward, opt.step)
                    loss = jitted_train_step(s_t, a_t, r_t, ns_t, d_t)
                    train_steps += 1
                    if train_steps % target_update == 0:
                        copy_params(qnet, qnet_target)

            # end if training
        # update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # periodic eval (greedy)
        if (ep + 1) % 20 == 0 or ep == 0:
            eval_reward = evaluate_policy(qnet, eval_env, episodes=5)
            elapsed = time.time() - start
            print(f"Ep {ep:03d} | Steps {steps:03d} | EpReward {ep_reward:.2f} | EvalReward {eval_reward:.2f} | Eps {epsilon:.3f} | Time {elapsed:.1f}s")
        else:
            print(f"Ep {ep:03d} | EpReward {ep_reward:.2f} | Eps {epsilon:.3f}")

    print("Training finished")

def evaluate_policy(qnet:QNet, env:FlickEnv, episodes:int=10):
    # Use non-jitted qnet for evaluation to avoid relying on outer-scope jitted functions.
    total = 0.0
    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            qvals = qnet(Tensor(s.astype(np.float32))).numpy().reshape(-1, NUM_ACTIONS)
            a = int(np.argmax(qvals[0]))
            s, r, done, _ = env.step(a)
            ep_r += r
        total += ep_r
    return total / episodes

if __name__ == "__main__":
    train()
