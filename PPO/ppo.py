from minimal_phosani import MinimalPhosaniEnv
from typing import Tuple
import time
from tinygrad import Tensor, TinyJit, nn
import gymnasium as gym
from tinygrad.helpers import trange
import numpy as np

BATCH_SIZE = 256
ENTROPY_SCALE = 0.0005
REPLAY_BUFFER_SIZE = 2000
PPO_EPSILON = 0.2
HIDDEN_UNITS = 32
LEARNING_RATE = 1e-2
TRAIN_STEPS = 5
EPISODES = 40
DISCOUNT_FACTOR = 0.99

class ActorCritic:
  def __init__(self, in_features, out_features, hidden_state=HIDDEN_UNITS):
    self.l1 = nn.Linear(in_features, hidden_state)
    self.l2 = nn.Linear(hidden_state, out_features)

    self.c1 = nn.Linear(in_features, hidden_state)
    self.c2 = nn.Linear(hidden_state, 1)

  def __call__(self, obs:Tensor) -> Tuple[Tensor, Tensor]:
    x = self.l1(obs).tanh()
    act = self.l2(x).log_softmax()
    x = self.c1(obs).relu()
    return act, self.c2(x)

def evaluate(model:ActorCritic, test_env:gym.Env) -> float:
  (obs, _), terminated, truncated = test_env.reset(), False, False
  total_rew = 0.0
  while not terminated and not truncated:
    act = model(Tensor(obs))[0].argmax().item()
    obs, rew, terminated, truncated, _ = test_env.step(act)
    total_rew += float(rew)
  return total_rew


if __name__ == "__main__":
    env = MinimalPhosaniEnv()
    model = ActorCritic(env.observation_space.shape[0],int(env.action_space.n))
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr=LEARNING_RATE)

