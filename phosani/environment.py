# Phosani environment + DQN with replay buffer, target network, batched updates and JIT'd training step.
#
# This replaces the simple per-step Q-update with a proper DQN workflow:
#  - ReplayBuffer (experience replay)
#  - Target network (periodically updated)
#  - Batched training updates
#  - JIT-captured batched train function (TinyJit)
#
# Notes:
# - This remains a simple, didactic implementation using tinygrad. It is not
#   production-grade but is organized so you can iterate.
# - JIT capture requires inputs to be tinygrad Tensors and will warm up after a
#   couple calls. The jitted function performs forward, target calculation,
#   MSE loss, backward, and optimizer.step() for a batch.
#
# Run:
#   python3 phosani/environment.py
#
# Recommended further improvements:
# - Add prioritized replay, n-step returns, double DQN, and observation normalization.
# - Profile to confirm the JIT is captured and provides speedup.

import random
from typing import Tuple, Dict, Any, List
import numpy as np
import time

# tinygrad imports
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
import tinygrad.nn.optim as optim

# -----------------------
# Environment / actions
# -----------------------
ACTIONS = {
    0: "attack",
    1: "prayer_magic",
    2: "prayer_ranged",
    3: "prayer_none",
    4: "eat_food",
    5: "change_gear",
}
ACTION_SIZE = len(ACTIONS)


def perform_action(state: Dict[str, Any], action_id: int) -> Dict[str, Any]:
    action_name = ACTIONS.get(action_id, None)
    if action_name is None:
        return state

    player = state

    if action_name == "attack":
        if player["player_cooldown"] <= 0 and not player["is_eating"]:
            dmg = random.randint(player["weapon_min"], player["weapon_max"])
            player["boss_hp"] = max(0, player["boss_hp"] - dmg)
            player["reward"] += dmg * player["reward_per_damage"]
            player["player_cooldown"] = player["weapon_speed"]
            player["last_action"] = "attack"
        else:
            player["reward"] += player["idle_penalty"]
            player["last_action"] = "attack_failed"

    elif action_name == "prayer_magic":
        player["active_prayer"] = 0
        player["last_action"] = "prayer_magic"

    elif action_name == "prayer_ranged":
        player["active_prayer"] = 1
        player["last_action"] = "prayer_ranged"

    elif action_name == "prayer_none":
        player["active_prayer"] = -1
        player["last_action"] = "prayer_none"

    elif action_name == "eat_food":
        if player["food_count"] > 0 and not player["is_eating"]:
            heal = player["eat_heal"]
            player["player_hp"] = min(player["max_player_hp"], player["player_hp"] + heal)
            player["food_count"] -= 1
            player["is_eating"] = True
            player["eat_timer"] = player["eat_recovery"]
            player["player_cooldown"] = max(player["player_cooldown"], player["eat_recovery"])
            player["reward"] += player["eat_reward"]
            player["last_action"] = "eat"
        else:
            player["reward"] += player["idle_penalty"]
            player["last_action"] = "eat_failed"

    elif action_name == "change_gear":
        player["gear_defensive"] = not player.get("gear_defensive", False)
        player["reward"] += player["gear_change_penalty"]
        player["last_action"] = "change_gear"

    return state


class PhosaniEnv:
    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.reset()

    def reset(self) -> Tuple[float, ...]:
        boss_hp = 400
        player_hp = 99
        prayer_points = 99

        boss_attack_speed = 4
        weapon_speed = 4
        weapon_min = 0
        weapon_max = 20

        eat_heal = 20
        eat_recovery = 2

        # reward shaping tuned: stronger reward per damage, smaller idle penalty
        reward_per_damage = 1.5
        eat_reward = 0.2
        idle_penalty = -0.005  # small penalty for invalid/wasted actions
        gear_change_penalty = -0.1

        self.state = {
            "boss_hp": boss_hp,
            "player_hp": player_hp,
            "prayer_points": prayer_points,
            "boss_attack_timer": boss_attack_speed,
            "boss_attack_speed": boss_attack_speed,
            "boss_attack_type": random.choice([0, 1, 2]),
            "player_cooldown": 0,
            "weapon_speed": weapon_speed,
            "weapon_min": weapon_min,
            "weapon_max": weapon_max,
            "weapon_damage_mod": 0,
            "is_eating": False,
            "eat_timer": 0,
            "eat_heal": eat_heal,
            "eat_recovery": eat_recovery,
            "food_count": 5,
            "active_prayer": -1,
            "prayer_drain_per_tick": 0.5,
            "max_player_hp": player_hp,
            "tick": 0,
            "reward": 0.0,
            "reward_per_damage": reward_per_damage,
            "idle_penalty": idle_penalty,
            "eat_reward": eat_reward,
            "gear_defensive": False,
            "gear_change_penalty": gear_change_penalty,
            "last_action": None,
        }
        return self._get_obs()

    def _get_obs(self) -> Tuple[float, ...]:
        s = self.state
        return (
            float(s["boss_hp"]),
            float(s["player_hp"]),
            float(s["prayer_points"]),
            float(s["boss_attack_timer"]),
            float(s["boss_attack_type"]),
            float(s["player_cooldown"]),
            float(s["active_prayer"]),
            float(s["food_count"]),
        )

    def step(self, action: int) -> Tuple[Tuple[float, ...], float, bool, Dict[str, Any]]:
        s = self.state
        s["tick"] += 1
        s["reward"] = 0.0

        perform_action(s, action)

        s["boss_attack_timer"] -= 1
        if s["boss_attack_timer"] <= 0:
            self._resolve_boss_attack()

        if s["player_cooldown"] > 0:
            s["player_cooldown"] = max(0, s["player_cooldown"] - 1)

        if s["is_eating"]:
            s["eat_timer"] -= 1
            if s["eat_timer"] <= 0:
                s["is_eating"] = False
                s["eat_timer"] = 0

        if s["active_prayer"] != -1:
            s["prayer_points"] = max(0.0, s["prayer_points"] - s["prayer_drain_per_tick"])
            if s["prayer_points"] <= 0:
                s["active_prayer"] = -1

        done = False
        reward = s["reward"]
        info: Dict[str, Any] = {"tick": s["tick"], "last_action": s["last_action"]}

        if s["boss_hp"] <= 0:
            # larger positive reward for finishing the fight
            reward += 200.0
            done = True
            info["result"] = "victory"
        elif s["player_hp"] <= 0:
            # reduce harshness of death penalty so gradients aren't dominated by it
            reward -= 50.0
            done = True
            info["result"] = "death"

        return self._get_obs(), float(reward), done, info

    def _resolve_boss_attack(self) -> None:
        s = self.state
        if s["active_prayer"] == s["boss_attack_type"]:
            dmg = 0
        else:
            base_min, base_max = 5, 30
            dmg = random.randint(base_min, base_max)
            if s.get("gear_defensive", False):
                dmg = int(dmg * 0.85)

        s["player_hp"] = max(0, s["player_hp"] - dmg)
        s["reward"] -= float(dmg)

        s["boss_attack_timer"] = s["boss_attack_speed"]
        s["boss_attack_type"] = random.choice([0, 1, 2])

    def render(self) -> None:
        s = self.state
        print(
            f"Tick {s['tick']:04d} | Boss HP: {s['boss_hp']:3d} | "
            f"Player HP: {s['player_hp']:3d} | Food: {s['food_count']} | "
            f"Cooldown: {s['player_cooldown']} | EatTimer: {s['eat_timer']} | "
            f"Prayer: {s['active_prayer']} | BossAttackIn: {s['boss_attack_timer']}"
        )


# -----------------------
# Replay Buffer
# -----------------------
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        idx = self.ptr % self.capacity
        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = 1.0 if done else 0.0
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idxs],
                self.actions[idxs],
                self.rewards[idxs],
                self.next_states[idxs],
                self.dones[idxs])

    def __len__(self):
        return self.size


# -----------------------
# tinygrad Q-network
# -----------------------
class QNet:
    def __init__(self, state_size, action_size):
        self.l1 = Tensor.randn(state_size, 128) * 0.1
        self.l2 = Tensor.randn(128, 128) * 0.1
        self.l3 = Tensor.randn(128, action_size) * 0.1
        # ensure parameters require gradients (tinygrad expects this for autograd)
        self.l1.requires_grad_(True)
        self.l2.requires_grad_(True)
        self.l3.requires_grad_(True)

    def __call__(self, x: Tensor) -> Tensor:
        # Accept x shape (batch, state_size) or (state_size,) as Tensor
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.reshape(1, x.shape[0])
        x = x.dot(self.l1).relu()
        x = x.dot(self.l2).relu()
        x = x.dot(self.l3)
        return x

    def parameters(self):
        return [self.l1, self.l2, self.l3]


# -----------------------
# Utilities
# -----------------------
def one_hot(actions: np.ndarray, num_actions: int) -> np.ndarray:
    b = actions.shape[0]
    oh = np.zeros((b, num_actions), dtype=np.float32)
    oh[np.arange(b), actions] = 1.0
    return oh


# -----------------------
# Training script (DQN)
# -----------------------
if __name__ == "__main__":
    # env and dims
    env = PhosaniEnv(seed=0)
    obs = env.reset()
    state_dim = len(obs)
    action_dim = ACTION_SIZE

    # model + target
    qnet = QNet(state_dim, action_dim)
    qnet_target = QNet(state_dim, action_dim)
    # copy params from qnet to target
    def copy_params(src: QNet, dst: QNet):
        for s, d in zip(src.parameters(), dst.parameters()):
            # copy parameter contents by replacing the destination tensor's uop with the source's uop
            # use Tensor.replace to ensure shape checks and device handling are respected
            d.replace(s)
    copy_params(qnet, qnet_target)

    # enable training
    Tensor.training = True
    # lower learning rate for more stable updates
    optimizer = optim.Adam(qnet.parameters(), lr=5e-4)

    # create JIT wrappers (forward inference and jitted batched training)
    jitted_q = TinyJit(qnet.__call__)

    # -----------------------
    # Evaluation helper
    # -----------------------
    def evaluate_policy(qnet_jit, eval_env, eval_episodes=5, max_steps=2000):
        """Run greedy (epsilon=0) episodes and return average reward and average length."""
        total_reward = 0.0
        total_steps = 0
        for _ in range(eval_episodes):
            obs = eval_env.reset()
            state = np.array(obs, dtype=np.float32)
            done = False
            ep_r = 0.0
            steps = 0
            while not done and steps < max_steps:
                qvals = qnet_jit(Tensor(state)).numpy()
                qvals = qvals.reshape(-1, action_dim)
                action = int(np.argmax(qvals[0]))
                next_obs, reward, done, _ = eval_env.step(action)
                state = np.array(next_obs, dtype=np.float32)
                ep_r += reward
                steps += 1
            total_reward += ep_r
            total_steps += steps
        return total_reward / eval_episodes, total_steps / eval_episodes

    gamma = 0.99
    batch_size = 32
    replay_capacity = 10000
    min_replay_size = 500
    target_update_steps = 1000
    train_updates_per_env_step = 1

    # Replay buffer
    replay = ReplayBuffer(replay_capacity, state_dim)

    # jitted batched training function
    # inputs: state_batch (Tensor: batch x state_dim),
    #         action_onehot (Tensor: batch x action_dim),
    #         reward (Tensor: batch x 1),
    #         next_state_batch (Tensor: batch x state_dim),
    #         done (Tensor: batch x 1)
    def train_step_batch(state_batch_t: Tensor, action_onehot_t: Tensor, reward_t: Tensor, next_state_batch_t: Tensor, done_t: Tensor):
        # predicted Q-values for current states: (batch, action_dim)
        q_values = qnet(state_batch_t)  # (B, A)
        # selected Q for taken actions: sum over actions using one-hot -> (B, 1)
        selected_q = (q_values * action_onehot_t).sum(axis=1).reshape(-1, 1)

        # Double DQN target:
        # 1) use online network to select next action (argmax)
        # 2) use target network to evaluate that action
        next_q_online = qnet(next_state_batch_t)                 # (B, A)
        # Double DQN: select next action (argmax) with online net, then evaluate with target net.
        # detach argmax indices so they don't require grad (prevents integer tensors from appearing in grad targets)
        next_actions = next_q_online.argmax(axis=1).reshape(-1).detach()  # (B,) int tensor detached
        # convert indices to one-hot (float) and use it to select Q from target (avoids integer tensors in autograd targets)
        next_actions_oh = next_actions.one_hot(ACTION_SIZE).reshape(-1, ACTION_SIZE)  # (B, A) float
        next_q_target = qnet_target(next_state_batch_t)          # (B, A)
        selected_next = (next_q_target * next_actions_oh).sum(axis=1).reshape(-1, 1)  # (B,1)

        target = reward_t + (1.0 - done_t) * (gamma * selected_next)
        # loss (MSE)
        diff = (selected_q - target)
        loss = (diff * diff).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    jitted_train_step = TinyJit(train_step_batch)

    # epsilon schedule
    epsilon = 1.0
    # slightly faster decay and higher final epsilon to keep some exploration during training
    epsilon_min = 0.10
    epsilon_decay = 0.990

    total_env_steps = 0
    total_train_steps = 0
    episodes = 400
    start_time = time.time()

    for ep in range(episodes):
        obs = env.reset()
        state = np.array(obs, dtype=np.float32)
        done = False
        ep_reward = 0.0
        steps = 0

        while not done and steps < 2000:
            # select action (epsilon-greedy) using jitted forward
            if np.random.rand() < epsilon:
                action = int(np.random.randint(0, action_dim))
            else:
                qvals = jitted_q(Tensor(state.astype(np.float32))).numpy()
                # qvals shape may be (1, action_dim) if single state - handle that
                qvals = qvals.reshape(-1, action_dim)
                action = int(np.argmax(qvals[0]))

            next_obs, reward, done, info = env.step(action)
            next_state = np.array(next_obs, dtype=np.float32)
            ep_reward += reward

            # store in replay
            replay.push(state, action, reward, next_state, done)

            state = next_state
            steps += 1
            total_env_steps += 1

            # training
            if len(replay) >= min_replay_size:
                for _ in range(train_updates_per_env_step):
                    s_b, a_b, r_b, ns_b, d_b = replay.sample(batch_size)
                    # convert to Tensors
                    s_t = Tensor(s_b)
                    ns_t = Tensor(ns_b)
                    # one-hot actions
                    a_oh = one_hot(a_b, action_dim)
                    a_t = Tensor(a_oh)
                    r_t = Tensor(r_b.reshape(-1, 1))
                    d_t = Tensor(d_b.reshape(-1, 1))
                    # run jitted batched training
                    loss_t = jitted_train_step(s_t, a_t, r_t, ns_t, d_t)
                    total_train_steps += 1

                    # update target
                    if total_train_steps % target_update_steps == 0:
                        copy_params(qnet, qnet_target)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        elapsed = time.time() - start_time
        print(f"Episode {ep:03d} | Steps {steps:03d} | EpReward {ep_reward:.2f} | Replay {len(replay):04d} | Eps {epsilon:.3f} | Time {elapsed:.1f}s")

        # Periodic greedy evaluation
        eval_every = 20
        eval_episodes = 5
        if (ep + 1) % eval_every == 0:
            eval_env = PhosaniEnv(seed=None)
            avg_reward, avg_len = evaluate_policy(jitted_q, eval_env, eval_episodes=eval_episodes)
            print(f"  >>> Eval over {eval_episodes} greedy episodes: AvgReward {avg_reward:.2f}, AvgLen {avg_len:.1f}")

    # Final save / info
    print("Training complete")
