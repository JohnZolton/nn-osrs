import numpy as np
import random
from tinygrad import Tensor, TinyJit, nn
from tinygrad.helpers import trange
import time
import os

ATTACKS = {0: "melee", 1: "range", 2: "magic"}
ACTIONS = {0: "noop", 1: "melee prayer", 2: "range prayer", 3: "magic prayer"}

class Prayer_Flick_Env:
    def __init__(self):
        # No pre-generated sequence - attacks will be chosen randomly
        self.reset()
    def reset(self):
        self.ticks_until_damage = 4
        self.attack_type = random.randint(0, 2)  # Pick first attack randomly
        self.prayer_state = 0
        self.total_reward = 0
        return self._get_obs()
    
    def step(self, action):
        reward = 0
        
        # Set prayer state if action is taken
        if action > 0:
            self.prayer_state = action - 1  # 1->0 (melee), 2->1 (range), 3->2 (magic)
        else:
            self.prayer_state = -1  # No prayer (use -1 to distinguish from melee prayer)
        
        # Always advance the timer
        self.ticks_until_damage -= 1
        
        # Check if damage is dealt this step
        if self.ticks_until_damage <= 0:
            if self.prayer_state >= 0 and self.prayer_state == self.attack_type:
                reward += 2.0  # Big reward for blocking
            else:
                reward -= 2.0  # Big penalty for getting hit
            
            # Reset for next attack (pick randomly)
            self.attack_type = random.randint(0, 2)  # Pick next attack randomly
            self.ticks_until_damage = 4
        else:
            if self.prayer_state == -1: # no prayer active
                reward += 0.25 # saving prayer points
        
        self.total_reward += reward
        return self._get_obs(), reward
    
    def _get_obs(self):
        attach_1h = np.eye(3)[self.attack_type]
        prayer_1h = np.eye(3)[max(0, self.prayer_state)]  # Handle -1 case
        return np.concatenate([attach_1h, [self.ticks_until_damage], prayer_1h])

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

    def __call__(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        x = self.l1(obs).tanh()
        act = self.l2(x).log_softmax()
        x = self.c1(obs).relu()
        return act, self.c2(x)

def evaluate(model: ActorCritic, test_env: Prayer_Flick_Env) -> float:
    obs = test_env.reset()
    total_rew = 0.0
    for _ in range(100):  # Run for 100 steps
        act = model(Tensor(obs))[0].argmax().item()
        obs, rew = test_env.step(act)
        total_rew += float(rew)
    return total_rew

def visualize_battle(model: ActorCritic, env: Prayer_Flick_Env, steps: int = 50):
    """Visualize the agent fighting a boss with prayer flicking"""
    obs = env.reset()
    total_reward = 0
    successful_prayers = 0
    failed_prayers = 0
    
    print("\n" + "="*60)
    print("🎮 PRAYER FLICKING BOSS BATTLE 🎮")
    print("="*60)
    print("Legend: 🗡️=Melee  🏹=Range  🔮=Magic  🛡️=Protected  💥=Hit  ❌=Miss")
    print("="*60)
    
    for step in range(steps):
        # Clear screen (works on most terminals)
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Get action from model
        logits = model(Tensor(obs))[0]
        probs = logits.softmax()
        act = probs.argmax().item()
        
        # Get current state info BEFORE step
        attack_type = env.attack_type
        prayer_state = env.prayer_state
        ticks_left = env.ticks_until_damage
        
        # Display battle scene
        print("\n" + "="*60)
        print("🎮 PRAYER FLICKING BOSS BATTLE 🎮")
        print("="*60)
        print("Legend: 🗡️=Melee  🏹=Range  🔮=Magic  🛡️=Protected  💥=Hit  ❌=Miss")
        print("="*60)
        
        # Boss section
        print(f"\n👹 BOSS (Step {step+1}/{steps})")
        print("─" * 40)
        
        # Show what attack is coming
        attack_icons = ["🗡️", "🏹", "🔮"]
        attack_names = ["MELEE", "RANGE", "MAGIC"]
        print(f"Boss is preparing: {attack_icons[attack_type]} {attack_names[attack_type]} ATTACK")
        print(f"Ticks until damage: {ticks_left}")
        
        # Player section
        print(f"\n⚔️  PLAYER")
        print("─" * 40)
        
        # Show current prayer state
        prayer_icons = ["🛡️", "🛡️", "🛡️"]
        prayer_names = ["MELEE PRAYER", "RANGE PRAYER", "MAGIC PRAYER"]
        
        if prayer_state == -1:
            current_prayer = "❌ NO PRAYER"
        else:
            current_prayer = f"{prayer_icons[prayer_state]} {prayer_names[prayer_state]}"
        
        print(f"Current prayer: {current_prayer}")
        
        # Show action taken
        action_names = ["WAIT", "MELEE PRAYER", "RANGE PRAYER", "MAGIC PRAYER"]
        print(f"Agent action: {action_names[act]}")
        
        # Take step and check if damage was dealt
        obs, reward = env.step(act)
        total_reward += reward
        
        # Show result
        print(f"\n💥 BATTLE RESULT")
        print("─" * 40)
        
        # Check if damage was dealt this step
        if ticks_left == 1:  # Timer was at 1, now it's 0
            # Use the prayer state AFTER the step function has updated it
            if env.prayer_state >= 0 and env.prayer_state == attack_type:
                print("✅ PERFECT! Prayer blocked the attack!")
                successful_prayers += 1
            else:
                print("❌ OOF! Attack got through!")
                failed_prayers += 1
        else:
            print("⏳ Waiting for attack...")
        
        # Stats
        print(f"\n📊 STATISTICS")
        print("─" * 40)
        print(f"Total Reward: {total_reward:.1f}")
        print(f"Successful Prayers: {successful_prayers}")
        print(f"Failed Prayers: {failed_prayers}")
        if successful_prayers + failed_prayers > 0:
            success_rate = (successful_prayers / (successful_prayers + failed_prayers)) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        # Wait a bit for dramatic effect
        time.sleep(0.5)
    
    print("\n" + "="*60)
    print("🎉 BATTLE COMPLETE! 🎉")
    print("="*60)
    print(f"Final Score: {total_reward:.1f}")
    print(f"Prayer Success Rate: {(successful_prayers / max(1, successful_prayers + failed_prayers)) * 100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    env = Prayer_Flick_Env()
    model = ActorCritic(7, 4)  # 7 observation features, 4 actions
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr=LEARNING_RATE)

    @TinyJit
    def train_step(x: Tensor, selected_action: Tensor, reward: Tensor, old_log_dist: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        with Tensor.train():
            log_dist, value = model(x)
            action_mask = (selected_action.reshape(-1, 1) == Tensor.arange(log_dist.shape[1]).reshape(1, -1).expand(selected_action.shape[0], -1)).float()

            # get real advantage using the value function
            advantage = reward.reshape(-1, 1) - value
            masked_advantage = action_mask * advantage.detach()

            # PPO
            ratios = (log_dist - old_log_dist).exp()
            unclipped_ratio = masked_advantage * ratios
            clipped_ratio = masked_advantage * ratios.clip(1-PPO_EPSILON, 1+PPO_EPSILON)
            action_loss = -unclipped_ratio.minimum(clipped_ratio).sum(-1).mean()

            entropy_loss = (log_dist.exp() * log_dist).sum(-1).mean()   # this encourages diversity
            critic_loss = advantage.square().mean()
            opt.zero_grad()
            (action_loss + entropy_loss*ENTROPY_SCALE + critic_loss).backward()
            opt.step()
            return action_loss.realize(), entropy_loss.realize(), critic_loss.realize()

    @TinyJit
    def get_action(obs: Tensor) -> Tensor:
        ret = model(obs)[0].exp().multinomial().realize()
        return ret

    st, steps = time.perf_counter(), 0
    Xn, An, Rn = [], [], []
    
    for episode_number in (t:=trange(EPISODES)):
        get_action.reset()   # NOTE: if you don't reset the jit here it captures the wrong model on the first run through

        obs = env.reset()
        rews, terminated = [], False
        
        # Run episode
        for _ in range(100):  # Fixed episode length
            # pick actions
            act = get_action(Tensor(obs)).item()

            # save this state action pair
            Xn.append(np.copy(obs))
            An.append(act)

            obs, rew = env.step(act)
            rews.append(float(rew))
        steps += len(rews)

        # reward to go
        discounts = np.power(DISCOUNT_FACTOR, np.arange(len(rews)))
        Rn += [np.sum(rews[i:] * discounts[:len(rews)-i]) for i in range(len(rews))]

        Xn, An, Rn = Xn[-REPLAY_BUFFER_SIZE:], An[-REPLAY_BUFFER_SIZE:], Rn[-REPLAY_BUFFER_SIZE:]
        X, A, R = Tensor(Xn), Tensor(An), Tensor(Rn)

        old_log_dist = model(X)[0].detach()
        for i in range(TRAIN_STEPS):
            if len(Xn) >= BATCH_SIZE:
                samples = Tensor.randint(BATCH_SIZE, high=X.shape[0]).realize()
                action_loss, entropy_loss, critic_loss = train_step(X[samples], A[samples], R[samples], old_log_dist[samples])
            else:
                action_loss, entropy_loss, critic_loss = train_step(X, A, R, old_log_dist)
        
        t.set_description(f"sz: {len(Xn):5d} steps/s: {steps/(time.perf_counter()-st):7.2f} action_loss: {action_loss.item():7.3f} entropy_loss: {entropy_loss.item():7.3f} critic_loss: {critic_loss.item():8.3f} reward: {sum(rews):6.2f}")

    test_rew = evaluate(model, Prayer_Flick_Env())
    print(f"test reward: {test_rew}")
    
    # Show the visual battle demonstration
    print("\n🎮 Starting visual battle demonstration...")
    time.sleep(2)
    visualize_battle(model, Prayer_Flick_Env(), steps=30)
