"""
minimal_phosani.py

Minimal Phosani-like environment suitable for plugging into a PPO agent.
This file implements only the environment (no model/training code).

- Action space vector (int -> action). All actions are available every tick,
  but some are no-ops depending on the current phase (boss vs pillar).
- Observation is a flat tuple of floats (state vector) suitable as input to an
  NN policy. The exact ordering is documented in `observation_description`.
- Pillar phase: when boss_hp is reduced to 0, the boss
  becomes invulnerable and 4 pillars become vulnerable. The player must kill all 4
  pillars (attack pillars) to damage the boss. Pillars are targets the player
  must whittle down.

Usage:
    env = MinimalPhosaniEnv()
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    env.render()
"""

import random
from typing import Tuple, Dict, Any, List
import numpy as np

# Action definitions (vectorized action space)
ACTIONS = {
    0: "attack_boss",   # in boss-phase: damages boss; in pillar-phase: no-op (prefer attack_pillar)
    1: "attack_pillar", # damages current target pillar (only useful in pillar-phase)
    2: "attack_husk",   # damages one existing husk (prefers mage then melee)
    3: "eat_food",
    4: "pray_melee",
    5: "pray_range",
    6: "pray_magic",
    7: "pray_none",
}
ACTION_SIZE = len(ACTIONS)


def action_space() -> Dict[int, str]:
    """Return the action id -> name mapping."""
    return ACTIONS.copy()


    def observation_description() -> List[str]:
        """
        Returns a list describing the observation vector (in order).
        Use this in your PPO actor to map indices to meanings.
        """
        return [
            "boss_hp",            # 0
            "player_hp",          # 1
            "prayer_points",      # 2
            "boss_attack_timer",  # 3
            "boss_attack_type",   # 4 (0=melee,1=ranged,2=magic)
            "player_cooldown",    # 5 (ticks until next attack)
            "active_prayer",      # 6 (-1 none, 0 melee, 1 range, 2 magic)
            "food_count",         # 7
            "husk_melee_alive",   # 8 (0 or 1)
            "husk_mage_alive",    # 9 (0 or 1)
            "pillars_remaining",  # 10 (0..4)
            "current_pillar_hp",  # 11 (hp of the pillar currently being attacked or 0)
        ]


class MinimalPhosaniEnv:
    def __init__(self):
        self.reset()

    def reset(self) -> Tuple[float, ...]:
        # Base stats
        self.boss_max_hp = 400
        self.boss_revive_hp = int(self.boss_max_hp * 0.5)  # boss returns with 50% HP
        self.player_max_hp = 99

        # speeds & ranges
        boss_attack_speed = 4
        weapon_speed = 4
        weapon_min = 0
        weapon_max = 50

        # eat
        eat_heal = 20
        eat_recovery = 3

        # reward shaping (simple)
        reward_per_damage = 1.0
        eat_reward = 0.1
        idle_penalty = -0.01

        # pillar stats
        self.pillar_max_hp = 80
        self.num_pillars = 4

        # initialize state
        self.state: Dict[str, Any] = {
            # boss-phase fields
            "boss_hp": self.boss_max_hp,
            "boss_max_hp": self.boss_max_hp,
            "phase": "boss",  # "boss" or "pillars"
            "boss_attack_timer": boss_attack_speed,
            "boss_attack_speed": boss_attack_speed,
            "boss_attack_type": random.choice([0, 1, 2]),

            # player
            "player_hp": self.player_max_hp,
            "player_cooldown": 0,
            "weapon_speed": weapon_speed,
            "weapon_min": weapon_min,
            "weapon_max": weapon_max,
            "is_eating": False,
            "eat_timer": 0,
            "eat_recovery": eat_recovery,
            "eat_heal": eat_heal,
            "food_count": 5,
            "active_prayer": -1,
            "prayer_points": 99,
            "prayer_drain_per_tick": 0.5,

            "pillars": [],  # list of pillar hp values
            "pillars_remaining": 0,
            "current_pillar_idx": 0,

            # husks: explicit flags for two husks
            "husk_melee_alive": 0,
            "husk_mage_alive": 0,
            # husk attack timing / spawn mechanics (spawn chance checked every 6 boss attacks)
            "husk_attack_speed": 4,
            "husk_attack_timer": 4,
            "boss_attack_count": 0,
            "husk_spawn_chance": 0.5,
            # per-husk damage ranges
            "husk_damage": {
                0: (2, 6),   # melee husk
                2: (3, 8),   # mage husk
            },

            # bookkeeping
            "tick": 0,
            "reward": 0.0,
            "reward_per_damage": reward_per_damage,
            "eat_reward": eat_reward,
            "idle_penalty": idle_penalty,
            "last_action": None,
        }
        return self._get_obs()

    def _get_obs(self) -> Tuple[float, ...]:
        s = self.state
        phase_is_pillars = 1.0 if s["phase"] == "pillars" else 0.0
        current_pillar_hp = 0.0
        if s["pillars_remaining"] > 0 and s["pillars"]:
            # ensure index is in range
            idx = min(s["current_pillar_idx"], len(s["pillars"]) - 1)
            current_pillar_hp = float(s["pillars"][idx])
        obs = (
            float(s["boss_hp"]),
            # float(phase_is_pillars), # no the model should just figure it out
            float(s["player_hp"]),
            float(s["prayer_points"]),
            float(s["boss_attack_timer"]),
            float(s["boss_attack_type"]),
            float(s["player_cooldown"]),
            float(s["active_prayer"]),
            float(s["food_count"]),
            float(s["husk_melee_alive"]),
            float(s["husk_mage_alive"]),
            float(s["pillars_remaining"]),
            float(current_pillar_hp),
        )
        return obs

    def step(self, action: int) -> Tuple[Tuple[float, ...], float, bool, Dict[str, Any]]:
        s = self.state
        s["tick"] += 1
        s["reward"] = 0.0
        s["last_action"] = ACTIONS.get(action, None)

        # helper: small idle penalty for invalid/wasted actions
        idle_penalty = s["idle_penalty"]

        # ACTION HANDLING
        # Attack boss: only damages boss when in boss-phase. In pillar-phase it's a no-op
        if action == 0:  # attack_boss
            if s["phase"] == "boss":
                if s["player_cooldown"] <= 0 and not s["is_eating"]:
                    dmg = random.randint(s["weapon_min"], s["weapon_max"])
                    s["boss_hp"] = max(0, s["boss_hp"] - dmg)
                    s["reward"] += dmg * s["reward_per_damage"]
                    s["player_cooldown"] = s["weapon_speed"]
                    s["last_action"] = "attack_boss"
                else:
                    s["reward"] += idle_penalty
                    s["last_action"] = "attack_boss_failed"
            else:
                # in pillar phase, encourage using attack_pillar instead
                s["reward"] += idle_penalty
                s["last_action"] = "attack_boss_noop_in_pillars"

        # Attack pillar: only useful in pillar-phase
        elif action == 1:  # attack_pillar
            if s["phase"] == "pillars" and s["pillars_remaining"] > 0:
                if s["player_cooldown"] <= 0 and not s["is_eating"]:
                    idx = s["current_pillar_idx"]
                    # clamp index
                    idx = min(idx, max(0, len(s["pillars"]) - 1))
                    dmg = random.randint(s["weapon_min"], s["weapon_max"])
                    s["pillars"][idx] = max(0, s["pillars"][idx] - dmg)
                    s["reward"] += dmg * (s["reward_per_damage"] * 0.7)  # slightly lower reward than boss damage
                    s["player_cooldown"] = s["weapon_speed"]
                    s["last_action"] = "attack_pillar"
                    # if pillar destroyed
                    if s["pillars"][idx] <= 0:
                        s["pillars_remaining"] -= 1
                        s["last_action"] = "pillar_destroyed"
                        # move to next pillar index if any remain
                        if s["pillars_remaining"] > 0:
                            s["current_pillar_idx"] = min(s["current_pillar_idx"] + 1, len(s["pillars"]) - 1)
                        else:
                            # all pillars down -> return to boss-phase and revive boss
                            s["phase"] = "boss"
                            s["boss_hp"] = self.boss_revive_hp
                            # husks are removed when pillar phase finishes
                            s["husk_melee_alive"] = 0
                            s["husk_mage_alive"] = 0
                            s["pillars"] = []
                            s["current_pillar_idx"] = 0
                            s["last_action"] = "pillars_cleared_boss_revived"
                else:
                    s["reward"] += idle_penalty
                    s["last_action"] = "attack_pillar_failed"
            else:
                s["reward"] += idle_penalty
                s["last_action"] = "attack_pillar_noop"

        # Attack husk: damages and kills one husk (prefers mage then melee)
        elif action == 2:  # attack_husk
            if (s["husk_mage_alive"] or s["husk_melee_alive"]) and s["player_cooldown"] <= 0 and not s["is_eating"]:
                # prefer mage
                if s["husk_mage_alive"]:
                    s["husk_mage_alive"] = 0
                    dmg = random.randint(6, 12)
                    s["reward"] += dmg * 0.4
                    s["last_action"] = "kill_husk_mage"
                else:
                    s["husk_melee_alive"] = 0
                    dmg = random.randint(4, 10)
                    s["reward"] += dmg * 0.4
                    s["last_action"] = "kill_husk_melee"
                s["player_cooldown"] = s["weapon_speed"]
            else:
                s["reward"] += idle_penalty
                s["last_action"] = "attack_husk_failed"

        # Eat food
        elif action == 3:  # eat_food
            if s["food_count"] > 0 and not s["is_eating"]:
                s["player_hp"] = min(self.player_max_hp, s["player_hp"] + s["eat_heal"])
                s["food_count"] -= 1
                s["is_eating"] = True
                s["eat_timer"] = s["eat_recovery"]
                s["player_cooldown"] = max(s["player_cooldown"], s["eat_recovery"])
                s["reward"] += s["eat_reward"]
                s["last_action"] = "eat"
            else:
                s["reward"] += idle_penalty
                s["last_action"] = "eat_failed"

        # Prayer switches (instant)
        elif action == 4:
            s["active_prayer"] = 0
            s["last_action"] = "pray_melee"
        elif action == 5:
            s["active_prayer"] = 1
            s["last_action"] = "pray_range"
        elif action == 6:
            s["active_prayer"] = 2
            s["last_action"] = "pray_magic"
        elif action == 7:
            s["active_prayer"] = -1
            s["last_action"] = "pray_none"
        else:
            s["reward"] += idle_penalty
            s["last_action"] = "invalid_action"

        # ---- Environment dynamics after action ----

        # boss attack timing (boss attacks only in boss-phase)
        if s["phase"] == "boss":
            s["boss_attack_timer"] -= 1
            if s["boss_attack_timer"] <= 0:
                self._resolve_boss_attack()

        # cooldowns and eating timers
        if s["player_cooldown"] > 0:
            s["player_cooldown"] = max(0, s["player_cooldown"] - 1)

        if s["is_eating"]:
            s["eat_timer"] -= 1
            if s["eat_timer"] <= 0:
                s["is_eating"] = False
                s["eat_timer"] = 0

        # prayer drain
        if s["active_prayer"] != -1:
            s["prayer_points"] = max(0.0, s["prayer_points"] - s["prayer_drain_per_tick"])
            if s["prayer_points"] <= 0:
                s["active_prayer"] = -1

        # husk attack timing (husks spawn during boss-phase checks; when alive they attack every husk_attack_speed ticks)
        s["husk_attack_timer"] = max(0, s.get("husk_attack_timer", s.get("husk_attack_speed", 4)) - 1)
        if s["husk_attack_timer"] <= 0:
            total_husk_damage = 0
            # melee husk attack
            if s["husk_melee_alive"]:
                lo, hi = s["husk_damage"][0]
                total_husk_damage += random.randint(lo, hi)
            # mage husk attack
            if s["husk_mage_alive"]:
                lo, hi = s["husk_damage"][2]
                total_husk_damage += random.randint(lo, hi)
            if total_husk_damage > 0:
                s["player_hp"] = max(0, s["player_hp"] - total_husk_damage)
                s["reward"] -= float(total_husk_damage) * 0.5
                s["last_husk_hit"] = total_husk_damage
            # reset husk attack timer
            s["husk_attack_timer"] = s.get("husk_attack_speed", 4)

        # husk spawn chance (checked every 6 boss attacks)
        s["boss_attack_count"] = s.get("boss_attack_count", 0)
        if s["boss_attack_count"] >= 6:
            s["boss_attack_count"] = 0
            # spawn husks if not already alive
            if s["husk_melee_alive"] == 0 and random.random() < s["husk_spawn_chance"]:
                s["husk_melee_alive"] = 1
            if s["husk_mage_alive"] == 0 and random.random() < s["husk_spawn_chance"]:
                s["husk_mage_alive"] = 1

        # Check for phase transition: boss -> pillars
        if s["phase"] == "boss" and s["boss_hp"] <= 0:
            # enter pillar-phase: boss is invulnerable, spawn pillars
            s["phase"] = "pillars"
            s["pillars"] = [self.pillar_max_hp for _ in range(self.num_pillars)]
            s["pillars_remaining"] = self.num_pillars
            s["current_pillar_idx"] = 0
            # small penalty for entering pillar phase (challenge)
            s["reward"] -= 5.0
            s["last_action"] = "enter_pillar_phase"

        # termination conditions
        done = False
        info: Dict[str, Any] = {"tick": s["tick"], "last_action": s["last_action"]}

        # if player dead
        if s["player_hp"] <= 0:
            done = True
            s["reward"] -= 80.0
            info["result"] = "death"

        # if boss finally dead after being revived and then killed again (allow boss_hp 0 in boss-phase)
        if s["phase"] == "boss" and s["boss_hp"] <= 0:
            done = True
            s["reward"] += 200.0
            info["result"] = "victory"

        return self._get_obs(), float(s["reward"]), done, info

    def _resolve_boss_attack(self) -> None:
        s = self.state
        # if correct prayer active, block the boss attack
        if s["active_prayer"] == s["boss_attack_type"]:
            dmg = 0
        else:
            base_min, base_max = 6, 30
            dmg = random.randint(base_min, base_max)
        s["player_hp"] = max(0, s["player_hp"] - dmg)
        s["reward"] -= float(dmg)
        # reset timer and choose a new attack type
        s["boss_attack_timer"] = s["boss_attack_speed"]
        s["boss_attack_type"] = random.choice([0, 1, 2])
        # increment boss attack count for husk spawn checks
        s["boss_attack_count"] = s.get("boss_attack_count", 0) + 1

    def render(self) -> None:
        s = self.state
        print_str = (
            f"Tick {s['tick']:03d} | Phase: {s['phase']:7s} | Boss HP: {s['boss_hp']:3d} | "
            f"Player HP: {s['player_hp']:3d} | Food: {s['food_count']} | CD: {s['player_cooldown']:1d} | "
            f"Prayer: {s['active_prayer']:2d} | BossAtkIn: {s['boss_attack_timer']:1d} | "
            f"Husks M/Mg: {s['husk_melee_alive']}/{s['husk_mage_alive']} | "
            f"Pillars rem: {s['pillars_remaining']:d}"
        )
        # show current pillar HP if in pillar phase
        if s["pillars_remaining"] > 0 and s["pillars"]:
            idx = min(s["current_pillar_idx"], len(s["pillars"]) - 1)
            print_str += f" | CurrPillarHP: {s['pillars'][idx]:3d}"
        print(print_str)


# Quick demo (random policy) - only runs when executed directly
if __name__ == "__main__":
    env = MinimalPhosaniEnv()
    obs = env.reset()
    done = False
    steps = 0
    while not done and steps < 400:
        # simple heuristic: if pillar-phase prefer attack_pillar; if husks alive prefer attack_husk
        if env.state["phase"] == "pillars":
            if env.state["husk_mage_alive"] or env.state["husk_melee_alive"]:
                action = 2  # attack_husk
            else:
                action = 1  # attack_pillar
        else:
            # boss phase: eat if low hp, else attack
            if env.state["player_hp"] < 40 and env.state["food_count"] > 0:
                action = 3
            else:
                action = 0
        obs, rew, done, info = env.step(action)
        env.render()
        steps += 1
    print("Demo finished:", info)
