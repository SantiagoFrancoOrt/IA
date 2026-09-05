import random
import pickle
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_pos_bins=20,
        n_vel_bins=20,
        n_actions=11,
        alpha=0.5,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    ):
        self.n_pos_bins = n_pos_bins
        self.n_vel_bins = n_vel_bins
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Boundary points for digitize; n boundaries create n+1 bins (indices 0..n)
        self.pos_bins = np.linspace(-1.2, 0.6, n_pos_bins)
        self.vel_bins = np.linspace(-0.07, 0.07, n_vel_bins)
        self.action_values = np.linspace(-1.0, 1.0, n_actions)

        # Optimistic initialization: incentiva exploración en entornos con recompensa dispersa
        self.q_table = np.ones((n_pos_bins + 1, n_vel_bins + 1, n_actions)) * 5.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def discretize(self, obs):
        pos, vel = obs
        return int(np.digitize(pos, self.pos_bins)), int(np.digitize(vel, self.vel_bins))

    def next_action(self, obs):
        if random.random() < self.epsilon:
            return self.action_values[random.randint(0, self.n_actions - 1)]
        state = self.discretize(obs)
        return self.action_values[int(np.argmax(self.q_table[state]))]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_agent(self, env, episodes=15000, epsilon=None, gamma=None, alpha=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if gamma is not None:
            self.gamma = gamma
        if alpha is not None:
            self.alpha = alpha

        rewards_per_episode = []

        for ep in range(episodes):
            obs, _ = env.reset()
            state = self.discretize(obs)
            total_reward = 0.0
            done = False

            while not done:
                if random.random() < self.epsilon:
                    action_idx = random.randint(0, self.n_actions - 1)
                else:
                    action_idx = int(np.argmax(self.q_table[state]))

                action = self.action_values[action_idx]
                next_obs, reward, terminated, truncated, _ = env.step(np.array([action]))
                done = terminated or truncated
                next_state = self.discretize(next_obs)

                # Q-Learning update (sin bootstrap si el episodio terminó de verdad)
                if terminated:
                    target = reward
                else:
                    best_next = float(np.max(self.q_table[next_state]))
                    target = reward + self.gamma * best_next
                td_error = target - self.q_table[state][action_idx]
                self.q_table[state][action_idx] += self.alpha * td_error

                state = next_state
                total_reward += reward  # acumulamos recompensa real (sin shaping)

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards_per_episode.append(total_reward)

            if (ep + 1) % 1000 == 0:
                avg = np.mean(rewards_per_episode[-100:])
                print(
                    f"  Ep {ep+1:5d}/{episodes} | avg(100): {avg:8.2f}"
                    f" | ε: {self.epsilon:.4f}"
                )

        return rewards_per_episode

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def test_agent(self, env, episodes=100):
        saved_eps = self.epsilon
        self.epsilon = 0.0

        rewards, successes = [], 0
        for _ in range(episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = self.next_action(obs)
                obs, reward, terminated, truncated, _ = env.step(np.array([action]))
                done = terminated or truncated
                total_reward += reward
            rewards.append(total_reward)
            if total_reward > 90:
                successes += 1

        self.epsilon = saved_eps
        return rewards, successes / episodes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "n_pos_bins": self.n_pos_bins,
                    "n_vel_bins": self.n_vel_bins,
                    "n_actions": self.n_actions,
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "epsilon_decay": self.epsilon_decay,
                    "epsilon_min": self.epsilon_min,
                },
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            n_pos_bins=data["n_pos_bins"],
            n_vel_bins=data["n_vel_bins"],
            n_actions=data["n_actions"],
            alpha=data["alpha"],
            gamma=data["gamma"],
            epsilon=data["epsilon"],
            epsilon_decay=data["epsilon_decay"],
            epsilon_min=data["epsilon_min"],
        )
        agent.q_table = data["q_table"]
        return agent


# ==============================================================================
# Dyna-Q  (Sutton & Barto, caps. 8.1–8.2)
# ==============================================================================

class DynaQAgent(QLearningAgent):
    """Q-Learning + simulated experience from a tabular environment model."""

    def __init__(self, n_planning_steps=10, **kwargs):
        super().__init__(**kwargs)
        self.n_planning_steps = n_planning_steps
        # model: (state, action_idx) -> (reward, next_state, terminated)
        self.model: dict = {}

    def train_agent(self, env, episodes=15000, epsilon=None, gamma=None, alpha=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if gamma is not None:
            self.gamma = gamma
        if alpha is not None:
            self.alpha = alpha

        rewards_per_episode = []

        for ep in range(episodes):
            obs, _ = env.reset()
            state = self.discretize(obs)
            total_reward = 0.0
            done = False

            while not done:
                if random.random() < self.epsilon:
                    action_idx = random.randint(0, self.n_actions - 1)
                else:
                    action_idx = int(np.argmax(self.q_table[state]))

                action = self.action_values[action_idx]
                next_obs, reward, terminated, truncated, _ = env.step(np.array([action]))
                done = terminated or truncated
                next_state = self.discretize(next_obs)

                # (a) Direct RL update from real experience (sin bootstrap si terminó de verdad)
                if terminated:
                    target = reward
                else:
                    best_next = float(np.max(self.q_table[next_state]))
                    target = reward + self.gamma * best_next
                td_error = target - self.q_table[state][action_idx]
                self.q_table[state][action_idx] += self.alpha * td_error

                # (b) Update tabular model
                self.model[(state, action_idx)] = (reward, next_state, terminated)

                # (c) Planning: n updates from simulated experience
                experienced = list(self.model.keys())
                for s, a in random.sample(experienced, min(self.n_planning_steps, len(experienced))):
                    r, ns, is_terminal = self.model[(s, a)]
                    if is_terminal:
                        plan_target = r
                    else:
                        best_plan = float(np.max(self.q_table[ns]))
                        plan_target = r + self.gamma * best_plan
                    self.q_table[s][a] += self.alpha * (plan_target - self.q_table[s][a])

                state = next_state
                total_reward += reward  # acumulamos recompensa real (sin shaping)

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards_per_episode.append(total_reward)

            if (ep + 1) % 1000 == 0:
                avg = np.mean(rewards_per_episode[-100:])
                print(
                    f"  Ep {ep+1:5d}/{episodes} | avg(100): {avg:8.2f}"
                    f" | ε: {self.epsilon:.4f} | model: {len(self.model)} entries"
                )

        return rewards_per_episode
