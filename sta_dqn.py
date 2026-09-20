import gymnasium as gym
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import ptan  # PyTorch AgentNet

from pathlib import Path


# ============================================================
# Hyperparameters
# ============================================================

GAMMA = 0.95

BATCH_SIZE = 8

REPLAY_SIZE = 200
REPLAY_START_SIZE = 32

LEARNING_RATE = 1e-3

# Synchronize online network -> target network
# every TARGET_SYNC ECO interactions.
TARGET_SYNC = 20

EPSILON_START = 1.0
EPSILON_FINAL = 0.05
EPSILON_DECAY = 200

HIDDEN_SIZE = 64

# Maximum number of ECO attempts in one episode.
MAX_ECO_STEPS = 10

# Training stops after this many consecutive
# successful timing-closure episodes.
SUCCESS_STREAK_REQUIRED = 10


# ============================================================
# STA Reinforcement-Learning Environment
# ============================================================

class STAEnv(gym.Env):

    def __init__(
        self,
        initial_sizes,
        initial_slack,
        path,
        required_time
    ):
        super().__init__()

        # ----------------------------------------------------
        # Load toy cell-delay model
        # ----------------------------------------------------

        # cell_timing.json is expected in the same directory
        # as this Python script.
        timing_file = Path(__file__).parent / "cell_timing.json"

        with open(timing_file, "r") as f:
            self.cell_delay = json.load(f)

        # ----------------------------------------------------
        # Initial circuit configuration
        # ----------------------------------------------------

        self.initial_sizes = initial_sizes.copy()
        self.initial_slack = initial_slack

        # Legal cell-size progression.
        self.cell_sizes = ["X1", "X2", "X4"]

        # ----------------------------------------------------
        # Timing information
        # ----------------------------------------------------

        self.clk_to_q = 40.0
        self.path = path
        self.required_time = required_time

        # ----------------------------------------------------
        # Episode results
        #
        # True  -> timing closure achieved
        # False -> ECO budget exhausted
        # ----------------------------------------------------

        self.episode_results = []

        # ----------------------------------------------------
        # Store successful ECO solutions
        #
        # IMPORTANT:
        # Do not clear this in reset().
        #
        # PTAN may reset the environment immediately after
        # an episode completes.
        # ----------------------------------------------------

        self.successful_solutions = []

        # ----------------------------------------------------
        # Action Space
        #
        # 0 -> NO_OP
        # 1 -> UPSIZE G1
        # 2 -> UPSIZE G2
        # 3 -> UPSIZE G3
        # ----------------------------------------------------

        self.action_space = gym.spaces.Discrete(4)

        self.action_map = {
            0: ("NO_OP", None),
            1: ("UPSIZE", "G1"),
            2: ("UPSIZE", "G2"),
            3: ("UPSIZE", "G3")
        }

        # ----------------------------------------------------
        # Observation
        #
        # [G1_size, G2_size, G3_size, slack]
        #
        # X1 -> 0
        # X2 -> 1
        # X4 -> 2
        # ----------------------------------------------------

        self.observation_space = gym.spaces.Box(
            low=np.array(
                [0, 0, 0, -500],
                dtype=np.float32
            ),
            high=np.array(
                [2, 2, 2, 500],
                dtype=np.float32
            ),
            dtype=np.float32
        )

    # ========================================================
    # Reset Environment
    # ========================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Restore original circuit.
        self.sizes = self.initial_sizes.copy()
        self.slack = self.initial_slack

        # ECO count is local to each episode.
        self.step_count = 0

        observation = self.get_observation()

        return observation, {}

    # ========================================================
    # Calculate Current Slack
    # ========================================================

    def calculate_slack(self):

        # Start with launching flop clk->Q delay.
        path_delay = self.clk_to_q

        # Add combinational gate delay.
        for gate in self.path:

            gate_size = self.sizes[gate]

            gate_delay = \
                self.cell_delay[gate][gate_size]

            path_delay += gate_delay

        # Setup/max timing:
        #
        # slack = required - arrival
        slack = self.required_time - path_delay

        return slack

    # ========================================================
    # Convert Internal State to Observation
    # ========================================================

    def get_observation(self):

        size_to_idx = {
            "X1": 0,
            "X2": 1,
            "X4": 2
        }

        observation = np.array(
            [
                size_to_idx[self.sizes[gate]]
                for gate in self.path
            ] + [self.slack],
            dtype=np.float32
        )

        return observation

    # ========================================================
    # Execute One ECO Action
    # ========================================================

    def step(self, action):

        # One step corresponds to one ECO attempt.
        self.step_count += 1

        # Convert DQN action number into operation/gate.
        operation, gate = self.action_map[action]

        # Slack before ECO.
        old_slack = self.slack

        # ----------------------------------------------------
        # Apply ECO
        # ----------------------------------------------------

        if operation == "UPSIZE":

            current_size = self.sizes[gate]

            current_index = \
                self.cell_sizes.index(current_size)

            # Legal progression:
            #
            # X1 -> X2 -> X4
            #
            # X4 cannot be upsized further.
            if current_index < len(self.cell_sizes) - 1:

                self.sizes[gate] = \
                    self.cell_sizes[current_index + 1]

        # NO_OP does nothing.

        # ----------------------------------------------------
        # Re-run toy STA
        # ----------------------------------------------------

        self.slack = self.calculate_slack()

        # ----------------------------------------------------
        # Reward
        #
        # Reward = improvement in slack
        # ----------------------------------------------------

        reward = self.slack - old_slack

        # ----------------------------------------------------
        # Episode termination
        # ----------------------------------------------------

        # Timing closure achieved.
        terminated = self.slack >= 0

        # ECO budget exhausted.
        truncated = self.step_count >= MAX_ECO_STEPS

        # ----------------------------------------------------
        # Successful episode
        # ----------------------------------------------------

        if terminated:

            # Timing closure bonus.
            reward += 50

            # Record episode result.
            self.episode_results.append(True)

            # ------------------------------------------------
            # SAVE FINAL ECO SOLUTION HERE
            #
            # This happens BEFORE PTAN gets a chance to reset
            # the environment.
            # ------------------------------------------------

            self.successful_solutions.append(
                {
                    "sizes": self.sizes.copy(),
                    "slack": self.slack,
                    "steps": self.step_count
                }
            )

        # ----------------------------------------------------
        # Failed episode
        # ----------------------------------------------------

        elif truncated:

            self.episode_results.append(False)

        # State after ECO.
        observation = self.get_observation()

        return (
            observation,
            reward,
            terminated,
            truncated,
            {}
        )


# ============================================================
# DQN
# ============================================================

class DQN(nn.Module):

    def __init__(self, obs_size, n_actions):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                obs_size,
                HIDDEN_SIZE
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_SIZE,
                n_actions
            )
        )

    def forward(self, x):

        return self.net(x)


# ============================================================
# Convert Replay Batch into PyTorch Tensors
# ============================================================

def batch_to_tensor(batch):

    states = []
    actions = []
    rewards = []
    last_states = []
    done_masks = []

    for exp in batch:

        states.append(exp.state)

        actions.append(exp.action)

        rewards.append(exp.reward)

        # PTAN sets last_state=None for an episode-ending
        # transition.
        #
        # We still need something numeric to construct the
        # tensor, so use exp.state as a placeholder.
        #
        # done_masks later forces its bootstrap value to zero.

        if exp.last_state is None:

            last_states.append(exp.state)

        else:

            last_states.append(exp.last_state)

        done_masks.append(
            exp.last_state is None
        )

    states_v = torch.as_tensor(
        np.stack(states),
        dtype=torch.float32
    )

    actions_v = torch.tensor(
        actions,
        dtype=torch.int64
    )

    rewards_v = torch.tensor(
        rewards,
        dtype=torch.float32
    )

    last_states_v = torch.as_tensor(
        np.stack(last_states),
        dtype=torch.float32
    )

    done_masks_v = torch.tensor(
        done_masks,
        dtype=torch.bool
    )

    return (
        states_v,
        actions_v,
        rewards_v,
        last_states_v,
        done_masks_v
    )


# ============================================================
# Bellman Target
# ============================================================

@torch.no_grad()
def Q_S_A_Value(
    net,
    gamma,
    last_states_v,
    rewards_v,
    done_masks_v
):

    # Target network calculates:
    #
    # Q_target(S', A')
    #
    # for every possible action.
    last_states_q_vals_v = net(last_states_v)

    # For every S', choose:
    #
    # max_A' Q_target(S', A')
    best_last_q_vals_v = \
        last_states_q_vals_v.max(dim=1)[0]

    # Terminal transition:
    #
    # no next-state bootstrap.
    best_last_q_vals_v[done_masks_v] = 0.0

    # Bellman target:
    #
    # y = R + gamma * max Q_target(S', A')
    target_q_vals_v = (
        rewards_v
        + gamma * best_last_q_vals_v
    )

    return target_q_vals_v


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = STAEnv(

        initial_sizes={
            "G1": "X2",
            "G2": "X1",
            "G3": "X4"
        },

        initial_slack=-50.0,

        path=[
            "G1",
            "G2",
            "G3"
        ],

        required_time=240
    )

    # --------------------------------------------------------
    # Observation and action dimensions
    # --------------------------------------------------------

    # Observation:
    #
    # [G1_size, G2_size, G3_size, slack]
    #
    # shape = (4,)
    obs_size = env.observation_space.shape[0]

    # Number of actions = 4
    n_actions = env.action_space.n

    # --------------------------------------------------------
    # Online DQN
    # --------------------------------------------------------

    net = DQN(
        obs_size,
        n_actions
    )

    # --------------------------------------------------------
    # Target DQN
    # --------------------------------------------------------

    target_net = \
        ptan.agent.TargetNet(net)

    # --------------------------------------------------------
    # Epsilon-Greedy Selector
    # --------------------------------------------------------

    selector = \
        ptan.actions.EpsilonGreedyActionSelector(
            epsilon=EPSILON_START
        )

    # --------------------------------------------------------
    # PTAN Agent
    # --------------------------------------------------------

    agent = ptan.agent.DQNAgent(
        net,
        selector,
        device="cpu"
    )

    # --------------------------------------------------------
    # Experience Source
    #
    # steps_count=1:
    #
    # S, A, R, S'
    # --------------------------------------------------------

    exp_source = \
        ptan.experience.ExperienceSourceFirstLast(
            env=env,
            agent=agent,
            gamma=GAMMA,
            steps_count=1
        )

    # --------------------------------------------------------
    # Replay Buffer
    # --------------------------------------------------------

    buffer = \
        ptan.experience.ExperienceReplayBuffer(
            exp_source,
            buffer_size=REPLAY_SIZE
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=LEARNING_RATE
    )

    # ========================================================
    # Counters
    # ========================================================

    # Total ECO interactions across all episodes.
    eco_count = 0

    # Number of completed episodes.
    episode = 0

    # Number of consecutive timing-closure episodes.
    consecutive_success = 0

    solved = False

    # ========================================================
    # Training Loop
    # ========================================================

    while True:

        # ----------------------------------------------------
        # Generate one new experience
        # ----------------------------------------------------

        eco_count += 1

        buffer.populate(1)

        # ----------------------------------------------------
        # Check completed episodes
        # ----------------------------------------------------

        for reward, steps in \
                exp_source.pop_rewards_steps():

            episode += 1

            # Get oldest environment result.
            result = \
                env.episode_results.pop(0)

            # ------------------------------------------------
            # Consecutive-success tracking
            # ------------------------------------------------

            if result:

                consecutive_success += 1

            else:

                consecutive_success = 0

            print(
                f"steps={steps}, "
                f"episode={episode}, "
                f"reward={reward:.2f}, "
                f"epsilon={selector.epsilon:.2f}, "
                f"success_streak={consecutive_success}"
            )

            # ------------------------------------------------
            # Training stopping condition
            # ------------------------------------------------

            if consecutive_success >= \
                    SUCCESS_STREAK_REQUIRED:

                solved = True

                print(
                    "Solved: timing closed in "
                    f"{SUCCESS_STREAK_REQUIRED} "
                    "consecutive episodes within "
                    f"{MAX_ECO_STEPS} ECO steps."
                )

                break

        # The previous break exits only the for-loop.
        if solved:
            break

        # ----------------------------------------------------
        # Epsilon Decay
        # ----------------------------------------------------

        selector.epsilon = max(
            EPSILON_FINAL,
            EPSILON_START
            - eco_count / EPSILON_DECAY
        )

        # ----------------------------------------------------
        # Replay Buffer Warm-Up
        # ----------------------------------------------------

        if len(buffer) < REPLAY_START_SIZE:
            continue

        # ----------------------------------------------------
        # Sample replay batch
        # ----------------------------------------------------

        batch = buffer.sample(BATCH_SIZE)

        (
            states_v,
            actions_v,
            rewards_v,
            last_states_v,
            done_masks_v
        ) = batch_to_tensor(batch)

        # ----------------------------------------------------
        # Clear previous gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ====================================================
        # TARGET SIDE
        # ====================================================

        q_state_action_target_v = Q_S_A_Value(

            target_net.target_model,

            GAMMA,

            last_states_v,

            rewards_v,

            done_masks_v
        )

        # ====================================================
        # ONLINE SIDE
        # ====================================================

        # Network predicts Q-values for all four actions.
        #
        # Shape:
        #
        # (BATCH_SIZE, 4)
        q_state_action_online_v = \
            net(states_v)

        # Select the Q-value corresponding to the action
        # actually taken in each replay experience.
        q_state_action_online_v = (
            q_state_action_online_v
            .gather(
                1,
                actions_v.unsqueeze(-1)
            )
            .squeeze(-1)
        )

        # ====================================================
        # Loss
        # ====================================================

        loss_v = F.mse_loss(
            q_state_action_online_v,
            q_state_action_target_v
        )

        # Calculate gradients.
        loss_v.backward()

        # Update online DQN weights.
        optimizer.step()

        # ----------------------------------------------------
        # Target Network Synchronization
        # ----------------------------------------------------

        if eco_count % TARGET_SYNC == 0:

            target_net.sync()

    # ========================================================
    # Print All Successful ECO Solutions
    # ========================================================

    print()
    print("========================================")
    print("Successful ECO Solutions")
    print("========================================")

    for index, solution in enumerate(
        env.successful_solutions,
        start=1
    ):

        sizes = solution["sizes"]

        slack = solution["slack"]

        steps = solution["steps"]

        print(
            f"Success {index}: "
            f"G1={sizes['G1']}, "
            f"G2={sizes['G2']}, "
            f"G3={sizes['G3']}, "
            f"slack={slack:.1f} ps, "
            f"steps={steps}"
        )