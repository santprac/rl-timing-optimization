# RL Timing Optimization with DQN

A compact educational example showing how a semiconductor timing-optimization problem can be formulated and solved using Reinforcement Learning.

The project uses a fully synthetic timing path and synthetic timing data. Its goal is to demonstrate the end-to-end RL workflow rather than model the complexity of production static timing analysis or commercial timing-closure flows.

## Problem

The timing path is:

`Flop1 -> G1 -> G2 -> G3 -> Flop2`

Clock-to-Q delay is 40 ps. The initial configuration is:

| Cell | Size | Delay (ps) |
|---|---|---:|
| G1 | X2 | 75 |
| G2 | X1 | 120 |
| G3 | X4 | 55 |

The total path delay is 290 ps. With a required arrival time of 240 ps, the initial slack is -50 ps.

The allowed ECO actions are:

- `NO_OP`
- `G1_UPSIZE`
- `G2_UPSIZE`
- `G3_UPSIZE`

Cell sizes progress as `X1 -> X2 -> X4`.

## RL Formulation

- **State:** `[G1 size, G2 size, G3 size, slack]`
- **Action:** one of the four ECO actions above
- **Reward:** slack improvement, plus a timing-closure bonus when slack >= 0
- **Episode termination:** timing closure achieved
- **Episode truncation:** maximum ECO budget reached

## DQN

The Q-network is implemented in PyTorch with:

`4 inputs -> 64 hidden neurons -> 4 Q-values`

PTAN provides the surrounding RL components, including the DQN agent, epsilon-greedy action selection, target network support, experience generation, replay buffer, and interaction with the Gymnasium environment.

## Training

Key settings include:

- Gamma: `0.95`
- Batch size: `8`
- Replay size: `200`
- Replay warm-up: `32`
- Learning rate: `1e-3`
- Target synchronization: every `20` ECO interactions
- Epsilon: `1.0 -> 0.05`
- Maximum ECOs per episode: `10`
- Training stop criterion: `10` consecutive timing-closure episodes

## Run

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python sta_dqn.py
```

The script prints episode progress and the successful final ECO configurations discovered during training.

## Scope and Limitations

This is intentionally a simplified experiment:

- single timing path
- three resizable cells
- cell upsizing as the only ECO action
- synthetic delay model
- reward focused primarily on slack improvement

A more realistic extension could include multiple interacting paths, additional ECO actions, and reward penalties for power and area so that the agent learns timing-closure strategies with lower PPA impact.

## Files

- `sta_dqn.py` - Gymnasium environment, DQN, PTAN setup, and training loop
- `cell_timing.json` - synthetic cell-delay model
