# ARC-AGI-3 — Latent World Model Agent

Research prototype developed during an R&D internship at **SynaLinks** for the **ARC-AGI-3 / ARC Prize 2026** interactive benchmark.

The objective is to build an agent that can enter an unseen game, learn how its environment reacts to actions, and plan useful behaviour without knowing the rules in advance.

> **Core idea:** keep perception stable, learn transition dynamics online, and separate *how to reach a state* from *which state is worth pursuing*.

![Current ARC-AGI-3 agent architecture](docs/architecture_arc_agi3_actuelle.webp)

## Why ARC-AGI-3 is interesting

ARC-AGI-3 turns abstract reasoning into an interactive problem. At each step, the agent receives a 64×64 grid, chooses an action, observes the consequence, and has to infer the mechanics of a game it has never seen before.

That naturally splits the problem into three levels:

- **Model** — how does the environment evolve after an action?
- **Plan** — which action sequence can reach a target state?
- **Goal selection / supervision** — which target state actually represents progress?

A large part of this project was about making those responsibilities explicit instead of forcing a single model to solve all three at once.

## Current agent

The implementation on `main` follows a model-predictive-control loop:

1. **Frozen visual encoder**  
   The current frame is mapped to a latent representation by the LeWM visual encoder.

2. **Frozen latent adapter**  
   A pretrained linear adapter maps the encoder representation into the latent space used by the dynamics model and planner.

3. **Online World Model**  
   An action-conditioned dynamics model predicts the next latent state. During interaction, only the dynamics are updated from real transitions collected in the current game.

4. **Observed waypoint memory**  
   Latent states actually visited by the agent are stored as candidate intermediate goals.

5. **Dominated Novelty Search (DNS)**  
   DNS can filter waypoint candidates to preserve useful diversity in latent space without adding another training objective to the World Model.

6. **CEM planner**  
   A categorical Cross-Entropy Method planner rolls out candidate action sequences through the learned dynamics and selects the trajectory that best approaches the active waypoint.

7. **Receding-horizon execution**  
   Only the first action is executed. The agent observes the new frame and replans from the updated state.

The controller also contains coordinate-aware handling for **ACTION6**, allowing the planner to reason about both the action and its click coordinates.

## Architecture

A simplified view of the data flow is:

```text
Observation
   ↓
Frozen visual encoder
   ↓
Frozen linear adapter
   ↓
Current latent state ───────────────┐
   ↓                                │
Waypoint memory + DNS               │
   ↓                                │
Goal latent                         │
   ↓                                │
CEM planner ← Online World Model ←──┘
   ↓
Action
   ↓
ARC-AGI-3 environment
   ↓
Real transition → Replay buffer → Dynamics update only
```

This separation is deliberate: the **World Model learns how the environment changes**, while the **waypoint layer decides where the planner should try to go**.

## Research evolution

Several directions were tested while iterating on the agent:

- auxiliary reward/progress heads;
- online adaptation to unseen games;
- direct encoder fine-tuning;
- frozen encoder + dense latent adapter;
- simplified dynamics-only World Model training;
- waypoint-based goal selection;
- Dominated Novelty Search for waypoint diversity;
- an experimental **RLM/LLM supervision** layer for proposing semantic goals;
- links with **neuro-symbolic reasoning** and hierarchical supervision in robotics.

The public controller on `main` currently corresponds to the **waypoints + DNS + CEM** architecture. The RLM supervision work is documented as a research direction rather than presented as part of the current production controller.

For the full reasoning behind these iterations, see:

**[Research note — World Models, supervision, neuro-symbolic reasoning and robotics (FR)](docs/article_arc_agi_3.md)**

## Repository structure

```text
.
├── agent/
│   ├── my_agent.py
│   ├── monitor.py
│   └── models/
│       ├── pretrained_heads-deterministic-linear.pt
│       └── lewm-pusht/
│           ├── config.json
│           ├── README.md
│           └── weights.pt
├── docs/
│   ├── article_arc_agi_3_linkedin_v3.md
│   └── architecture_arc_agi3_actuelle.webp
├── scripts/
│   ├── build_notebook.py
│   ├── play_local.py
│   └── slim_framework.py
├── pretrain.py
├── Makefile
└── README.md
```

Large datasets, local recordings, environment files, credentials and generated notebooks are intentionally excluded through `.gitignore`.

## Run locally

### Requirements

- Python **3.12**
- Git
- A Kaggle account only if you want to build/push a competition submission

### Setup

```bash
git clone https://github.com/31nidal/ARC-AGI-3.git
cd ARC-AGI-3
make setup
```

### Play

Run the agent across the available games:

```bash
make play-local
```

Run a single game while debugging:

```bash
make play-local GAME=ls20
```

Change the action budget:

```bash
make play-local GAME=ls20 STEPS=500
```

### Visual monitor

```bash
make monitor GAME=ls20
```

The monitor displays the current game frame, a PCA view of the latent trajectory, waypoint information and decoded imagined CEM rollouts.

### Pretraining

```bash
make pretrain
```

The resulting adapter/dynamics checkpoint is stored under `agent/models/` and can be compared against a cold start with:

```bash
make play-local GAME=ls20 COLD=1
```

### Kaggle submission

Create the local notebook:

```bash
make notebook
```

Push it to Kaggle:

```bash
make submit
```

Kaggle credentials belong in `.kaggle/`, which is ignored by Git.

## Current limitation

The main lesson from the waypoint experiments is that **latent novelty is not the same thing as task progress**.

The planner can become better at reaching a selected latent state while that state still has little semantic value for solving the game. This is why goal selection and higher-level supervision became a separate research question in the later part of the project.

## Origin and attribution

This repository started from the official **[ARC-AGI-3 Kaggle Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)** and was extended with the World Model, latent adaptation, online learning, waypoint/DNS memory, CEM planning and monitoring work described above.

ARC-AGI-3 documentation: **[docs.arcprize.org](https://docs.arcprize.org/)**

This is a personal research repository and not an official ARC Prize repository.
