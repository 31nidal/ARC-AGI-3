# AUTORESEARCH — latent world-model agent for ARC-AGI-3

Working research log for the agent in `agent/my_agent.py`. The README is the
*how-to-run*; this file is the *why* and *what-next*: the research thesis, the
current architecture, what's been tried, and the experiment backlog. Keep it
current so any session (human or agent) can pick up the thread without
re-deriving the design from the code.

**North star:** score above the random baseline on the hidden ARC-AGI-3 game
set — an agent that *generalises to unseen games* rather than memorising the
25 games we have replays for. Every design choice below is justified by that
generalisation requirement, not by fitting the training games.

---

## The thesis

ARC-AGI-3 is a set of short interactive grid games. The agent sees a 64×64
cell grid (16 colours), picks one of ~7 actions per step, and must reach a WIN
state within a tight action budget. The games at test time are *unseen*, so a
policy that memorises action sequences cannot transfer.

Our bet: **plan in a frozen, pretrained latent space toward a learned
subgoal.** Concretely —

1. **Frozen perception.** A pretrained JEPA-from-pixels encoder (LeWM, a
   ViT/DINO-style backbone) turns each frame into a latent `z`. The encoder is
   never trained here — it already carries object/layout structure, and
   freezing it is what keeps perception game-agnostic. We use `CLS ⊕
   patch-mean` so `z` carries both the global summary and spatial layout.
2. **Learned dynamics.** A small residual MLP predicts `ẑ' = z + f(z, action)`.
   Cheap enough to keep training *online, within a single game*, so the model
   adapts to each game's action semantics on the fly.
3. **Learned subgoal.** A GRU goal head reads the recent latent history of the
   *current episode* and predicts a latent ~`GOAL_HORIZON` steps ahead — "where
   a competent player heads next". Conditioning on in-episode history (not a
   single Markov state) is the generalisation lever: it specialises to the
   game being played from context alone, so it can steer on games it never saw
   in training.
4. **Planning as goal-reaching.** A CEM solver
   (`CategoricalCEMSolver`) rolls candidate action sequences through the
   dynamics and picks the one whose imagined final latent lands nearest the
   subgoal. Distance-to-goal *is* the (negative) reward. Engine-declared
   unavailable actions get a hard cost penalty so no illegal plan survives.

If the thesis holds, the agent needs no per-game reward signal at test time: the
goal head supplies direction, the dynamics supply a model to plan against, and
online adaptation closes the gap the frozen pieces leave.

---

## Architecture at a glance

| Component | Where | Trained | Role |
|---|---|---|---|
| LeWM encoder | `WorldModel.encode` | **frozen** (pretrained) | frame → latent `z` (`CLS ⊕ patch-mean`) |
| Dynamics MLP | `WorldModel.dynamics` | pretrained **+ online** | `(z, action) → ẑ'`, residual |
| Goal head (GRU) | `GoalHead` | pretrained | recent latent history → k-ahead subgoal |
| Decoder probe | `WorldModel.decoder` | online (monitor only) | latent → imagined frame, 16-way per-cell |
| CEM solver | `CategoricalCEMSolver` | — | search action sequences → min distance-to-goal |

Data flow per step (`MyAgent._goal_directed_action`):
`frame → encode → train dynamics on t-1 transition → propose subgoal →
CEM plan → ε-greedy pick → remember (z, action) for next step`.

Fallbacks, in order: no encoder → **random available action**; encoder but no
trained goal head → **random** (won't steer toward an untrained noise goal);
goal head present → **CEM plan** with `EPSILON` off-plan exploration.

Key hyper-parameters live at the top of `MyAgent` (`PLAN_HORIZON=5`,
`NUM_SAMPLES=256`, `CEM_STEPS=5`, `TOPK=32`, `EPSILON=0.1`,
`LEARNING_RATE=3e-3`) and `GOAL_CONTEXT=8`, `GOAL_HORIZON=5` in
`my_agent.py` (shared with `pretrain.py`).

---

## Pretraining (`pretrain.py` → `agent/models/pretrained_heads-deterministic-linear.pt`)

The reference architecture is always:

`frozen encoder → linear latent adapter → World Model → CEM`.

Pretraining gathers random-exploration traces, keeps the visual encoder fully
frozen, and trains the **latent adapter**, **dynamics** (predict next latent),
and **decoder probe** (class-weighted cross-entropy over the 16-colour palette).
Frame encodings are cached under `dataset/.enc_cache/`. The adapter is frozen
before the artefact is saved and remains frozen during inference. The artefact
is auto-loaded at agent startup; `COLD=1` skips the warm start while retaining
the same architecture.

---

## How we evaluate

- `make play-local` — score across every game (fast, the primary signal).
- `make play-local GAME=<id>` — single game while debugging.
- `make monitor GAME=<id>` — live arcade window: current frame, 3-D PCA of the
  latent trajectory, the decoded CEM plan (imagination strip), and the button
  row. The main tool for *seeing* whether the latent geometry and plans are
  sane, not just whether the score moved.
- A/B knobs: `COLD=1` (from-scratch vs warmed), `OBJECTIVE=…`
  (novelty / displacement / surprise / reward — legacy planner objectives kept
  for comparison against the current goal-reaching cost).

**Metrics to watch, beyond raw score:** dynamics prediction error (is the model
learning?), fraction of steps where CEM beats random (is planning helping?),
goal-distance trajectory (does it actually decrease?), and generalisation gap
(held-out-game score vs training-game score).

---

## Discrete iCEM-inspired planner study (2026-08-05)

Four planner-only changes were implemented and evaluated sequentially against
the latent-adapter baseline. Each change was removed before testing the next.
The fixed gate used all 25 public games, seed 101, 200 actions, the
`pretrained_heads-deterministic-linear.pt` checkpoint, separate ACTION6 mode,
and identical waypoint, exploration, World Model, encoder, and adapter settings.
The ACTION6 joint paths were also covered by targeted solver tests.

| Final CEM variant | Levels | Aggregate score | Waypoints reached |
|---|---:|---:|---:|
| Marginal baseline | 1 (`r11l`) | 0.0051342435 | 20 |
| Best evaluated complete trajectory | 1 (`r11l`) | 0.0051342435 | 25 |
| Shifted previous-plan warm-start | 1 (`r11l`) | 0.0051342435 | 29 |
| Within-solve elite reuse (32/256) | 1 (`r11l`) | 0.0051342435 | 37 |
| Categorical momentum (`alpha=0.1`) | 1 (`r11l`) | 0.0051342435 | 37 |

The installed `CategoricalCEMSolver` already supports `alpha`, but defaults to
zero. It ignores `init_action`, so shifted warm-start requires an adapter around
the solver. Exact elite reuse exists only in the continuous `ICEMSolver`. The
agent's passive `CEMEliteTracker` already exposes complete sampled trajectories,
but the production planner intentionally returns the marginal vote.

All four variants changed navigation and three substantially increased internal
waypoint completions, but none improved either primary retention metric: game
score or levels completed. Per the predeclared gate, no variant qualified for
the five-seed extension, and all experimental flags and code were removed. The
production baseline therefore remains cold-start categorical CEM, no elite
reuse, zero momentum, and marginal final selection.

---

## Open questions / experiment backlog

Roughly ordered by expected value. Log results inline as they land.

- [ ] **Encoder checkpoint is a placeholder.** `LEWM_CHECKPOINT` defaults to
  `galilai-group/lewm` (a TODO) with a checked-in `agent/models/lewm-pusht`
  for local runs. Confirm which checkpoint actually transfers to ARC grids;
  the whole thesis rests on the frozen encoder carrying useful structure.
- [ ] **Does the goal head generalise?** Measure the training-vs-held-out-game
  gap. If it memorises, the in-context conditioning isn't doing its job —
  consider stronger history encoders or contrastive goal training.
- [ ] **ACTION6 coordinates are random.** Complex actions get random `(x, y)`.
  Have the planner choose coordinates too (the biggest obvious hole for any
  game that needs pointing).
- [ ] **Does online dynamics adaptation help within a game?** A/B `COLD=1` vs
  warm, and warm-frozen-dynamics vs warm-online. Isolate the contribution of
  each trained part.
- [ ] **CEM budget vs latency.** `NUM_SAMPLES×CEM_STEPS×PLAN_HORIZON` is the
  cost per step against the action budget. Find the knee.
- [ ] **Subgoal horizon.** `GOAL_HORIZON=5` is a guess; sweep it against
  `PLAN_HORIZON`. A subgoal the planner can't reach within its horizon is
  wasted.
- [ ] **Per-game priors.** LS20's ACTION4 2× bias is a hand-tuned example;
  decide whether such priors help enough to keep or are a generalisation trap.
- [ ] **Reward head, retired.** Earlier objectives (novelty/surprise/reward)
  are kept as `OBJECTIVE=…` knobs. Decide whether any beats pure goal-reaching
  and either fold it into the cost or drop the dead code.

---

## Notes for the next session

- The only file that ships to Kaggle is `agent/my_agent.py` (spliced into the
  notebook by `scripts/build_notebook.py`). `agent/monitor.py` and
  `pretrain.py` are **local-only** — nothing monitor-gated (`ARC_MONITOR`) runs
  on Kaggle. Keep it that way: no monitor imports at module top level.
- Kaggle accelerated sessions have **no internet**, so the encoder checkpoint
  and pretrained heads must be self-contained / checked in, not downloaded at
  runtime.
- `pretrain.py` deliberately points `PRETRAINED_HEADS` at a non-existent path
  so it can't warm-start from its own previous output.
- Prefer `make monitor` when a change *should* alter behaviour but the score
  doesn't move — the geometry usually shows why before the score does.
