"""World-model agent for the ARC-AGI-3 interactive benchmark.

The controller follows a model-predictive-control loop:

1. A frozen LeWM visual encoder maps the latest grid to a latent state.
2. A pretrained linear adapter maps that representation to the latent
   space used by the rest of the controller.
3. An action-conditioned dynamics model predicts how that state will evolve.
4. A bounded waypoint memory stores diverse states observed during interaction.
5. An optional Dominated Novelty Search (DNS) filter selects useful, diverse
   waypoint candidates without training an additional goal network.
6. Cross-Entropy Method (CEM) planning imagines short action sequences and
   executes only the first action before replanning from the next observation.
7. For ACTION6, the planner can optimise the action and its click coordinate as
   one structured decision instead of choosing the coordinate afterwards.

The dynamics model learns transition structure, while the waypoint layer supplies
the target used by the planner. These responsibilities intentionally remain
separate: predicting the next state does not by itself define which state should
be pursued.

`scripts/build_notebook.py` embeds this file directly in the Kaggle submission.
The class name and the `is_done` / `choose_action` methods implement the contract
required by the official ARC-AGI-3-Agents framework.
"""
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver import CategoricalCEMSolver

from arcengine import FrameData, GameAction, GameState

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent


# ── Dominated Novelty Search (DNS) ───────────────────────────────────────────
#
# DNS is used only as a waypoint pre-filter. It does not alter the learned world
# model, predict rewards, or directly execute actions. Each stored waypoint has:
#   - an objective: its distance from the memory at creation time;
#   - a descriptor: its latent representation.
# A candidate competes only against strictly fitter candidates. Its DNS score is
# the mean distance to the k nearest members of that fitter set. Keeping the
# highest DNS scores favours candidates that combine quality with behavioural
# diversity, after which the existing cyclic/least-targeted policy picks one.
@dataclass(frozen=True)
class DNSCandidate:
    """DNS diagnostics for one candidate."""

    index: int
    objective: float
    fitter_indices: tuple[int, ...]
    neighbor_indices: tuple[int, ...]
    dns_score: float


@dataclass(frozen=True)
class DNSSelection:
    """Complete result of DNS competition followed by truncation selection."""

    candidates: tuple[DNSCandidate, ...]
    retained_indices: tuple[int, ...]
    capacity: int
    k_neighbors: int


class DNSSelector:
    """Dominated Novelty Search competition and top-capacity selection.

    Objectives follow the maximisation convention: a larger value is fitter.
    Descriptors may have any shape after the leading candidate axis; they are
    flattened before Euclidean distances are computed.
    """

    def __init__(self, k_neighbors: int = 5) -> None:
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be at least 1")
        self.k_neighbors = int(k_neighbors)

    def select(
        self,
        objectives: Sequence[float],
        descriptors: torch.Tensor,
        capacity: int | None = None,
    ) -> DNSSelection:
        """Compute DNS scores and retain the top ``capacity`` candidates."""
        objective_values = tuple(float(value) for value in objectives)
        count = len(objective_values)
        if count == 0:
            return DNSSelection((), (), 0, self.k_neighbors)
        if any(not math.isfinite(value) for value in objective_values):
            raise ValueError("DNS objectives must contain only finite values")

        descriptor_tensor = torch.as_tensor(
            descriptors, dtype=torch.float64
        ).detach()
        if descriptor_tensor.ndim < 2 or descriptor_tensor.shape[0] != count:
            raise ValueError(
                "descriptors must have shape (num_candidates, ...), got "
                f"{tuple(descriptor_tensor.shape)} for {count} candidates"
            )
        descriptor_tensor = descriptor_tensor.reshape(count, -1).cpu()
        if not bool(torch.isfinite(descriptor_tensor).all()):
            raise ValueError("DNS descriptors must contain only finite values")

        distances = torch.cdist(
            descriptor_tensor, descriptor_tensor, p=2
        )
        rows: list[DNSCandidate] = []
        for index, objective in enumerate(objective_values):
            # Strict comparison is part of DNS: equal-quality candidates do not
            # dominate one another and therefore are not in the fitter set.
            fitter = tuple(
                other
                for other, other_objective in enumerate(objective_values)
                if other_objective > objective
            )
            if not fitter:
                # A locally best candidate has no fitter reference. The DNS
                # convention assigns +inf so truncation cannot discard it.
                neighbors: tuple[int, ...] = ()
                dns_score = math.inf
            else:
                ordered = sorted(
                    fitter,
                    key=lambda other: (float(distances[index, other]), other),
                )
                neighbors = tuple(ordered[: self.k_neighbors])
                dns_score = sum(
                    float(distances[index, other]) for other in neighbors
                ) / len(neighbors)
            rows.append(
                DNSCandidate(
                    index=index,
                    objective=objective,
                    fitter_indices=fitter,
                    neighbor_indices=neighbors,
                    dns_score=dns_score,
                )
            )

        if capacity is None:
            capacity = (count + 1) // 2
        if not 1 <= capacity <= count:
            raise ValueError(
                f"capacity must be in [1, {count}], got {capacity}"
            )
        # Python's stable ordering gives deterministic index-order tie handling
        # without adding objective quality as an undocumented second criterion.
        ranked = sorted(
            range(count),
            key=lambda index: -rows[index].dns_score,
        )
        retained = tuple(ranked[:capacity])
        return DNSSelection(
            candidates=tuple(rows),
            retained_indices=retained,
            capacity=capacity,
            k_neighbors=self.k_neighbors,
        )


# ── Observation encoding ─────────────────────────────────────────────────────
# ARC-AGI-3 16-colour palette (cell id → RGB), matching the engine's renderer.
# We render frames to RGB so a pretrained world model can consume them directly.
ARC_PALETTE = torch.tensor(
    [
        (255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102),
        (51, 51, 51),    (0, 0, 0),       (229, 58, 163),  (255, 123, 204),
        (249, 60, 49),   (30, 147, 255),  (136, 216, 241), (255, 220, 0),
        (255, 133, 27),  (146, 18, 49),   (79, 204, 48),   (163, 86, 214),
    ],
    dtype=torch.float32,
) / 255.0
NUM_COLORS = ARC_PALETTE.shape[0]  # 16

# Pretrained LeWM (LeWorldModel, JEPA-from-pixels) checkpoint to take the frozen
# encoder from. Accepts a HuggingFace repo id, a local folder, or a `.pt` file
# (see stable_worldmodel.wm.load_pretrained). Overridable via env so local
# targets (`make monitor`) can point at the checked-in copy under agent/models/.
LEWM_CHECKPOINT = os.environ.get("LEWM_CHECKPOINT", "galilai-group/lewm")  # TODO: point at the actual checkpoint

# Pretrained ViT/DINO backbones expect ImageNet-normalised RGB in [0, 1].
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ── Behaviour knobs (env-overridable) ─────────────────────────────────────────
#   ARC_DEVICE          — 'cuda' / 'cpu' for the world model (default: cuda if available).
def _env_num(name: str, default: float) -> float:
    try:
        return type(default)(os.environ[name])
    except (KeyError, ValueError):
        return default


def frame_data_to_tensor(frame_data: FrameData) -> torch.Tensor:
    """Render an ARC-AGI-3 frame to an RGB tensor of shape ``(1, 3, H, W)``.

    ``frame_data.frame`` is a *stack* of 2D integer grids; we take the most
    recent grid (the current screen), map each cell id through `ARC_PALETTE` to
    RGB in ``[0, 1]``, and add the leading batch dim that flows into
    ``WorldModel.encode``.
    """
    grids = frame_data.frame
    if not grids:
        # No observation yet → an all-background frame so encode() still works.
        grid = torch.zeros(1, 1, dtype=torch.long)
    else:
        grid = torch.tensor(grids[-1], dtype=torch.long)   # (H, W)
    return grid_to_rgb(grid)


def grid_to_rgb(grid: torch.Tensor) -> torch.Tensor:
    """Cell-id grid ``(H, W)`` → RGB tensor ``(1, 3, H, W)`` in [0, 1] through
    the ARC palette."""
    grid = grid.clamp(0, NUM_COLORS - 1)
    rgb = ARC_PALETTE[grid]                                 # (H, W, 3) in [0, 1]
    return rgb.permute(2, 0, 1).unsqueeze(0)                # (1, 3, H, W)


def frame_data_to_grid(frame_data: FrameData) -> torch.Tensor:
    """The most recent grid as raw cell ids, shape ``(1, H, W)`` long — the
    classification target for the decoder probe."""
    grids = frame_data.frame
    if not grids:
        return torch.zeros(1, 1, 1, dtype=torch.long)
    return torch.tensor(grids[-1], dtype=torch.long).clamp_(0, NUM_COLORS - 1).unsqueeze(0)


GRID_EVENT_MOVED = 1
GRID_EVENT_APPEARED = 2
GRID_EVENT_DISAPPEARED = 4
GRID_EVENT_COLOR_CHANGED = 8
GRID_EVENT_MAX_COMPONENT_FRACTION = 1 / 16
GRID_EVENT_NAMES = (
    (GRID_EVENT_MOVED, "moved"),
    (GRID_EVENT_APPEARED, "appeared"),
    (GRID_EVENT_DISAPPEARED, "disappeared"),
    (GRID_EVENT_COLOR_CHANGED, "color_changed"),
)


def grid_event_names(mask: int) -> list[str]:
    """Stable names for the four non-trainable V0 grid event bits."""
    return [name for bit, name in GRID_EVENT_NAMES if mask & bit]


def _grid_components(
    grid: torch.Tensor,
    background: int,
) -> list[dict[str, Any]]:
    """Extract same-colour 4-connected components from one small ARC grid."""
    cells = grid.detach().reshape(grid.shape[-2], grid.shape[-1]).long().cpu()
    height, width = cells.shape
    visited = torch.zeros_like(cells, dtype=torch.bool)
    components: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            colour = int(cells[y, x].item())
            if colour == background or visited[y, x]:
                continue
            visited[y, x] = True
            pending = [(y, x)]
            component_cells: list[tuple[int, int]] = []
            while pending:
                cy, cx = pending.pop()
                component_cells.append((cy, cx))
                for ny, nx in (
                    (cy - 1, cx), (cy + 1, cx),
                    (cy, cx - 1), (cy, cx + 1),
                ):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not visited[ny, nx]
                        and int(cells[ny, nx].item()) == colour
                    ):
                        visited[ny, nx] = True
                        pending.append((ny, nx))
            min_y = min(yx[0] for yx in component_cells)
            min_x = min(yx[1] for yx in component_cells)
            absolute = tuple(sorted(component_cells))
            shape = tuple(sorted(
                (cy - min_y, cx - min_x) for cy, cx in component_cells
            ))
            components.append({
                "colour": colour,
                "origin": (min_y, min_x),
                "shape": shape,
                "absolute": absolute,
            })
    return components


def detect_simple_grid_events(
    previous: torch.Tensor,
    current: torch.Tensor,
) -> tuple[int, dict[str, Any]]:
    """Detect only exact V0 component moves, appearances, losses, and recolours.

    Matching is deliberately conservative and deterministic: unchanged objects
    are removed first, then stationary recolours, then translations preserving
    colour and exact normalised shape. Anything unmatched is an appearance or
    disappearance. Simultaneous recolour-plus-motion is intentionally outside
    V0 and appears as one disappearance plus one appearance.
    """
    before = previous.detach().reshape(
        previous.shape[-2], previous.shape[-1]
    ).long().cpu()
    after = current.detach().reshape(
        current.shape[-2], current.shape[-1]
    ).long().cpu()
    if before.shape != after.shape:
        raise ValueError(
            f"Grid event shapes differ: {tuple(before.shape)} vs "
            f"{tuple(after.shape)}"
        )
    background = int(torch.bincount(
        before.reshape(-1), minlength=NUM_COLORS
    ).argmax().item())
    max_component_cells = max(
        1, math.floor(before.numel() * GRID_EVENT_MAX_COMPONENT_FRACTION)
    )
    border_width = max(1, math.floor(min(before.shape) / 16))
    previous_components_all = _grid_components(before, background)
    current_components_all = _grid_components(after, background)
    # Large regions are terrain/background surfaces. A sprite crossing them
    # changes their exact connected shape and otherwise creates a false
    # disappearance+appearance pair on nearly every move. Components contained
    # wholly in a thin outer band are status/HUD candidates; ls20, for example,
    # advances a two-row action counter after every command.
    def is_trackable(component: dict[str, Any]) -> bool:
        absolute = component["absolute"]
        ys = [cell[0] for cell in absolute]
        xs = [cell[1] for cell in absolute]
        in_border_band = (
            max(ys) < border_width
            or min(ys) >= before.shape[0] - border_width
            or max(xs) < border_width
            or min(xs) >= before.shape[1] - border_width
        )
        return (
            len(component["shape"]) <= max_component_cells
            and not in_border_band
        )

    previous_components = [
        component for component in previous_components_all
        if is_trackable(component)
    ]
    current_components = [
        component for component in current_components_all
        if is_trackable(component)
    ]
    unmatched_previous = set(range(len(previous_components)))
    unmatched_current = set(range(len(current_components)))

    def consume_matches(key: str, include_colour: bool = True) -> int:
        matches = 0
        groups_previous: dict[Any, list[int]] = {}
        groups_current: dict[Any, list[int]] = {}
        for index in sorted(unmatched_previous):
            component = previous_components[index]
            group_key = (
                component["colour"] if include_colour else None,
                component[key],
            )
            groups_previous.setdefault(group_key, []).append(index)
        for index in sorted(unmatched_current):
            component = current_components[index]
            group_key = (
                component["colour"] if include_colour else None,
                component[key],
            )
            groups_current.setdefault(group_key, []).append(index)
        for group_key in sorted(
            set(groups_previous) & set(groups_current),
            key=repr,
        ):
            left = groups_previous[group_key]
            right = groups_current[group_key]
            for previous_index, current_index in zip(left, right):
                unmatched_previous.remove(previous_index)
                unmatched_current.remove(current_index)
                matches += 1
        return matches

    # Exact same colour and occupied cells: unchanged.
    consume_matches("absolute")

    # Same occupied cells but another colour: stationary recolour.
    recoloured = consume_matches("absolute", include_colour=False)

    # Same colour and exact shape at another origin: translation.
    moved = consume_matches("shape")

    mask = 0
    if moved:
        mask |= GRID_EVENT_MOVED
    if unmatched_current:
        mask |= GRID_EVENT_APPEARED
    if unmatched_previous:
        mask |= GRID_EVENT_DISAPPEARED
    if recoloured:
        mask |= GRID_EVENT_COLOR_CHANGED
    details = {
        "background": background,
        "max_component_cells": max_component_cells,
        "border_width": border_width,
        "previous_components_raw": len(previous_components_all),
        "current_components_raw": len(current_components_all),
        "previous_components": len(previous_components),
        "current_components": len(current_components),
        "moved_matches": moved,
        "appeared_components": len(unmatched_current),
        "disappeared_components": len(unmatched_previous),
        "color_changed_matches": recoloured,
        "changed_cells": int((before != after).sum().item()),
    }
    return mask, details


def simple_object_transition_predicates(
    before: torch.Tensor,
    after: torch.Tensor,
    crop_border: int = 4,
) -> torch.Tensor:
    """Three bounded grid-transition predicates for a batch of imagined states.

    ``before`` is one grid ``(H, W)`` and ``after`` is ``(A, H, W)``.  The
    returned columns are:

    1. fill ratio of the changed cells' bounding box;
    2. change in the fraction of adjacent foreground cells with different
       colours (a vectorised contact-density proxy);
    3. L1 colour-histogram change, normalised to ``[0, 1]``.

    These are the smallest inexpensive counterparts of the predicates that
    transferred most consistently in the offline LOGO study.  They use no
    parameters and no future real observation.
    """
    if before.ndim != 2 or after.ndim != 3:
        raise ValueError(
            "Expected before=(H,W) and after=(A,H,W), got "
            f"{tuple(before.shape)} and {tuple(after.shape)}"
        )
    if tuple(after.shape[-2:]) != tuple(before.shape):
        raise ValueError(
            f"Grid shapes differ: {tuple(before.shape)} vs "
            f"{tuple(after.shape[-2:])}"
        )
    height, width = before.shape
    border = max(0, int(crop_border))
    if border and height > 2 * border and width > 2 * border:
        before = before[border:-border, border:-border]
        after = after[:, border:-border, border:-border]

    before = before.long()
    after = after.long()
    actions = after.shape[0]
    cell_count = max(1, before.numel())
    changed = after != before.unsqueeze(0)
    changed_count = changed.flatten(1).sum(dim=1).float()
    bbox_fill = torch.zeros(actions, device=after.device, dtype=torch.float32)
    # Only seven action grids are evaluated, so this tiny loop avoids building
    # large coordinate tensors while keeping the expensive decoder batched.
    for index in range(actions):
        positions = changed[index].nonzero(as_tuple=False)
        if positions.numel():
            y0, x0 = positions.min(dim=0).values
            y1, x1 = positions.max(dim=0).values
            area = (y1 - y0 + 1) * (x1 - x0 + 1)
            bbox_fill[index] = changed_count[index] / area.float().clamp_min(1.0)

    all_grids = torch.cat([before.unsqueeze(0), after], dim=0)
    one_hot = F.one_hot(
        all_grids.clamp(0, NUM_COLORS - 1), num_classes=NUM_COLORS
    )
    histograms = one_hot.flatten(1, 2).sum(dim=1).float()
    backgrounds = histograms.argmax(dim=1)
    foreground = all_grids != backgrounds[:, None, None]

    def contact_fraction(grids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        horizontal_valid = mask[:, :, :-1] & mask[:, :, 1:]
        vertical_valid = mask[:, :-1, :] & mask[:, 1:, :]
        horizontal_contact = horizontal_valid & (
            grids[:, :, :-1] != grids[:, :, 1:]
        )
        vertical_contact = vertical_valid & (
            grids[:, :-1, :] != grids[:, 1:, :]
        )
        contacts = (
            horizontal_contact.flatten(1).sum(dim=1)
            + vertical_contact.flatten(1).sum(dim=1)
        ).float()
        adjacent = (
            horizontal_valid.flatten(1).sum(dim=1)
            + vertical_valid.flatten(1).sum(dim=1)
        ).float()
        return contacts / adjacent.clamp_min(1.0)

    contacts = contact_fraction(all_grids, foreground)
    contact_gain = contacts[1:] - contacts[0]
    histogram_change = (
        (histograms[1:] - histograms[0]).abs().sum(dim=1)
        / (2.0 * cell_count)
    )
    return torch.stack([bbox_fill, contact_gain, histogram_change], dim=1)


def normalise_object_predicates(
    predicates: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    """Equal-weight, per-decision min-max score over available actions only."""
    if predicates.ndim != 2:
        raise ValueError(f"Expected predicates=(A,P), got {tuple(predicates.shape)}")
    available = available.reshape(-1).bool().to(predicates.device)
    utility = torch.zeros(
        predicates.shape[0], device=predicates.device, dtype=predicates.dtype
    )
    if not available.any():
        return utility
    selected = predicates[available]
    low = selected.min(dim=0).values
    high = selected.max(dim=0).values
    span = high - low
    informative = span > 1e-8
    scaled = torch.zeros_like(selected)
    if informative.any():
        scaled[:, informative] = (
            selected[:, informative] - low[informative]
        ) / span[informative]
    utility[available] = scaled.mean(dim=1)
    return utility


def cell_class_weights(targets: torch.Tensor) -> torch.Tensor:
    """Inverse-sqrt-frequency weight per colour class within a batch of cell-id
    targets. A small moving object is a sliver of the cells, so unweighted
    losses let the decoder paint pure background and call it a day — this makes
    rare colours (the objects) matter. Shared by pretrain.py and the online
    decoder step."""
    counts = torch.bincount(targets.flatten(), minlength=NUM_COLORS).float()
    weight = (counts.sum() / counts.clamp_min(1.0)).sqrt()
    return weight / weight.mean()


# ── Latent world model ───────────────────────────────────────────────────────
class WorldModel(nn.Module):
    """Action-conditioned latent world model with three components:

      - ``encode``: frame tensor → controller latent ``z`` (frozen pretrained
        encoder, standardisation, and frozen linear adapter);
      - ``predict_next`` (dynamics): ``(z, action) → ẑ'`` predicted next latent
        (trained online);
      - ``decode`` (probe): latent → imagined frame, for the monitor only.

    Implements the stable-worldmodel ``Costable`` protocol via ``get_cost`` so it
    plugs into ``CategoricalCEMSolver``. The cost is a **goal-reaching** objective:
    roll the plan through the dynamics and reward the plan whose imagined trajectory
    comes closest to ``self.goal_latent`` (distance in the standardised latent
    space). ``MyAgent`` sets the goal from its waypoint memory. With no active
    waypoint, ``get_cost`` is a no-op and the agent explores randomly.
    """

    # Cost added per rollout step using an action the engine declared unavailable
    # — large enough to push any such plan out of the elite set.
    UNAVAILABLE_COST = 1e3

    # Dynamics MLP geometry. A shallow 1×256 predictor on the raw latent
    # collapsed to identity (AUTORESEARCH 2026-07-11); a deeper 4×768 trunk with
    # LayerNorm, trained in the *standardised* latent space, keeps a real
    # action-conditioned signal. LayerNorm lets it train at the base LR.
    DYN_WIDTH = 768
    DYN_DEPTH = 4
    # ACTION6 is structurally different from the other actions: its outcome
    # depends on both the action identity and a grid coordinate. The coordinate
    # is Fourier-encoded (COORD_FREQS bands per axis) and accompanied by a valid
    # flag. Non-spatial actions receive an all-zero coordinate feature vector.
    COORD_FREQS = 6
    COORD_DIM = 1 + 4 * COORD_FREQS  # valid flag + sin/cos × 2 axes × COORD_FREQS
    @staticmethod
    def _build_dynamics(latent_dim: int, n_actions: int, width: int, depth: int,
                        coord_dim: int = 0) -> nn.Module:
        """Residual dynamics trunk: (latent_dim + n_actions + coord_dim) →
        latent_dim, `depth` Linear+LayerNorm+ReLU blocks of `width`. The residual
        `z + trunk(·)` is applied in `predict_next`."""
        layers: list[nn.Module] = []
        c = latent_dim + n_actions + coord_dim
        for _ in range(depth):
            layers += [nn.Linear(c, width), nn.LayerNorm(width), nn.ReLU(inplace=True)]
            c = width
        layers += [nn.Linear(c, latent_dim)]
        return nn.Sequential(*layers)

    def __init__(
        self,
        n_actions: int = 6,
        fallback_latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        # The waypoint manager writes an observed latent here through `set_goal`;
        # CEM ranks imagined plans by their distance to it. None means that
        # memory collection is still in progress, so the agent explores instead.
        self.goal_latent: Optional[torch.Tensor] = None

        # Encoder: the frozen pretrained LeWM vision encoder (a ViT/DINO-style
        # backbone). We use its CLS token concatenated with the patch mean. The
        # encoder never receives gradients; only the adapter is allowed
        # to refine this representation, and only during offline pretraining.
        self.encoder = None
        latent_dim = fallback_latent_dim
        try:
            import stable_worldmodel as swm

            lewm = swm.wm.load_pretrained(LEWM_CHECKPOINT)
            self.encoder = lewm.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            latent_dim = self._infer_latent_dim()
        except Exception as exc:  # no checkpoint / no transformers / offline
            # encode() will signal not-ready and the agent falls back to random.
            self._encoder_error = exc

        self.latent_dim = latent_dim

        # Per-dimension standardisation of the frozen encoder's pooled latent,
        # applied before the adapter. The raw ViT latent has uneven
        # per-dim scale, so an MSE dynamics loss is dominated by a few
        # high-variance (largely action-independent) dims and the predictor
        # collapses toward identity (AUTORESEARCH 2026-07-11). Buffers default to
        # identity and are overwritten by the corpus statistics pretrain.py stores.
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))

        # Latent-adapter boundary: Encoder -> Standardisation -> Linear Adapter.
        # The adapter preserves dimensionality so the
        # dynamics, decoder, waypoints, and CEM interfaces remain unchanged.
        # Identity initialisation gives pretraining the baseline representation
        # at step zero. The layer is frozen by default; only pretrain.py opts in
        # to updating it, and its learned weights are frozen again before saving.
        self.latent_adapter = nn.Linear(latent_dim, latent_dim)
        with torch.no_grad():
            nn.init.eye_(self.latent_adapter.weight)
            nn.init.zeros_(self.latent_adapter.bias)
        self.set_latent_adapter_trainable(False)

        # Derive the coordinate-bearing action from the enum rather than
        # hard-coding ACTION6's numerical index. The ordering matches both
        # `_candidate_actions` and the action indexing used during pretraining.
        cand = [a for a in GameAction if a is not GameAction.RESET][:n_actions]
        coord_mask = torch.zeros(n_actions)
        for i, a in enumerate(cand):
            if a.is_complex():
                coord_mask[i] = 1.0
        self.register_buffer("coord_action_mask", coord_mask)
        self.register_buffer("_coord_freqs",
                             (2.0 ** torch.arange(self.COORD_FREQS)) * math.pi)

        # Dynamics: a deep residual MLP over concat([z, action, coord_features])
        # in the controller latent space (standardised baseline or adapted).
        self.dynamics = self._build_dynamics(
            latent_dim, n_actions, width=self.DYN_WIDTH, depth=self.DYN_DEPTH,
            coord_dim=self.COORD_DIM,
        )

        # Pixel decoder: latent z → 16-way colour logits per cell of a 64×64
        # imagined frame — a learned probe of the frozen latent space, trained
        # (class-weighted cross-entropy) on the (z, grid) pairs the agent observes
        # and used to render the monitor. Classification not RGB regression: MSE
        # hedges by averaging colours, which erases small moving objects.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16×16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32×32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, NUM_COLORS, 4, stride=2, padding=1),  # 64×64
        )

        self._load_pretrained_heads()

    def _load_pretrained_heads(self) -> None:
        """Warm-start dynamics, decoder, adapter, and standardisation stats from
        `pretrain.py` output when present: PRETRAINED_HEADS env var, else
        agent/models/pretrained_heads-deterministic-linear.pt. Silently skipped when absent or
        shape-incompatible — online learning then starts from scratch. Any extra
        keys an older checkpoint carries (goal head, distance head) are ignored."""
        path = os.environ.get("PRETRAINED_HEADS")
        if not path:
            try:
                base = os.path.dirname(os.path.abspath(__file__))
            except NameError:  # spliced into the Kaggle notebook, no __file__
                base = os.path.join(os.getcwd(), "agent")
            path = os.path.join(
                base, "models", "pretrained_heads-deterministic-linear.pt"
            )
        if not os.path.exists(path):
            return
        try:
            # A checkpoint is accepted only when every component that defines
            # the planner's latent space is present and shape-compatible.
            # Partial loading would mix incompatible latent geometries.
            blob = torch.load(path, map_location="cpu", weights_only=True)
            meta = blob["meta"]
            if meta.get("latent_dim") != self.latent_dim or meta["n_actions"] != self.n_actions:
                print(f"[world-model] pretrained heads at {path} don't match "
                      f"(latent {meta.get('latent_dim')} vs {self.latent_dim}, "
                      f"actions {meta['n_actions']} vs {self.n_actions}), skipped")
                return
            if meta.get("coord_dim") != self.COORD_DIM:
                print(f"[world-model] pretrained heads at {path} are not "
                      f"coordinate-conditioned (coord_dim {meta.get('coord_dim')} "
                      f"vs {self.COORD_DIM}), skipped — retrain with pretrain.py")
                return
            if "latent_adapter" not in blob:
                print(
                    "[world-model] latent-adapter weights are missing; "
                    "skipped"
                )
                return
            # Corpus standardisation stats: encode() applies these, so they must
            # load before (and match) the dynamics/decoder trained in that space.
            if "latent_mean" not in blob:
                print(f"[world-model] {path} predates latent standardisation, "
                      f"skipped (retrain with the current pretrain.py)")
                return
            # Restore the representation transform before the modules trained
            # against it. The adapter is explicitly frozen even if construction
            # or a caller changed its gradient state before loading.
            self.latent_mean.copy_(blob["latent_mean"])
            self.latent_std.copy_(blob["latent_std"])
            self.latent_adapter.load_state_dict(blob["latent_adapter"])
            self.set_latent_adapter_trainable(False)
            self.dynamics.load_state_dict(blob["dynamics"])
            try:
                self.decoder.load_state_dict(blob["decoder"])
            except RuntimeError:
                # Checkpoint predates the 16-way classification decoder (was
                # RGB regression) — probe starts fresh, dynamics still load.
                print(f"[world-model] decoder in {path} has the old RGB shape, skipped")
            print(f"[world-model] warm-started latent adapter + dynamics + decoder "
                  f"from {path} "
                  f"({meta['transitions']} transitions, {meta['epochs']} epochs)")
        except Exception as exc:
            print(f"[world-model] failed to load pretrained heads: {exc!r}")

    def set_goal(self, latent: Optional[torch.Tensor]) -> None:
        """Set the target the planner drives toward. ``latent`` is a *standardised*
        latent (an ``encode`` output), shape ``(D,)`` or ``(1, D)`` — stored as
        ``(D,)`` on the model's device. ``None`` clears the goal and returns the
        agent to waypoint collection or exploratory control."""
        if latent is None:
            self.goal_latent = None
            return
        device = self.latent_mean.device
        self.goal_latent = latent.detach().reshape(-1).to(device)

    def clear_goal(self) -> None:
        """Drop the current goal (planner reverts to the random-exploration base)."""
        self.goal_latent = None

    @torch.no_grad()
    def _infer_latent_dim(self) -> int:
        """Probe the pretrained encoder with a dummy frame to get the latent dim."""
        z = self.encode_base(torch.zeros(1, 3, 64, 64))
        return z.shape[-1]

    def set_latent_adapter_trainable(self, trainable: bool) -> None:
        """Control adapter gradients without ever changing encoder gradients.

        Runtime construction leaves the adapter frozen. ``pretrain.py`` is the
        only caller that enables gradients and freezes the layer again once the
        offline optimisation is complete.
        """
        self.latent_adapter.requires_grad_(trainable)
        self.latent_adapter.train(trainable)

    def encode_base(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode RGB frames with the completely frozen visual backbone.

        The returned ``CLS ⊕ patch-mean`` representation is raw: corpus
        standardisation and the linear adapter are intentionally kept
        outside this method so pretraining can cache encoder outputs once.
        """
        if self.encoder is None:
            raise NotImplementedError("pretrained LeWM encoder unavailable")
        frame = (frame - IMAGENET_MEAN.to(frame)) / IMAGENET_STD.to(frame)
        with torch.no_grad():
            hidden = self.encoder(frame, interpolate_pos_encoding=True).last_hidden_state
            cls, patch_mean = hidden[:, 0], hidden[:, 1:].mean(dim=1)
            z = torch.cat([cls, patch_mean], dim=-1)  # (B, 2 * encoder_dim)
        return z

    def adapt_latent(self, raw_latent: torch.Tensor) -> torch.Tensor:
        """Standardise an encoder output and apply the latent adapter."""
        # This is the single projection path used during pretraining and
        # inference; downstream planning never consumes raw encoder features.
        z = (raw_latent - self.latent_mean.to(raw_latent)) / (
            self.latent_std.to(raw_latent)
        )
        return self.latent_adapter(z)

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        """Map an RGB frame to the latent space consumed by the controller.

        The frozen pretrained adapter is applied after corpus standardisation,
        so dynamics, waypoints, and CEM all operate in the adapted space.
        """
        return self.adapt_latent(self.encode_base(frame))

    def decode_logits(self, latent: torch.Tensor) -> torch.Tensor:
        """Latent ``(B, latent_dim)`` → per-cell colour logits
        ``(B, NUM_COLORS, 64, 64)`` — the decoder probe's training output."""
        return self.decoder(latent)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Latent → imagined RGB frame ``(B, 3, 64, 64)`` in [0, 1]: per-cell
        argmax through the ARC palette. Only as good as the trained decoder probe."""
        idx = self.decode_logits(latent).argmax(dim=1)      # (B, 64, 64)
        return ARC_PALETTE.to(latent.device)[idx].permute(0, 3, 1, 2)

    def predict_next(self, latent: torch.Tensor, action_onehot: torch.Tensor,
                     coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Dynamics: predict the next latent ``ẑ'``, shape ``(..., latent_dim)``.
        Residual so the model predicts the *change* the action causes.

        ``coords`` is the ACTION6 click ``(x, y)``, with shape ``(..., 2)`` in
        ``[0, 63]``; negative values mean "no coordinate". If omitted, the
        legacy action-only planner uses a neutral centre click for ACTION6 and
        the absent-coordinate sentinel for every simple action."""
        if coords is None:
            coords = self._default_coords(action_onehot)
        x = torch.cat([latent, action_onehot, self._coord_features(coords)], dim=-1)
        return latent + self.dynamics(x)

    def _default_coords(self, action_onehot: torch.Tensor) -> torch.Tensor:
        """Return the coordinate convention used by action-only CEM.

        ACTION6 receives the grid centre because this legacy solver does not
        optimise coordinates. Simple actions receive ``(-1, -1)``, which is an
        absence marker rather than a real off-grid click. Online transitions and
        coordinate-aware planning always pass an explicit coordinate.
        """
        lead = action_onehot.shape[:-1]
        is_coord = (action_onehot * self.coord_action_mask.to(action_onehot)
                    ).sum(dim=-1, keepdim=True) > 0
        centre = torch.full((*lead, 2), 31.0, device=action_onehot.device)
        none = torch.full((*lead, 2), -1.0, device=action_onehot.device)
        return torch.where(is_coord, centre, none)

    def _coord_features(self, coords: torch.Tensor) -> torch.Tensor:
        """Click ``(..., 2)`` → ``(..., COORD_DIM)``: a validity flag plus Fourier
        features of the normalised coordinate, zeroed where the click is absent
        (any component ``< 0``). The validity flag distinguishes a real click at
        the origin from the all-zero representation of a missing coordinate."""
        c = coords.float()
        valid = (c >= 0).all(dim=-1, keepdim=True).float()
        xy = c.clamp(min=0.0) / 63.0
        ang = xy.unsqueeze(-1) * self._coord_freqs.to(c)         # (..., 2, F)
        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).flatten(-2)
        return torch.cat([valid, feats * valid], dim=-1)

    @staticmethod
    def prediction_error(predicted_latent: torch.Tensor, actual_latent: torch.Tensor) -> torch.Tensor:
        """Latent prediction error between the dynamics' prediction and the
        actual encoding one step later — the training target the dynamics
        minimise. Shape ``(...,)``."""
        return ((predicted_latent - actual_latent) ** 2).mean(dim=-1)

    @torch.no_grad()
    def _decode_grid(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latents to argmax cell-id grids, preserving leading dims:
        ``(..., D) → (..., 64, 64)``."""
        lead = latent.shape[:-1]
        logits = self.decode_logits(latent.reshape(-1, latent.shape[-1]))
        return logits.argmax(dim=1).reshape(*lead, 64, 64)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """stable-worldmodel ``Costable`` hook for the CEM solver — the
        **goal-reaching** objective. Roll each candidate plan forward through the
        dynamics and score it by how close its imagined trajectory gets to
        ``self.goal_latent`` (squared latent distance, in the standardised space).

        We take the **minimum distance over the horizon**, not the frontier only:
        a plan is good if it reaches the goal at *any* point along the rollout, and
        the receding horizon (execute the first action, re-plan next step) means a
        plan that touches the goal early and drifts after is still the right first
        move. Distance is the same per-dim MSE the dynamics train on, so goal-space
        and training-space are commensurate.

        With no goal set (``self.goal_latent is None``) this returns a flat zero
        cost — the agent should be falling back to random exploration and not
        calling the planner, but this stays well-defined if it does.

        Args:
            info_dict: ``"latent"`` of shape ``(B, N, latent_dim)`` (current latent
                expanded over the ``N`` candidates) and optionally ``"available"``
                of shape ``(B, N, n_actions)``.
            action_candidates: one-hot actions ``(B, N, horizon, n_actions)``.

        Returns:
            cost ``(B, N)`` — lower is better. Every step using an
            engine-unavailable action adds ``UNAVAILABLE_COST``.
        """
        latent = info_dict["latent"]
        available = info_dict.get("available")
        horizon = action_candidates.shape[2]
        penalty = torch.zeros(latent.shape[:2], device=latent.device, dtype=latent.dtype)

        if self.goal_latent is None:
            return penalty

        goal = self.goal_latent.to(latent)                    # (D,)
        z = latent                                            # (B, N, D)
        best = torch.full(latent.shape[:2], float("inf"),
                          device=latent.device, dtype=latent.dtype)
        for h in range(horizon):
            action_onehot = action_candidates[:, :, h, :]     # (B, N, A)
            z = self.predict_next(z, action_onehot)           # (B, N, D)
            dist = ((z - goal) ** 2).mean(dim=-1)             # (B, N)
            best = torch.minimum(best, dist)
            if available is not None:
                legal = (action_onehot * available).sum(dim=-1)  # (B, N) ∈ {0, 1}
                penalty = penalty + self.UNAVAILABLE_COST * (1.0 - legal)
        return best + penalty


# ── Goal-conditioned planning objective ──────────────────────────────────────
class WaypointCEMObjective:
    """Non-trainable CEM objective for reaching the active waypoint latent.

    This adapter deliberately lives outside ``WorldModel``. It uses the frozen
    interface already available to the planner (``predict_next`` and the goal
    installed by ``set_goal``), owns no tensor parameters, and performs no
    optimisation. Plans are ranked by final latent MSE. A small bounded
    stagnation penalty breaks ties in favour of rollouts whose predicted latent
    actually changes.
    """

    def __init__(
        self,
        world_model: WorldModel,
        stagnation_weight: float,
        stagnation_margin: float,
        object_score_weight: float = 0.0,
        uncertainty_weight: float = 0.0,
        uncertainty_k: int = 8,
    ) -> None:
        self.world_model = world_model
        self.stagnation_weight = stagnation_weight
        self.stagnation_margin = stagnation_margin
        self.object_score_weight = object_score_weight
        self.uncertainty_weight = uncertainty_weight
        self.uncertainty_k = uncertainty_k

    def parameters(self):
        """Expose the existing WM dtype to CEM; this object adds no parameters."""
        return self.world_model.parameters()

    def _local_replay_error(
        self,
        latent: torch.Tensor,
        action_onehot: torch.Tensor,
        coords: Optional[torch.Tensor],
        replay: dict[str, Any],
    ) -> torch.Tensor:
        """Vectorised same-action replay k-NN error for rollout states.

        The replay residuals were computed once, before CEM, using only real
        transitions already observed under the current dynamics. Distances
        exactly match the passive diagnostic: standardised-latent MSE plus
        normalised coordinate MSE for ACTION6. The result is divided by the
        replay residual P95 and clipped to ``[0, 1]``.
        """
        leading_shape = latent.shape[:-1]
        flat_latent = latent.reshape(-1, latent.shape[-1])
        flat_actions = action_onehot.argmax(dim=-1).reshape(-1)
        flat_coords = None if coords is None else coords.reshape(-1, 2)
        # An unseen action is maximally uncertain once some replay exists.
        result = torch.ones(
            flat_latent.shape[0], device=latent.device, dtype=latent.dtype
        )
        replay_latents = replay["latents"].to(latent)
        replay_actions = replay["action_indices"].to(
            device=latent.device
        )
        replay_coords = replay["coords"].to(latent)
        residuals = replay["residuals"].to(latent)
        scale = replay["normalization_scale"].to(latent).clamp_min(1e-12)
        complex_action_index = int(replay["complex_action_index"])

        for action_index in flat_actions.unique().tolist():
            query_indices = torch.nonzero(
                flat_actions == action_index, as_tuple=False
            ).reshape(-1)
            reference_indices = torch.nonzero(
                replay_actions == action_index, as_tuple=False
            ).reshape(-1)
            if reference_indices.numel() == 0:
                continue
            queries = flat_latent[query_indices]
            references = replay_latents[reference_indices]
            # Pairwise per-dimension squared distance without constructing a
            # Q x R x D tensor.
            distances = (
                queries.square().mean(dim=-1, keepdim=True)
                + references.square().mean(dim=-1).unsqueeze(0)
                - 2.0
                * (queries @ references.transpose(0, 1))
                / max(queries.shape[-1], 1)
            ).clamp_min(0.0)
            if action_index == complex_action_index and flat_coords is not None:
                query_coords = flat_coords[query_indices] / 63.0
                reference_coords = replay_coords[reference_indices] / 63.0
                distances = distances + (
                    (query_coords[:, None, :] - reference_coords[None, :, :])
                    .square()
                    .mean(dim=-1)
                )
            neighbour_count = min(
                self.uncertainty_k, int(reference_indices.numel())
            )
            neighbour_indices = torch.topk(
                distances, k=neighbour_count, dim=1, largest=False
            ).indices
            local_error = residuals[reference_indices][
                neighbour_indices
            ].mean(dim=1)
            result[query_indices] = (local_error / scale).clamp(0.0, 1.0)
        return result.reshape(leading_shape)

    def trajectory_uncertainty(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
        coordinate_candidates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mean bounded replay uncertainty along every imagined trajectory."""
        latent = info_dict["latent"]
        replay = info_dict.get("uncertainty_replay")
        result = torch.zeros(
            latent.shape[:2], device=latent.device, dtype=latent.dtype
        )
        if replay is None or not replay.get("size", 0):
            return result
        z = latent
        horizon = action_candidates.shape[2]
        for h in range(horizon):
            action_onehot = action_candidates[:, :, h, :]
            coords = (
                None
                if coordinate_candidates is None
                else coordinate_candidates[:, :, h, :]
            )
            result = result + self._local_replay_error(
                z, action_onehot, coords, replay
            )
            if coords is None:
                z = self.world_model.predict_next(z, action_onehot)
            else:
                z = self.world_model.predict_next(z, action_onehot, coords)
        return result / max(horizon, 1)

    def get_cost(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
        coordinate_candidates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Roll candidates through the unchanged WM and score goal proximity.

        ``coordinate_candidates`` is optional so the legacy action-only CEM
        remains bit-for-bit on its old path.  The joint ACTION6 planner supplies
        an explicit ``(x, y)`` for every rollout position; simple actions carry
        ``(-1, -1)`` and are therefore unchanged by coordinate conditioning.
        """
        latent = info_dict["latent"]
        available = info_dict.get("available")
        penalty = torch.zeros(
            latent.shape[:2], device=latent.device, dtype=latent.dtype
        )
        goal = self.world_model.goal_latent
        if goal is None:
            return penalty

        z = latent
        stagnation = torch.zeros_like(penalty)
        horizon = action_candidates.shape[2]
        margin = max(self.stagnation_margin, 1e-12)
        for h in range(horizon):
            action_onehot = action_candidates[:, :, h, :]
            coords = (
                None
                if coordinate_candidates is None
                else coordinate_candidates[:, :, h, :]
            )
            if coords is None:
                # Preserve compatibility with the legacy WorldModel protocol
                # and light-weight test doubles that accept actions only.
                z_next = self.world_model.predict_next(z, action_onehot)
            else:
                z_next = self.world_model.predict_next(
                    z, action_onehot, coords
                )
            movement = ((z_next - z) ** 2).mean(dim=-1)
            # Normalised and bounded in [0, 1] per step. It penalises only
            # near-static predictions; it never rewards arbitrarily large moves.
            stagnation = stagnation + (margin - movement).clamp_min(0.0) / margin
            z = z_next
            if available is not None:
                legal = (action_onehot * available).sum(dim=-1)
                penalty = penalty + self.world_model.UNAVAILABLE_COST * (1.0 - legal)

        final_goal_distance = ((z - goal.to(z)) ** 2).mean(dim=-1)
        stagnation = stagnation / max(horizon, 1)
        cost = (
            final_goal_distance
            + self.stagnation_weight * stagnation
            + penalty
        )
        # Receding-horizon execution uses only the first planned action.  Its
        # object utility is precomputed once from the seven decoded one-step
        # outcomes and shared by all sequences beginning with that action.
        # This leaves CEM sampling/refitting and the WM rollout untouched.
        object_utility = info_dict.get("object_utility")
        if self.object_score_weight and object_utility is not None:
            first_action = action_candidates[:, :, 0, :]
            first_utility = (first_action * object_utility).sum(dim=-1)
            cost = cost - self.object_score_weight * first_utility
        if self.uncertainty_weight:
            uncertainty = self.trajectory_uncertainty(
                info_dict, action_candidates, coordinate_candidates
            )
            cost = cost + self.uncertainty_weight * uncertainty
        return cost


class CEMEliteTracker:
    """Capture the best complete sequence sampled by categorical CEM.

    ``stable-worldmodel`` exposes iteration callbacks but returns only the
    per-position marginal mode. This callback is passive: it records the best
    evaluated complete candidate and its cost so the caller can measure the
    reconstruction regret without changing planner behaviour.
    """

    output_key = "elite_tracker"

    def __init__(self) -> None:
        self.history: list[list[dict[str, Any]]] = []
        self._current: list[dict[str, Any]] = []
        self.best_actions: Optional[torch.Tensor] = None
        self.best_costs: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.history = []
        self._current = []
        self.best_actions = None
        self.best_costs = None

    def start_batch(self) -> None:
        if self._current:
            self.history.append(self._current)
        self._current = []

    def end_solve(self) -> None:
        if self._current:
            self.history.append(self._current)
            self._current = []

    def __call__(self, **state: Any) -> None:
        candidates = state["candidates"]
        costs = state["costs"]
        best_indices = costs.argmin(dim=1)
        batch = torch.arange(costs.shape[0], device=costs.device)
        best_candidates = candidates[batch, best_indices]
        # action_block is one in this agent, so the flattened final dimension
        # is exactly the categorical action dimension.
        self.best_actions = best_candidates.argmax(dim=-1).detach().cpu()
        self.best_costs = costs[batch, best_indices].detach().cpu()
        self._current.append(
            {
                "step": int(state["step"]),
                "best_costs": self.best_costs.tolist(),
                "best_first_actions": self.best_actions[:, 0].tolist(),
            }
        )


# ── ACTION6-aware CEM solver ─────────────────────────────────────────────────
class JointActionCoordinateCEMSolver:
    """CEM over the structured decision ``(action, coordinate)``.

    ACTION6 is not a single categorical command: it is a command paired with an
    ``(x, y)`` click. Treating the coordinate as an afterthought can select an
    action whose subsequently chosen click was never evaluated with the rest of
    the plan. This solver keeps the pair coupled throughout sampling, rollout,
    elite selection, and execution.

    The distribution factorises as ``p(action) * p(coord | ACTION6)``. This is
    still one joint decision per horizon position: sampled coordinates travel
    with their sampled ACTION6 through the objective and the selected first
    pair is executed unchanged.  Keeping two categorical factors avoids the
    severe prior bias and sample dilution of flattening six simple actions plus
    64 click locations into 70 unrelated categories.

    Coordinates can use either the fixed categorical lattice of the validated
    joint baseline or a continuous coarse-to-fine distribution. In the latter
    mode, the first CEM iteration samples the complete 64x64 domain uniformly;
    subsequent iterations refit a Gaussian from ACTION6 elites only. If no
    elite uses ACTION6 at a horizon position, its distribution is retained.
    """

    def __init__(
        self,
        model: WaypointCEMObjective,
        coordinate_candidates: torch.Tensor,
        complex_action_index: int,
        num_samples: int,
        n_steps: int,
        topk: int,
        device: torch.device,
        seed: int,
        smoothing: float = 0.05,
        coordinate_mode: str = "categorical",
    ) -> None:
        self.model = model
        self.coordinate_candidates = coordinate_candidates
        self.complex_action_index = complex_action_index
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.device = device
        self.smoothing = smoothing
        if coordinate_mode not in {"categorical", "multires"}:
            raise ValueError(
                "coordinate_mode must be 'categorical' or 'multires'"
            )
        self.coordinate_mode = coordinate_mode
        # Keep the action stream identical to the legacy categorical CEM.
        # Coordinate sampling uses an independent deterministic stream so merely
        # enabling joint planning cannot perturb simple-action candidates.
        self.action_torch_gen = torch.Generator(
            device=device
        ).manual_seed(seed)
        self.coordinate_torch_gen = torch.Generator(
            device=device
        ).manual_seed(seed + 1_000_003)
        try:
            self.dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self.dtype = torch.float32
        self._configured = False

    def configure(
        self, *, action_space: gym.Space, n_envs: int, config: Any
    ) -> None:
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError(
                "JointActionCoordinateCEMSolver requires Discrete actions"
            )
        self.n_actions = int(action_space.n)
        self.n_envs = n_envs
        self.horizon = int(config.horizon)
        self._configured = True

    def _uniform(
        self, batch_size: int, categories: int
    ) -> torch.Tensor:
        return torch.full(
            (batch_size, self.horizon, categories),
            1.0 / categories,
            device=self.device,
            dtype=self.dtype,
        )

    def _sample(
        self,
        probabilities: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Gumbel-max samples shaped ``(B, samples, horizon)``."""
        log_probs = probabilities.clamp_min(1e-10).log()
        expanded = log_probs.unsqueeze(1).expand(
            -1, self.num_samples, -1, -1
        )
        uniform = torch.rand(
            expanded.shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        ).clamp_min(1e-10)
        gumbel = -(-uniform.log()).log()
        return (expanded + gumbel).argmax(dim=-1)

    def _smooth(self, probabilities: torch.Tensor) -> torch.Tensor:
        if self.smoothing <= 0:
            return probabilities
        probabilities = probabilities + self.smoothing
        return probabilities / probabilities.sum(dim=-1, keepdim=True)

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: Any = None) -> dict:
        del init_action
        if not self._configured:
            raise RuntimeError("Joint CEM must be configured before solve()")
        started = time.time()
        batch_size = len(next(iter(info_dict.values())))
        coord_count = int(self.coordinate_candidates.shape[0])
        action_probs = self._uniform(batch_size, self.n_actions)
        coord_probs = self._uniform(batch_size, coord_count)
        coord_mean = torch.full(
            (batch_size, self.horizon, 2),
            31.5,
            device=self.device,
            dtype=self.dtype,
        )
        coord_std = torch.full_like(coord_mean, 31.5)

        expanded_info: dict[str, Any] = {}
        for key, value in info_dict.items():
            if torch.is_tensor(value):
                batch = value.to(
                    device=self.device,
                    dtype=(
                        self.dtype if value.is_floating_point() else None
                    ),
                )
                expanded_info[key] = batch.unsqueeze(1).expand(
                    batch_size, self.num_samples, *batch.shape[1:]
                )
            else:
                expanded_info[key] = value

        final_costs: Optional[torch.Tensor] = None
        for step in range(self.n_steps):
            # Sample both factors for every horizon position. Coordinates are
            # later masked out for simple actions, so they affect neither their
            # dynamics input nor the coordinate posterior update.
            action_indices = self._sample(
                action_probs, self.action_torch_gen
            )
            if self.coordinate_mode == "categorical":
                coordinate_indices = self._sample(
                    coord_probs, self.coordinate_torch_gen
                )
                sampled_coords = self.coordinate_candidates[
                    coordinate_indices
                ].to(device=self.device, dtype=self.dtype)
            elif step == 0:
                sampled_coords = torch.rand(
                    (
                        batch_size,
                        self.num_samples,
                        self.horizon,
                        2,
                    ),
                    generator=self.coordinate_torch_gen,
                    device=self.device,
                    dtype=self.dtype,
                ) * 63.0
                coordinate_indices = None
            else:
                noise = torch.randn(
                    (
                        batch_size,
                        self.num_samples,
                        self.horizon,
                        2,
                    ),
                    generator=self.coordinate_torch_gen,
                    device=self.device,
                    dtype=self.dtype,
                )
                sampled_coords = (
                    coord_mean.unsqueeze(1)
                    + coord_std.unsqueeze(1) * noise
                ).clamp(0.0, 63.0)
            # Include the current joint mode deterministically. This guarantees
            # that an iteration evaluates its present best hypothesis instead of
            # relying entirely on stochastic samples.
            action_indices[:, 0] = action_probs.argmax(dim=-1)
            if self.coordinate_mode == "categorical":
                assert coordinate_indices is not None
                coordinate_indices[:, 0] = coord_probs.argmax(dim=-1)
                sampled_coords[:, 0] = self.coordinate_candidates[
                    coordinate_indices[:, 0]
                ].to(device=self.device, dtype=self.dtype)
            else:
                sampled_coords[:, 0] = coord_mean

            actions = F.one_hot(
                action_indices, num_classes=self.n_actions
            ).to(self.dtype)
            coords = sampled_coords.round()
            no_coords = torch.full_like(coords, -1)
            # Only ACTION6 owns a coordinate. The -1 sentinel makes the same
            # rollout tensor safe for all actions and maps to zero coord features.
            coords = torch.where(
                (action_indices == self.complex_action_index)
                .unsqueeze(-1),
                coords,
                no_coords,
            )
            costs = self.model.get_cost(
                expanded_info, actions, coords
            )
            if costs.shape != (batch_size, self.num_samples):
                raise ValueError(
                    "Joint CEM objective returned "
                    f"{tuple(costs.shape)}, expected "
                    f"{(batch_size, self.num_samples)}"
                )
            final_costs, elite_indices = torch.topk(
                costs, k=self.topk, dim=1, largest=False
            )
            batch_indices = torch.arange(
                batch_size, device=self.device
            ).unsqueeze(1)
            elite_actions = action_indices[
                batch_indices, elite_indices
            ]
            elite_coords = coords[batch_indices, elite_indices]

            action_frequency = F.one_hot(
                elite_actions, num_classes=self.n_actions
            ).to(self.dtype).mean(dim=1)
            action_probs = self._smooth(action_frequency)

            complex_mask = (
                elite_actions == self.complex_action_index
            ).to(self.dtype)
            complex_count = complex_mask.sum(
                dim=1, keepdim=False
            ).unsqueeze(-1)
            if self.coordinate_mode == "categorical":
                assert coordinate_indices is not None
                elite_coordinate_indices = coordinate_indices[
                    batch_indices, elite_indices
                ]
                coordinate_counts = (
                    F.one_hot(
                        elite_coordinate_indices,
                        num_classes=coord_count,
                    ).to(self.dtype)
                    * complex_mask.unsqueeze(-1)
                ).sum(dim=1)
                conditional_frequency = coordinate_counts / (
                    complex_count.clamp_min(1.0)
                )
                updated_coords = self._smooth(
                    conditional_frequency
                )
                # Refit p(coord | ACTION6) only from elites that actually chose
                # ACTION6. Simple-action elites contain no evidence about clicks.
                coord_probs = torch.where(
                    complex_count > 0, updated_coords, coord_probs
                )
            else:
                weights = complex_mask.unsqueeze(-1)
                new_mean = (elite_coords * weights).sum(dim=1) / (
                    complex_count.clamp_min(1.0)
                )
                squared = (
                    (elite_coords - new_mean.unsqueeze(1)) ** 2
                    * weights
                ).sum(dim=1) / complex_count.clamp_min(1.0)
                new_std = squared.sqrt().clamp_min(1.0)
                has_elites = complex_count > 0
                coord_mean = torch.where(
                    has_elites,
                    0.1 * coord_mean + 0.9 * new_mean,
                    coord_mean,
                )
                coord_std = torch.where(
                    has_elites,
                    (0.1 * coord_std + 0.9 * new_std).clamp_min(1.0),
                    coord_std,
                )

        action_modes = action_probs.argmax(dim=-1)
        if self.coordinate_mode == "categorical":
            coordinate_modes = coord_probs.argmax(dim=-1)
            selected_coords = self.coordinate_candidates[
                coordinate_modes
            ].to(device=self.device)
        else:
            selected_coords = coord_mean.round().clamp(0, 63).long()
        selected_coords = torch.where(
            (action_modes == self.complex_action_index).unsqueeze(-1),
            selected_coords,
            torch.full_like(selected_coords, -1),
        )
        print(
            f"Joint {self.coordinate_mode} CEM solve time: "
            f"{time.time() - started:.4f} seconds"
        )
        return {
            "actions": action_modes.detach().cpu().unsqueeze(-1),
            "coordinates": selected_coords.detach().cpu(),
            "costs": (
                []
                if final_costs is None
                else final_costs.mean(dim=1).detach().cpu().tolist()
            ),
            "action_probs": action_probs.detach().cpu(),
            "coordinate_probs": (
                coord_probs.detach().cpu()
                if self.coordinate_mode == "categorical"
                else None
            ),
            "coordinate_mean": (
                coord_mean.detach().cpu()
                if self.coordinate_mode == "multires"
                else None
            ),
            "coordinate_std": (
                coord_std.detach().cpu()
                if self.coordinate_mode == "multires"
                else None
            ),
        }


# ── Agent orchestration ──────────────────────────────────────────────────────
class MyAgent(Agent):
    """Online world-model controller with non-trainable waypoint supervision.

    A random warm-up builds a bounded memory of observed latent states. The
    waypoint manager then selects one memory as the current target, optionally
    using DNS as a diversity-aware pre-filter. At every subsequent observation,
    CEM imagines short trajectories, the controller executes the first decision,
    and the process repeats from the newly observed state.

    The encoder remains frozen. Only the transition model and monitor-only
    decoder learn online; waypoint selection and CEM are optimisation procedures,
    not learned goal predictors.
    """

    # Upper bound on actions per game; the framework also enforces global limits.
    MAX_ACTIONS = int(os.environ.get("ARC_MAX_ACTIONS", "2000"))

    # (Latent dim is taken from the pretrained encoder; this is only the fallback
    # used when the encoder can't be loaded.)
    FALLBACK_LATENT_DIM = 128
    # CEM planner (stable-worldmodel CategoricalCEMSolver).
    PLAN_HORIZON = 8          # rollout length
    NUM_SAMPLES = 256         # candidate action sequences per CEM iteration
    CEM_STEPS = 5             # CEM refinement iterations
    TOPK = 32                 # elites kept per iteration
    EPSILON = _env_num("ARC_EPSILON", 0.1)
                              # ε-greedy: take a random available action this often
                              # so planning never fully starves exploration
    CEM_STAGNATION_WEIGHT = 0.01
    CEM_STAGNATION_MARGIN = 1e-4
    OBJECT_SCORE_WEIGHT = 0.01
    OBJECT_SCORE_CROP_BORDER = 4
    # Online learning: one SGD step per action on a minibatch drawn from a
    # sliding replay window, so the dynamics keep improving as the agent plays.
    TRAIN_WINDOW = 256        # transitions kept for replay
    TRAIN_BATCH = 128         # minibatch per SGD step
    TRAIN_MIN = 8             # start training once this many transitions exist
    LEARNING_RATE = 3e-3      # online dynamics LR
    # Passive uncertainty calibration. These values never enter CEM cost or
    # action selection unless the explicit ARC_CEM_KNN_LAMBDA experiment is
    # enabled. The default remains exactly zero.
    UNCERTAINTY_K = 8
    UNCERTAINTY_EMA_ALPHA = 0.1
    # Decoder probe (monitor runs only): one SGD step per action on a replay of
    # observed (z, frame) pairs, so the imagined frame sharpens live.
    DECODER_LR = 1e-3
    RECON_BUFFER = 256        # (z, frame) pairs kept for decoder replay
    RECON_BATCH = 16          # pairs per decoder SGD step
    # Minimal waypoint proof of concept. Distances are per-dimension MSE in the
    # already-standardised latent space returned by WorldModel.encode(). For V1,
    # sample the observed trajectory at a fixed interval: a diversity threshold
    # can otherwise reject every state and prevent the warm-up transition.
    WAYPOINT_WARMUP = 32      # random actions used to collect the initial list
    WAYPOINT_MAX = 24         # bounded non-trainable memory
    WAYPOINT_SAMPLE_INTERVAL = 4
    # MSE=0.05 was far too permissive for the standardised encoder: distinct
    # observed states commonly differ by only 0.005–0.02. Keep only genuinely
    # separated memories, and require an almost exact latent revisit to reach.
    WAYPOINT_MIN_SEPARATION = 0.01
    WAYPOINT_REACHED_DISTANCE = 0.001
    WAYPOINT_TIMEOUT = 64     # skip an unreachable waypoint instead of stalling
    WAYPOINT_RECENT_EXCLUSION = 4
    # Diagnostic-only single-link threshold. It is deliberately twice the
    # insertion separation so the cluster count is not forced to equal N.
    WAYPOINT_CLUSTER_DISTANCE = 0.02
    WAYPOINT_POST_REACH_HORIZON = 8

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Seed per game_id so runs of the same game are reproducible but
        # different games explore independently. ARC_SEED makes paired research
        # runs exactly reproducible.
        explicit_seed = os.environ.get("ARC_SEED")
        seed = (
            int(explicit_seed)
            if explicit_seed is not None
            else int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        )
        random.seed(seed)

        # Waypoint-memory policy. These switches are retained for reproducible
        # ablations; the default path uses a dynamic memory cleared on reset.
        self._waypoint_memory_mode = os.environ.get(
            "ARC_WAYPOINT_MEMORY", "dynamic"
        ).lower()
        if self._waypoint_memory_mode not in {"dynamic", "frozen"}:
            raise ValueError(
                "ARC_WAYPOINT_MEMORY must be 'dynamic' or 'frozen', got "
                f"{self._waypoint_memory_mode!r}"
            )
        self._waypoint_reset_mode = os.environ.get(
            "ARC_WAYPOINT_RESET", "clear"
        ).lower()
        if self._waypoint_reset_mode not in {"clear", "persist"}:
            raise ValueError(
                "ARC_WAYPOINT_RESET must be 'clear' or 'persist', got "
                f"{self._waypoint_reset_mode!r}"
            )
        self._waypoint_selection_mode = os.environ.get(
            "ARC_WAYPOINT_SELECTION", "cyclic"
        ).lower()
        if self._waypoint_selection_mode not in {"cyclic", "least_targeted"}:
            raise ValueError(
                "ARC_WAYPOINT_SELECTION must be 'cyclic' or 'least_targeted', got "
                f"{self._waypoint_selection_mode!r}"
            )

        # DNS is an optional first-stage filter. "legacy" sends the complete
        # memory to the final cyclic/least-targeted policy; "dns" first retains
        # the upper half according to dominated novelty. ARC_DNS_K controls how
        # many strictly fitter neighbours define each novelty score.
        self._waypoint_selector_mode = os.environ.get(
            "ARC_WAYPOINT_SELECTOR", "legacy"
        ).lower()
        if self._waypoint_selector_mode not in {"legacy", "dns"}:
            raise ValueError(
                "ARC_WAYPOINT_SELECTOR must be 'legacy' or 'dns', got "
                f"{self._waypoint_selector_mode!r}"
            )
        self._dns_k_neighbors = int(os.environ.get("ARC_DNS_K", "5"))
        if self._dns_k_neighbors < 1:
            raise ValueError(
                "ARC_DNS_K must be at least 1, got "
                f"{self._dns_k_neighbors!r}"
            )
        self._dns_selector = DNSSelector(self._dns_k_neighbors)
        self._waypoint_creation_mode = os.environ.get(
            "ARC_WAYPOINT_CREATION", "global"
        ).lower()
        if self._waypoint_creation_mode not in {"global", "recent"}:
            raise ValueError(
                "ARC_WAYPOINT_CREATION must be 'global' or 'recent', got "
                f"{self._waypoint_creation_mode!r}"
            )
        self._waypoint_recent_size = int(
            os.environ.get("ARC_WAYPOINT_RECENT_N", "50")
        )
        if self._waypoint_recent_size <= 0:
            raise ValueError(
                "ARC_WAYPOINT_RECENT_N must be positive, got "
                f"{self._waypoint_recent_size!r}"
            )
        self._waypoint_recent_threshold = float(
            os.environ.get(
                "ARC_WAYPOINT_RECENT_THRESHOLD",
                str(self.WAYPOINT_MIN_SEPARATION),
            )
        )
        if self._waypoint_recent_threshold < 0:
            raise ValueError(
                "ARC_WAYPOINT_RECENT_THRESHOLD must be non-negative, got "
                f"{self._waypoint_recent_threshold!r}"
            )
        self._frontier_burst_length = int(
            os.environ.get("ARC_FRONTIER_BURST", "0")
        )
        if self._frontier_burst_length not in {0, 8, 16}:
            raise ValueError(
                "ARC_FRONTIER_BURST must be 0, 8, or 16, got "
                f"{self._frontier_burst_length!r}"
            )
        self._object_score_mode = os.environ.get(
            "ARC_OBJECT_SCORE", "off"
        ).lower()
        if self._object_score_mode not in {"off", "simple"}:
            raise ValueError(
                "ARC_OBJECT_SCORE must be 'off' or 'simple', got "
                f"{self._object_score_mode!r}"
            )
        self._cem_diagnostic_enabled = (
            os.environ.get("ARC_CEM_DIAGNOSTIC") == "1"
        )
        # ACTION6 planning modes:
        #   separate       — CEM chooses the action sequence, then a one-step
        #                    search chooses the first ACTION6 coordinate;
        #   joint          — CEM jointly optimises actions and a 64-point click
        #                    lattice across the full planning horizon;
        #   joint_multires — CEM refines continuous click distributions from
        #                    coarse global sampling to local Gaussian sampling.
        self._cem_action6_mode = os.environ.get(
            "ARC_CEM_ACTION6_MODE", "separate"
        ).lower()
        if self._cem_action6_mode not in {
            "separate", "joint", "joint_multires"
        }:
            raise ValueError(
                "ARC_CEM_ACTION6_MODE must be 'separate', 'joint', or "
                "'joint_multires', got "
                f"{self._cem_action6_mode!r}"
            )
        self._uncertainty_diagnostic_enabled = (
            os.environ.get("ARC_UNCERTAINTY_DIAGNOSTIC") == "1"
        )
        self._cem_knn_lambda = float(
            os.environ.get("ARC_CEM_KNN_LAMBDA", "0")
        )
        if self._cem_knn_lambda < 0 or self._cem_knn_lambda > 1:
            raise ValueError(
                "ARC_CEM_KNN_LAMBDA must be in [0, 1], got "
                f"{self._cem_knn_lambda!r}"
            )
        self._planner_transition_diagnostic_enabled = (
            os.environ.get("ARC_PLANNER_TRANSITION_DIAGNOSTIC") == "1"
            or self._uncertainty_diagnostic_enabled
        )
        print(
            f"[waypoint] experiment seed={seed} "
            f"memory={self._waypoint_memory_mode} "
            f"reset={self._waypoint_reset_mode} "
            f"selection={self._waypoint_selection_mode} "
            f"selector={self._waypoint_selector_mode} "
            f"dns_k={self._dns_k_neighbors} "
            f"creation={self._waypoint_creation_mode} "
            f"recent_n={self._waypoint_recent_size} "
            f"recent_threshold={self._waypoint_recent_threshold:.6f} "
            f"frontier_burst={self._frontier_burst_length} "
            f"object_score={self._object_score_mode} "
            f"object_weight="
            f"{self.OBJECT_SCORE_WEIGHT if self._object_score_mode == 'simple' else 0.0} "
            f"cem_goal_cost=terminal "
            f"cem_final_selection=marginal "
            f"cem_action6_mode={self._cem_action6_mode} "
            f"cem_diagnostic={self._cem_diagnostic_enabled} "
            f"planner_transition_diagnostic="
            f"{self._planner_transition_diagnostic_enabled} "
            f"uncertainty_diagnostic="
            f"{self._uncertainty_diagnostic_enabled} "
            f"cem_knn_lambda={self._cem_knn_lambda:.6f} "
            f"latent_adapter=linear "
            f"event_diagnostic="
            f"{os.environ.get('ARC_WAYPOINT_EVENT_DIAGNOSTIC') == '1'}"
        )

        # RESET is handled by the episode lifecycle, not by CEM. Exactly one of
        # the remaining enum values must be complex so its coordinate channel
        # can be represented by a single conditional distribution.
        self._candidate_actions = [a for a in GameAction if a is not GameAction.RESET]
        n_actions = len(self._candidate_actions)
        complex_indices = [
            index
            for index, action in enumerate(self._candidate_actions)
            if action.is_complex()
        ]
        if len(complex_indices) != 1:
            raise ValueError(
                "Joint coordinate planning currently requires exactly one "
                f"complex action, got {complex_indices}"
            )
        self._complex_action_index = complex_indices[0]

        self._device = torch.device(
            os.environ.get("ARC_DEVICE",
                           "cuda" if torch.cuda.is_available() else "cpu"))
        self.world_model = WorldModel(
            n_actions=n_actions, fallback_latent_dim=self.FALLBACK_LATENT_DIM,
        ).to(self._device)
        self._cem_objective = WaypointCEMObjective(
            self.world_model,
            stagnation_weight=self.CEM_STAGNATION_WEIGHT,
            stagnation_margin=self.CEM_STAGNATION_MARGIN,
            object_score_weight=(
                self.OBJECT_SCORE_WEIGHT
                if self._object_score_mode == "simple"
                else 0.0
            ),
            uncertainty_weight=self._cem_knn_lambda,
            uncertainty_k=self.UNCERTAINTY_K,
        )
        self._cem_elite_tracker = CEMEliteTracker()
        self._cem_diagnostic_decisions = 0
        self._cem_diagnostic_action_disagreements = 0
        self._cem_diagnostic_positive_regrets = 0
        self._cem_diagnostic_regret_sum = 0.0
        self._cem_diagnostic_regret_max = 0.0
        # Passive calibration journal for CEM decisions. One prediction made
        # immediately before an executed planner action is paired with the
        # latent encoded from the observation returned by that action.
        self._planner_transition_next_id = 1
        self._planner_transition_pending: Optional[dict[str, Any]] = None
        self._planner_transition_records: list[dict[str, Any]] = []
        self._planner_transition_censored: dict[str, int] = {}
        self._uncertainty_action_ema: list[Optional[float]] = [
            None for _ in range(n_actions)
        ]
        self._uncertainty_action_counts = [0 for _ in range(n_actions)]
        self._cem_elite_cost_sum = 0.0
        self._cem_elite_cost_count = 0
        self._cem_action6_elite_cost_sum = 0.0
        self._cem_action6_elite_cost_count = 0
        self._cem_other_elite_cost_sum = 0.0
        self._cem_other_elite_cost_count = 0
        self._cem_selected_uncertainty_sum = 0.0
        self._cem_selected_uncertainty_count = 0

        # Shared 8×8 coordinate lattice covering the 64×64 board at cell-centre
        # offsets. Separate mode searches these anchors after choosing ACTION6;
        # joint mode samples them inside every CEM rollout position.
        anchors = [4 + 8 * k for k in range(8)]
        self._click_candidates = torch.tensor(
            [(x, y) for x in anchors for y in anchors],
            dtype=torch.long,
            device=self._device,
        )

        # CEM is driven directly without a Gymnasium environment loop;
        # `gym.spaces.Discrete` only describes the action set to the solver.
        # Both solver variants share the same non-trainable waypoint objective.
        if self._cem_action6_mode in {"joint", "joint_multires"}:
            self._solver = JointActionCoordinateCEMSolver(
                model=self._cem_objective,
                coordinate_candidates=self._click_candidates,
                complex_action_index=self._complex_action_index,
                num_samples=self.NUM_SAMPLES,
                n_steps=self.CEM_STEPS,
                topk=self.TOPK,
                device=self._device,
                seed=seed,
                smoothing=0.05,
                coordinate_mode=(
                    "multires"
                    if self._cem_action6_mode == "joint_multires"
                    else "categorical"
                ),
            )
        else:
            self._solver = CategoricalCEMSolver(
                model=self._cem_objective,
                num_samples=self.NUM_SAMPLES,
                n_steps=self.CEM_STEPS,
                topk=self.TOPK,
                device=self._device,
                seed=seed,
                # Laplace smoothing on the elite refit: without it a hair-thin
                # cost edge collapses the distribution on the 1st iteration.
                smoothing=0.05,
                callbacks=(
                    [self._cem_elite_tracker]
                    if self._cem_diagnostic_enabled
                    else None
                ),
            )
        self._solver.configure(
            action_space=gym.spaces.Discrete(n_actions),
            n_envs=1,
            config=PlanConfig(horizon=self.PLAN_HORIZON, receding_horizon=1, action_block=1),
        )

        # Online learning state. Buffer rows: (grid_{t-1}, latent_{t-1},
        # action_onehot_{t-1}, coords_{t-1}, grid_t, latent_t). The dynamics
        # regress latent_t from (latent_{t-1}, action, click coords).
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._buffer: list[tuple[torch.Tensor, ...]] = []
        # (grid, latent, action_onehot, coords) carried to the next step so we
        # can form the transition once the next frame is observed.
        self._prev: Optional[tuple[torch.Tensor, ...]] = None

        # V1 waypoint state: a plain chronological list of observed latents.
        # Tensors are detached snapshots; this memory has no parameters and is
        # never optimised. The active entry is copied into WorldModel via the
        # existing set_goal() API.
        self._waypoints: list[torch.Tensor] = []
        self._waypoint_created_at: list[int] = []
        self._waypoint_creation_distance: list[float] = []
        self._waypoint_target_counts: list[int] = []
        self._recent_waypoint_indices: list[int] = []
        self._waypoint_selections = 0
        self._waypoint_immediate_loops = 0
        self._waypoint_index: Optional[int] = None
        self._waypoint_steps = 0
        self._waypoint_observations = 0
        self._waypoints_reached = 0
        self._waypoints_timed_out = 0
        self._waypoint_distance: Optional[float] = None
        self._waypoint_summary_logged = False
        # Planner-efficiency instrumentation. Unlike the legacy counters above,
        # this journal deliberately survives episode resets so a complete game
        # run can be summarised without reconstructing events from console text.
        self._goal_metric_action_clock = 0
        self._goal_metric_next_id = 1
        self._goal_metric_active: Optional[dict[str, Any]] = None
        self._goal_metric_records: list[dict[str, Any]] = []
        self._goal_metric_reach_clocks: list[int] = []
        self._goal_metric_observations = 0
        self._goal_metric_active_sum = 0
        self._goal_metric_memory_sum = 0
        self._waypoint_debug_latents = os.environ.get("ARC_WAYPOINT_DEBUG_LATENTS") == "1"
        self._waypoint_geometry_enabled = (
            os.environ.get("ARC_WAYPOINT_GEOMETRY") == "1"
        )
        self._waypoint_frontier_diagnostic_enabled = (
            os.environ.get("ARC_WAYPOINT_FRONTIER_DIAGNOSTIC") == "1"
        )
        self._grid_event_diagnostic_enabled = (
            os.environ.get("ARC_WAYPOINT_EVENT_DIAGNOSTIC") == "1"
        )
        self._grid_event_transition_count = 0
        self._grid_event_eventful_count = 0
        self._grid_event_type_counts = {
            name: 0 for _, name in GRID_EVENT_NAMES
        }
        self._grid_event_waypoint_count = 0
        self._grid_event_eventful_waypoint_count = 0
        self._grid_event_examples_logged = 0
        self._episode_initial_latent: Optional[torch.Tensor] = None
        self._waypoint_reached_counts: list[int] = []
        self._waypoint_event_masks: list[int] = []
        # Creation-only rolling reference. It contains the N observations
        # immediately preceding the current one and is reset at episode
        # boundaries. It is intentionally separate from `_visited_latents`,
        # which feeds monitor diagnostics and has a different retention limit.
        self._waypoint_recent_latents: list[torch.Tensor] = []
        self._pending_post_reach: list[dict[str, Any]] = []
        self._completed_post_reach: list[dict[str, Any]] = []
        self._frontier_burst_remaining = 0
        self._frontier_burst_resume_index: Optional[int] = None
        self._active_frontier_burst: Optional[dict[str, Any]] = None
        self._completed_frontier_bursts: list[dict[str, Any]] = []
        self._frontier_bursts_started = 0
        self._visited_latents: list[torch.Tensor] = []
        self._last_current_latent: Optional[torch.Tensor] = None
        self._last_plan_latents: Optional[torch.Tensor] = None
        self._object_score_plans = 0
        self._object_score_reconstruction_sum = 0.0

        # Live arcade monitor, local dev only (`make monitor` sets ARC_MONITOR=1).
        # agent/monitor.py is not spliced into the Kaggle notebook.
        self._monitor = None
        self._last_plan = None    # decoded CEM plan rollout [(rgb, action_name), …]
        self._recon_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._decoder_optimizer: Optional[torch.optim.Optimizer] = None
        if os.environ.get("ARC_MONITOR"):
            try:
                from agent.monitor import ArcadeMonitor

                self._monitor = ArcadeMonitor(
                    [a.name for a in self._candidate_actions] + ["RESET"],
                    game_id=self.game_id,
                )
            except Exception as exc:
                print(f"[monitor] disabled, ArcadeMonitor failed: {exc!r}")
                self._monitor = None

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Stop once we win. Don't stop on GAME_OVER — we want to RESET and retry.
        won = latest_frame.state is GameState.WIN
        if won and not self._waypoint_summary_logged:
            self._cancel_planner_transition("win")
            self._finish_goal_metric("win")
            self._grid_event_summary("win")
            self._log_waypoint_geometry("win")
            self._log_waypoint_frontier("win")
            print(
                f"[waypoint] game_end result=WIN created={len(self._waypoints)} "
                f"reached={self._waypoints_reached} "
                f"timeouts={self._waypoints_timed_out}"
            )
            self._waypoint_summary_logged = True
        return won

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # First call or after a death → reset. No transition spans a reset.
        # In the persistent A/B condition only GAME_OVER keeps the non-trainable
        # waypoint snapshots; the active target is always cleared because no
        # transition or plan may span the environment reset.
        if latest_frame.state is GameState.NOT_PLAYED:
            self._cancel_planner_transition("initial_reset")
            self._prev = None
            self._reset_waypoints()
            return self._monitored(latest_frame, GameAction.RESET)
        if latest_frame.state is GameState.GAME_OVER:
            self._cancel_planner_transition("game_over")
            self._prev = None
            if self._waypoint_reset_mode == "persist":
                self._preserve_waypoints_after_reset()
            else:
                self._reset_waypoints()
            return self._monitored(latest_frame, GameAction.RESET)

        action = self._act(latest_frame)

        if action is None:
            # World model unavailable → random baseline so the agent still
            # returns valid actions and ships.
            action = self._random_action(latest_frame)
            action.reasoning = f"random fallback: {action.value}"
            if action.is_complex():
                # No model to choose a click → random (x, y) on the 64×64 grid.
                action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
        # For a model-chosen action, `_act` has already set the click coordinate
        # (via the coordinate search), so nothing more to do here.
        return self._monitored(latest_frame, action)

    def _random_action(self, latest_frame: FrameData) -> GameAction:
        """A uniformly-random *available* action, with a light per-game bias
        (LS20 favours ACTION4 2×) as an example heuristic."""
        mask = self._available_mask(latest_frame)
        candidates = [a for i, a in enumerate(self._candidate_actions) if mask[0, i] > 0]
        if self.game_id.split("-")[0] == "ls20":
            weights = [2 if a is GameAction.ACTION4 else 1 for a in candidates]
            return random.choices(candidates, weights=weights, k=1)[0]
        return random.choice(candidates)

    def _monitored(self, latest_frame: FrameData, action: GameAction) -> GameAction:
        """Pass-through that reports (frame, latent, action) to the arcade
        monitor when it's on. The latent is consumed once so the PCA cloud only
        gets points for frames the world model actually encoded."""
        if self._monitor is not None:
            rgb = frame_data_to_tensor(latest_frame)[0].permute(1, 2, 0).numpy()
            click = None
            if action.is_complex():
                data = getattr(action, "action_data", None)
                if data is not None and hasattr(data, "x"):
                    click = (int(data.x), int(data.y))
            self._monitor.update(
                rgb, action.name, plan=self._last_plan, click=click,
                waypoint_status=self._waypoint_status(),
                latent_debug=self._latent_debug_payload(),
            )
            self._last_plan = None
            self._last_plan_latents = None
        return action

    @torch.no_grad()
    def _object_action_utilities(
        self,
        latent: torch.Tensor,
        actual_grid: torch.Tensor,
        available: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Score decoded one-step outcomes using the fixed V0 predicates."""
        action_count = len(self._candidate_actions)
        actions = torch.eye(
            action_count, device=latent.device, dtype=latent.dtype
        )
        starts = latent.expand(action_count, -1)
        predicted = self.world_model.predict_next(starts, actions)
        decoded = self.world_model._decode_grid(
            torch.cat([latent, predicted], dim=0)
        )
        reference = decoded[0]
        candidates = decoded[1:]
        predicates = simple_object_transition_predicates(
            reference,
            candidates,
            crop_border=self.OBJECT_SCORE_CROP_BORDER,
        )
        utility = normalise_object_predicates(
            predicates, available.reshape(-1)
        )

        # Decoder quality is not used to gate or tune the score.  Logging its
        # agreement with the real current grid lets the benchmark distinguish
        # a bad objective from an unusable latent-to-grid interface.
        border = self.OBJECT_SCORE_CROP_BORDER
        real = actual_grid.detach().reshape(
            actual_grid.shape[-2], actual_grid.shape[-1]
        ).to(reference)
        if (
            border
            and real.shape[0] > 2 * border
            and real.shape[1] > 2 * border
        ):
            real = real[border:-border, border:-border]
            reconstructed = reference[border:-border, border:-border]
        else:
            reconstructed = reference
        agreement = float((real == reconstructed).float().mean().item())
        diagnostics = {
            "reconstruction_agreement": agreement,
            "predicates": predicates.detach().cpu().tolist(),
            "utilities": utility.detach().cpu().tolist(),
        }
        return utility.unsqueeze(0), diagnostics

    # ── Plan loop ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _uncertainty_replay_context(self) -> Optional[dict[str, Any]]:
        """Snapshot strictly past replay errors once for one CEM decision.

        Residuals are refreshed under the current online dynamics. The P95 is
        a robust, causal scale: the trajectory estimator divides by it and
        clips at one, so the total cost addition is always in ``[0, lambda]``.
        """
        if not self._buffer:
            return None
        latents = torch.cat([row[1] for row in self._buffer], dim=0)
        actions = torch.cat([row[2] for row in self._buffer], dim=0)
        coords = torch.cat([row[3] for row in self._buffer], dim=0)
        actual = torch.cat([row[5] for row in self._buffer], dim=0)
        predicted = self.world_model.predict_next(latents, actions, coords)
        residuals = self.world_model.prediction_error(predicted, actual)
        # Avoid a zero normaliser on an initially exact/repeated replay. This
        # does not create a floor in the estimate itself: zero residual stays 0.
        scale = torch.quantile(residuals.float(), 0.95).clamp_min(1e-8)
        return {
            "size": int(latents.shape[0]),
            "latents": latents.detach(),
            "action_indices": actions.argmax(dim=-1).detach(),
            "coords": coords.detach(),
            "residuals": residuals.detach(),
            "normalization_scale": scale.detach(),
            "complex_action_index": self._complex_action_index,
        }

    @torch.no_grad()
    def _cem_plan_cost(
        self,
        solver_info: dict[str, Any],
        plan: list[int],
        plan_coords: Optional[list[list[int]]] = None,
    ) -> float:
        """Re-evaluate one complete discrete plan with the active CEM cost."""
        candidate = F.one_hot(
            torch.tensor(plan, device=self._device),
            num_classes=len(self._candidate_actions),
        ).to(dtype=solver_info["latent"].dtype)
        candidate = candidate.unsqueeze(0).unsqueeze(0)
        expanded_info = {
            key: (value.unsqueeze(1) if torch.is_tensor(value) else value)
            for key, value in solver_info.items()
        }
        coordinate_candidates = None
        if plan_coords is not None:
            coordinate_candidates = torch.tensor(
                plan_coords,
                device=self._device,
                dtype=solver_info["latent"].dtype,
            ).unsqueeze(0).unsqueeze(0)
        return float(
            self._cem_objective.get_cost(
                expanded_info, candidate, coordinate_candidates
            )[0, 0].item()
        )

    @torch.no_grad()
    def _record_cem_trajectory_metrics(
        self,
        outputs: dict[str, Any],
        solver_info: dict[str, Any],
        plan: list[int],
        plan_coords: Optional[list[list[int]]],
        action_idx: int,
    ) -> None:
        """Record comparable final-elite cost and chosen-plan uncertainty."""
        costs = outputs.get("costs") or []
        if costs:
            elite_cost = float(costs[0])
            self._cem_elite_cost_sum += elite_cost
            self._cem_elite_cost_count += 1
            if action_idx == self._complex_action_index:
                self._cem_action6_elite_cost_sum += elite_cost
                self._cem_action6_elite_cost_count += 1
            else:
                self._cem_other_elite_cost_sum += elite_cost
                self._cem_other_elite_cost_count += 1
        replay = solver_info.get("uncertainty_replay")
        if replay is None:
            return
        candidate = F.one_hot(
            torch.tensor(plan, device=self._device),
            num_classes=len(self._candidate_actions),
        ).to(dtype=solver_info["latent"].dtype).unsqueeze(0).unsqueeze(0)
        coordinates = None
        if plan_coords is not None:
            coordinates = torch.tensor(
                plan_coords,
                device=self._device,
                dtype=solver_info["latent"].dtype,
            ).unsqueeze(0).unsqueeze(0)
        expanded_info = {
            key: (value.unsqueeze(1) if torch.is_tensor(value) else value)
            for key, value in solver_info.items()
        }
        uncertainty = float(
            self._cem_objective.trajectory_uncertainty(
                expanded_info, candidate, coordinates
            )[0, 0].item()
        )
        self._cem_selected_uncertainty_sum += uncertainty
        self._cem_selected_uncertainty_count += 1

    @torch.no_grad()
    def _select_cem_plan(
        self,
        outputs: dict[str, Any],
        solver_info: dict[str, torch.Tensor],
    ) -> list[int]:
        """Return the marginal-mode plan and optionally record passive regret."""
        marginal = [
            int(index)
            for index in outputs["actions"][0, :, 0]
        ]
        best = self._cem_elite_tracker.best_actions
        best_costs = self._cem_elite_tracker.best_costs
        if best is None or best_costs is None:
            return marginal

        elite = [int(index) for index in best[0]]
        elite_cost = float(best_costs[0].item())
        marginal_cost = self._cem_plan_cost(solver_info, marginal)
        regret = max(0.0, marginal_cost - elite_cost)
        first_action_differs = marginal[0] != elite[0]

        self._cem_diagnostic_decisions += 1
        self._cem_diagnostic_action_disagreements += int(first_action_differs)
        self._cem_diagnostic_positive_regrets += int(regret > 1e-9)
        self._cem_diagnostic_regret_sum += regret
        self._cem_diagnostic_regret_max = max(
            self._cem_diagnostic_regret_max, regret
        )
        if self._cem_diagnostic_enabled:
            print(
                "[cem-diagnostic] "
                f"decision={self._cem_diagnostic_decisions} "
                "selection=marginal "
                f"marginal_cost={marginal_cost:.8f} "
                f"elite_cost={elite_cost:.8f} "
                f"regret={regret:.8f} "
                f"first_action_differs={first_action_differs} "
                f"marginal_plan={marginal} elite_plan={elite}"
            )
        return marginal

    def cem_diagnostic_summary(self) -> dict[str, Any]:
        decisions = self._cem_diagnostic_decisions
        return {
            "final_selection": "marginal",
            "decisions": decisions,
            "first_action_disagreements": (
                self._cem_diagnostic_action_disagreements
            ),
            "first_action_disagreement_rate": (
                self._cem_diagnostic_action_disagreements / decisions
                if decisions
                else 0.0
            ),
            "positive_regrets": self._cem_diagnostic_positive_regrets,
            "positive_regret_rate": (
                self._cem_diagnostic_positive_regrets / decisions
                if decisions
                else 0.0
            ),
            "mean_regret": (
                self._cem_diagnostic_regret_sum / decisions
                if decisions
                else 0.0
            ),
            "max_regret": self._cem_diagnostic_regret_max,
        }

    @torch.no_grad()
    def _passive_uncertainty_estimates(
        self,
        latent: torch.Tensor,
        action_idx: int,
        coords: torch.Tensor,
    ) -> dict[str, Any]:
        """Estimate reliability from strictly past real transitions.

        Historical residuals are recomputed with the *current* dynamics, so
        online SGD does not leave stale error labels. Neighbours must share the
        executed action. For ACTION6, normalised coordinate MSE is added to the
        standardised-latent MSE used as the support metric.
        """
        result: dict[str, Any] = {
            "uncertainty_knn_error": None,
            "uncertainty_action_ema": self._uncertainty_action_ema[
                action_idx
            ],
            "uncertainty_support_distance": None,
            "uncertainty_replay_matches": 0,
            "uncertainty_knn_k": 0,
            "uncertainty_action_history": self._uncertainty_action_counts[
                action_idx
            ],
        }
        matching = [
            row
            for row in self._buffer
            if int(row[2].argmax(dim=-1).item()) == action_idx
        ]
        if not matching:
            return result

        starts = torch.cat([row[1] for row in matching], dim=0)
        actions = torch.cat([row[2] for row in matching], dim=0)
        history_coords = torch.cat([row[3] for row in matching], dim=0)
        actual = torch.cat([row[5] for row in matching], dim=0)
        predicted = self.world_model.predict_next(
            starts, actions, history_coords
        )
        residuals = self.world_model.prediction_error(predicted, actual)
        query = latent.detach().reshape(1, -1).to(starts)
        state_distance = ((starts - query) ** 2).mean(dim=-1)
        if action_idx == self._complex_action_index:
            query_coords = coords.detach().reshape(1, 2).to(history_coords)
            coordinate_distance = (
                ((history_coords - query_coords) / 63.0) ** 2
            ).mean(dim=-1)
        else:
            coordinate_distance = torch.zeros_like(state_distance)
        support_distance = state_distance + coordinate_distance
        neighbour_count = min(self.UNCERTAINTY_K, len(matching))
        neighbour_indices = torch.topk(
            support_distance,
            k=neighbour_count,
            largest=False,
        ).indices
        result.update(
            {
                "uncertainty_knn_error": float(
                    residuals[neighbour_indices].mean().item()
                ),
                "uncertainty_support_distance": float(
                    support_distance[neighbour_indices[0]].item()
                ),
                "uncertainty_replay_matches": len(matching),
                "uncertainty_knn_k": neighbour_count,
            }
        )
        return result

    @staticmethod
    def _categorical_entropy(probabilities: torch.Tensor) -> float:
        probabilities = probabilities.detach().float().clamp_min(1e-12)
        return float((-(probabilities * probabilities.log()).sum()).item())

    def _cem_entropy_diagnostic(self, outputs: dict) -> dict[str, Any]:
        """Return first-step entropy of the unchanged joint CEM posterior."""
        action_probs = outputs.get("action_probs")
        coordinate_probs = outputs.get("coordinate_probs")
        if not torch.is_tensor(action_probs):
            return {
                "cem_action_entropy": None,
                "cem_coordinate_entropy": None,
                "cem_joint_entropy": None,
            }
        action_first = action_probs[0, 0]
        action_entropy = self._categorical_entropy(action_first)
        coordinate_entropy = (
            self._categorical_entropy(coordinate_probs[0, 0])
            if torch.is_tensor(coordinate_probs)
            else None
        )
        joint_entropy = action_entropy
        if coordinate_entropy is not None:
            joint_entropy += (
                float(action_first[self._complex_action_index].item())
                * coordinate_entropy
            )
        return {
            "cem_action_entropy": action_entropy,
            "cem_coordinate_entropy": coordinate_entropy,
            "cem_joint_entropy": joint_entropy,
        }

    @torch.no_grad()
    def _start_planner_transition_diagnostic(
        self,
        latent: torch.Tensor,
        action_idx: int,
        coords: torch.Tensor,
        plan: list[int],
        available: torch.Tensor,
        plan_coords: Optional[list[list[int]]] = None,
        uncertainty: Optional[dict[str, Any]] = None,
    ) -> None:
        """Snapshot one CEM prediction before its first action is executed.

        This method is diagnostic-only. It neither changes the selected action
        nor feeds any value back into CEM or online dynamics training.
        """
        if not self._planner_transition_diagnostic_enabled:
            return
        goal = self.world_model.goal_latent
        if goal is None:
            return
        if self._planner_transition_pending is not None:
            self._cancel_planner_transition("replaced_pending")

        action_count = len(self._candidate_actions)
        action_eye = torch.eye(
            action_count, device=latent.device, dtype=latent.dtype
        )
        starts = latent.expand(action_count, -1)
        one_step_latents = self.world_model.predict_next(starts, action_eye)
        goal_batch = goal.to(one_step_latents).reshape(1, -1)
        one_step_distances_tensor = (
            (one_step_latents - goal_batch) ** 2
        ).mean(dim=-1)
        action_onehot = self._one_hot(action_idx)
        predicted_executed = self.world_model.predict_next(
            latent, action_onehot, coords
        )
        current_distance = self._latent_distance(latent, goal)
        predicted_distance = self._latent_distance(
            predicted_executed, goal
        )
        # For joint ACTION6 planning, rank the chosen action using the exact
        # coordinate that will be executed rather than the legacy centre click.
        one_step_distances_tensor[action_idx] = predicted_distance
        available_mask = available.reshape(-1).bool().to(
            one_step_distances_tensor.device
        )
        ranked = sorted(
            (
                (float(one_step_distances_tensor[index].item()), index)
                for index in range(action_count)
                if bool(available_mask[index].item())
            ),
            key=lambda item: (item[0], item[1]),
        )
        chosen_rank = next(
            (
                rank
                for rank, (_, index) in enumerate(ranked, start=1)
                if index == action_idx
            ),
            None,
        )
        one_step_best = ranked[0][1] if ranked else None

        imagined = latent
        for step, index in enumerate(plan):
            imagined_coords = (
                None
                if plan_coords is None
                else torch.tensor(
                    [plan_coords[step]],
                    device=imagined.device,
                    dtype=imagined.dtype,
                )
            )
            imagined = self.world_model.predict_next(
                imagined, self._one_hot(index), imagined_coords
            )
        plan_final_distance = self._latent_distance(imagined, goal)

        one_step_distances = [
            (
                float(one_step_distances_tensor[index].item())
                if bool(available_mask[index].item())
                else None
            )
            for index in range(action_count)
        ]
        goal_metric = self._goal_metric_active or {}
        pending = {
            "transition_id": self._planner_transition_next_id,
            "goal_id": goal_metric.get("goal_id"),
            "waypoint_index": goal_metric.get("waypoint_index"),
            "action_clock": self._goal_metric_action_clock,
            "action_index": action_idx,
            "action_name": self._candidate_actions[action_idx].name,
            "coords": [
                int(value)
                for value in coords.detach().reshape(-1).cpu().tolist()
            ],
            "plan": list(plan),
            "plan_coords": plan_coords,
            "current_goal_distance": current_distance,
            "predicted_goal_distance": predicted_distance,
            "predicted_progress": current_distance - predicted_distance,
            "cem_default_coord_goal_distance": float(
                one_step_distances_tensor[action_idx].item()
            ),
            "one_step_best_action_index": one_step_best,
            "one_step_best_action_name": (
                None
                if one_step_best is None
                else self._candidate_actions[one_step_best].name
            ),
            "cem_action_one_step_rank": chosen_rank,
            "one_step_goal_distances": one_step_distances,
            "plan_final_goal_distance": plan_final_distance,
            **(uncertainty or {}),
            "_goal": goal.detach().reshape(-1).cpu(),
            "_predicted_latent": predicted_executed.detach().reshape(-1).cpu(),
        }
        self._planner_transition_next_id += 1
        self._planner_transition_pending = pending
        printable = {
            key: value
            for key, value in pending.items()
            if not key.startswith("_")
        }
        print(
            "[planner-transition-prediction] "
            f"{json.dumps(printable, separators=(',', ':'))}"
        )

    def _complete_planner_transition_diagnostic(
        self,
        actual_latent: torch.Tensor,
    ) -> None:
        """Pair the pending pre-action prediction with the returned real state."""
        pending = self._planner_transition_pending
        if pending is None:
            return
        goal = pending.pop("_goal")
        predicted_latent = pending.pop("_predicted_latent")
        actual = actual_latent.detach().reshape(-1).float().cpu()
        goal = goal.float()
        predicted_latent = predicted_latent.float()
        actual_distance = self._latent_distance(actual, goal)
        actual_progress = (
            float(pending["current_goal_distance"]) - actual_distance
        )
        predicted_progress = float(pending["predicted_progress"])
        predicted_improves = predicted_progress > 0.0
        actual_improves = actual_progress > 0.0
        record = {
            **pending,
            "actual_goal_distance": actual_distance,
            "actual_progress": actual_progress,
            "progress_error": predicted_progress - actual_progress,
            "absolute_progress_error": abs(
                predicted_progress - actual_progress
            ),
            "latent_prediction_mse": self._latent_distance(
                predicted_latent, actual
            ),
            "predicted_improves": predicted_improves,
            "actual_improves": actual_improves,
            "direction_agreement": predicted_improves == actual_improves,
            "false_progress": predicted_improves and not actual_improves,
            "missed_progress": actual_improves and not predicted_improves,
        }
        self._planner_transition_records.append(record)
        if getattr(self, "_uncertainty_diagnostic_enabled", False):
            action_index = int(record["action_index"])
            residual = float(record["latent_prediction_mse"])
            previous = self._uncertainty_action_ema[action_index]
            self._uncertainty_action_ema[action_index] = (
                residual
                if previous is None
                else (
                    (1.0 - self.UNCERTAINTY_EMA_ALPHA) * previous
                    + self.UNCERTAINTY_EMA_ALPHA * residual
                )
            )
            self._uncertainty_action_counts[action_index] += 1
        self._planner_transition_pending = None
        print(
            "[planner-transition-result] "
            f"{json.dumps(record, separators=(',', ':'))}"
        )

    def _cancel_planner_transition(self, reason: str) -> None:
        """Censor a prediction whose post-action frame cannot be encoded."""
        if self._planner_transition_pending is None:
            return
        self._planner_transition_censored[reason] = (
            self._planner_transition_censored.get(reason, 0) + 1
        )
        printable = {
            key: value
            for key, value in self._planner_transition_pending.items()
            if not key.startswith("_")
        }
        printable["censored_reason"] = reason
        print(
            "[planner-transition-censored] "
            f"{json.dumps(printable, separators=(',', ':'))}"
        )
        self._planner_transition_pending = None

    @staticmethod
    def _diagnostic_pearson(
        left: list[float],
        right: list[float],
    ) -> Optional[float]:
        if len(left) < 2 or len(left) != len(right):
            return None
        left_mean = sum(left) / len(left)
        right_mean = sum(right) / len(right)
        numerator = sum(
            (x - left_mean) * (y - right_mean)
            for x, y in zip(left, right)
        )
        left_scale = math.sqrt(
            sum((x - left_mean) ** 2 for x in left)
        )
        right_scale = math.sqrt(
            sum((y - right_mean) ** 2 for y in right)
        )
        denominator = left_scale * right_scale
        return numerator / denominator if denominator > 0 else None

    def planner_transition_diagnostic_summary(
        self,
        *,
        finalize_pending: bool = False,
        final_reason: str = "action_budget_end",
    ) -> dict[str, Any]:
        """Aggregate prediction-vs-reality calibration without changing policy."""
        if finalize_pending:
            self._cancel_planner_transition(final_reason)
        records = self._planner_transition_records
        predicted = [
            float(record["predicted_progress"]) for record in records
        ]
        actual = [
            float(record["actual_progress"]) for record in records
        ]
        count = len(records)

        def rate(key: str) -> float:
            return (
                sum(bool(record[key]) for record in records) / count
                if count else 0.0
            )

        ranks = [
            int(record["cem_action_one_step_rank"])
            for record in records
            if record["cem_action_one_step_rank"] is not None
        ]
        latent_errors = [
            float(record["latent_prediction_mse"]) for record in records
        ]
        absolute_errors = [
            float(record["absolute_progress_error"]) for record in records
        ]
        return {
            "evaluated_transitions": count,
            "censored_transitions": sum(
                self._planner_transition_censored.values()
            ),
            "censored_reasons": dict(self._planner_transition_censored),
            "predicted_progress_rate": rate("predicted_improves"),
            "actual_progress_rate": rate("actual_improves"),
            "direction_agreement_rate": rate("direction_agreement"),
            "false_progress_rate": rate("false_progress"),
            "missed_progress_rate": rate("missed_progress"),
            "mean_predicted_progress": (
                sum(predicted) / count if count else None
            ),
            "mean_actual_progress": (
                sum(actual) / count if count else None
            ),
            "mean_progress_bias": (
                sum(
                    predicted_value - actual_value
                    for predicted_value, actual_value in zip(
                        predicted, actual
                    )
                ) / count
                if count else None
            ),
            "mean_absolute_progress_error": (
                sum(absolute_errors) / count if count else None
            ),
            "predicted_actual_progress_correlation": (
                self._diagnostic_pearson(predicted, actual)
            ),
            "mean_latent_prediction_mse": (
                sum(latent_errors) / count if count else None
            ),
            "cem_matches_one_step_best_rate": (
                sum(rank == 1 for rank in ranks) / len(ranks)
                if ranks else None
            ),
            "mean_cem_one_step_rank": (
                sum(ranks) / len(ranks) if ranks else None
            ),
            "records": list(records),
        }

    def _act(self, latest_frame: FrameData) -> Optional[GameAction]:
        """Encode the frame, learn the dynamics from the previous transition
        (online), then — if a goal is set — plan with the goal-reaching CEM and
        execute its first action (ε-greedy random otherwise). With no goal set,
        explore with a random available action while the dynamics keep learning.

        Returns ``None`` if the world model isn't available, so `choose_action`
        falls back to the random baseline.
        """
        try:
            # Phase 1 — Observe: encode the real frame returned by the engine.
            # Planning never advances from a cached prediction between actions.
            frame = frame_data_to_tensor(latest_frame).to(self._device)
            grid = frame_data_to_grid(latest_frame).to(self._device)  # (1, 64, 64)
            latent = self.world_model.encode(frame)  # (1, D)
            # Complete the prediction made immediately before the action that
            # produced this exact observation. This happens before any online
            # dynamics update or waypoint/goal transition.
            self._complete_planner_transition_diagnostic(latent.detach())
            self._last_current_latent = latent.detach().reshape(-1).cpu()
            self._visited_latents.append(self._last_current_latent.clone())
            if len(self._visited_latents) > self.TRAIN_WINDOW:
                self._visited_latents.pop(0)
            if self._monitor is not None:
                self._train_decoder(latent.detach(), grid)

            # Phase 2 — Learn: pair the action taken at t-1 with the newly
            # observed state at t, then update the dynamics from replay.
            event_mask = 0
            if self._prev is not None:
                grid_prev, latent_prev, action_prev, coords_prev = self._prev
                self._buffer.append((grid_prev, latent_prev.detach(),
                                     action_prev.detach(), coords_prev,
                                     grid, latent.detach()))
                if len(self._buffer) > self.TRAIN_WINDOW:
                    self._buffer.pop(0)
                if len(self._buffer) >= self.TRAIN_MIN:
                    self._train_step()
                if self._grid_event_diagnostic_enabled:
                    event_mask = self._record_grid_event_transition(
                        grid_prev, grid
                    )

            # Phase 3 — Supervise: update the external waypoint manager. During
            # warm-up it only records diverse observations; afterwards it sets
            # one remembered latent at a time as the existing CEM goal.
            # `latent` was encoded from *latest_frame* above. When `_prev` is
            # present, latest_frame is the observation returned after executing
            # the previous action; it is never a predicted or cached waypoint.
            self._update_waypoints(
                latent.detach(), grid.detach(),
                after_action=self._prev is not None,
                event_mask=event_mask,
            )
            self._goal_metric_observations += 1
            self._goal_metric_active_sum += int(
                self._goal_metric_active is not None
            )
            self._goal_metric_memory_sum += len(self._waypoints)

            mask = self._available_mask(latest_frame)
            avail_indices = [i for i in range(len(self._candidate_actions)) if mask[0, i] > 0]
            outputs: Optional[dict] = None

            if self.world_model.goal_latent is None:
                # Warm-up (or too few distinct states) → random exploration.
                action_idx = random.choice(avail_indices)
                why = (
                    "random (frontier burst)"
                    if self._frontier_burst_remaining > 0
                    else "random (collecting waypoints)"
                )
                explored = True
                plan = None
                plan_coords = None
                object_utility = None
            else:
                # Phase 4 — Plan: optimise a short imagined trajectory, execute
                # only its first decision, then replan after the next real frame.
                # Epsilon-greedy exploration prevents complete policy collapse
                # and is a safety net for an unavailable first planned action.
                solver_info = {
                    "latent": latent.detach(),
                    "available": mask.to(self._device),
                }
                if self._cem_knn_lambda:
                    replay_context = self._uncertainty_replay_context()
                    if replay_context is not None:
                        solver_info["uncertainty_replay"] = replay_context
                object_utility = None
                if self._object_score_mode == "simple":
                    object_utility, diagnostics = self._object_action_utilities(
                        latent.detach(), grid, mask.to(self._device)
                    )
                    solver_info["object_utility"] = object_utility
                    self._object_score_plans += 1
                    self._object_score_reconstruction_sum += (
                        diagnostics["reconstruction_agreement"]
                    )
                    if (
                        self._object_score_plans == 1
                        or self._object_score_plans % 25 == 0
                    ):
                        print(
                            "[object-score] "
                            f"plans={self._object_score_plans} "
                            f"weight={self.OBJECT_SCORE_WEIGHT:.6f} "
                            f"reconstruction_agreement="
                            f"{diagnostics['reconstruction_agreement']:.4f} "
                            f"mean_reconstruction_agreement="
                            f"{self._object_score_reconstruction_sum / self._object_score_plans:.4f} "
                            f"utilities="
                            f"{[round(value, 4) for value in diagnostics['utilities']]} "
                            f"predicates={diagnostics['predicates']}"
                        )
                outputs = self._solver.solve(solver_info)
                plan = self._select_cem_plan(outputs, solver_info)
                plan_coords = (
                    outputs.get("coordinates")
                    if self._cem_action6_mode in {"joint", "joint_multires"}
                    else None
                )
                if torch.is_tensor(plan_coords):
                    plan_coords = plan_coords[0].tolist()
                action_idx = plan[0]  # receding horizon: execute only the first step
                explored = random.random() < self.EPSILON or action_idx not in avail_indices
                if explored:
                    action_idx = random.choice(avail_indices)
                why = "ε-greedy" if explored else "waypoint CEM"

            # Phase 5 — Bind ACTION6 coordinates. Joint modes execute the first
            # action-coordinate pair evaluated by CEM. Separate mode performs a
            # one-step coordinate search only after CEM has selected ACTION6.
            # Exploratory clicks remain uniformly random over the full grid.
            action = self._candidate_actions[action_idx]
            if action.is_complex():
                if explored or self.world_model.goal_latent is None:
                    x, y = random.randint(0, 63), random.randint(0, 63)
                elif (
                    self._cem_action6_mode in {"joint", "joint_multires"}
                    and plan_coords is not None
                ):
                    x, y = (
                        int(plan_coords[0][0]),
                        int(plan_coords[0][1]),
                    )
                else:
                    x, y = self._choose_click(latent.detach(), action_idx)
                action.set_data({"x": x, "y": y})
                coords_prev = torch.tensor([[x, y]], device=self._device)
            else:
                coords_prev = torch.full((1, 2), -1, device=self._device)

            action_onehot = self._one_hot(action_idx)
            if plan is not None and not explored:
                assert outputs is not None
                self._record_cem_trajectory_metrics(
                    outputs,
                    solver_info,
                    plan,
                    plan_coords,
                    action_idx,
                )
                uncertainty = None
                if self._uncertainty_diagnostic_enabled:
                    uncertainty = self._passive_uncertainty_estimates(
                        latent.detach(), action_idx, coords_prev
                    )
                    uncertainty.update(
                        self._cem_entropy_diagnostic(outputs)
                    )
                self._start_planner_transition_diagnostic(
                    latent.detach(),
                    action_idx,
                    coords_prev,
                    plan,
                    mask.to(self._device),
                    plan_coords=plan_coords,
                    uncertainty=uncertainty,
                )

            # Monitor: decode the CEM plan's imagined rollout (t+1 … t+H) — what
            # the world model expects each planned step to look like.
            if self._monitor is not None and plan is not None:
                self._last_plan = self._imagine_plan(
                    latent, plan, plan_coords=plan_coords
                )

            # Phase 6 — Commit: retain the exact executed action and coordinate.
            # The transition becomes complete only when the engine returns the
            # next real observation on the following call.
            self._prev = (grid, latent.detach(), action_onehot, coords_prev)
        except NotImplementedError:
            return None

        action.reasoning = {
            "why": why,
            "action_idx": action_idx,
            "cem_action6_mode": self._cem_action6_mode,
            "planned_coords": (
                None
                if plan_coords is None
                else plan_coords[0]
            ),
            "waypoint": self._waypoint_status(),
            "object_utility": (
                None
                if object_utility is None
                else float(object_utility[0, action_idx].item())
            ),
        }
        return action

    # ── Waypoint memory and target supervision ──────────────────────────────
    @staticmethod
    def _latent_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        """Per-dimension squared distance, matching WorldModel.get_cost()."""
        return float(((a.reshape(-1) - b.reshape(-1)) ** 2).mean().item())

    def _grid_event_summary(self, event: str) -> None:
        """Log cumulative detector-only counts without changing behaviour."""
        if not self._grid_event_diagnostic_enabled:
            return
        transition_fraction = (
            self._grid_event_eventful_count / self._grid_event_transition_count
            if self._grid_event_transition_count else 0.0
        )
        waypoint_fraction = (
            self._grid_event_eventful_waypoint_count
            / self._grid_event_waypoint_count
            if self._grid_event_waypoint_count else 0.0
        )
        payload = {
            "event": event,
            "transitions": self._grid_event_transition_count,
            "eventful_transitions": self._grid_event_eventful_count,
            "eventful_transition_fraction": round(
                transition_fraction, 6
            ),
            **self._grid_event_type_counts,
            "waypoints": self._grid_event_waypoint_count,
            "eventful_waypoints": self._grid_event_eventful_waypoint_count,
            "eventful_waypoint_fraction": round(waypoint_fraction, 6),
            "examples_logged": self._grid_event_examples_logged,
        }
        print(
            "[grid-event-summary] "
            f"{json.dumps(payload, separators=(',', ':'))}"
        )

    def _record_grid_event_transition(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
    ) -> int:
        """Run and log the passive V0 detector on one real transition."""
        mask, details = detect_simple_grid_events(previous, current)
        self._grid_event_transition_count += 1
        names = grid_event_names(mask)
        if mask:
            self._grid_event_eventful_count += 1
        for bit, name in GRID_EVENT_NAMES:
            if mask & bit:
                self._grid_event_type_counts[name] += 1
        transition_payload = {
            "transition": self._grid_event_transition_count,
            "eventful": bool(mask),
            "mask": mask,
            "events": names,
            **details,
        }
        print(
            "[grid-event] "
            f"{json.dumps(transition_payload, separators=(',', ':'))}"
        )
        if mask and self._grid_event_examples_logged < 8:
            self._grid_event_examples_logged += 1
            before = previous.detach().reshape(
                previous.shape[-2], previous.shape[-1]
            ).long().cpu()
            after = current.detach().reshape(
                current.shape[-2], current.shape[-1]
            ).long().cpu()
            example_payload = {
                "example": self._grid_event_examples_logged,
                **transition_payload,
                "before": before.tolist(),
                "after": after.tolist(),
            }
            print(
                "[grid-event-example] "
                f"{json.dumps(example_payload, separators=(',', ':'))}"
            )
        if self._grid_event_transition_count % 25 == 0:
            self._grid_event_summary("periodic")
        return mask

    def _remember_waypoint(
        self,
        latent: torch.Tensor,
        event_mask: int = 0,
    ) -> None:
        """Sample the chronological trajectory into the bounded waypoint list."""
        if len(self._waypoints) >= self.WAYPOINT_MAX:
            return
        if (
            self._waypoint_memory_mode == "frozen"
            and self._waypoint_observations > self.WAYPOINT_WARMUP
        ):
            return
        if (self._waypoint_observations != 1
                and self._waypoint_observations % self.WAYPOINT_SAMPLE_INTERVAL != 0):
            return
        candidate = latent.detach().reshape(-1).clone()
        if self._waypoint_creation_mode == "recent":
            references = self._waypoint_recent_latents[
                -self._waypoint_recent_size:
            ]
            reference_name = "recent_states"
            minimum = self._waypoint_recent_threshold
        else:
            references = self._waypoints
            reference_name = "global_waypoints"
            minimum = self.WAYPOINT_MIN_SEPARATION
        nearest = (
            min(self._latent_distance(candidate, z) for z in references)
            if references else float("inf")
        )
        previous = (
            self._latent_distance(candidate, self._waypoints[-1])
            if self._waypoints else float("inf")
        )
        if nearest < minimum:
            print(
                f"[waypoint] rejected observation={self._waypoint_observations} "
                f"reason=near_reference reference={reference_name} "
                f"nearest_distance={nearest:.6f} minimum={minimum:.6f} "
                f"reference_count={len(references)} "
                f"candidate={self._latent_fingerprint(candidate)}"
            )
            return
        self._waypoints.append(candidate)
        self._waypoint_created_at.append(self._waypoint_observations)
        self._waypoint_creation_distance.append(nearest)
        self._waypoint_target_counts.append(0)
        self._waypoint_reached_counts.append(0)
        self._waypoint_event_masks.append(event_mask)
        if self._grid_event_diagnostic_enabled:
            self._grid_event_waypoint_count += 1
            if event_mask:
                self._grid_event_eventful_waypoint_count += 1
        print(
            f"[waypoint] created index={len(self._waypoints)}/{self.WAYPOINT_MAX} "
            f"observation={self._waypoint_observations} "
            f"reference={reference_name} reference_count={len(references)} "
            f"reference_distance={nearest:.6f} minimum={minimum:.6f} "
            f"previous_waypoint_distance={previous:.6f} "
            f"grid_events={grid_event_names(event_mask)} "
            f"latent={self._latent_fingerprint(candidate)}"
        )
        self._print_full_latent("created", candidate)
        self._log_waypoint_geometry("created")
        self._log_waypoint_frontier("created")

    def _activate_waypoint(self, index: int) -> None:
        """Select one list entry and expose it through the existing WM API."""
        if not self._waypoints:
            self._finish_goal_metric("replaced_no_waypoint")
            self._waypoint_index = None
            self.world_model.clear_goal()
            return
        if self._goal_metric_active is not None:
            self._finish_goal_metric("replaced")
        self._waypoint_index = index % len(self._waypoints)
        immediate_loop = self._waypoint_index in self._recent_waypoint_indices
        if immediate_loop:
            self._waypoint_immediate_loops += 1
        self._waypoint_target_counts[self._waypoint_index] += 1
        self._waypoint_selections += 1
        self._recent_waypoint_indices.append(self._waypoint_index)
        self._recent_waypoint_indices = self._recent_waypoint_indices[
            -self.WAYPOINT_RECENT_EXCLUSION:
        ]
        self._waypoint_steps = 0
        waypoint = self._waypoints[self._waypoint_index]
        self.world_model.set_goal(waypoint)
        self._goal_metric_active = {
            "goal_id": self._goal_metric_next_id,
            "waypoint_index": self._waypoint_index + 1,
            "waypoint_created_observation": self._waypoint_created_at[
                self._waypoint_index
            ],
            "selected_action_clock": self._goal_metric_action_clock,
            "selected_observation": self._waypoint_observations,
        }
        self._goal_metric_next_id += 1
        # Do not dump the full high-dimensional vector: shape, norm, and the
        # identity check below are enough to prove set_goal received this entry.
        goal_matches = (
            self.world_model.goal_latent is not None
            and torch.equal(self.world_model.goal_latent, waypoint)
        )
        print(
            f"[waypoint] set_goal index={self._waypoint_index + 1}/{len(self._waypoints)} "
            f"captured_observation={self._waypoint_created_at[self._waypoint_index]} "
            f"shape={tuple(waypoint.shape)} latent_norm={waypoint.norm().item():.4f} "
            f"stored_goal_matches={goal_matches} "
            f"target_count={self._waypoint_target_counts[self._waypoint_index]} "
            f"selection_number={self._waypoint_selections} "
            f"immediate_loop={immediate_loop} "
            f"immediate_loops={self._waypoint_immediate_loops} "
            f"memory_coverage={self._waypoint_selection_coverage():.4f} "
            f"selection_counts={self._waypoint_target_counts}"
        )
        self._print_full_latent("set_goal", waypoint)

    def _finish_goal_metric(self, outcome: str) -> None:
        """Close the active goal attempt without affecting planner behaviour."""
        active = self._goal_metric_active
        if active is None:
            return
        actions = max(
            0,
            self._goal_metric_action_clock
            - int(active["selected_action_clock"]),
        )
        record = {
            **active,
            "outcome": outcome,
            "reached": outcome == "reached",
            "actions": actions,
            "finished_action_clock": self._goal_metric_action_clock,
            "finished_observation": self._waypoint_observations,
        }
        self._goal_metric_records.append(record)
        if record["reached"]:
            self._goal_metric_reach_clocks.append(
                self._goal_metric_action_clock
            )
        self._goal_metric_active = None
        print(
            "[planner-goal-result] "
            f"{json.dumps(record, separators=(',', ':'))}"
        )

    @staticmethod
    def _metric_percentile(values: list[int], percentile: float) -> Optional[float]:
        """Linearly interpolated percentile, matching NumPy's default rule."""
        if not values:
            return None
        ordered = sorted(float(value) for value in values)
        position = (len(ordered) - 1) * percentile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return (
            ordered[lower] * (1.0 - fraction)
            + ordered[upper] * fraction
        )

    def planner_efficiency_summary(
        self,
        *,
        finalize_active: bool = False,
        final_reason: str = "benchmark_end",
    ) -> dict[str, Any]:
        """Return run-global goal timing metrics and optionally close censoring."""
        if finalize_active:
            self._finish_goal_metric(final_reason)
        reached_records = [
            record for record in self._goal_metric_records
            if record["reached"]
        ]
        reached_actions = [
            int(record["actions"]) for record in reached_records
        ]
        intervals = [
            current - previous
            for previous, current in zip(
                self._goal_metric_reach_clocks,
                self._goal_metric_reach_clocks[1:],
            )
        ]
        total = len(self._goal_metric_records)
        reached = len(reached_records)
        abandoned = total - reached
        outcomes: dict[str, int] = {}
        for record in self._goal_metric_records:
            outcome = str(record["outcome"])
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        observations = self._goal_metric_observations
        return {
            "total_goals_selected": total,
            "goals_reached": reached,
            "goals_abandoned": abandoned,
            "never_reached_rate": abandoned / total if total else 0.0,
            "outcomes": outcomes,
            "actions_to_reach": reached_actions,
            "actions_to_reach_stats": {
                "mean": (
                    sum(reached_actions) / len(reached_actions)
                    if reached_actions else None
                ),
                "median": self._metric_percentile(reached_actions, 0.5),
                "minimum": min(reached_actions) if reached_actions else None,
                "maximum": max(reached_actions) if reached_actions else None,
                "p95": self._metric_percentile(reached_actions, 0.95),
            },
            "mean_actions_between_reached_goals": (
                sum(intervals) / len(intervals) if intervals else None
            ),
            "actions_between_reached_goals": intervals,
            "mean_active_goals": (
                self._goal_metric_active_sum / observations
                if observations else 0.0
            ),
            "mean_waypoint_memory_size": (
                self._goal_metric_memory_sum / observations
                if observations else 0.0
            ),
            "cem_knn_lambda": getattr(self, "_cem_knn_lambda", 0.0),
            "cem_elite_cost_count": getattr(
                self, "_cem_elite_cost_count", 0
            ),
            "mean_cem_elite_trajectory_cost": (
                self._cem_elite_cost_sum / self._cem_elite_cost_count
                if getattr(self, "_cem_elite_cost_count", 0) else None
            ),
            "cem_action6_decisions": getattr(
                self, "_cem_action6_elite_cost_count", 0
            ),
            "mean_cem_elite_cost_action6": (
                self._cem_action6_elite_cost_sum
                / self._cem_action6_elite_cost_count
                if getattr(self, "_cem_action6_elite_cost_count", 0)
                else None
            ),
            "cem_other_action_decisions": getattr(
                self, "_cem_other_elite_cost_count", 0
            ),
            "mean_cem_elite_cost_other_actions": (
                self._cem_other_elite_cost_sum
                / self._cem_other_elite_cost_count
                if getattr(self, "_cem_other_elite_cost_count", 0)
                else None
            ),
            "mean_selected_trajectory_knn_uncertainty": (
                self._cem_selected_uncertainty_sum
                / self._cem_selected_uncertainty_count
                if getattr(self, "_cem_selected_uncertainty_count", 0)
                else None
            ),
            "mean_selected_trajectory_knn_penalty": (
                getattr(self, "_cem_knn_lambda", 0.0)
                * self._cem_selected_uncertainty_sum
                / self._cem_selected_uncertainty_count
                if getattr(self, "_cem_selected_uncertainty_count", 0)
                else None
            ),
            "instrumented_observations": observations,
            "instrumented_action_clock": self._goal_metric_action_clock,
            "goal_records": list(self._goal_metric_records),
        }

    def _legacy_waypoint_index(
        self,
        eligible: Optional[list[int]] = None,
    ) -> int:
        """Apply the unchanged final-choice policy to an optional candidate set."""
        assert self._waypoint_index is not None
        candidates = (
            list(range(len(self._waypoints)))
            if eligible is None else list(eligible)
        )
        if not candidates:
            raise ValueError("waypoint candidate set must not be empty")
        if self._waypoint_selection_mode == "cyclic":
            candidate_set = set(candidates)
            for offset in range(1, len(self._waypoints) + 1):
                index = (
                    self._waypoint_index + offset
                ) % len(self._waypoints)
                if index in candidate_set:
                    return index
            raise AssertionError("cyclic selection found no eligible waypoint")

        recent = set(self._recent_waypoint_indices)
        non_recent = [i for i in candidates if i not in recent]
        if non_recent:
            candidates = non_recent
        return min(
            candidates,
            key=lambda i: (self._waypoint_target_counts[i], i),
        )

    @staticmethod
    def _dns_log_number(value: float) -> float | str:
        """Represent infinities explicitly so DNS JSON logs remain portable."""
        if math.isinf(value):
            return "+inf" if value > 0 else "-inf"
        return round(value, 6)

    def _dns_candidates(self) -> DNSSelection:
        """Apply DNS to the current waypoint memory.

        Creation distance acts as the quality objective: a waypoint that opened
        a new region of latent space is considered fitter. The stored latent is
        the behavioural descriptor used for novelty comparisons. DNS returns a
        survivor set; it does not choose the final waypoint by itself.
        """
        descriptors = torch.stack(self._waypoints).detach().reshape(
            len(self._waypoints), -1
        )
        # The first stored creation distance is +inf only because no reference
        # existed yet. It is a sentinel, not evidence of infinite quality.
        # DNS assumes finite objective fitness, so map only that undefined
        # input to neutral zero without changing the stored score generator.
        objectives = [
            score if math.isfinite(score) else 0.0
            for score in self._waypoint_creation_distance
        ]
        return self._dns_selector.select(
            objectives=objectives,
            descriptors=descriptors,
        )

    def _log_dns_selection(
        self,
        result: DNSSelection,
        chosen_index: int,
        decision: str,
        eligible_indices: Optional[list[int]] = None,
        fallback_reason: Optional[str] = None,
    ) -> None:
        """Log objectives, fitter groups, DNS groups, survivors, and choice."""
        retained = set(result.retained_indices)
        payload = {
            "decision": decision,
            "total_waypoints": len(result.candidates),
            "objective": "stored_creation_distance",
            "descriptor": "waypoint_latent",
            "distance": "euclidean_l2",
            "strictly_fitter": True,
            "k_neighbors": result.k_neighbors,
            "capacity": result.capacity,
            "waypoints": [
                {
                    "index": row.index + 1,
                    "score": self._dns_log_number(
                        self._waypoint_creation_distance[row.index]
                    ),
                    "dns_objective": self._dns_log_number(row.objective),
                    "fitter_group": [
                        index + 1 for index in row.fitter_indices
                    ],
                    "nearest_fitter_group": [
                        index + 1 for index in row.neighbor_indices
                    ],
                    "dns_score": self._dns_log_number(row.dns_score),
                    "retained": row.index in retained,
                }
                for row in result.candidates
            ],
            "retained_count": len(result.retained_indices),
            "retained_indices": [
                index + 1 for index in result.retained_indices
            ],
            "eligible_after_active_exclusion": [
                index + 1
                for index in (
                    result.retained_indices
                    if eligible_indices is None
                    else eligible_indices
                )
            ],
            "fallback_reason": fallback_reason,
            "final_policy": self._waypoint_selection_mode,
            "chosen_index": chosen_index + 1,
        }
        print(
            "[dns-selector] "
            f"{json.dumps(payload, separators=(',', ':'))}"
        )

    def _next_waypoint_index(self) -> int:
        """Choose the next target, optionally using DNS as a pre-filter.

        The final cyclic or least-targeted policy is deliberately shared by both
        modes. This isolates the effect of DNS to candidate retention and keeps
        the downstream target-selection behaviour comparable.
        """
        if self._waypoint_selector_mode == "legacy":
            return self._legacy_waypoint_index()
        result = self._dns_candidates()
        assert self._waypoint_index is not None
        eligible = [
            index
            for index in result.retained_indices
            if index != self._waypoint_index
        ]
        fallback_reason = None
        if not eligible:
            # DNS normally feeds reproduction, where a retained individual
            # generates a new candidate. This waypoint adapter has no
            # reproduction step: retargeting the just-reached singleton would
            # be a trivial self-loop rather than DNS exploration. Fall back to
            # the unchanged policy over other stored waypoints.
            eligible = [
                index
                for index in range(len(self._waypoints))
                if index != self._waypoint_index
            ]
            fallback_reason = "no_other_dns_survivor"
        chosen = self._legacy_waypoint_index(eligible)
        self._log_dns_selection(
            result,
            chosen,
            "advance",
            eligible_indices=eligible,
            fallback_reason=fallback_reason,
        )
        return chosen

    def _waypoint_selection_coverage(self) -> float:
        """Fraction of the current memory that has been targeted at least once."""
        if not self._waypoint_target_counts:
            return 0.0
        targeted = sum(count > 0 for count in self._waypoint_target_counts)
        return targeted / len(self._waypoint_target_counts)

    def _isolated_waypoint_indices(self) -> tuple[set[int], list[float]]:
        """Top quarter by mean MSE to the current waypoint memory."""
        count = len(self._waypoints)
        if not count:
            return set(), []
        latents = torch.stack(self._waypoints).detach().reshape(count, -1)
        distances = (
            (latents[:, None, :] - latents[None, :, :]).square().mean(dim=-1)
        )
        scores = []
        for index in range(count):
            others = torch.cat((
                distances[index, :index],
                distances[index, index + 1:],
            ))
            scores.append(float(others.mean().item()) if len(others) else 0.0)
        isolated_count = max(1, math.ceil(count / 4))
        ranked = sorted(range(count), key=lambda i: (-scores[i], i))
        return set(ranked[:isolated_count]), scores

    @staticmethod
    def _count_latent_clusters(
        latents: list[torch.Tensor],
        threshold: float,
    ) -> int:
        """Single-link cluster count for a small diagnostic latent list."""
        if not latents:
            return 0
        stacked = torch.stack(latents).reshape(len(latents), -1).float().cpu()
        distances = (
            (stacked[:, None, :] - stacked[None, :, :]).square().mean(dim=-1)
        )
        labels = [-1] * len(latents)
        clusters = 0
        for start in range(len(latents)):
            if labels[start] >= 0:
                continue
            labels[start] = clusters
            stack = [start]
            while stack:
                node = stack.pop()
                neighbours = torch.nonzero(
                    distances[node] <= threshold, as_tuple=False
                ).reshape(-1).tolist()
                for neighbour in neighbours:
                    if labels[neighbour] < 0:
                        labels[neighbour] = clusters
                        stack.append(neighbour)
            clusters += 1
        return clusters

    def _start_frontier_burst(
        self,
        reached_index: int,
        resume_index: int,
        latent: torch.Tensor,
        isolation_score: float,
    ) -> None:
        """Suspend the goal and start the configured fixed random burst."""
        self._frontier_bursts_started += 1
        self._frontier_burst_remaining = self._frontier_burst_length
        self._frontier_burst_resume_index = resume_index
        self._active_frontier_burst = {
            "burst_id": self._frontier_bursts_started,
            "reached_waypoint": reached_index + 1,
            "resume_waypoint": resume_index + 1,
            "reach_observation": self._waypoint_observations,
            "length": self._frontier_burst_length,
            "isolation_score": isolation_score,
            "known_latents": torch.stack([
                waypoint.detach().reshape(-1).float().cpu()
                for waypoint in self._waypoints
            ]),
            "reach_latent": latent.detach().reshape(-1).float().cpu().clone(),
            "start_waypoint_count": len(self._waypoints),
            "observed_latents": [],
            "nearest_known": [],
            "latent_from_reach": [],
        }
        self._waypoint_index = None
        self._waypoint_steps = 0
        self._waypoint_distance = None
        self.world_model.clear_goal()
        print(
            f"[waypoint-burst-start] id={self._frontier_bursts_started} "
            f"reached={reached_index + 1} resume={resume_index + 1} "
            f"length={self._frontier_burst_length} "
            f"isolation_score={isolation_score:.6f} "
            f"known_waypoints={len(self._waypoints)}"
        )

    def _observe_frontier_burst(self, latent: torch.Tensor) -> bool:
        """Record one real post-action state; return True after the final one."""
        if self._active_frontier_burst is None:
            return False
        current = latent.detach().reshape(-1).float().cpu()
        burst = self._active_frontier_burst
        nearest = float(
            (burst["known_latents"] - current).square().mean(dim=1).min().item()
        )
        burst["observed_latents"].append(current.clone())
        burst["nearest_known"].append(nearest)
        burst["latent_from_reach"].append(
            self._latent_distance(current, burst["reach_latent"])
        )
        self._frontier_burst_remaining -= 1
        print(
            f"[waypoint-burst-step] id={burst['burst_id']} "
            f"step={len(burst['observed_latents'])}/{burst['length']} "
            f"nearest_known={nearest:.6f} "
            f"remaining={self._frontier_burst_remaining}"
        )
        return self._frontier_burst_remaining == 0

    def _finish_frontier_burst(self) -> None:
        """Log burst coverage, then resume the deferred normal target."""
        assert self._active_frontier_burst is not None
        assert self._frontier_burst_resume_index is not None
        burst = self._active_frontier_burst
        novel_latents = [
            latent
            for latent, distance in zip(
                burst["observed_latents"], burst["nearest_known"]
            )
            if distance >= self.WAYPOINT_MIN_SEPARATION
        ]
        payload = {
            "burst_id": burst["burst_id"],
            "reached_waypoint": burst["reached_waypoint"],
            "resume_waypoint": burst["resume_waypoint"],
            "reach_observation": burst["reach_observation"],
            "length": burst["length"],
            "isolation_score": round(burst["isolation_score"], 6),
            "novel_steps": len(novel_latents),
            "novel_fraction": round(
                len(novel_latents) / burst["length"], 6
            ),
            "new_region_clusters": self._count_latent_clusters(
                novel_latents, self.WAYPOINT_CLUSTER_DISTANCE
            ),
            "max_distance_to_known": round(
                max(burst["nearest_known"], default=0.0), 6
            ),
            "final_distance_to_known": round(
                burst["nearest_known"][-1] if burst["nearest_known"] else 0.0,
                6,
            ),
            "max_latent_from_reach": round(
                max(burst["latent_from_reach"], default=0.0), 6
            ),
            "new_waypoints_created": (
                len(self._waypoints) - burst["start_waypoint_count"]
            ),
        }
        self._completed_frontier_bursts.append(payload)
        print(
            "[waypoint-burst] "
            f"{json.dumps(payload, separators=(',', ':'))}"
        )
        resume_index = self._frontier_burst_resume_index
        self._active_frontier_burst = None
        self._frontier_burst_remaining = 0
        self._frontier_burst_resume_index = None
        self._activate_waypoint(resume_index)

    def _log_waypoint_geometry(self, event: str) -> None:
        """Emit deterministic, diagnostic-only geometry for the current memory."""
        if not self._waypoint_geometry_enabled or not self._waypoints:
            return

        latents = torch.stack(self._waypoints).detach().reshape(
            len(self._waypoints), -1
        ).double().cpu()
        differences = latents[:, None, :] - latents[None, :, :]
        distances = differences.square().mean(dim=-1)
        count = len(latents)

        if count > 1:
            nearest = distances.clone()
            nearest.fill_diagonal_(float("inf"))
            nearest_distances = nearest.min(dim=1).values
            mean_nearest = float(nearest_distances.mean().item())
            diameter = float(distances.max().item())
        else:
            nearest_distances = torch.zeros(1, dtype=torch.float64)
            mean_nearest = 0.0
            diameter = 0.0

        # Single-link connected components under the fixed MSE threshold.
        cluster_labels = [-1] * count
        cluster_sizes: list[int] = []
        for start in range(count):
            if cluster_labels[start] >= 0:
                continue
            cluster_id = len(cluster_sizes)
            cluster_labels[start] = cluster_id
            stack = [start]
            size = 0
            while stack:
                node = stack.pop()
                size += 1
                neighbours = torch.nonzero(
                    distances[node] <= self.WAYPOINT_CLUSTER_DISTANCE,
                    as_tuple=False,
                ).reshape(-1).tolist()
                for neighbour in neighbours:
                    if cluster_labels[neighbour] < 0:
                        cluster_labels[neighbour] = cluster_id
                        stack.append(neighbour)
            cluster_sizes.append(size)

        centered = latents - latents.mean(dim=0, keepdim=True)
        pca = torch.zeros((count, 2), dtype=torch.float64)
        explained_variance_2d = 0.0
        if count > 1 and bool(torch.any(centered != 0)):
            _, singular_values, vh = torch.linalg.svd(
                centered, full_matrices=False
            )
            components = min(2, vh.shape[0])
            pca[:, :components] = centered @ vh[:components].T
            total_variance = singular_values.square().sum()
            if total_variance > 0:
                explained_variance_2d = float(
                    singular_values[:components].square().sum().div(
                        total_variance
                    ).item()
                )

        def rounded(values: torch.Tensor) -> Any:
            return values.round(decimals=6).tolist()

        payload = {
            "event": event,
            "observation": self._waypoint_observations,
            "count": count,
            "distance_metric": "per_dimension_mse",
            "cluster_threshold": self.WAYPOINT_CLUSTER_DISTANCE,
            "distance_matrix": rounded(distances),
            "nearest_distances": rounded(nearest_distances),
            "mean_nearest_distance": round(mean_nearest, 6),
            "diameter": round(diameter, 6),
            "cluster_count": len(cluster_sizes),
            "cluster_labels": cluster_labels,
            "cluster_sizes": cluster_sizes,
            "pca_explained_variance_2d": round(explained_variance_2d, 6),
            "pca_coordinates": rounded(pca),
        }
        print(f"[waypoint-geometry] {json.dumps(payload, separators=(',', ':'))}")

    def _start_post_reach_window(
        self,
        waypoint_index: int,
        latent: torch.Tensor,
        grid: torch.Tensor,
    ) -> None:
        """Start a passive eight-observation diagnostic after a real reach."""
        if not self._waypoint_frontier_diagnostic_enabled:
            return
        event_id = len(self._completed_post_reach) + len(
            self._pending_post_reach
        ) + 1
        self._pending_post_reach.append({
            "event_id": event_id,
            "waypoint_index": waypoint_index,
            "reach_observation": self._waypoint_observations,
            "reach_latent": latent.detach().reshape(-1).float().cpu().clone(),
            "last_latent": latent.detach().reshape(-1).float().cpu().clone(),
            "reach_grid": grid.detach().reshape(-1).cpu().clone(),
            "known_latents": torch.stack([
                waypoint.detach().reshape(-1).float().cpu()
                for waypoint in self._waypoints
            ]),
            "steps": 0,
            "changed_cells": [],
            "changed_union": torch.zeros(
                grid.numel(), dtype=torch.bool
            ),
            "latent_from_reach": [],
            "latent_step_change": [],
            "distance_to_known": [],
            "first_new_step": None,
            "return_step": None,
        })

    def _update_post_reach_windows(
        self,
        latent: torch.Tensor,
        grid: torch.Tensor,
    ) -> None:
        """Update every active post-reach window using the new real state."""
        if not self._waypoint_frontier_diagnostic_enabled:
            return
        current_latent = latent.detach().reshape(-1).float().cpu()
        current_grid = grid.detach().reshape(-1).cpu()
        completed: list[dict[str, Any]] = []
        for event in self._pending_post_reach:
            event["steps"] += 1
            step = event["steps"]
            changed = current_grid != event["reach_grid"]
            event["changed_union"] |= changed
            event["changed_cells"].append(int(changed.sum().item()))
            event["latent_from_reach"].append(
                self._latent_distance(current_latent, event["reach_latent"])
            )
            event["latent_step_change"].append(
                self._latent_distance(current_latent, event["last_latent"])
            )
            event["last_latent"] = current_latent.clone()
            known_distances = (
                (event["known_latents"] - current_latent).square().mean(dim=1)
            )
            nearest_known = float(known_distances.min().item())
            event["distance_to_known"].append(nearest_known)
            is_new = nearest_known >= self.WAYPOINT_MIN_SEPARATION
            if is_new and event["first_new_step"] is None:
                event["first_new_step"] = step
            elif (
                not is_new
                and event["first_new_step"] is not None
                and event["return_step"] is None
            ):
                event["return_step"] = step
            if step >= self.WAYPOINT_POST_REACH_HORIZON:
                completed.append(event)

        for event in completed:
            self._pending_post_reach.remove(event)
            payload = self._finalize_post_reach_event(event)
            self._completed_post_reach.append(payload)
            print(
                "[waypoint-frontier-event] "
                f"{json.dumps(payload, separators=(',', ':'))}"
            )
        if completed:
            self._log_waypoint_frontier("window_complete")

    def _finalize_post_reach_event(
        self,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert one tensor-backed pending window to a serialisable record."""
        changed_cells = event["changed_cells"]
        latent_from_reach = event["latent_from_reach"]
        latent_step_change = event["latent_step_change"]
        distance_to_known = event["distance_to_known"]
        first_new = event["first_new_step"]
        return_step = event["return_step"]
        return {
            "event_id": event["event_id"],
            "waypoint_index": event["waypoint_index"] + 1,
            "reach_observation": event["reach_observation"],
            "steps": event["steps"],
            "changed_cells": changed_cells,
            "max_changed_cells": max(changed_cells, default=0),
            "final_changed_cells": changed_cells[-1] if changed_cells else 0,
            "unique_changed_cells": int(event["changed_union"].sum().item()),
            "latent_from_reach": [
                round(value, 6) for value in latent_from_reach
            ],
            "max_latent_from_reach": round(
                max(latent_from_reach, default=0.0), 6
            ),
            "final_latent_from_reach": round(
                latent_from_reach[-1] if latent_from_reach else 0.0, 6
            ),
            "mean_step_latent_change": round(
                sum(latent_step_change) / len(latent_step_change)
                if latent_step_change else 0.0,
                6,
            ),
            "distance_to_known": [
                round(value, 6) for value in distance_to_known
            ],
            "novel_steps": sum(
                value >= self.WAYPOINT_MIN_SEPARATION
                for value in distance_to_known
            ),
            "first_new_step": first_new,
            "return_step": return_step,
            "entered_new_region": first_new is not None,
            "returned_to_known": return_step is not None,
        }

    def _log_waypoint_frontier(self, event: str) -> None:
        """Emit per-waypoint frontier utility without influencing selection."""
        if (
            not self._waypoint_frontier_diagnostic_enabled
            or not self._waypoints
            or self._episode_initial_latent is None
        ):
            return
        latents = torch.stack(self._waypoints).detach().reshape(
            len(self._waypoints), -1
        ).float().cpu()
        distances = (
            (latents[:, None, :] - latents[None, :, :]).square().mean(dim=-1)
        )
        distance_to_initial = (
            (latents - self._episode_initial_latent).square().mean(dim=1)
        )
        waypoint_rows = []
        for index in range(len(self._waypoints)):
            others = torch.cat((distances[index, :index], distances[index, index + 1:]))
            reached_events = [
                item for item in self._completed_post_reach
                if item["waypoint_index"] == index + 1
            ]

            def event_mean(key: str) -> Optional[float]:
                if not reached_events:
                    return None
                return round(
                    sum(float(item[key]) for item in reached_events)
                    / len(reached_events),
                    6,
                )

            entered = sum(item["entered_new_region"] for item in reached_events)
            returned = sum(item["returned_to_known"] for item in reached_events)
            created_after_reach = sum(
                any(
                    item["reach_observation"] < created_at
                    <= item["reach_observation"] + self.WAYPOINT_POST_REACH_HORIZON
                    for created_at in self._waypoint_created_at
                )
                for item in reached_events
            )
            waypoint_rows.append({
                "index": index + 1,
                "captured_observation": self._waypoint_created_at[index],
                "distance_to_initial": round(
                    float(distance_to_initial[index].item()), 6
                ),
                "mean_distance_to_others": round(
                    float(others.mean().item()) if len(others) else 0.0, 6
                ),
                "selections": self._waypoint_target_counts[index],
                "reached": self._waypoint_reached_counts[index],
                "completed_windows": len(reached_events),
                "mean_unique_changed_cells": event_mean(
                    "unique_changed_cells"
                ),
                "mean_max_changed_cells": event_mean("max_changed_cells"),
                "mean_final_changed_cells": event_mean(
                    "final_changed_cells"
                ),
                "mean_max_latent_from_reach": event_mean(
                    "max_latent_from_reach"
                ),
                "mean_final_latent_from_reach": event_mean(
                    "final_latent_from_reach"
                ),
                "mean_step_latent_change": event_mean(
                    "mean_step_latent_change"
                ),
                "mean_novel_steps": event_mean("novel_steps"),
                "new_region_rate": (
                    round(entered / len(reached_events), 6)
                    if reached_events else None
                ),
                "return_rate_after_new": (
                    round(returned / entered, 6) if entered else None
                ),
                "new_waypoint_after_reach_rate": (
                    round(created_after_reach / len(reached_events), 6)
                    if reached_events else None
                ),
            })
        payload = {
            "event": event,
            "observation": self._waypoint_observations,
            "post_reach_horizon": self.WAYPOINT_POST_REACH_HORIZON,
            "new_region_threshold": self.WAYPOINT_MIN_SEPARATION,
            "waypoints": waypoint_rows,
            "completed_events": len(self._completed_post_reach),
            "pending_events": [
                {
                    "event_id": item["event_id"],
                    "waypoint_index": item["waypoint_index"] + 1,
                    "observed_steps": item["steps"],
                }
                for item in self._pending_post_reach
            ],
        }
        print(
            "[waypoint-frontier] "
            f"{json.dumps(payload, separators=(',', ':'))}"
        )

    def _update_waypoints(
        self,
        latent: torch.Tensor,
        grid: torch.Tensor,
        after_action: bool,
        event_mask: int = 0,
    ) -> None:
        """Collect, activate, and advance the sequential waypoint list."""
        if after_action:
            self._goal_metric_action_clock += 1
        self._waypoint_observations += 1
        if self._episode_initial_latent is None:
            self._episode_initial_latent = (
                latent.detach().reshape(-1).float().cpu().clone()
            )
        self._update_post_reach_windows(latent, grid)
        burst_finished = self._observe_frontier_burst(latent)
        self._remember_waypoint(latent, event_mask)
        # Evaluate the current observation against the *previous* N states,
        # then retain it for future candidates. Including it before the test
        # would make every nearest distance exactly zero.
        recent = latent.detach().reshape(-1).clone()
        self._waypoint_recent_latents.append(recent)
        if len(self._waypoint_recent_latents) > self._waypoint_recent_size:
            self._waypoint_recent_latents.pop(0)
        if burst_finished:
            self._finish_frontier_burst()

        if self._waypoint_observations < self.WAYPOINT_WARMUP:
            self._waypoint_distance = None
            print(
                f"[waypoint] status observation={self._waypoint_observations} "
                f"created={len(self._waypoints)} active=none "
                f"phase=collecting warmup={self._waypoint_observations}/"
                f"{self.WAYPOINT_WARMUP}"
            )
            return

        if len(self._waypoints) < 2:
            # This can only happen with an extremely short/invalid episode. Keep
            # the state explicit instead of incorrectly reporting warm-up.
            self._waypoint_distance = None
            print(
                f"[waypoint] status observation={self._waypoint_observations} "
                f"created={len(self._waypoints)} active=none "
                f"phase=waiting_for_waypoints"
            )
            return

        if self._waypoint_observations == self.WAYPOINT_WARMUP:
            print(
                f"[waypoint] warmup_complete observations="
                f"{self._waypoint_observations} created={len(self._waypoints)} "
                f"memory={self._waypoint_memory_mode}"
            )

        if self._frontier_burst_remaining > 0:
            self._waypoint_distance = None
            print(
                f"[waypoint] status observation={self._waypoint_observations} "
                f"created={len(self._waypoints)} active=none "
                f"phase=frontier_burst "
                f"remaining={self._frontier_burst_remaining}/"
                f"{self._frontier_burst_length}"
            )
            return

        if self._waypoint_index is None:
            # Avoid starting on the current state: begin with the remembered
            # latent farthest from it. DNS changes only the eligible set.
            distances = [self._latent_distance(latent, z) for z in self._waypoints]
            eligible = list(range(len(self._waypoints)))
            dns_result: Optional[DNSSelection] = None
            if self._waypoint_selector_mode == "dns":
                dns_result = self._dns_candidates()
                eligible = list(dns_result.retained_indices)
            chosen = max(eligible, key=distances.__getitem__)
            if dns_result is not None:
                self._log_dns_selection(dns_result, chosen, "initial")
            self._activate_waypoint(chosen)

        target = self._waypoints[self._waypoint_index]
        self._waypoint_distance = self._latent_distance(latent, target)
        current_flat = latent.reshape(-1)
        same_storage = current_flat.data_ptr() == target.data_ptr()
        exact_equal = torch.equal(current_flat, target)
        goal_matches_target = (
            self.world_model.goal_latent is not None
            and torch.equal(self.world_model.goal_latent, target)
        )
        print(
            f"[waypoint] status observation={self._waypoint_observations} "
            f"source={'latest_frame_after_action' if after_action else 'initial_frame'} "
            f"created={len(self._waypoints)} "
            f"active={self._waypoint_index + 1}/{len(self._waypoints)} "
            f"target_captured_observation="
            f"{self._waypoint_created_at[self._waypoint_index]} "
            f"distance={self._waypoint_distance:.6f} "
            f"threshold={self.WAYPOINT_REACHED_DISTANCE:.6f} "
            f"target_steps={self._waypoint_steps}/{self.WAYPOINT_TIMEOUT} "
            f"memory_coverage={self._waypoint_selection_coverage():.4f} "
            f"selection_counts={self._waypoint_target_counts} "
            f"immediate_loops={self._waypoint_immediate_loops} "
            f"same_storage={same_storage} exact_equal={exact_equal} "
            f"goal_matches_target={goal_matches_target} "
            f"current={self._latent_fingerprint(current_flat)} "
            f"target={self._latent_fingerprint(target)}"
        )
        self._print_full_latent("compare_current", current_flat)
        self._print_full_latent("compare_target", target)
        if self._waypoint_distance <= self.WAYPOINT_REACHED_DISTANCE:
            reached_index = self._waypoint_index
            self._waypoints_reached += 1
            self._waypoint_reached_counts[reached_index] += 1
            self._finish_goal_metric("reached")
            print(
                f"[waypoint] reached index={reached_index + 1}/"
                f"{len(self._waypoints)} distance={self._waypoint_distance:.6f} "
                f"threshold={self.WAYPOINT_REACHED_DISTANCE:.6f} "
                f"total_reached={self._waypoints_reached}"
            )
            self._start_post_reach_window(
                reached_index, latent, grid
            )
            resume_index = self._next_waypoint_index()
            burst_started = False
            if self._frontier_burst_length > 0:
                isolated_indices, isolation_scores = (
                    self._isolated_waypoint_indices()
                )
                if reached_index in isolated_indices:
                    self._start_frontier_burst(
                        reached_index,
                        resume_index,
                        latent,
                        isolation_scores[reached_index],
                    )
                    burst_started = True
            if not burst_started:
                self._activate_waypoint(resume_index)
                self._waypoint_distance = self._latent_distance(
                    latent, self._waypoints[self._waypoint_index]
                )
            self._log_waypoint_frontier("reached")
        else:
            self._waypoint_steps += 1
            if self._waypoint_steps >= self.WAYPOINT_TIMEOUT:
                self._waypoints_timed_out += 1
                self._finish_goal_metric("timeout")
                print(
                    f"[waypoint] timeout index={self._waypoint_index + 1}/"
                    f"{len(self._waypoints)} distance={self._waypoint_distance:.6f} "
                    f"steps={self._waypoint_steps} "
                    f"total_timeouts={self._waypoints_timed_out}"
                )
                self._activate_waypoint(self._next_waypoint_index())
                self._waypoint_distance = self._latent_distance(
                    latent, self._waypoints[self._waypoint_index]
                )

    def _reset_waypoints(self) -> None:
        """Clear waypoint memory and navigation state (baseline condition)."""
        self._finish_goal_metric("reset")
        if self._waypoints or self._waypoint_observations:
            self._grid_event_summary("reset_clear")
            if self._active_frontier_burst is not None:
                print(
                    f"[waypoint-burst-aborted] "
                    f"id={self._active_frontier_burst['burst_id']} "
                    f"observed={len(self._active_frontier_burst['observed_latents'])}/"
                    f"{self._active_frontier_burst['length']} reason=reset"
                )
            self._log_waypoint_geometry("episode_end")
            self._log_waypoint_frontier("episode_end")
            print(
                f"[waypoint] reset_cleared created={len(self._waypoints)} "
                f"reached={self._waypoints_reached} "
                f"timeouts={self._waypoints_timed_out} "
                f"selections={self._waypoint_selections} "
                f"immediate_loops={self._waypoint_immediate_loops} "
                f"memory_coverage={self._waypoint_selection_coverage():.4f} "
                f"selection_counts={self._waypoint_target_counts}"
            )
        self._waypoints.clear()
        self._waypoint_created_at.clear()
        self._waypoint_creation_distance.clear()
        self._waypoint_target_counts.clear()
        self._waypoint_reached_counts.clear()
        self._waypoint_event_masks.clear()
        self._recent_waypoint_indices.clear()
        self._waypoint_selections = 0
        self._waypoint_immediate_loops = 0
        self._waypoint_index = None
        self._waypoint_steps = 0
        self._waypoint_observations = 0
        self._waypoints_reached = 0
        self._waypoints_timed_out = 0
        self._waypoint_distance = None
        self._waypoint_summary_logged = False
        self._episode_initial_latent = None
        self._pending_post_reach.clear()
        self._completed_post_reach.clear()
        self._frontier_burst_remaining = 0
        self._frontier_burst_resume_index = None
        self._active_frontier_burst = None
        self._completed_frontier_bursts.clear()
        self._frontier_bursts_started = 0
        self._waypoint_recent_latents.clear()
        self._visited_latents.clear()
        self._last_current_latent = None
        self._last_plan_latents = None
        self.world_model.clear_goal()

    def _preserve_waypoints_after_reset(self) -> None:
        """Keep latent snapshots but discard the pre-reset navigation state."""
        self._finish_goal_metric("reset")
        self._grid_event_summary("reset_persist")
        if self._active_frontier_burst is not None:
            print(
                f"[waypoint-burst-aborted] "
                f"id={self._active_frontier_burst['burst_id']} "
                f"observed={len(self._active_frontier_burst['observed_latents'])}/"
                f"{self._active_frontier_burst['length']} reason=reset"
            )
        self._log_waypoint_geometry("episode_end")
        self._log_waypoint_frontier("episode_end")
        print(
            f"[waypoint] reset_preserved created={len(self._waypoints)} "
            f"observations={self._waypoint_observations} "
            f"reached={self._waypoints_reached} "
            f"timeouts={self._waypoints_timed_out} "
            f"selections={self._waypoint_selections} "
            f"immediate_loops={self._waypoint_immediate_loops} "
            f"memory_coverage={self._waypoint_selection_coverage():.4f} "
            f"selection_counts={self._waypoint_target_counts}"
        )
        self._waypoint_index = None
        self._waypoint_reached_counts = [0] * len(self._waypoints)
        self._recent_waypoint_indices.clear()
        self._waypoint_steps = 0
        self._waypoint_distance = None
        self._waypoint_summary_logged = False
        self._episode_initial_latent = None
        self._pending_post_reach.clear()
        self._completed_post_reach.clear()
        self._frontier_burst_remaining = 0
        self._frontier_burst_resume_index = None
        self._active_frontier_burst = None
        self._completed_frontier_bursts.clear()
        self._frontier_bursts_started = 0
        # A reset starts a new local trajectory even when the old waypoint
        # snapshots are intentionally preserved.
        self._waypoint_recent_latents.clear()
        self._visited_latents.clear()
        self._last_current_latent = None
        self._last_plan_latents = None
        self.world_model.clear_goal()

    @staticmethod
    def _latent_fingerprint(latent: torch.Tensor) -> str:
        """Compact, human-readable identity for a high-dimensional latent."""
        z = latent.detach().reshape(-1).float().cpu()
        preview = ",".join(f"{v:.4f}" for v in z[:6].tolist())
        return (
            f"shape={tuple(z.shape)},norm={z.norm().item():.4f},"
            f"mean={z.mean().item():.4f},first6=[{preview}]"
        )

    def _print_full_latent(self, label: str, latent: torch.Tensor) -> None:
        """Optionally dump all components for a one-off forensic run."""
        if self._waypoint_debug_latents:
            values = latent.detach().reshape(-1).float().cpu().tolist()
            print(f"[waypoint-latent] {label} observation="
                  f"{self._waypoint_observations} values={values}")

    def _waypoint_status(self) -> dict[str, Any]:
        """Small diagnostics payload for action logs and the local monitor."""
        active = None if self._waypoint_index is None else self._waypoint_index + 1
        return {
            "active": active,
            "count": len(self._waypoints),
            "distance": self._waypoint_distance,
            "steps": self._waypoint_steps,
            "reached": self._waypoints_reached,
            "timed_out": self._waypoints_timed_out,
            "warming_up": self._waypoint_observations < self.WAYPOINT_WARMUP,
        }

    def _latent_debug_payload(self) -> dict[str, Any]:
        """CPU latents consumed only by the local PCA monitor."""
        waypoints = (
            torch.stack([z.detach().cpu() for z in self._waypoints]).numpy()
            if self._waypoints else None
        )
        visited = (
            torch.stack(self._visited_latents).numpy()
            if self._visited_latents else None
        )
        active = (
            None if self._waypoint_index is None
            else self._waypoints[self._waypoint_index].detach().cpu().numpy()
        )
        current = (
            None if self._last_current_latent is None
            else self._last_current_latent.numpy()
        )
        predicted = (
            None if self._last_plan_latents is None
            else self._last_plan_latents.numpy()
        )
        return {
            "visited": visited,
            "waypoints": waypoints,
            "active": active,
            "current": current,
            "predicted": predicted,
        }

    def _train_step(self) -> None:
        """One inline SGD step training the latent dynamics to predict the next
        latent from (z, action), on a replay minibatch. Only `dynamics` moves."""
        params = [p for p in self.world_model.dynamics.parameters() if p.requires_grad]
        if not params:
            return
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(params, lr=self.LEARNING_RATE)

        batch = random.sample(self._buffer, min(self.TRAIN_BATCH, len(self._buffer)))
        latent_prev = torch.cat([b[1] for b in batch], dim=0)
        action_prev = torch.cat([b[2] for b in batch], dim=0)
        coords_prev = torch.cat([b[3] for b in batch], dim=0)
        actual = torch.cat([b[5] for b in batch], dim=0)

        predicted = self.world_model.predict_next(latent_prev, action_prev, coords_prev)
        loss = self.world_model.prediction_error(predicted, actual).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    @torch.no_grad()
    def _choose_click(self, latent: torch.Tensor, action_idx: int) -> tuple[int, int]:
        """Pick the ``(x, y)`` for ACTION6 in ``separate`` planning mode.

        Score each candidate
        anchor with the coordinate-conditioned dynamics: roll one step per anchor
        and take the click whose imagined next latent lands closest to the goal —
        the one-step, coordinate-space analogue of the planner's goal objective.
        This method is not used by either joint ACTION6 mode, because those modes
        already optimise and return the coordinate inside CEM.
        """
        goal = self.world_model.goal_latent
        cand = self._click_candidates                        # (M, 2)
        m = cand.shape[0]
        z = latent.expand(m, -1)                             # (M, D)
        a = self._one_hot(action_idx).expand(m, -1)          # (M, n_actions)
        z_next = self.world_model.predict_next(z, a, cand)   # (M, D)
        dist = ((z_next - goal.to(z_next)) ** 2).mean(dim=-1)  # (M,)
        best = int(dist.argmin())
        return int(cand[best, 0]), int(cand[best, 1])

    @torch.no_grad()
    def _imagine_plan(
        self,
        latent: torch.Tensor,
        plan: list[int],
        plan_coords: Optional[list[list[int]]] = None,
    ) -> list[tuple[Any, str]]:
        """Roll the CEM's chosen action sequence through the dynamics and decode
        each imagined latent to pixels: [(rgb 64×64×3, action_name), …], one per
        horizon step. This is the agent's *imagination* — what it expects each
        planned step to look like, not what the game will actually show. Monitor
        only."""
        z, zs = latent, []
        for step, idx in enumerate(plan):
            coords = (
                None
                if plan_coords is None
                else torch.tensor(
                    [plan_coords[step]],
                    device=z.device,
                    dtype=z.dtype,
                )
            )
            z = self.world_model.predict_next(
                z, self._one_hot(idx), coords
            )
            zs.append(z)
        self._last_plan_latents = torch.cat(zs, dim=0).detach().cpu()
        rgbs = self.world_model.decode(torch.cat(zs, dim=0)).permute(0, 2, 3, 1).cpu().numpy()
        return [(rgbs[h], self._candidate_actions[idx].name) for h, idx in enumerate(plan)]

    def _train_decoder(self, latent: torch.Tensor, grid: torch.Tensor) -> None:
        """One cheap SGD step teaching the decoder probe to invert the frozen
        encoder, on a replay of observed (z, cell-id grid) pairs. Called once per
        action, only while the monitor is on."""
        target = F.interpolate(grid.unsqueeze(1).float(), size=(64, 64),
                               mode="nearest").squeeze(1).long()
        self._recon_buffer.append((latent, target))
        if len(self._recon_buffer) > self.RECON_BUFFER:
            self._recon_buffer.pop(0)

        if self._decoder_optimizer is None:
            self._decoder_optimizer = torch.optim.Adam(
                self.world_model.decoder.parameters(), lr=self.DECODER_LR
            )
        batch = random.sample(self._recon_buffer, min(self.RECON_BATCH, len(self._recon_buffer)))
        zs = torch.cat([b[0] for b in batch], dim=0)
        targets = torch.cat([b[1] for b in batch], dim=0)
        loss = F.cross_entropy(self.world_model.decode_logits(zs), targets,
                               weight=cell_class_weights(targets))
        self._decoder_optimizer.zero_grad()
        loss.backward()
        self._decoder_optimizer.step()

    def _available_mask(self, latest_frame: FrameData) -> torch.Tensor:
        """0/1 mask ``(1, n_actions)`` over `_candidate_actions` from the engine's
        per-frame ``available_actions`` declaration. All-ones when the engine
        doesn't say (empty list) — never mask on missing information."""
        declared = set(latest_frame.available_actions or [])
        if not declared:
            return torch.ones(1, len(self._candidate_actions))
        mask = torch.tensor(
            [[1.0 if a.value in declared else 0.0 for a in self._candidate_actions]]
        )
        # Defensive: if the declaration excludes everything we can send, ignore it.
        return mask if mask.any() else torch.ones_like(mask)

    def _one_hot(self, action_idx: int) -> torch.Tensor:
        """One-hot encode an action index as ``(1, n_actions)`` on the model device."""
        onehot = torch.zeros(1, len(self._candidate_actions), device=self._device)
        onehot[0, action_idx] = 1.0
        return onehot
