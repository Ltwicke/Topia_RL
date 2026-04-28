"""Smoke renders for the V2.0 EnvWrapper renderer.

Three modes are exercised:
    1. Single POV — `env.render()` baseline.
    2. Trajectory — single-POV with `joint_probs` / `traj_actions` overlay.
    3. Hidden estimation — dual-POV with the actual `policy.estimate_hidden_dual(obs)`
       API (no synthetic random arrays).

The policy is built with random init weights — that's enough to drive every
forward path end-to-end and confirm the rendering cascade still works.
"""

import os
import random

import numpy as np
import torch

from env.wrapper import EnvWrapper
from game.enums import BoardType, Tribes
from RL.models.policy import PolicyNetwork


OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _seed(s: int = 42):
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)


def _build_env():
    env = EnvWrapper(
        {"board_size": (8, 8), "board_type": BoardType.Dummy, "n_players": 2},
        [Tribes.Omaji, Tribes.Yaddak],
    )
    env.reset()
    return env


class _Cfg:
    """PolicyNetwork pulls hyperparams via getattr — empty cfg = all defaults."""
    pass


def _build_policy() -> PolicyNetwork:
    return PolicyNetwork(_Cfg())


def smoke_single():
    _seed()
    env = _build_env()
    out = os.path.join(OUT_DIR, "smoke_single.png")
    env.render(save_path=out, show=False)
    print(f"[ok] single POV -> {out}")


def smoke_trajectory():
    _seed()
    env    = _build_env()
    policy = _build_policy()
    obs    = env._get_obs()
    mask   = env.get_action_mask()

    with torch.no_grad():
        action, joint_probs, traj_actions, _, _, value = policy(obs, mask)

    out = os.path.join(OUT_DIR, "smoke_traj.png")
    env.render(
        action=action,
        joint_probs=joint_probs,
        traj_actions=traj_actions,
        critic_value=value,
        save_path=out, show=False,
    )
    print(f"[ok] trajectory  -> {out}  (sampled action: {action})")


def smoke_hidden():
    _seed()
    env    = _build_env()
    policy = _build_policy()
    obs    = env._get_obs()

    est_a, est_b = policy.estimate_hidden_dual(obs)
    print(f"     est_a shape: {est_a.shape}, est_b shape: {est_b.shape}")

    out = os.path.join(OUT_DIR, "smoke_hidden.png")
    env.render(
        show_hidden=True,
        hidden_estimate=(est_a, est_b),
        save_path=out, show=False,
    )
    print(f"[ok] hidden POV  -> {out}")


if __name__ == "__main__":
    smoke_single()
    smoke_trajectory()
    smoke_hidden()
