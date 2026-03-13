import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

# set the path for imports:

sys.path.insert(0, r"C:\Users\laure\1own_projects\1polytopia_score")

from env.wrapper import EnvWrapper
from game.enums import BoardType, Tribes
from RL.ppo.policy import DummyPolicy


# ── Hyperparameters ────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:

    # --- Parallelism ---
    n_processes:         int   = 8      # independent worker processes
    n_envs_per_process:  int   = 4      # envs stepped sequentially inside each worker

    # --- Environment ---
    board_config_dict = {
        "board_size" : (10,10),
        "board_type" : BoardType.Dummy,
        "n_players" : 2,
    }
    player_tribes = [Tribes.Omaji, Tribes.Imperius]
    max_turns_per_game = 100

    # --- Rollout ---
    n_steps:             int   = 256    # steps collected per env before each update

    # --- PPO epochs & batching ---
    n_epochs:            int   = 10      # passes over the collected batch
    n_minibatches:       int   = 16      # minibatch splits per epoch

    # --- PPO loss coefficients ---
    clip_eps:            float = 0.2    # clipping range for probability ratio
    vf_coef:             float = 0.5    # value loss weight
    ent_coef:            float = 0.01   # entropy bonus weight
    max_grad_norm:       float = 0.5    # gradient clipping

    # --- GAE / discount ---
    gamma:               float = 0.99
    gae_lambda:          float = 0.95

    # --- Optimizer ---
    lr:                  float = 3e-4

    # --- Network ---
    embed_dim:           int   = 16

    # --- Training ---
    n_updates:           int   = 5
    log_interval:        int   = 1

    # ── derived ───────────────────────────────────────────────────────────────

    @property
    def n_envs_total(self):
        return self.n_processes * self.n_envs_per_process

    @property
    def batch_size(self):
        """Total transitions per update = steps × total envs."""
        return self.n_steps * self.n_envs_total

    @property
    def board_size(self):
        return self.board_config_dict["board_size"]

    @property
    def minibatch_size(self):
        return self.batch_size // self.n_minibatches

    @property
    def board_config(self):
        return {"board_size": list(self.board_config_dict["board_size"]),
                "board_type": self.board_config_dict["board_type"],
                "n_players": self.board_config_dict["n_players"]}
        


# ── Worker process ─────────────────────────────────────────────────────────────

def worker_fn(worker_id: int, cfg: PPOConfig, conn):
    """
    Owns cfg.n_envs_per_process independent game instances.
    Loop:
      1. Receives updated policy weights from main.
      2. Collects cfg.n_steps steps across all envs.
      3. Sends trajectory buffer back to main.
    """
    envs    = [EnvWrapper(cfg.board_config, cfg.player_tribes,
                          max_turns_per_game=cfg.max_turns_per_game,
                          dense_reward=True)
               for _ in range(cfg.n_envs_per_process)]
    obs_buf = [env.reset() for env in envs]

    policy  = DummyPolicy(cfg)
    policy.eval()

    M = cfg.n_envs_per_process
    T = cfg.n_steps

    while True:
        cmd, payload = conn.recv()
        if cmd == 'stop':
            break

        policy.load_state_dict(payload)

        # Pre-allocate buffers  shape (T, M)
        obs_sizes = [[None] * M for _ in range(T)]   # (n_u, n_e, n_c) per step
        actions   = [[None] * M for _ in range(T)]
        masks_buf = [[None] * M for _ in range(T)]
        log_probs = np.zeros((T, M), dtype=np.float32)
        values    = np.zeros((T, M), dtype=np.float32)
        rewards   = np.zeros((T, M), dtype=np.float32)
        dones     = np.zeros((T, M), dtype=np.float32)

        with torch.no_grad():
            for t in range(T):
                for e, env in enumerate(envs):
                    obs  = obs_buf[e]
                    mask = env.get_action_mask()

                    action, lp, _, val         = policy(obs, mask)
                    next_obs, rew, done, _info = env.step(action)

                    obs_sizes[t][e] = (len(obs["units"]),
                                       len(obs["enemy_units"]),
                                       len(obs["cities"]))
                    actions[t][e]   = action
                    masks_buf[t][e] = mask
                    log_probs[t, e] = lp.item()
                    values[t, e]    = val.item()
                    rewards[t, e]   = rew
                    dones[t, e]     = float(done)

                    obs_buf[e] = env.reset() if done else next_obs

            # Bootstrap value for the step just AFTER the rollout ends.
            # In a real policy this would encode obs_buf[e]; dummy ignores obs.
            last_values = np.array([policy._value().item()
                                    for _ in range(M)], dtype=np.float32)

        conn.send(('data', {
            'obs_sizes':   obs_sizes,
            'actions':     actions,
            'masks':       masks_buf,
            'log_probs':   log_probs,
            'values':      values,
            'rewards':     rewards,
            'dones':       dones,
            'last_values': last_values,
        }))


# ── GAE ────────────────────────────────────────────────────────────────────────

def compute_gae(rewards, values, dones, last_values, gamma, gae_lam):
    """
    Generalized Advantage Estimation.

    rewards, values, dones : (T, N)  numpy float32
    last_values            : (N,)    value of obs right after the rollout ends

    When dones[t] == 1 the episode terminated; the multiplier (1-done) zeroes
    out both the next-value bootstrap and the GAE carry-over, so advantages
    never leak across episode boundaries.

    Returns
    -------
    advantages : (T, N)
    returns    : (T, N)   = advantages + values  (critic regression targets)
    """
    T, N       = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    gae        = np.zeros(N,      dtype=np.float32)

    for t in reversed(range(T)):
        next_val = last_values if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta    = rewards[t] + gamma * next_val * not_done - values[t]
        gae      = delta + gamma * gae_lam * not_done * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# ── PPO update ─────────────────────────────────────────────────────────────────

def ppo_update(policy: DummyPolicy, optimizer, batch: dict, cfg: PPOConfig):
    """
    Runs cfg.n_epochs passes over the batch, each split into cfg.n_minibatches.
    Returns mean losses for logging.
    """
    B = cfg.batch_size

    # Flatten (T, N) list-of-lists → (B,) flat lists
    flat_obs   = [s for step in batch['obs_sizes'] for s in step]
    flat_acts  = [a for step in batch['actions']   for a in step]
    flat_masks = [m for step in batch['masks']      for m in step]

    log_old = torch.tensor(batch['log_probs'].reshape(-1),  dtype=torch.float32)
    adv     = torch.tensor(batch['advantages'].reshape(-1), dtype=torch.float32)
    ret     = torch.tensor(batch['returns'].reshape(-1),    dtype=torch.float32)

    # Normalise advantages over the whole batch (not per minibatch)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    indices = np.arange(B)
    pl_log, vl_log, el_log = [], [], []

    for _ in range(cfg.n_epochs):
        np.random.shuffle(indices)

        for start in range(0, B, cfg.minibatch_size):
            mb = indices[start : start + cfg.minibatch_size]

            new_lp, new_ent, new_val = policy.evaluate_actions(
                [flat_acts[i]  for i in mb],
                [flat_masks[i] for i in mb],
            )

            ratio      = torch.exp(new_lp - log_old[mb])
            mb_adv     = adv[mb]

            # Clipped surrogate objective
            loss_clip  = torch.min(
                ratio * mb_adv,
                ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
            ).mean()

            loss_val   = F.mse_loss(new_val.squeeze(), ret[mb])
            loss_ent   = new_ent.mean()

            loss = -loss_clip + cfg.vf_coef * loss_val - cfg.ent_coef * loss_ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            pl_log.append(loss_clip.item())
            vl_log.append(loss_val.item())
            el_log.append(loss_ent.item())

    return np.mean(pl_log), np.mean(vl_log), np.mean(el_log)


# ── Main training loop ─────────────────────────────────────────────────────────

def main():
    cfg       = PPOConfig()
    policy    = DummyPolicy(cfg)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    print(f"Batch size: {cfg.batch_size} steps  "
          f"({cfg.n_processes} proc × {cfg.n_envs_per_process} envs × {cfg.n_steps} steps)\n"
          f"Minibatch : {cfg.minibatch_size}  |  "
          f"Epochs    : {cfg.n_epochs}  |  "
          f"Updates   : {cfg.n_updates}\n")

    # Spawn worker processes
    parent_conns, child_conns = zip(*[mp.Pipe() for _ in range(cfg.n_processes)])
    workers = [
        mp.Process(target=worker_fn, args=(i, cfg, child_conns[i]), daemon=True)
        for i in range(cfg.n_processes)
    ]
    for w in workers:
        w.start()

    for update in range(cfg.n_updates):

        # ── 1. Distribute weights ──────────────────────────────────────────
        state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
        for conn in parent_conns:
            conn.send(('collect', state_dict))

        # ── 2. Collect rollouts ────────────────────────────────────────────
        chunks = [conn.recv()[1] for conn in parent_conns]

        # Merge worker chunks along the env axis
        batch = {
            'obs_sizes': [sum([c['obs_sizes'][t] for c in chunks], []) for t in range(cfg.n_steps)],
            'actions':   [sum([c['actions'][t]   for c in chunks], []) for t in range(cfg.n_steps)],
            'masks':     [sum([c['masks'][t]      for c in chunks], []) for t in range(cfg.n_steps)],
            'log_probs': np.concatenate([c['log_probs']   for c in chunks], axis=1),  # (T, N_total)
            'values':    np.concatenate([c['values']      for c in chunks], axis=1),
            'rewards':   np.concatenate([c['rewards']     for c in chunks], axis=1),
            'dones':     np.concatenate([c['dones']       for c in chunks], axis=1),
            'last_values': np.concatenate([c['last_values'] for c in chunks]),         # (N_total,)
        }

        # ── 3. GAE ────────────────────────────────────────────────────────
        adv, ret = compute_gae(
            batch['rewards'], batch['values'], batch['dones'],
            batch['last_values'], cfg.gamma, cfg.gae_lambda
        )
        batch['advantages'] = adv
        batch['returns']    = ret

        # ── 4. PPO update ─────────────────────────────────────────────────
        pl, vl, el = ppo_update(policy, optimizer, batch, cfg)

        # ── 5. Logging ────────────────────────────────────────────────────
        if update % cfg.log_interval == 0:
            n_ep  = int(batch['dones'].sum())
            avg_r = batch['rewards'].sum() / max(n_ep, 1)
            print(f"update {update:4d} | "
                  f"episodes {n_ep:4d} | "
                  f"avg_reward/ep {avg_r:6.2f} | "
                  f"policy_loss {pl:.4f} | "
                  f"value_loss {vl:.4f} | "
                  f"entropy {el:.4f}")

    # ── Shutdown ───────────────────────────────────────────────────────────────
    for conn in parent_conns:
        conn.send(('stop', None))
    for w in workers:
        w.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()