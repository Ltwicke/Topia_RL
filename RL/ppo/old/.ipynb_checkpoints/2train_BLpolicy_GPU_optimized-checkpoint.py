"""
train_BLpolicy.py
────────────────────────────────────────────────────────────────────────────────
Parallel PPO training for the Polytopia RL project.

GPU-efficiency changes
───────────────────────
• ppo_update() now pre-converts the flat snapshot list to a single stacked
  numpy array (B, N, 26) ONCE before the epoch loop, instead of re-stacking
  inside every evaluate_actions call.  This moves the data-prep cost out of
  the hot loop entirely.
• evaluate_actions() in BLpolicy.py receives these pre-stacked arrays and
  runs a single batched MPNN forward pass per minibatch (see that file for
  details).
• log_old / adv / ret are moved to GPU once, before the epoch loop.
"""

import sys, os, time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm

sys.path.insert(0, r"C:\Users\laure\1own_projects\1polytopia_score")

from env.wrapper import EnvWrapper
from game.enums import BoardType, Tribes
from RL.ppo.BLpolicy import PolicyNetwork, model_summary


# ── Checkpoint bookkeeping ─────────────────────────────────────────────────────
MAX_CKPT_KEEP = 3
CKPT_DIR      = "checkpoints"


def _save_checkpoint(policy: PolicyNetwork, update: int,
                     ckpt_queue: deque) -> None:
    """
    Persist state_dict (no cfg object) and maintain a rolling window of the
    last MAX_CKPT_KEEP files.
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"policy_update_{update:05d}.pt")
    torch.save(policy.state_dict(), path)
    ckpt_queue.append(path)
    if len(ckpt_queue) > MAX_CKPT_KEEP:
        oldest = ckpt_queue.popleft()
        if os.path.exists(oldest):
            os.remove(oldest)
            print(f"  [ckpt] removed old checkpoint: {oldest}")
    print(f"  [ckpt] saved → {path}  (keeping last {MAX_CKPT_KEEP})")


# ── Hyperparameters ────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:

    # --- Parallelism ---
    n_processes:         int   = 16
    n_envs_per_process:  int   = 8

    # --- Environment ---
    board_config_dict = {
        "board_size" : (10, 10),
        "board_type" : BoardType.Dummy,
        "n_players"  : 2,
    }
    player_tribes      = [Tribes.Omaji, Tribes.Imperius]
    max_turns_per_game = 100

    # --- Rollout ---
    n_steps:             int   = 256

    # --- PPO epochs & batching ---
    n_epochs:            int   = 8
    n_minibatches:       int   = 32

    # --- PPO loss coefficients ---
    clip_eps:            float = 0.2
    vf_coef:             float = 0.5
    ent_coef:            float = 0.01
    max_grad_norm:       float = 0.5

    # --- GAE / discount ---
    gamma:               float = 0.99
    gae_lambda:          float = 0.95

    # --- Optimizer ---
    lr:                  float = 3e-4

    # --- Training ---
    n_updates:           int   = 20
    log_interval:        int   = 1

    # --- Model architecture ---
    mpnn_hidden_dim:     int   = 64
    mlp_hidden_dim:      int   = 128
    mlp_depth:           int   = 3

    @property
    def n_envs_total(self):
        return self.n_processes * self.n_envs_per_process

    @property
    def batch_size(self):
        return self.n_steps * self.n_envs_total

    @property
    def board_size(self):
        return self.board_config_dict["board_size"]

    @property
    def minibatch_size(self):
        return self.batch_size // self.n_minibatches

    @property
    def board_config(self):
        return {
            "board_size" : list(self.board_config_dict["board_size"]),
            "board_type" : self.board_config_dict["board_type"],
            "n_players"  : self.board_config_dict["n_players"],
        }


# ── Obs snapshot ───────────────────────────────────────────────────────────────

def _snapshot(obs: dict) -> dict:
    """
    Captured BEFORE env.step() — game objects are mutated in-place, so we
    must copy graph data and resolve tile IDs immediately.
    """
    return {
        'graph'     : np.asarray(obs['partial_graph']).copy(),
        'unit_ids'  : [u.tile.id for u in obs['units']],
        'enemy_ids' : [u.tile.id for u in obs['enemy_units']],
        'city_ids'  : [c.tile_id for c in obs['cities']],
    }


# ── Worker process — CPU only ──────────────────────────────────────────────────

def worker_fn(worker_id: int, cfg: PPOConfig, conn):
    """
    Collects T-step rollouts across M environments on CPU.
    Prints a progress line at 25 / 50 / 75 / 100 % of each rollout so the
    main console stays alive without 16 competing tqdm bars.
    """
    envs    = [EnvWrapper(cfg.board_config, cfg.player_tribes,
                          max_turns_per_game=cfg.max_turns_per_game,
                          dense_reward=True)
               for _ in range(cfg.n_envs_per_process)]
    obs_buf = [env.reset() for env in envs]

    policy  = PolicyNetwork(cfg)   # CPU only
    policy.eval()

    M          = cfg.n_envs_per_process
    T          = cfg.n_steps
    checkpoints = {T // 4, T // 2, 3 * T // 4}   # progress print triggers

    while True:
        cmd, payload = conn.recv()
        if cmd == 'stop':
            break

        policy.load_state_dict(payload)

        obs_snaps = [[None] * M for _ in range(T)]
        actions   = [[None] * M for _ in range(T)]
        masks_buf = [[None] * M for _ in range(T)]
        log_probs = np.zeros((T, M), dtype=np.float32)
        values    = np.zeros((T, M), dtype=np.float32)
        rewards   = np.zeros((T, M), dtype=np.float32)
        dones     = np.zeros((T, M), dtype=np.float32)
        won_flags = np.zeros((T, M), dtype=np.float32)

        t0 = time.time()
        with torch.no_grad():
            for t in range(T):
                for e, env in enumerate(envs):
                    obs  = obs_buf[e]
                    mask = env.get_action_mask()

                    snap                       = _snapshot(obs)
                    action, lp, _, val         = policy(obs, mask)
                    next_obs, rew, done, _info = env.step(action)

                    obs_snaps[t][e] = snap
                    actions[t][e]   = action
                    masks_buf[t][e] = mask
                    log_probs[t, e] = lp.item()
                    values[t, e]    = val.item()
                    rewards[t, e]   = rew
                    dones[t, e]     = float(done)

                    if done:
                        won_flags[t, e] = float(env.winner is not None)
                        obs_buf[e]      = env.reset()
                    else:
                        obs_buf[e] = next_obs

                if t in checkpoints:
                    pct     = int(100 * t / T)
                    elapsed = time.time() - t0
                    print(f"  [worker {worker_id:02d}] {pct:3d}%  "
                          f"step {t}/{T}  ({elapsed:.1f}s)", flush=True)

            # Bootstrap final value
            last_values = np.zeros(M, dtype=np.float32)
            for e in range(M):
                snap_last      = _snapshot(obs_buf[e])
                node_emb       = policy._encode(snap_last['graph'])
                last_values[e] = policy.critic_head(
                    node_emb.mean(dim=0, keepdim=True)).item()

        elapsed_total = time.time() - t0
        print(f"  [worker {worker_id:02d}] done — {elapsed_total:.2f}s", flush=True)

        conn.send(('data', {
            'obs_snaps':   obs_snaps,
            'actions':     actions,
            'masks':       masks_buf,
            'log_probs':   log_probs,
            'values':      values,
            'rewards':     rewards,
            'dones':       dones,
            'won_flags':   won_flags,
            'last_values': last_values,
        }))


# ── GAE ────────────────────────────────────────────────────────────────────────

def compute_gae(rewards, values, dones, last_values, gamma, gae_lam):
    T, N       = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    gae        = np.zeros(N,      dtype=np.float32)
    for t in reversed(range(T)):
        next_val = last_values if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta    = rewards[t] + gamma * next_val * not_done - values[t]
        gae      = delta + gamma * gae_lam * not_done * gae
        advantages[t] = gae
    return advantages, advantages + values


# ── Batch pre-processing ───────────────────────────────────────────────────────

def _preprocess_batch(flat_snaps: list, flat_acts: list,
                      flat_masks: list, log_probs_np: np.ndarray,
                      adv_np: np.ndarray, ret_np: np.ndarray,
                      device: torch.device):
    """
    Convert the flat rollout lists into GPU tensors ONCE per PPO update,
    before the epoch × minibatch loop.

    Specifically:
      graphs_all  : (B, N, 26)  float32 numpy — stays on CPU until sliced
                                into minibatches inside evaluate_actions
      log_old     : (B,)  GPU tensor
      adv         : (B,)  GPU tensor  (normalised)
      ret         : (B,)  GPU tensor

    Returning graphs_all as numpy avoids a single monolithic (B*N, 26) GPU
    allocation which would be sliced wastefully; evaluate_actions receives the
    minibatch slice and uploads only what it needs.
    """
    B = len(flat_snaps)
    N = flat_snaps[0]['graph'].shape[0]

    # Stack graphs into a contiguous array — one allocation, no Python loop
    # at forward time
    graphs_all = np.stack([s['graph'] for s in flat_snaps])  # (B, N, 26)

    log_old = torch.tensor(log_probs_np, dtype=torch.float32, device=device)
    adv     = torch.tensor(adv_np,       dtype=torch.float32, device=device)
    ret     = torch.tensor(ret_np,       dtype=torch.float32, device=device)

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return graphs_all, log_old, adv, ret


# ── PPO update ─────────────────────────────────────────────────────────────────

def ppo_update(policy: PolicyNetwork, optimizer: torch.optim.Optimizer,
               batch: dict, cfg: PPOConfig, device: torch.device):
    """
    Runs cfg.n_epochs passes over the minibatch-shuffled rollout data.

    GPU-efficiency notes
    ─────────────────────
    • The flat snapshot / action / mask lists are assembled once outside the
      epoch loop.
    • graphs_all (B, N, 26) is pre-stacked in numpy.  Each minibatch slice
      graphs_all[mb] is passed directly to evaluate_actions, which uploads it
      in one torch.tensor() call and runs a single MPNN pass.
    • log_old / adv / ret live on GPU for the full duration of ppo_update.
    """
    B = cfg.batch_size

    # ── Flatten rollout lists ──────────────────────────────────────────────
    t0 = time.time()
    flat_snaps = [s for step in batch['obs_snaps'] for s in step]
    flat_acts  = [a for step in batch['actions']   for a in step]
    flat_masks = [m for step in batch['masks']      for m in step]

    graphs_all, log_old, adv, ret = _preprocess_batch(
        flat_snaps, flat_acts, flat_masks,
        batch['log_probs'].reshape(-1),
        batch['advantages'].reshape(-1),
        batch['returns'].reshape(-1),
        device
    )
    print(f"    [ppo] batch prep : {time.time() - t0:.3f}s", flush=True)

    indices = np.arange(B)
    pl_log, vl_log, el_log = [], [], []

    total_steps = cfg.n_epochs * cfg.n_minibatches
    pbar = tqdm(total=total_steps, desc="  PPO epochs × minibatches",
                leave=False, unit="mb")

    for epoch in range(cfg.n_epochs):
        np.random.shuffle(indices)
        t_epoch = time.time()

        for start in range(0, B, cfg.minibatch_size):
            mb = indices[start : start + cfg.minibatch_size]

            # Build minibatch snapshot list — graph slice is a view into the
            # pre-stacked numpy array, so no extra copy occurs here
            mb_snaps = [{'graph'     : graphs_all[i],
                         'unit_ids'  : flat_snaps[i]['unit_ids'],
                         'enemy_ids' : flat_snaps[i]['enemy_ids'],
                         'city_ids'  : flat_snaps[i]['city_ids']}
                        for i in mb]
            mb_acts  = [flat_acts[i]  for i in mb]
            mb_masks = [flat_masks[i] for i in mb]

            new_lp, new_ent, new_val = policy.evaluate_actions(
                mb_snaps, mb_acts, mb_masks)

            ratio     = torch.exp(new_lp - log_old[mb])
            mb_adv    = adv[mb]

            loss_clip = torch.min(
                ratio * mb_adv,
                ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
            ).mean()

            loss_val  = F.mse_loss(new_val.squeeze(), ret[mb])
            loss_ent  = new_ent.mean()

            loss = -loss_clip + cfg.vf_coef * loss_val - cfg.ent_coef * loss_ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            pl_log.append(loss_clip.item())
            vl_log.append(loss_val.item())
            el_log.append(loss_ent.item())

            pbar.set_postfix(ep=epoch + 1,
                             p=f"{loss_clip.item():.4f}",
                             v=f"{loss_val.item():.4f}",
                             ent=f"{loss_ent.item():.4f}")
            pbar.update(1)

        print(f"    [ppo] epoch {epoch+1}/{cfg.n_epochs} — "
              f"{time.time() - t_epoch:.2f}s", flush=True)

    pbar.close()
    return np.mean(pl_log), np.mean(vl_log), np.mean(el_log)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg    = PPOConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy    = PolicyNetwork(cfg).to(device)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    # ── Startup banner ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  POLYTOPIA RL — PPO TRAINING")
    print("=" * 60)
    print(f"  Device          : {device}")
    print(f"  Batch size      : {cfg.batch_size:,}  "
          f"({cfg.n_processes} proc × {cfg.n_envs_per_process} envs "
          f"× {cfg.n_steps} steps)")
    print(f"  Minibatch size  : {cfg.minibatch_size:,}  |  "
          f"Epochs: {cfg.n_epochs}  |  Updates: {cfg.n_updates}")
    print(f"  MLP depth/width : {cfg.mlp_depth} hidden layers × {cfg.mlp_hidden_dim}")
    print(f"  MPNN hidden     : {cfg.mpnn_hidden_dim}")
    print()
    model_summary(policy)
    print()

    # ── Spawn workers ─────────────────────────────────────────────────────
    parent_conns, child_conns = zip(*[mp.Pipe() for _ in range(cfg.n_processes)])
    workers = [
        mp.Process(target=worker_fn, args=(i, cfg, child_conns[i]), daemon=True)
        for i in range(cfg.n_processes)
    ]
    for w in workers:
        w.start()

    ckpt_queue = deque()

    # ── Outer update loop ─────────────────────────────────────────────────
    outer_bar = tqdm(range(cfg.n_updates), desc="Updates", unit="upd")

    for update in outer_bar:
        t_update_start = time.time()

        # ── 1. Distribute weights ─────────────────────────────────────────
        t0 = time.time()
        state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
        for conn in parent_conns:
            conn.send(('collect', state_dict))
        t_dist = time.time() - t0
        print(f"\n[update {update:04d}] weights dispatched  ({t_dist:.3f}s)")

        # ── 2. Collect rollouts ───────────────────────────────────────────
        t0 = time.time()
        print(f"[update {update:04d}] waiting for {cfg.n_processes} workers …")
        chunks = []
        for conn in tqdm(parent_conns, desc="  collecting workers",
                         leave=False, unit="worker"):
            chunks.append(conn.recv()[1])
        t_collect = time.time() - t0
        print(f"[update {update:04d}] all workers done  ({t_collect:.2f}s)")

        # ── 3. Assemble batch ─────────────────────────────────────────────
        t0 = time.time()
        batch = {
            'obs_snaps': [sum([c['obs_snaps'][t] for c in chunks], [])
                          for t in range(cfg.n_steps)],
            'actions':   [sum([c['actions'][t]   for c in chunks], [])
                          for t in range(cfg.n_steps)],
            'masks':     [sum([c['masks'][t]      for c in chunks], [])
                          for t in range(cfg.n_steps)],
            'log_probs':   np.concatenate([c['log_probs']   for c in chunks], axis=1),
            'values':      np.concatenate([c['values']      for c in chunks], axis=1),
            'rewards':     np.concatenate([c['rewards']     for c in chunks], axis=1),
            'dones':       np.concatenate([c['dones']       for c in chunks], axis=1),
            'won_flags':   np.concatenate([c['won_flags']   for c in chunks], axis=1),
            'last_values': np.concatenate([c['last_values'] for c in chunks]),
        }

        # ── 4. GAE ────────────────────────────────────────────────────────
        adv, ret = compute_gae(
            batch['rewards'], batch['values'], batch['dones'],
            batch['last_values'], cfg.gamma, cfg.gae_lambda
        )
        batch['advantages'] = adv
        batch['returns']    = ret
        t_gae = time.time() - t0
        print(f"[update {update:04d}] batch assembled + GAE  ({t_gae:.3f}s)")

        # ── 5. PPO update ─────────────────────────────────────────────────
        t0 = time.time()
        print(f"[update {update:04d}] starting PPO update …")
        pl, vl, el = ppo_update(policy, optimizer, batch, cfg, device)
        t_ppo = time.time() - t0
        print(f"[update {update:04d}] PPO update done  ({t_ppo:.2f}s)")

        # ── 6. Checkpoint ─────────────────────────────────────────────────
        _save_checkpoint(policy, update, ckpt_queue)

        # ── 7. Stats & logging ────────────────────────────────────────────
        n_finished   = int(batch['dones'].sum())
        n_won        = int(batch['won_flags'].sum())
        n_timeout    = n_finished - n_won
        win_rate     = n_won     / max(n_finished, 1)
        avg_r        = batch['rewards'].sum() / max(n_finished, 1)
        avg_ep_len   = cfg.batch_size         / max(n_finished, 1)
        t_total      = time.time() - t_update_start

        if update % cfg.log_interval == 0:
            print()
            print(f"╔══ update {update:04d} ═══════════════════════════════════════")
            print(f"║  Wall time     : {t_total:.1f}s  "
                  f"(dist {t_dist:.2f}s | collect {t_collect:.2f}s "
                  f"| GAE {t_gae:.3f}s | PPO {t_ppo:.2f}s)")
            print(f"║  Games finished: {n_finished:6d}  "
                  f"(conquest {n_won}, timeout {n_timeout})")
            print(f"║  Win rate      : {win_rate:.3f}")
            print(f"║  Avg ep length : {avg_ep_len:.1f} steps")
            print(f"║  Avg reward/ep : {avg_r:.3f}")
            print(f"║  p_loss        : {pl:.4f}")
            print(f"║  v_loss        : {vl:.4f}")
            print(f"║  entropy       : {el:.4f}")
            print(f"╚{'═' * 52}")
            print()

        outer_bar.set_postfix(
            fin=n_finished, win=f"{win_rate:.2f}",
            p=f"{pl:.3f}", v=f"{vl:.3f}", ent=f"{el:.3f}"
        )

    # ── Final save ────────────────────────────────────────────────────────
    final_path = os.path.join(CKPT_DIR, "policy_FINAL.pt")
    torch.save(policy.state_dict(), final_path)
    print(f"\nFinal policy saved → {final_path}")

    # ── Shutdown workers ──────────────────────────────────────────────────
    for conn in parent_conns:
        conn.send(('stop', None))
    for w in workers:
        w.join()
    print("All workers shut down.  Training complete.")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()