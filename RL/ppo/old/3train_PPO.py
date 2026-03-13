import sys, os, time, random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm

sys.path.insert(0, r"C:\Users\laure\1own_projects\1polytopia_score")

from env.wrapper import EnvWrapper
from game.enums import BoardType, Tribes
from RL.ppo.ThrBLpolicy import PolicyNetwork, model_summary


# ── Value-return normaliser ────────────────────────────────────────────────────

class RunningMeanStd:
    """
    Welford online algorithm for tracking the running mean and variance of
    scalar return values.  Used to normalise targets for the value loss so
    that the critic always operates on a roughly unit-scale signal regardless
    of reward magnitude or discount horizon.
    """
    def __init__(self, epsilon: float = 1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon   # initialise with small count to avoid div-by-zero

    def update(self, x: np.ndarray) -> None:
        x            = np.asarray(x, dtype=np.float64).ravel()
        batch_mean   = x.mean()
        batch_var    = x.var()
        batch_count  = x.size
        delta        = batch_mean - self.mean
        tot_count    = self.count + batch_count
        self.mean    = self.mean + delta * batch_count / tot_count
        M2           = (self.var   * self.count
                        + batch_var * batch_count
                        + delta**2  * self.count * batch_count / tot_count)
        self.var     = M2 / tot_count
        self.count   = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# ── Checkpoint bookkeeping ─────────────────────────────────────────────────────
MAX_CKPT_KEEP = 3
CKPT_DIR      = "checkpoints_new"


def _save_checkpoint(policy: PolicyNetwork, update: int,
                     ckpt_queue: deque) -> None:
    """
    Save state_dict only (no cfg) and evict the oldest checkpoint when
    the rolling window is full.
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
    n_processes:         int   = 8
    n_envs_per_process:  int   = 4

    # --- Environment ---
    # board_size is chosen randomly per episode; only type and player count
    # are fixed here.
    board_config_dict = {
        "board_type" : BoardType.Dummy,
        "n_players"  : 2,
    }
    player_tribes      = [Tribes.Omaji, Tribes.Imperius]
    max_turns_per_game = 100

    # --- Board size curriculum: uniform random square in [min, max] ──────────
    board_size_range:    tuple = (10, 16)

    # --- Rollout ---
    n_steps:             int   = 256

    # --- PPO epochs & batching ---
    n_epochs:            int   = 2
    n_minibatches:       int   = 4

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
    n_updates:           int   = 1000
    log_interval:        int   = 1

    # ── Model architecture ────────────────────────────────────────────────────
    mpnn_hidden_dim:     int   = 128    # GraphSAGE hidden size
    mlp_hidden_dim:      int   = 64   # MLP hidden layer width
    mlp_depth:           int   = 3     # number of hidden layers in every MLP

    @property
    def n_envs_total(self):
        return self.n_processes * self.n_envs_per_process

    @property
    def batch_size(self):
        return self.n_steps * self.n_envs_total

    @property
    def minibatch_size(self):
        return self.batch_size // self.n_minibatches


# ── Environment helpers ────────────────────────────────────────────────────────

def _random_board_size(cfg: PPOConfig) -> tuple:
    """Sample a random square board size within the configured range."""
    n = random.randint(cfg.board_size_range[0], cfg.board_size_range[1])
    return (n, n)


def _make_env(cfg: PPOConfig) -> EnvWrapper:
    """Create a new EnvWrapper with a freshly sampled random board size."""
    board_size = _random_board_size(cfg)
    board_config = {
        "board_size" : list(board_size),
        "board_type" : cfg.board_config_dict["board_type"],
        "n_players"  : cfg.board_config_dict["n_players"],
    }
    return EnvWrapper(board_config, cfg.player_tribes,
                      max_turns_per_game=cfg.max_turns_per_game,
                      dense_reward=True)


# ── Obs snapshot ───────────────────────────────────────────────────────────────

def _snapshot(obs: dict, env: EnvWrapper) -> dict:
    """
    Called BEFORE env.step() so that tile IDs and graph are captured at
    decision time, not post-step (game objects are mutable in-place).

    Board dimensions are taken from the env so that evaluate_actions can
    reconstruct the correct edge topology and positional encoding.
    """
    return {
        'graph'     : np.asarray(obs['partial_graph']).copy(),
        'unit_ids'  : [u.tile.id for u in obs['units']],
        'enemy_ids' : [u.tile.id for u in obs['enemy_units']],
        'city_ids'  : [c.tile_id for c in obs['cities']],
        'Nx'        : env.Nx,
        'Ny'        : env.Ny,
        'player_id' : env.game.player_go_id,
    }


# ── Worker process — CPU only ──────────────────────────────────────────────────

def worker_fn(worker_id: int, cfg: PPOConfig, conn):
    """
    Collects T-step rollouts across M environments.

    Each environment is recreated with a new random board size whenever an
    episode ends, providing natural curriculum diversity.

    Extra arrays sent back:
      won_flags  (T, M)  float32 — 1 if the episode ended by conquest
                                   (winner != None), 0 otherwise.
    """
    envs    = [_make_env(cfg) for _ in range(cfg.n_envs_per_process)]
    obs_buf = [env.reset() for env in envs]

    policy  = PolicyNetwork(cfg)   # CPU — no .to(device)
    policy.eval()

    M = cfg.n_envs_per_process
    T = cfg.n_steps

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

                    snap                       = _snapshot(obs, env)
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
                        # Recreate env with a new random board size for next episode
                        envs[e]    = _make_env(cfg)
                        obs_buf[e] = envs[e].reset()
                    else:
                        obs_buf[e] = next_obs

            # Bootstrap — use current env's board size for each slot
            last_values = np.zeros(M, dtype=np.float32)
            for e in range(M):
                snap_last      = _snapshot(obs_buf[e], envs[e])
                node_emb       = policy._encode(snap_last['graph'],
                                                snap_last['Nx'], snap_last['Ny'])
                last_values[e] = policy.critic_head(
                    node_emb.mean(dim=0, keepdim=True)).item()
                
        player_ids = np.array([[obs_snaps[t][e]['player_id'] for e in range(M)] for t in range(T)], dtype=np.int32)

        elapsed_total = time.time() - t0
        print(f"  [worker {worker_id:02d}] rollout done — {elapsed_total:.2f}s", flush=True)

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
            "player_ids" : player_ids,
        }))


# ── GAE ────────────────────────────────────────────────────────────────────────

def compute_gae(rewards, values, dones, last_values, gamma, gae_lam, player_ids):
    T, N       = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    gae        = np.zeros(N,      dtype=np.float32)

    for t in reversed(range(T)):
        next_val = last_values if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        if t < T - 1:
            player_switched = (player_ids[t] != player_ids[t + 1]).astype(np.float32)
            not_done        = not_done * (1.0 - player_switched)

        delta    = rewards[t] + gamma * next_val * not_done - values[t]
        gae      = delta + gamma * gae_lam * not_done * gae
        advantages[t] = gae

    return advantages, advantages + values


# ── PPO update — GPU ───────────────────────────────────────────────────────────

def ppo_update(policy: PolicyNetwork, optimizer, batch: dict,
               cfg: PPOConfig, device: torch.device,
               ret_normalizer: RunningMeanStd):
    """
    PPO update with two additions:
      • Value-function normalisation: returns are normalised with a running
        mean/std before computing the MSE value loss.
      • Per-epoch return recomputation: at the start of every epoch, fresh
        value estimates are computed with torch.no_grad() and GAE is re-run.
        This has been shown to reduce variance and improve sample efficiency.
    """
    B = cfg.batch_size

    flat_snaps = [s for step in batch['obs_snaps'] for s in step]
    flat_acts  = [a for step in batch['actions']   for a in step]
    flat_masks = [m for step in batch['masks']      for m in step]

    log_old = torch.tensor(batch['log_probs'].reshape(-1),
                           dtype=torch.float32).to(device)

    # ── Seed advantages/returns from the rollout-time GAE ─────────────────────
    # These will be replaced at the start of every epoch below.
    adv_np = batch['advantages'].reshape(-1).astype(np.float32)
    ret_np = batch['returns'].reshape(-1).astype(np.float32)

    # Update running return statistics once per ppo_update call (before epochs).
    ret_normalizer.update(ret_np)

    indices = np.arange(B)
    pl_log, vl_log, el_log = [], [], []

    total_steps = cfg.n_epochs * cfg.n_minibatches
    pbar = tqdm(total=total_steps, desc="  PPO epochs × minibatches",
                leave=False, unit="mb")

    for epoch in range(cfg.n_epochs):

        # ── Recompute returns at the start of every epoch ─────────────────────
        # Fresh critic values → re-run GAE → re-normalise advantages.
        # last_values stay fixed (bootstrap obs not stored) which is a safe
        # approximation; only the T in-trajectory values are refreshed.
        with torch.no_grad():
            fresh_vals_chunks = []
            for start in range(0, B, cfg.minibatch_size):
                mb_snaps = flat_snaps[start : start + cfg.minibatch_size]
                fresh_vals_chunks.append(
                    policy.compute_values_batch(mb_snaps).cpu().numpy()
                )
            fresh_vals_flat = np.concatenate(fresh_vals_chunks)   # (B,)

        # Reshape to (T, N_total) that compute_gae expects
        N_total       = cfg.n_envs_total
        fresh_vals_2d = fresh_vals_flat.reshape(cfg.n_steps, N_total)

        adv_2d, ret_2d = compute_gae(
            batch['rewards'], fresh_vals_2d, batch['dones'],
            batch['last_values'], cfg.gamma, cfg.gae_lambda,
            batch['player_ids'],
        )
        adv_np = adv_2d.reshape(-1).astype(np.float32)
        ret_np = ret_2d.reshape(-1).astype(np.float32)

        adv = torch.tensor(adv_np, dtype=torch.float32).to(device)
        ret = torch.tensor(ret_np, dtype=torch.float32).to(device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Normalised return targets for the value loss
        ret_norm = torch.tensor(
            ret_normalizer.normalize(ret_np),
            dtype=torch.float32
        ).to(device)

        np.random.shuffle(indices)

        for start in range(0, B, cfg.minibatch_size):
            mb = indices[start : start + cfg.minibatch_size]

            new_lp, new_ent, new_val = policy.evaluate_actions(
                [flat_snaps[i] for i in mb],
                [flat_acts[i]  for i in mb],
                [flat_masks[i] for i in mb],
            )

            ratio     = torch.exp(new_lp - log_old[mb])
            mb_adv    = adv[mb]

            loss_clip = torch.min(
                ratio * mb_adv,
                ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
            ).mean()

            # Value loss against normalised returns
            loss_val  = F.mse_loss(new_val.squeeze(), ret_norm[mb])
            loss_ent  = new_ent.mean()

            loss = -loss_clip + cfg.vf_coef * loss_val - cfg.ent_coef * loss_ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            pl_log.append(loss_clip.item())
            vl_log.append(loss_val.item())
            el_log.append(loss_ent.item())

            pbar.set_postfix(epoch=epoch + 1,
                             p_loss=f"{loss_clip.item():.4f}",
                             v_loss=f"{loss_val.item():.4f}")
            pbar.update(1)

    pbar.close()
    return np.mean(pl_log), np.mean(vl_log), np.mean(el_log)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg    = PPOConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy    = PolicyNetwork(cfg).to(device)
    #pre_load_weights = torch.load(r'./checkpoints_new/policy_update_00042.pt', weights_only=True, map_location=device)
    #policy.load_state_dict(pre_load_weights)
    policy.train()
    optimizer     = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    ret_normalizer = RunningMeanStd()   # tracks running mean/var of returns

    # ── Startup banner ────────────────────────────────────────────────────────
    lo, hi = cfg.board_size_range
    print("\n" + "=" * 60)
    print("  POLYTOPIA RL — PPO TRAINING")
    print("=" * 60)
    print(f"  Device          : {device}")
    print(f"  Board size      : random square [{lo}×{lo} … {hi}×{hi}]")
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

    # ── Spawn workers ─────────────────────────────────────────────────────────
    parent_conns, child_conns = zip(*[mp.Pipe() for _ in range(cfg.n_processes)])
    workers = [
        mp.Process(target=worker_fn, args=(i, cfg, child_conns[i]), daemon=True)
        for i in range(cfg.n_processes)
    ]
    for w in workers:
        w.start()

    ckpt_queue = deque()   # rolling checkpoint tracker

    # ── Outer update loop ─────────────────────────────────────────────────────
    outer_bar = tqdm(range(0, cfg.n_updates), desc="Updates", unit="upd")

    for update in outer_bar:
        t_update_start = time.time()

        # ── 1. Distribute weights ─────────────────────────────────────────
        t0 = time.time()
        state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
        for conn in parent_conns:
            conn.send(('collect', state_dict))
        t_dist = time.time() - t0
        print(f"\n[update {update:04d}] weights dispatched in {t_dist:.3f}s")

        # ── 2. Collect rollouts ───────────────────────────────────────────
        t0 = time.time()
        print(f"[update {update:04d}] waiting for {cfg.n_processes} workers …")
        chunks = []
        for i, conn in enumerate(tqdm(parent_conns,
                                       desc="  collecting workers",
                                       leave=False, unit="worker")):
            chunks.append(conn.recv()[1])
        t_collect = time.time() - t0
        print(f"[update {update:04d}] all workers done in {t_collect:.2f}s")

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
            'player_ids':  np.concatenate([c['player_ids']  for c in chunks], axis=1),
        }

        # ── 4. GAE ────────────────────────────────────────────────────────
        adv, ret = compute_gae(
            batch['rewards'], batch['values'], batch['dones'],
            batch['last_values'], cfg.gamma, cfg.gae_lambda,
            batch["player_ids"],
        )
        batch['advantages'] = adv
        batch['returns']    = ret
        t_gae = time.time() - t0
        print(f"[update {update:04d}] GAE computed in {t_gae:.3f}s")

        # ── 5. PPO update (GPU) ───────────────────────────────────────────
        t0 = time.time()
        pl, vl, el = ppo_update(policy, optimizer, batch, cfg, device, ret_normalizer)
        t_ppo = time.time() - t0
        print(f"[update {update:04d}] PPO update done in {t_ppo:.2f}s")

        # ── 6. Checkpoint ─────────────────────────────────────────────────
        _save_checkpoint(policy, update, ckpt_queue)

        # ── 7. Stats & logging ────────────────────────────────────────────
        n_finished = int(batch['dones'].sum())
        n_won      = int(batch['won_flags'].sum())
        n_timeout  = n_finished - n_won

        total_reward = batch['rewards'].sum()
        avg_r        = total_reward / max(n_finished, 1)
        win_rate     = n_won / max(n_finished, 1)
        avg_ep_len   = (cfg.batch_size / max(n_finished, 1))

        t_total = time.time() - t_update_start

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

    # ── Final save (always kept regardless of rolling window) ────────────────
    final_path = os.path.join(CKPT_DIR, "policy_FINAL.pt")
    torch.save(policy.state_dict(), final_path)
    print(f"\nFinal policy saved → {final_path}")

    # ── Shutdown workers ──────────────────────────────────────────────────────
    for conn in parent_conns:
        conn.send(('stop', None))
    for w in workers:
        w.join()
    print("All workers shut down. Training complete.")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()