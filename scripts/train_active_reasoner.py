"""Train the Active Reasoner policy on the HypothesisSearchEnv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
import torch.nn.functional as F
import yaml

from training.reasoner import ActiveReasonerPolicy, ActiveReasonerPolicyConfig, HypothesisSearchEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Training config YAML")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/active_reasoner"),
        help="Where to store metrics/checkpoints",
    )
    parser.add_argument("--iterations", type=int, default=None, help="Override iteration count")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("config must be a mapping")
    return dict(data)


def obs_to_tensors(obs: Mapping[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "current_latent": torch.from_numpy(obs["current_latent"]).to(device),
        "target_latent": torch.from_numpy(obs["target_latent"]).to(device),
        "program_ids": torch.from_numpy(obs["program_ids"]).to(device),
        "program_params": torch.from_numpy(obs["program_params"]).to(device),
        "program_mask": torch.from_numpy(obs["program_mask"]).to(device),
        "steps": torch.from_numpy(obs["steps"]).to(device),
    }


def stack_observations(batch: Iterable[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    stacked: Dict[str, list[torch.Tensor]] = {}
    for obs in batch:
        for key, tensor in obs.items():
            stacked.setdefault(key, []).append(tensor)
    return {key: torch.cat(tensors, dim=0) for key, tensors in stacked.items()}


def rollout_episode(env: HypothesisSearchEnv, policy: ActiveReasonerPolicy, *, gamma: float, device: torch.device):
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs, _ = reset_out
    else:
        obs = reset_out
    transitions = []
    done = False
    episode_reward = 0.0
    steps = 0
    last_info: Mapping[str, Any] = {}

    while not done:
        obs_tensor = obs_to_tensors(obs, device)
        action, log_prob, value, _ = policy.act(obs_tensor)
        step_out = env.step(action.item())
        if len(step_out) == 5:
            next_obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_out
        transitions.append(
            {
                "obs": obs_tensor,
                "action": action,
                "log_prob": log_prob,
                "value": value.squeeze(-1),
                "reward": torch.tensor([float(reward)], device=device),
            }
        )
        obs = next_obs
        episode_reward += float(reward)
        steps += 1
        last_info = info

    returns = []
    ret = torch.zeros(1, device=device)
    for transition in reversed(transitions):
        ret = transition["reward"] + gamma * ret
        returns.insert(0, ret.clone())
    for transition, target in zip(transitions, returns):
        transition["return"] = target.squeeze(-1)

    info = {
        "reward": episode_reward,
        "length": steps,
        "success": bool(last_info.get("success")),
    }
    return transitions, info


def train_active_reasoner(
    env: HypothesisSearchEnv,
    policy: ActiveReasonerPolicy,
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    iterations_override: int | None = None,
) -> None:
    trainer_cfg = dict(config.get("trainer", {}))
    device = next(policy.parameters()).device
    gamma = float(trainer_cfg.get("gamma", 0.95))
    entropy_coeff = float(trainer_cfg.get("entropy_coeff", 0.01))
    value_coeff = float(trainer_cfg.get("value_coeff", 0.5))
    lr = float(trainer_cfg.get("lr", 3e-4))
    max_grad_norm = float(trainer_cfg.get("max_grad_norm", 1.0))
    episodes_per_iter = int(trainer_cfg.get("episodes_per_iter", 32))
    iterations = iterations_override or int(trainer_cfg.get("iterations", 50))

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    metrics_path = output_dir / "metrics.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for iteration in range(1, iterations + 1):
            batch_transitions = []
            rewards = []
            lengths = []
            successes = 0
            sigreg_sums = []

            for _ in range(episodes_per_iter):
                transitions, info = rollout_episode(env, policy, gamma=gamma, device=device)
                batch_transitions.extend(transitions)
                rewards.append(info["reward"])
                lengths.append(info["length"])
                successes += 1 if info["success"] else 0
                if info.get("sigreg_penalty") is not None:
                    sigreg_sums.append(float(info["sigreg_penalty"]))

            batch_obs = stack_observations(transition["obs"] for transition in batch_transitions)
            actions = torch.cat([t["action"] for t in batch_transitions], dim=0)
            returns = torch.stack([t["return"] for t in batch_transitions], dim=0).to(device)

            optimizer.zero_grad()
            logits, values = policy(batch_obs)
            distribution = torch.distributions.Categorical(logits=logits)
            log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()

            advantages = returns - values.detach()
            actor_loss = -(log_probs * advantages).mean()
            value_loss = F.mse_loss(values, returns)
            loss = actor_loss + value_coeff * value_loss - entropy_coeff * entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            summary = {
                "iteration": iteration,
                "reward_mean": float(sum(rewards) / max(len(rewards), 1)),
                "success_rate": float(successes / max(len(rewards), 1)),
                "avg_chain_length": float(sum(lengths) / max(len(lengths), 1)),
                "actor_loss": float(actor_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
            }
            if sigreg_sums:
                summary["sigreg_penalty_mean"] = float(sum(sigreg_sums) / len(sigreg_sums))
            metrics_file.write(json.dumps(summary) + "\n")
            metrics_file.flush()
            print(
                f"[iter {iteration}] reward={summary['reward_mean']:.3f} "
                f"success={summary['success_rate']:.2f} steps={summary['avg_chain_length']:.2f}"
            )

    checkpoint = {
        "policy_state": policy.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, output_dir / "policy.pt")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    env_cfg = dict(config.get("env", {}))
    env = HypothesisSearchEnv(env_cfg)

    policy_cfg_data = dict(config.get("policy", {}))
    policy_cfg = ActiveReasonerPolicyConfig(
        latent_dim=env.latent_dim,
        action_dim=env.action_space.n,
        program_embedding_dim=int(policy_cfg_data.get("program_embedding_dim", 128)),
        program_hidden_dim=int(policy_cfg_data.get("program_hidden_dim", 128)),
        program_layers=int(policy_cfg_data.get("program_layers", 1)),
        dropout=float(policy_cfg_data.get("dropout", 0.1)),
        hidden_dims=tuple(policy_cfg_data.get("hidden_dims", (512, 256))),
        activation=str(policy_cfg_data.get("activation", "relu")),
    )

    device = torch.device(policy_cfg_data.get("device") or env.device)
    policy = ActiveReasonerPolicy(
        env.vocab_size,
        env.parameter_dim,
        config=policy_cfg,
    ).to(device)

    train_active_reasoner(
        env,
        policy,
        config=config,
        output_dir=args.output_dir,
        iterations_override=args.iterations,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
