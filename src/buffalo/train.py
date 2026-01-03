"""Training loop for a Buffalo Q-network using saved game CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import torch
from torch.utils.data import DataLoader

from .dataloader import BuffaloGameDataset
from .encoders import BoardStateEncoder
from .models import BuffaloQNetwork


def _max_next_q(
    model: BuffaloQNetwork,
    next_states: torch.Tensor,
    action_size: int,
) -> torch.Tensor:
    batch_size = next_states.size(0)
    actions = torch.eye(action_size, device=next_states.device)
    expanded_states = next_states.unsqueeze(1).expand(batch_size, action_size, -1)
    expanded_actions = actions.unsqueeze(0).expand(batch_size, action_size, -1)
    flat_states = expanded_states.reshape(-1, next_states.size(1))
    flat_actions = expanded_actions.reshape(-1, action_size)
    q_values = model(flat_states, flat_actions).view(batch_size, action_size)
    return q_values.max(dim=1).values


def train(
    data_dir: str | Path,
    save_path: str | Path,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    gamma: float = 0.99,
    device: Optional[str] = None,
) -> BuffaloQNetwork:
    encoder = BoardStateEncoder()
    dataset = BuffaloGameDataset(data_dir, encoder=encoder)
    loader = DataLoader(dataset, batch_size=batch_size)

    torch_device = torch.device(device) if device else None
    model = BuffaloQNetwork(encoder.state_size, encoder.buffalo_action_size)
    if torch_device:
        model.to(torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for _ in range(epochs):
        for state, action, reward, next_state in loader:
            if torch_device:
                state = state.to(torch_device)
                action = action.to(torch_device)
                reward = reward.to(torch_device)
                next_state = next_state.to(torch_device)

            reward = reward.float()
            q_pred = model(state, action)
            with torch.no_grad():
                max_next_q = _max_next_q(model, next_state, encoder.buffalo_action_size)
                target = reward + gamma * max_next_q

            loss = loss_fn(q_pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    save_payload = {
        "state_dict": model.state_dict(),
        "state_size": encoder.state_size,
        "action_size": encoder.buffalo_action_size,
    }
    torch.save(save_payload, Path(save_path))
    return model


@click.command()
@click.option("--data-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--save-path", type=click.Path(path_type=Path), required=True)
@click.option("--epochs", type=int, default=5, show_default=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--gamma", type=float, default=0.99, show_default=True)
@click.option("--device", type=str, default=None)
def main(
    data_dir: Path,
    save_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    gamma: float,
    device: Optional[str],
) -> None:
    train(
        data_dir=data_dir,
        save_path=save_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        device=device,
    )


if __name__ == "__main__":
    main()
