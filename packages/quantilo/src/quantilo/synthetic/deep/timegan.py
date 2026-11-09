"""TimeGAN (Yoon, Jarrett, van der Schaar, NeurIPS 2019), written from the paper.

Five GRU networks: embedder (real -> latent), recovery (latent -> real), generator (noise ->
latent), supervisor (latent at t -> latent at t + 1) and discriminator (latent -> real or not).
Three phases: autoencoder, supervised dynamics, then joint adversarial training with moment
matching.
"""

from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional

BATCH = 128
GAMMA = 1.0
LEARNING_RATE = 1e-3
DISCRIMINATOR_THRESHOLD = 0.15

Progress = Callable[[str, int, int], None]  # phase, step, steps of the phase
StopCheck = Callable[[], bool]


class Recurrent(nn.Module):
    def __init__(
        self, size_in: int, hidden: int, size_out: int, layers: int, sigmoid: bool = True
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(size_in, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, size_out)
        self.sigmoid = sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(x)
        out = self.out(hidden)
        return torch.sigmoid(out) if self.sigmoid else out


def hidden_size(width: int) -> int:
    return int(min(64, max(24, 2 * width)))


class TimeGAN:
    def __init__(self, width: int, device: torch.device, seed: int, layers: int = 3) -> None:
        torch.manual_seed(seed)
        self.width = width
        self.device = device
        hidden = hidden_size(width)
        self.embedder = Recurrent(width, hidden, hidden, layers).to(device)
        self.recovery = Recurrent(hidden, hidden, width, layers).to(device)
        self.generator = Recurrent(width, hidden, hidden, layers).to(device)
        self.supervisor = Recurrent(hidden, hidden, hidden, max(1, layers - 1)).to(device)
        self.discriminator = Recurrent(hidden, hidden, 1, layers, sigmoid=False).to(device)
        self.generator_rng = torch.Generator().manual_seed(seed)

    def _batch(self, data: torch.Tensor) -> torch.Tensor:
        index = torch.randint(0, len(data), (min(BATCH, len(data)),), generator=self.generator_rng)
        return data[index.to(self.device)]

    def _noise(self, count: int, length: int) -> torch.Tensor:
        return torch.rand(count, length, self.width, generator=self.generator_rng).to(self.device)

    def fit(
        self, windows: np.ndarray, steps: int, progress: Progress, should_stop: StopCheck
    ) -> bool:
        """Train; False if stopped."""
        data = torch.from_numpy(windows).to(self.device)
        length = data.shape[1]
        every = max(1, steps // 50)
        autoencoder = torch.optim.Adam(
            [*self.embedder.parameters(), *self.recovery.parameters()], lr=LEARNING_RATE
        )
        supervised = torch.optim.Adam(
            [*self.generator.parameters(), *self.supervisor.parameters()], lr=LEARNING_RATE
        )
        critic = torch.optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE)

        # 1. autoencoder
        for step in range(steps):
            if should_stop():
                return False
            x = self._batch(data)
            loss = 10 * torch.sqrt(functional.mse_loss(self.recovery(self.embedder(x)), x))
            autoencoder.zero_grad(set_to_none=True)
            loss.backward()
            autoencoder.step()
            if (step + 1) % every == 0 or step + 1 == steps:
                progress("Autoencoder", step + 1, steps)

        # 2. supervised dynamics: latent at t predicts latent at t + 1
        for step in range(steps):
            if should_stop():
                return False
            h = self.embedder(self._batch(data)).detach()
            loss = functional.mse_loss(self.supervisor(h)[:, :-1], h[:, 1:])
            supervised.zero_grad(set_to_none=True)
            loss.backward()
            supervised.step()
            if (step + 1) % every == 0 or step + 1 == steps:
                progress("Supervisor", step + 1, steps)

        # 3. joint training
        ones = lambda t: torch.ones_like(t)  # noqa: E731
        zeros = lambda t: torch.zeros_like(t)  # noqa: E731
        bce = functional.binary_cross_entropy_with_logits
        for step in range(steps):
            if should_stop():
                return False
            for _ in range(2):
                x = self._batch(data)
                e_hat = self.generator(self._noise(len(x), length))
                h_hat = self.supervisor(e_hat)
                h = self.embedder(x)
                x_hat = self.recovery(h_hat)
                y_fake, y_fake_e = self.discriminator(h_hat), self.discriminator(e_hat)
                supervised_loss = functional.mse_loss(self.supervisor(h)[:, :-1], h[:, 1:])
                moments = torch.mean(
                    torch.abs(torch.sqrt(x_hat.var(dim=0) + 1e-6) - torch.sqrt(x.var(dim=0) + 1e-6))
                ) + torch.mean(torch.abs(x_hat.mean(dim=0) - x.mean(dim=0)))
                g_loss = (
                    bce(y_fake, ones(y_fake))
                    + GAMMA * bce(y_fake_e, ones(y_fake_e))
                    + 100 * torch.sqrt(supervised_loss)
                    + 100 * moments
                )
                supervised.zero_grad(set_to_none=True)
                g_loss.backward()
                supervised.step()

                h = self.embedder(x)
                rebuilt = functional.mse_loss(self.recovery(h), x)
                dynamics = functional.mse_loss(self.supervisor(h)[:, :-1], h[:, 1:])
                e_loss = 10 * torch.sqrt(rebuilt) + 0.1 * dynamics
                autoencoder.zero_grad(set_to_none=True)
                e_loss.backward()
                autoencoder.step()

            x = self._batch(data)
            with torch.no_grad():
                h = self.embedder(x)
                e_hat = self.generator(self._noise(len(x), length))
                h_hat = self.supervisor(e_hat)
            y_real, y_fake, y_fake_e = (self.discriminator(t) for t in (h, h_hat, e_hat))
            d_loss = (
                bce(y_real, ones(y_real))
                + bce(y_fake, zeros(y_fake))
                + GAMMA * bce(y_fake_e, zeros(y_fake_e))
            )
            if d_loss.item() > DISCRIMINATOR_THRESHOLD:
                critic.zero_grad(set_to_none=True)
                d_loss.backward()
                critic.step()
            if (step + 1) % every == 0 or step + 1 == steps:
                progress("Joint training", step + 1, steps)
        return True

    @torch.no_grad()
    def sample(self, count: int, length: int) -> np.ndarray:
        out = []
        for start in range(0, count, 1_024):
            noise = self._noise(min(1_024, count - start), length)
            out.append(self.recovery(self.supervisor(self.generator(noise))).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, length, self.width), dtype=np.float32)


# --- The paper's post-hoc scores -------------------------------------------------------------


def _train(model: nn.Module, step: Callable[[], torch.Tensor], steps: int) -> None:
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for _ in range(steps):
        loss = step()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()


def discriminative_accuracy(
    real: np.ndarray, synthetic: np.ndarray, device: torch.device, seed: int, steps: int = 200
) -> float:
    """Accuracy of a small GRU telling real from synthetic windows, on 20 % kept aside."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    count = min(len(real), len(synthetic), 2_000)
    r = real[rng.choice(len(real), count, replace=False)]
    s = synthetic[rng.choice(len(synthetic), count, replace=False)]
    data = np.concatenate([r, s]).astype(np.float32)
    target = np.r_[np.ones(count), np.zeros(count)].astype(np.float32)
    order = rng.permutation(len(data))
    cut = int(len(data) * 0.8)
    fit, test = order[:cut], order[cut:]
    x = torch.from_numpy(data).to(device)
    y = torch.from_numpy(target).to(device)
    model = Recurrent(data.shape[2], max(8, data.shape[2] // 2), 1, 2, sigmoid=False).to(device)
    fit_index = torch.from_numpy(fit).to(device)

    def step() -> torch.Tensor:
        batch = fit_index[torch.randint(0, len(fit_index), (BATCH,), device=device)]
        logits = model(x[batch])[:, -1, 0]
        return functional.binary_cross_entropy_with_logits(logits, y[batch])

    _train(model, step, steps)
    with torch.no_grad():
        test_index = torch.from_numpy(test).to(device)
        predicted = (model(x[test_index])[:, -1, 0] > 0).float()
        return float((predicted == y[test_index]).float().mean().cpu())


def predictive_error(
    train: np.ndarray,
    test: np.ndarray,
    columns: list[int],
    device: torch.device,
    seed: int,
    steps: int = 200,
) -> float:
    """MAE of a small GRU predicting the next step of the numeric columns, trained on `train`
    windows and tested on `test` windows."""
    torch.manual_seed(seed)
    x = torch.from_numpy(train.astype(np.float32)).to(device)
    t = torch.from_numpy(test.astype(np.float32)).to(device)
    picked = torch.tensor(columns, device=device)
    model = Recurrent(train.shape[2], max(8, train.shape[2] // 2), len(columns), 2).to(device)

    def step() -> torch.Tensor:
        batch = x[torch.randint(0, len(x), (min(BATCH, len(x)),), device=device)]
        return functional.l1_loss(model(batch[:, :-1]), batch[:, 1:][:, :, picked])

    _train(model, step, steps)
    with torch.no_grad():
        return float(functional.l1_loss(model(t[:, :-1]), t[:, 1:][:, :, picked]).cpu())
