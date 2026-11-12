"""TVAE (Xu et al., NeurIPS 2019), written from the paper.

Encoder: two layers of 128 (ReLU) to a mean and a log-variance (latent size 128). Decoder: two
layers of 128 to the alphas (with a learned standard deviation per column) and the one-hot
logits. Loss: reconstruction (squared error weighted by the learned deviation for alphas,
cross-entropy for one-hots, factor 2) plus the Kullback-Leibler divergence.
"""

from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from quantilo.synthetic.deep import BATCH
from quantilo.synthetic.deep.training import Paper, Stepper, Training, ema_for
from quantilo.synthetic.deep.transformer import Span

HIDDEN = 128
LATENT = 128
LOSS_FACTOR = 2.0

Progress = Callable[[int, float, float], None]  # epoch, loss, reconstruction loss
StopCheck = Callable[[], bool]


class Encoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(width, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU()
        )
        self.mean = nn.Linear(HIDDEN, LATENT)
        self.log_variance = nn.Linear(HIDDEN, LATENT)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(x)
        return self.mean(hidden), self.log_variance(hidden)


class Decoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(LATENT, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, width),
        )
        self.sigma = nn.Parameter(torch.ones(width) * 0.1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.body(z)


def reconstruction(
    raw: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor, spans: list[Span]
) -> torch.Tensor:
    losses, start = [], 0
    for span in spans:
        end = start + span.size
        if span.activation == "tanh":
            std = sigma[start]
            losses.append(((target[:, start] - torch.tanh(raw[:, start])) ** 2 / 2 / std**2).sum())
            losses.append(torch.log(std) * target.shape[0])
        else:
            losses.append(
                functional.cross_entropy(
                    raw[:, start:end], target[:, start:end].argmax(dim=1), reduction="sum"
                )
            )
        start = end
    return torch.stack(losses).sum() / target.shape[0]


class TVAE:
    def __init__(
        self,
        spans: list[Span],
        data: np.ndarray,
        device: torch.device,
        seed: int,
        training: Training | None = None,
    ) -> None:
        self.spans = spans
        self.training = training
        self.data = data
        self.device = device
        width = sum(s.size for s in spans)
        torch.manual_seed(seed)
        self.generator = torch.Generator().manual_seed(seed)
        self.encoder = Encoder(width).to(device)
        self.decoder = Decoder(width).to(device)

    def fit(self, epochs: int, progress: Progress, should_stop: StopCheck) -> None:
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        data = torch.from_numpy(self.data).to(self.device)
        rows = len(data)
        batch = min(BATCH, rows)
        steps = len(range(0, rows - batch + 1, batch))
        optimiser = Stepper(
            parameters, self.training, Paper(1e-3, weight_decay=1e-5), epochs * steps
        )
        ema = ema_for(self.decoder, self.training)
        for epoch in range(epochs):
            if should_stop():
                return
            order = torch.randperm(rows, generator=self.generator).to(self.device)
            loss = rebuilt = torch.zeros(())
            for start in range(0, rows - batch + 1, batch):
                real = data[order[start : start + batch]]
                mean, log_variance = self.encoder(real)
                z = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
                raw = self.decoder(z)
                rebuilt = reconstruction(raw, real, self.decoder.sigma, self.spans)
                divergence = (
                    -0.5 * torch.sum(1 + log_variance - mean**2 - log_variance.exp()) / batch
                )
                loss = LOSS_FACTOR * rebuilt + divergence
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)
                if ema is not None:
                    ema.update(self.decoder)
            progress(epoch + 1, float(loss.detach().cpu()), float(rebuilt.detach().cpu()))
        if ema is not None:  # generate with the averaged weights
            ema.copy_to(self.decoder)

    @torch.no_grad()
    def sample(self, rows: int) -> np.ndarray:
        self.decoder.eval()
        out = []
        for start in range(0, rows, BATCH):
            z = torch.randn(min(BATCH, rows - start), LATENT, device=self.device)
            raw = self.decoder(z)
            parts, position = [], 0
            for span in self.spans:
                block = raw[:, position : position + span.size]
                if span.activation == "tanh":
                    parts.append(torch.tanh(block))
                else:
                    # drawn from the probabilities: the most likely level would erase the rare ones
                    drawn = torch.multinomial(torch.softmax(block, dim=1), 1).squeeze(1)
                    parts.append(functional.one_hot(drawn, span.size).float())
                position += span.size
            out.append(torch.cat(parts, dim=1).cpu().numpy())
        self.decoder.train()
        return np.concatenate(out)[:rows]
