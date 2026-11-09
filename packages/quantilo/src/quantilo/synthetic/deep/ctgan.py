"""CTGAN (Xu et al., NeurIPS 2019), written from the paper.

Generator: two residual blocks (linear 256, batch norm, ReLU, concatenated with their input),
tanh for alphas and Gumbel-softmax (tau 0.2) for modes and levels. Critic: PacGAN (10 rows judged
together), two linear layers of 256 (LeakyReLU 0.2, dropout 0.5). Loss: WGAN-GP (lambda 10) plus a
cross-entropy that makes the generator honour the conditional vector.
"""

from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from quantilo.synthetic.deep import batch_size
from quantilo.synthetic.deep.sampler import Sampler, TensorSampler
from quantilo.synthetic.deep.transformer import Span

NOISE = 128
HIDDEN = 256
PAC = 10
PENALTY = 10.0
TAU = 0.2

Progress = Callable[[int, float, float], None]  # epoch, generator loss, critic loss
StopCheck = Callable[[], bool]


class Residual(nn.Module):
    def __init__(self, size_in: int, size_out: int) -> None:
        super().__init__()
        self.linear = nn.Linear(size_in, size_out)
        self.norm = nn.BatchNorm1d(size_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([functional.relu(self.norm(self.linear(x))), x], dim=1)


class Generator(nn.Module):
    def __init__(self, size_in: int, width: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            Residual(size_in, HIDDEN),
            Residual(size_in + HIDDEN, HIDDEN),
            nn.Linear(size_in + 2 * HIDDEN, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Critic(nn.Module):
    def __init__(self, size_in: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(size_in * PAC, HIDDEN),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(HIDDEN, 1),
        )
        self.size_in = size_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x.view(-1, self.size_in * PAC))


class Layout:
    """Where each block of the output sits, with blocks of the same size grouped: one Gumbel-
    softmax or cross-entropy per size instead of one per column (a GPU spends most of a small
    step launching operations)."""

    def __init__(self, spans: list[Span], device: torch.device) -> None:
        tanh, softmax, discrete = [], {}, {}
        start = offset = 0
        for k, span in enumerate(spans):
            positions = list(range(start, start + span.size))
            if span.activation == "tanh":
                tanh.extend(positions)
            else:
                softmax.setdefault(span.size, []).append(positions)
            if span.discrete:
                entry = discrete.setdefault(span.size, ([], [], []))
                entry[0].append(positions)
                entry[1].append(list(range(offset, offset + span.size)))
                entry[2].append(sum(1 for s in spans[:k] if s.discrete))
                offset += span.size
            start += span.size

        def tensor(values: list) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.long, device=device)

        self.tanh = tensor(tanh)
        self.softmax = [tensor(group) for group in softmax.values()]
        self.discrete = [(tensor(d), tensor(c), tensor(m)) for d, c, m in discrete.values()]

    def activate(self, raw: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(raw)
        if len(self.tanh):
            out[:, self.tanh] = torch.tanh(raw[:, self.tanh])
        for index in self.softmax:
            out[:, index.flatten()] = functional.gumbel_softmax(
                raw[:, index], tau=TAU, dim=-1
            ).flatten(1)
        return out

    def condition_loss(
        self, raw: torch.Tensor, vector: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy between the generated levels of each discrete column and the condition,
        kept only for the column that was conditioned on."""
        total = torch.zeros((), device=raw.device)
        for data, condition, column in self.discrete:
            logits = raw[:, data]
            size = logits.shape[-1]
            target = vector[:, condition].argmax(dim=-1)
            loss = functional.cross_entropy(
                logits.reshape(-1, size), target.reshape(-1), reduction="none"
            )
            total = total + (loss.view(len(raw), -1) * mask[:, column]).sum()
        return total / len(raw)


def gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand(real.shape[0] // PAC, 1, 1, device=real.device)
    alpha = alpha.repeat(1, PAC, real.shape[1]).view(-1, real.shape[1])
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = critic(mixed)
    gradients = torch.autograd.grad(score, mixed, torch.ones_like(score), create_graph=True)[0]
    return ((gradients.view(-1, PAC * real.shape[1]).norm(2, dim=1) - 1) ** 2).mean() * PENALTY


class CTGAN:
    def __init__(
        self, spans: list[Span], sampler: Sampler, device: torch.device, seed: int
    ) -> None:
        self.spans = spans
        self.sampler = sampler
        self.device = device
        self.width = sum(s.size for s in spans)
        torch.manual_seed(seed)
        self.draw = TensorSampler(sampler, device)
        self.layout = Layout(spans, device)
        self.generator = Generator(NOISE + sampler.width, self.width).to(device)
        self.critic = Critic(self.width + sampler.width).to(device)

    def _batch(self, rows: int) -> int:
        return max(PAC, min(batch_size(self.device.type != "cpu"), rows) // PAC * PAC)

    def fit(self, epochs: int, progress: Progress, should_stop: StopCheck) -> None:
        g_opt = torch.optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6
        )
        c_opt = torch.optim.Adam(
            self.critic.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6
        )
        draw = self.draw
        batch = self._batch(draw.rows)
        steps = max(1, draw.rows // batch)
        device = self.device
        for epoch in range(epochs):
            if should_stop():
                return
            g_loss = c_loss = torch.zeros(())
            for _ in range(steps):
                # critic: real rows matching a shuffled copy of the conditions
                noise = torch.randn(batch, NOISE, device=device)
                condition = draw.train_condition(batch)
                if condition is None:
                    real_in = draw.real_rows(None, batch)
                    fake_in = self.layout.activate(self.generator(noise))
                else:
                    c1, _mask, groups = condition
                    order = torch.randperm(batch, device=device)
                    real = draw.real_rows(groups[order], batch)
                    fake = self.layout.activate(self.generator(torch.cat([noise, c1], dim=1)))
                    real_in = torch.cat([real, c1[order]], dim=1)
                    fake_in = torch.cat([fake, c1], dim=1)
                penalty = gradient_penalty(self.critic, real_in, fake_in.detach())
                c_loss = -(self.critic(real_in).mean() - self.critic(fake_in.detach()).mean())
                c_opt.zero_grad(set_to_none=True)
                (c_loss + penalty).backward()
                c_opt.step()
                # generator
                noise = torch.randn(batch, NOISE, device=device)
                condition = draw.train_condition(batch)
                if condition is None:
                    raw = self.generator(noise)
                    g_loss = -self.critic(self.layout.activate(raw)).mean()
                else:
                    c1, mask, _ = condition
                    raw = self.generator(torch.cat([noise, c1], dim=1))
                    fake_in = torch.cat([self.layout.activate(raw), c1], dim=1)
                    g_loss = -self.critic(fake_in).mean() + self.layout.condition_loss(
                        raw, c1, mask
                    )
                g_opt.zero_grad(set_to_none=True)
                g_loss.backward()
                g_opt.step()
            progress(epoch + 1, float(g_loss.detach().cpu()), float(c_loss.detach().cpu()))

    @torch.no_grad()
    def sample(self, rows: int) -> np.ndarray:
        self.generator.eval()
        batch = self._batch(max(rows, PAC))
        out = []
        for _ in range(0, rows, batch):
            noise = torch.randn(batch, NOISE, device=self.device)
            vector = self.draw.generation_condition(batch)
            if vector is not None:
                noise = torch.cat([noise, vector], dim=1)
            out.append(self.layout.activate(self.generator(noise)).cpu().numpy())
        self.generator.train()
        return np.concatenate(out)[:rows]
