"""Conditional vector and training-by-sampling (CTGAN paper, section 4.3).

For each row of a batch: a discrete column is drawn uniformly, then one of its levels with
probability proportional to the log of its frequency, and a real row having that level is taken.
Rare levels are learned too. To generate, levels are drawn at their real frequency.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quantilo.synthetic.deep.transformer import Transformer

if TYPE_CHECKING:
    import torch


@dataclass
class Discrete:
    start: int  # first column of its one-hot in the data matrix
    size: int
    offset: int  # first column of its block in the conditional vector
    log_shares: np.ndarray  # drawing probabilities while training
    shares: np.ndarray  # real frequencies, for generating
    rows: list[np.ndarray]  # real rows of each level


class Sampler:
    def __init__(
        self, data: np.ndarray, transformer: Transformer, rng: np.random.Generator
    ) -> None:
        self.data = data
        self.rng = rng
        self.discrete: list[Discrete] = []
        start = offset = 0
        for column in transformer.columns:
            if column.categorical:
                block = data[:, start : start + column.width]
                counts = block.sum(axis=0)
                logs = np.log(counts + 1)
                self.discrete.append(
                    Discrete(
                        start,
                        column.width,
                        offset,
                        logs / logs.sum(),
                        counts / counts.sum(),
                        [np.flatnonzero(block[:, j]) for j in range(column.width)],
                    )
                )
                offset += column.width
            start += column.width
        self.width = offset  # size of the conditional vector (0: nothing to condition on)
        # every (column, level) as one group: its rows laid end to end, for vectorised drawing
        groups = [rows for column in self.discrete for rows in column.rows]
        self.group_rows = np.concatenate(groups) if groups else np.zeros(0, dtype=np.int64)
        counts = np.array([len(g) for g in groups], dtype=np.int64)
        self.group_counts = counts
        self.offsets = np.cumsum([0] + [c.size for c in self.discrete])[:-1].astype(np.int64)
        self.group_starts = np.concatenate([[0], counts.cumsum()[:-1]]) if groups else counts

    def _draw(self, shares: np.ndarray, count: int) -> np.ndarray:
        cumulative = shares.cumsum()
        return np.minimum(
            np.searchsorted(cumulative, self.rng.random(count), side="right"), len(shares) - 1
        )

    def train_condition(
        self, batch: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """(conditional vectors, mask of the chosen column, column index, level index)."""
        if not self.discrete:
            return None
        columns = self.rng.integers(0, len(self.discrete), size=batch)
        levels = np.empty(batch, dtype=np.int64)
        vector = np.zeros((batch, self.width), dtype=np.float32)
        mask = np.zeros((batch, len(self.discrete)), dtype=np.float32)
        for k, column in enumerate(self.discrete):
            chosen = np.flatnonzero(columns == k)
            if chosen.size == 0:
                continue
            levels[chosen] = self._draw(column.log_shares, chosen.size)
            vector[chosen, column.offset + levels[chosen]] = 1
            mask[chosen, k] = 1
        return vector, mask, columns, levels

    def real_rows(
        self, columns: np.ndarray | None, levels: np.ndarray | None, batch: int
    ) -> np.ndarray:
        """Real rows matching each (column, level); random rows without conditions."""
        if columns is None or levels is None:
            return self.data[self.rng.integers(0, len(self.data), size=batch)]
        group = self.offsets[columns] + levels
        counts = self.group_counts[group]
        drawn = self.group_starts[group] + np.floor(self.rng.random(len(group)) * counts).astype(
            np.int64
        )
        fallback = self.rng.integers(0, len(self.data), size=len(group))
        index = np.where(
            counts > 0, self.group_rows[np.minimum(drawn, len(self.group_rows) - 1)], fallback
        )
        return self.data[index]

    def generation_condition(self, batch: int) -> np.ndarray | None:
        """Conditional vectors with levels at their real frequency."""
        if not self.discrete:
            return None
        columns = self.rng.integers(0, len(self.discrete), size=batch)
        vector = np.zeros((batch, self.width), dtype=np.float32)
        for k, column in enumerate(self.discrete):
            chosen = np.flatnonzero(columns == k)
            if chosen.size:
                vector[chosen, column.offset + self._draw(column.shares, chosen.size)] = 1
        return vector


class TensorSampler:
    """The same draws as `Sampler`, done with tensors on the training device: no copy from the
    CPU at each step (on a GPU, those copies cost more than the networks)."""

    def __init__(self, sampler: Sampler, device: "torch.device") -> None:
        import torch

        self.device = device
        self.data = torch.from_numpy(sampler.data).to(device)
        self.rows = len(sampler.data)
        self.width = sampler.width
        self.columns = len(sampler.discrete)
        if not self.columns:
            return
        longest = max(c.size for c in sampler.discrete)
        log_cumulative = np.ones((self.columns, longest), dtype=np.float32)
        cumulative = np.ones((self.columns, longest), dtype=np.float32)
        for k, column in enumerate(sampler.discrete):
            log_cumulative[k, : column.size] = column.log_shares.cumsum()
            cumulative[k, : column.size] = column.shares.cumsum()
        self.log_cumulative = torch.from_numpy(log_cumulative).to(device)
        self.cumulative = torch.from_numpy(cumulative).to(device)
        self.sizes = torch.tensor([c.size for c in sampler.discrete], device=device)
        self.offsets = torch.from_numpy(sampler.offsets).to(device)
        self.group_rows = torch.from_numpy(sampler.group_rows.astype(np.int64)).to(device)
        self.group_starts = torch.from_numpy(sampler.group_starts.astype(np.int64)).to(device)
        self.group_counts = torch.from_numpy(sampler.group_counts).to(device)

    def _groups(
        self, batch: int, cumulative: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        import torch

        columns = torch.randint(0, self.columns, (batch,), device=self.device)
        draws = torch.rand(batch, 1, device=self.device)
        levels = (cumulative[columns] < draws).sum(dim=1)
        levels = torch.minimum(levels, self.sizes[columns] - 1)
        return columns, self.offsets[columns] + levels

    def _vector(self, groups: "torch.Tensor") -> "torch.Tensor":
        import torch

        vector = torch.zeros(len(groups), self.width, device=self.device)
        vector[torch.arange(len(groups), device=self.device), groups] = 1
        return vector

    def train_condition(
        self, batch: int
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"] | None:
        """(conditional vectors, mask of the chosen column, group = column offset + level)."""
        import torch

        if not self.columns:
            return None
        columns, groups = self._groups(batch, self.log_cumulative)
        mask = torch.zeros(batch, self.columns, device=self.device)
        mask[torch.arange(batch, device=self.device), columns] = 1
        return self._vector(groups), mask, groups

    def real_rows(self, groups: "torch.Tensor | None", batch: int) -> "torch.Tensor":
        import torch

        if groups is None:
            return self.data[torch.randint(0, self.rows, (batch,), device=self.device)]
        counts = self.group_counts[groups]
        offset = (torch.rand(len(groups), device=self.device) * counts).long()
        return self.data[self.group_rows[self.group_starts[groups] + offset]]

    def generation_condition(self, batch: int) -> "torch.Tensor | None":
        if not self.columns:
            return None
        _, groups = self._groups(batch, self.cumulative)
        return self._vector(groups)
