"""PaCMAP and TriMAP maps (122), behind the optional `interpret` extra (R-XAI-10).

PaCMAP (Wang, Huang, Rudin, Shaposhnik, JMLR 2021) keeps the local and the global structure
better than t-SNE or UMAP; TriMAP (Amid, Warmuth, 2019) keeps the global one through triplets.
Both need numba (and faiss or annoy), absent from the browser version; neither is imported
before a map is asked for."""

from typing import Literal

import numpy as np

from quantilo.interpret import extras

Method = Literal["pacmap", "trimap"]
METHODS: tuple[Method, ...] = ("pacmap", "trimap")
SEED = 42


def embed(x: np.ndarray, method: Method, seed: int = SEED) -> np.ndarray:
    """The rows of `x` (prepared numbers) in two dimensions."""
    if method not in METHODS:
        raise ValueError(f"Unknown map: {method}")
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if method == "pacmap":
        pacmap = extras.require("pacmap")
        reducer = pacmap.PaCMAP(n_components=2, random_state=seed)
        found = reducer.fit_transform(x, init="pca")
    else:
        trimap = extras.require("trimap")
        found = trimap.TRIMAP(n_dims=2).fit_transform(x)
    found = np.asarray(found, dtype=float)
    if found.shape != (len(x), 2):
        raise ValueError(f"{method} returned {found.shape} instead of ({len(x)}, 2)")
    return found
