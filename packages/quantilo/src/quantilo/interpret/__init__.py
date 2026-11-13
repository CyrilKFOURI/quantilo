"""Models readable by construction (Duke, Cynthia Rudin's lab; study 2026-11-11 § 13).

No mandatory dependency: numpy, pandas and scikit-learn only, so the browser version (Pyodide)
runs every default. Each optional method says whether it can run here (`available()`): the
packages of the `interpret` extra are imported lazily, when the method is called, never when the
engine starts.

- `conditions`: the true / false conditions a scorecard or an optimal tree is built on (R-XAI-06).
- `risk_score`: FasterRisk re-implemented (121): sparse scorecards with small integer points.
- `sparse_tree`: a sparse CART tree, pruned by cross-validation with the 1-SE rule (122).
- `rashomon`: GOSDT's optimal tree and TreeFARMS's Rashomon set (122, extra `interpret`).
- `maps`: PaCMAP and TriMAP maps (122, extra `interpret`).
- `matching`: FLAME, almost-exact matching in DuckDB: matched groups and their effects (123).
- `compare`: the readable model against the run's model, on the same held-out rows.

Post-hoc readings of any model, under Explain › Advanced › More (126), numpy only as well:

- `kernel_shap`: SHAP values by sampling permutations, exactly additive (R-XAI-12).
- `interactions`: Friedman's H for pairs of columns (R-XAI-13).
- `anchors`: one row's IF … THEN rule with its precision and coverage (R-XAI-14).
- `tabular_prototypes`: prototypes and criticisms of each class, MMD-critic (R-XAI-15).
"""

from quantilo.interpret.extras import Availability, available

__all__ = ["Availability", "available"]
