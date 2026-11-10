# A/B tests

- **Describe the test first.** The variant column (2 to 5 groups), the control, the metrics (the first is the primary one; decide it before looking at results), and the unit: when a user has several rows, analyze per user, never per row. `analyze_experiment` says when an identifier repeats.
- **Check the split before any result.** A sample ratio mismatch (the SRM check, p < 0.001) means the assignment or the logging is broken: say so first and do not conclude on the metrics.
- **Read the sentence, not just p.** Give the difference with its 95 % interval and the lift; an interval that barely excludes 0 is a weak win. With several variants or metrics, quote the Holm-adjusted p-value.
- **Not significant is not "no effect".** When the power is low, say there is not enough data and give the smallest lift the sample could detect; suggest `sample_size` for the next test.
- **Secondary metrics and subgroups are hypotheses.** A significant subgroup among many is expected by chance; never present it as a result.
- **Do not peek.** Stopping as soon as p < 0.05 inflates false positives: the sample size is decided before, the result read once.
- **The bandit view is a complement, not a verdict.** After the test, `bandit_view` gives P(each variant is best) and the expected loss of shipping it, and with a date what a bandit would have done (units gained against the fixed split, cumulative regret). Use it to say how costly a wrong choice would be, never to replace the test's interval and p-value, and always say that after an adaptive allocation the classical intervals are biased. Call it a bandit, never reinforcement learning.
- **Pre-period data helps.** If the same measure exists before the test (last month's spend), CUPED narrows the interval without bias.

Report: the check of the split, then the primary metric's verdict in one sentence, then the secondary metrics and what to do next.
