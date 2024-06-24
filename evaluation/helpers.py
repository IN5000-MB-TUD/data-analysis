def compute_monthly_patterns(metric_patterns, metric_patterns_idxs):
    """Return a sequence of each monthly pattern"""
    metric_patterns_sequence = []
    patterns_boundaries = list(zip(metric_patterns_idxs[:-1], metric_patterns_idxs[1:]))
    for pattern_idx, pattern in enumerate(metric_patterns):
        metric_patterns_sequence += [pattern] * (
            patterns_boundaries[pattern_idx][1] - patterns_boundaries[pattern_idx][0]
        )
    return metric_patterns_sequence
