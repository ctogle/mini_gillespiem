"""Method to compute meanfield of a time-series without resampling."""

def meanfield(location, targets, batch, process_targets):
    """Compute the meanfield of a batch of trajectories."""
    which = [j for j, t in enumerate(targets) if t in process_targets]
    batch = batch[:, which, :].mean(axis=0)
    targets = [('Mean %s' % t) for t in process_targets]
    return targets, batch

