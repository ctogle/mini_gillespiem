import numpy as np
from scipy.stats import pearsonr


def steady_state_correlation(location, batch_targets, batch, process_targets,
                             transient=0.8):
    selection = [(t in process_targets) for t in batch_targets]
    selection = np.array(selection, dtype=bool)
    time = batch[:, 0, :]
    data = batch[:, selection, int(batch.shape[2] * transient):]

    #print(data.type)

    corr = np.array([pearsonr(traj[0, :], traj[1, :]) for traj in batch])
    mean = corr.mean(axis=0)

    #print(corr.shape, mean.shape, mean)
    #print(data.shape)

    result = [mean]

    #print(location, batch.shape, time.shape, targets.shape, batch.mean(axis=0).shape)
    return result


