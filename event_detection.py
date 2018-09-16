"""Functions to quantify bistable events in time series data."""

import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict


def event_detection(location, batch_targets, batch, process_targets,
                    transient=0.1, min_x_dt=5, z=0.2, w=0.125):
    """Detect and quantify high count events in a bistable/excitable system.

    Args:
        location (dict): Dictionary describing the location in parameter space.
        batch_targets (seq): List of target names 1-1 with the time series data.
        batch (numpy array): Time series data for a batch of trajectories. `batch`
            has three dimensions representing which trajectory in the batch, which
            target in a trajectory, and which time step in a target's trajectory.
        process_targets (seq): Subset of `batch_targets` on which to perform
            event detection and measurements.

    Keyword Args:
        transient (float): Fraction of time series data to ignore to ensure
            only measurements at steady state are returned.
        min_x_dt (int): Resolution limit in time index space for event detection.
        z (float): Fraction of the determined effective maximum count which will
            be the midpoint between the low and high transition thresholds.
        w (float): Fraction of the determined effective maximum count which will
            be the distance from the midpoint between the low and high transition
            thresholds to those thresholds.

    Returns:
        list: List of results which is 1-1 with `process_targets`. Each result is a
        dictionary of measurements. Each dictionary has the following results:

            - Target (str): The batch target associated with the measurements.
            - Batch (numpy array): The time series trajectories of the target.
            - Events (list): List of pairs of transition indices of detected events.
            - Parameters (dict): Dictionary of determined event detection parameters.
            - Mean Event Duration (float): The mean event duration computed by
                averaging event durations over each trajectory and then computing
                a weighted average of these averages where the event counts of the
                trajectories are the weights.
            - Mean Event Height (float): The mean event height computed by
                averaging event heights over each trajectory and then computing
                a weighted average of these averages where the event counts of the
                trajectories are the weights.
            - StdDev Event Height (float): The standard deviation of event height
                computed by a weighted average of event height standard deviations
                computed from each trajectory where the event counts of the
                trajectories are the weights.
            - Toxic Probability (float): The fractions of each trajectory spent in
                the high toxin state are weight averaged where the event counts of
                the trajectories are the weights.

    """
    transient = int(batch.shape[-1] * transient)
    steady = batch[:, :, transient:]

    results = []
    for target in process_targets:
        i = batch_targets.index(target)
        parameters = None
        events = []
        measurements = []
        if i == 0:
            time = steady[:, i, :]
        else:
            counts = steady[:, i, :]
            eff_max = np.percentile(counts, 99.9)

            z_0 = eff_max * z
            w_0 = eff_max * w
            high = min(eff_max - 1, z_0 + w_0)
            low = max(1, z_0 - w_0)
            parameters = {
                'high': high,
                'low': low,
                'min_dt': min_x_dt,
                'effective_max': eff_max
            }

            for j, trajectory in enumerate(counts):
                es = segment_events(time[0], trajectory, high, low, min_x_dt)
                measurements.append(measure_events(time[0], counts[j], es))
                events.append([(u + transient, v + transient) for u, v in es])

        measurements = pd.DataFrame(measurements)
        if events and sum([len(e) for e in events]):
            values = np.average(measurements,
                                weights=measurements['Event Count'], axis=0)
            values[-1] = measurements['Event Count'].sum()
        else:
            values = np.array((-1, -1, -1, 0, 0))

        entry = location.copy()
        entry['Target'] = target
        entry['Batch'] = batch[:, i, :]
        entry['Events'] = events
        entry['Parameters'] = parameters
        for j, m in enumerate(measurements):
            entry[m] = values[j]
        results.append(entry)

    return results


def segment_events(x, y, high, low, min_dt):
    """Determine the event transitions in a time series trajectory (x, y).
    An event identifies a "high count state" entry/exit for a species.
    When `y` passes from below `th` to above `th`, a low to high transition may occur.
    when `y` passes from above `tl` to below `tl`, a high to low transition may occur.
    An event is always defined as a pair of transitions (low to high and high to low).
    Events are filtered to meet the following standards:

        - At least one point is above th (i.e. sufficiently tall).
        - The end points are below tl (i.e. system is low before/after event).
        - The event contains at least `min_dt` points
          (i.e. sufficiently long compared to resolution).
        - There are at least `min_dt` points between any two events
          (i.e. system stabilizes between events).
        - Before and after any event there are at least `min_dt` points
          whose maximum value is below `th` (i.e. pre/post low state is stable).

    The word "stable" means no transition happened for `min_dt` consecutive points.

    Args:
        x (numpy array): The time axis of the trajectory.
        y (numpy array): The count axis of the trajectory.
        high (float): The low to high transition threshold.
        low (float): The high to low transition threshold.
        min_dt (int): A resolution limit for how events are defined.

    Returns:
        list: A list of transition pairs associated with each high count event.

    """
    if (high - low <= 4) or y.min() > low or y.max() < high:
        return []

    state = (1 if y[0] > high else (-1 if y[0] < low else 0))

    events = [None]
    for j in range(1, y.size - min_dt - 1):
        dy = y[j] - y[j - 1]
        dhigh = high - y[j - 1]
        dlow = low - y[j - 1]

        if state == 0:
            peek = y[(j + 1):(j + min_dt + 1)]
            if   dy >= dhigh and peek.max() > low: #change max to min?
                state = 1
            elif dy <= dlow and peek.max() < high:
                state = -1

        elif state > 0 and dy <= dlow:
            state = -1
            if events[-1]:
                events[-1] = (events[-1], j)
                events.append(None)

        elif state < 0 and dy >= dhigh:
            state = 1
            lastcross = j - 1
            while lastcross > 0 and not y[lastcross] < low:
                lastcross -= 1
            events[-1] = lastcross

    if not isinstance(events[-1], tuple):
        events.pop(-1)

    f = []
    for j, (e0, e1) in enumerate(events):
        last = f[-1] if f else None
        if last:
            f0, f1 = last
            # distance to last event is small
            if e0 - f1 < min_dt:
                f[-1] = (f0, e1)
            # mean since last event is too large
            elif y[f1 + 1:e0 - 1].mean() > high:
                f[-1] = (f0, e1)
            # event is sufficiently long
            elif e1 - e0 > min_dt:
                f.append((e0, e1))
        # event is sufficiently long and nothing to merge with
        elif e1 - e0 > min_dt:
            f.append((e0, e1))

    if f:
        i, j = f[0]
        if i < min_dt + 1:
            f.pop(0)
        elif y[i - min_dt - 1:i].max() >= high:
            f.pop(0)

    if f:
        i, j = f[-1]
        if j > x.size - min_dt - 1:
            f.pop(-1)
        elif y[j + 1:j + min_dt + 1].max() >= high:
            f.pop(-1)

    return f


def measure_events(time, counts, events):
    """Compute statistics averaged over a single trajectory.

    Args:
        time (numpy array): The time axis of a trajectory.
        counts (numpy array): The counts of a single target for a trajectory.
        events (list): List of tuples where each tuple has the start and stop
            index of a high count event.

    Returns:
        OrderedDict: Ordered dictionary with trajectory averaged measurements.

    """
    labels = [
        'Mean Event Duration',
        'Mean Event Height',
        'StdDev Event Height',
        'Toxic Probability',
        'Event Count',
    ]
    if events:
        durations = [(time[v] - time[u]) for u, v in events]
        event_counts = [counts[u:v] for u, v in events]
        event_means = [np.mean(e) for e in event_counts]
        event_stds = [np.std(e) for e in event_counts]
        count_mean = np.average(event_means, weights=durations)
        count_std = np.average(event_stds, weights=durations)
        prob_toxic = sum(durations) / (time[-1] - time[0])
        measurements = (np.mean(durations), count_mean, count_std,
                        prob_toxic, len(events))
    else:
        measurements = (-1, -1, -1, 0, 0)
    return OrderedDict(m for m in zip(labels, measurements))


