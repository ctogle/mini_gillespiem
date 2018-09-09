import numpy as np
from collections import defaultdict, OrderedDict


def event_detection(location, targets, batch,
                    transient=0.2, min_x_dt=5, z=0.2, w=0.125):
    transient = int(batch.shape[-1] * transient)
    steady = batch[:, :, transient:]

    outputs = {}
    outputs['batch'] = batch
    outputs['parameters'] = []
    outputs['events'] = []
    outputs['measurements'] = []
    for i, target in enumerate(targets):
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
            parameters = {'high': high, 'low': low, 'min_dt': min_x_dt}

            for j, trajectory in enumerate(counts):
                es = segment_events(time[0], trajectory, high, low, min_x_dt)
                measurements.append(measure_events(time[0], counts[j], es))
                events.append([(u + transient, v + transient) for u, v in es])
        outputs['parameters'].append(parameters)
        outputs['events'].append(events)
        outputs['measurements'].append(measurements)

    return outputs


def segment_events(x, y, high, low, min_dt):
    '''
    return measurements of events in a trajectory (x,y)
    an event identifies a "high count state" entry/exit for a species
    when y passes from below th to above th, a low->high transition may occur
    when y passes from above tl to below tl, a high->low transition may occur
    an event is always defined as a pair of transitions (low->high, high->low)

    filtered events have the following properties:
      1) at least one point is above th (sufficiently tall)
      2) the end points are below tl (system is low before/after event)
      3) the event contains at least min_x_dt points
            (sufficiently long compared to resolution)
      4) there are at least min_x_dt points between any two events
            (system stabilizes between events)
      5) before and after any event there are at least min_x_dt points
            whose maximum value is below th (pre/post low state is stable)

    * the word "stable" means no transition happened for min_dt consecutive points
    '''
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


