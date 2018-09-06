import multiprocessing as mp
import itertools
import tqdm
import numpy as np
from simulator import get_simulator


def mp_run(arg):
    run, s, kws = arg
    return run(s, **kws)


def simulate(system, processing=None, batchsize=1, axes=None,
             n_workers=4, maxseed=2147483647):
    np.random.seed(0)
    run = get_simulator(system)
    targets = system.get('targets', [])
    data = []

    pool = mp.Pool(processes=n_workers)

    if axes:
        axes_names, axes_values = zip(*axes)

        locations = list(itertools.product(*axes_values))
        location_count = np.cumprod([len(av) for av in axes_values])[-1]
        locations = dict((a, np.array(l)) for a, l in zip(axes_names, zip(*locations)))

        for j in tqdm.tqdm(range(location_count), total=location_count):
            seeds = np.random.randint(0, maxseed, size=(batchsize, ), dtype=int)
            location = dict((a, locations[a][j]) for a in axes_names)
            batch = ((run, s, location) for s in seeds)
            imap = pool.imap_unordered(mp_run, batch)
            batch = np.array(list(imap))
            if processing:
                batch = [p(location, targets, batch) for p in processing]
            data.append((seeds, batch))

    else:
        locations = {}
        seeds = np.random.randint(0, maxseed, size=(batchsize, ), dtype=int)
        batch = ((run, s, {}) for s in seeds)
        imap = pool.imap_unordered(mp_run, batch)
        batch = np.array(list(tqdm.tqdm(imap, total=batchsize)))
        if processing:
            batch = [p({}, targets, batch) for p in processing]
        data.append((seeds, batch))

    seeds, data = zip(*data)
    return locations, data


