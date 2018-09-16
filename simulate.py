import multiprocessing as mp
import subprocess
import itertools
import pickle
import json
import tqdm
import time
import os
import numpy as np
from simulator import get_simulator


maxseed = 2147483647
workpath = './.simulators'


def mp_run(arg):
    run, s, kws = arg
    return run(s, **kws)


def worker(n, in_q, out_q, seed, system, processing):
    targets = system.get('targets', [])

    np.random.seed(seed)
    run = get_simulator(system)

    msg = in_q.get()
    while not msg[0] == 'halt':
        if msg[0] == 'run':
            l, location, batchsize = msg[1], msg[2], msg[3]
            seeds = np.random.randint(0, maxseed, size=(batchsize, ), dtype=int)
            batch = np.array([run(s, **location) for s in seeds])
            if processing:
                batch = [p(location, targets, batch) for p in processing]
            out_q.put(('ran', l, batch))
        else:
            print('msg (%d):' % n, msg)
        msg = in_q.get()


def walk_space(axes):
    names, values = zip(*axes)
    n = np.cumprod([len(av) for av in values])[-1]
    locations = list(itertools.product(*values))
    locations = dict((a, l) for a, l in zip(names, zip(*locations)))
    trajectory = (dict((a, locations[a][l]) for a in names) for l in range(n))
    return locations, list(enumerate(trajectory))


def simulate(system, processing=None, batchsize=1, axes=None,
             n_workers=4):
    np.random.seed(0)
    seeds = np.random.randint(0, maxseed, size=(n_workers, ), dtype=int)

    workers, idle, busy = [], [], []
    for n, s in enumerate(seeds):
        iq, oq = mp.Queue(), mp.Queue()
        p = mp.Process(target=worker, args=(n, iq, oq, s, system, processing))
        p.start()
        iq.put(('hello world', ))
        workers.append((iq, oq, p))
        idle.append(n)

    if axes:
        locations, trajectory = walk_space(axes)

        data = [None] * len(trajectory)
        while None in data:
            if idle and trajectory:
                n = idle.pop(0)
                iq, oq, p = workers[n]
                l, loc = trajectory.pop(0)
                iq.put(('run', l, loc, batchsize))
                busy.append(n)
                print('deployed', idle, busy)
            elif busy:
                for n in busy:
                    iq, oq, p = workers[n]
                    if not oq.empty():
                        msg = oq.get()
                        if msg[0] == 'ran':
                            l, batch = msg[1], msg[2]
                            data[l] = batch
                            idle.append(n)
                            busy.remove(n)
                            print('returned', idle, busy)
                            break
                else:
                    print('sleep', idle, busy)
                    time.sleep(1)

    else:
        locations = {}
        seeds = np.random.randint(0, maxseed, size=(batchsize, ), dtype=int)
        batch = ((run, s, {}) for s in seeds)
        imap = pool.imap_unordered(mp_run, batch)
        batch = np.array(list(tqdm.tqdm(imap, total=batchsize)))
        if processing:
            batch = [p({}, targets, batch) for p in processing]
        data = [batch]

    for iq, oq, p in workers:
        iq.put(('halt', ))
    for iq, oq, p in workers:
        p.join()

    return locations, data


def mpi_simulate(system, processing=None, batchsize=1, axes=None, n_workers=8):
    mpirun_spec = {
        'seed': 0,
        'system': system,
        'processing': [(p.__module__, p.__name__) for p in processing],
        'batchsize': batchsize,
        'axes': tuple((a, tuple(v)) for a, v in axes),
    }
    mpirun_path = os.path.join(workpath, 'run.json')
    with open(mpirun_path, 'w') as f:
        f.write(json.dumps(mpirun_spec, indent=4))
    mpiout_path = os.path.join(workpath, 'run.pkl')
    mpiargs = (n_workers, 'cluster.py', mpirun_path, mpiout_path)
    cmd = 'mpiexec -n %d python %s %s %s' % mpiargs
    mpiprocess = subprocess.Popen(cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        shell=True, universal_newlines=True)
    for line in iter(mpiprocess.stdout.readline, ''):
        print(line, end='')
    mpiprocess.stdout.close()
    return_code = mpiprocess.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
    print('loading output data...')
    with open(mpiout_path, 'rb') as f:
        locations, data = pickle.loads(f.read())
    print('loaded output data')
    return locations, data


