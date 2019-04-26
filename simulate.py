from contextlib import contextmanager
import multiprocessing as mp
import subprocess
import pickle
import json
import tqdm
import time
import sys
import os
import re
import numpy as np
from simulator import get_simulator
from pscan import walk_space, organize_pscans
from config import maxseed, workpath
import mpi


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


def simulate(system, processing=[], batchsize=1, axes=None,
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
        while any([datum is None for datum in data]):
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


@contextmanager
def progress_bars(n_locations):

    dispatcher_start_re = re.compile(r'^dispatch \d+ start: \d+ workers$')
    worker_start_re = re.compile(r'^worker \d+ start$')
    setting_up_re = re.compile(r'^worker \d+ setting up$')
    setup_re = re.compile(r'^worker \d+ ready to simulate$')
    update_re = re.compile(r'^worker \d+ ran location \d+$')

    update_bar = tqdm.tqdm(total=n_locations, desc='Scanning Parameters')

    def handle_input_line(l, n_workers=-1):
        if update_re.match(l):
            r = l.split()[1]
            d = 'worker %s / %d ran location' % (r, n_workers)
            update_bar.set_description(d)
            update_bar.update(1)
        elif dispatcher_start_re.match(l):
            n_workers = int(l.split()[3])
        elif worker_start_re.match(l):
            r = l.split()[1]
            d = 'started worker %s / %d' % (r, n_workers)
            update_bar.set_description(d)
        elif setting_up_re.match(l):
            r = l.split()[1]
            d = 'worker %s / %d setting up' % (r, n_workers)
            update_bar.set_description(d)
        elif setup_re.match(l):
            r = l.split()[1]
            d = 'worker %s / %d setup up' % (r, n_workers)
            update_bar.set_description(d)
        else:
            print(l, end='')

    try:
        yield handle_input_line
    finally:
        update_bar.close()


def mpi_simulate(system, processing=[], batchsize=1, axes=None,
                 n_workers=8, hostfile=None):
    mpirun_spec = {
        'seed': 0,
        'system': system,
        'processing': [((p.__module__, p.__name__), t) for p, t in processing],
        'batchsize': batchsize,
        'axes': tuple((a, tuple(v)) for a, v in axes),
    }

    mpirun_path = os.path.join(workpath, 'run.json')
    mpiout_path = os.path.join(workpath, 'run.pkl')

    os.makedirs(os.path.dirname(mpirun_path), exist_ok=True)
    with open(mpirun_path, 'w') as f:
        f.write(json.dumps(mpirun_spec, indent=4))

    if hostfile:
        mpiconfig = '--nooversubscribe --hostfile %s' % hostfile
    else:
        mpiconfig = '-n %d' % n_workers
    mpiargs = (mpiconfig, sys.executable, 'cluster.py',
               mpirun_path, mpiout_path)
    cmd = 'mpiexec %s %s %s %s %s' % mpiargs

    n_locations = np.cumprod([len(v) for a, v in axes])[-1]
    with progress_bars(n_locations) as line_handler:
        mpiprocess = subprocess.Popen(cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True, universal_newlines=True)
        for line in iter(mpiprocess.stdout.readline, ''):
            line_handler(line)
        mpiprocess.stdout.close()

    return_code = mpiprocess.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

    print('loading output data...')
    with open(mpiout_path, 'rb') as f:
        locations, data = pickle.loads(f.read())
    print('loaded output data')

    pspace, pscans = organize_pscans(locations, data, len(processing))
    return pspace, pscans


