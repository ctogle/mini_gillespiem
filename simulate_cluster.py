import subprocess
import itertools
import numpy as np
import pickle
import json
import time
import sys
import os
import mpi
from simulator import get_simulator


maxseed = 2147483647
workpath = './.simulators'


def worker():
    print('worker %d start' % mpi.rank())
    while True:
        msg = mpi.pollrecv(mpi.mpiroot)
        if msg == 'halt':
            print('worker %d halt' % mpi.rank())
            break
        elif msg == 'host':
            mpi.broadcast((mpi.host(), ), mpi.mpiroot)
            #print('worker %d host' % mpi.rank())
        elif msg == 'seed':
            seed = mpi.pollrecv(mpi.mpiroot)
            np.random.seed(seed)
            #print('worker %d seed %d' % (mpi.rank(), seed))
        elif msg == 'setup':
            system = mpi.pollrecv(mpi.mpiroot)
            processing = mpi.pollrecv(mpi.mpiroot)
            install = mpi.pollrecv(mpi.mpiroot)
            if install:
                print('worker %d setting up' % mpi.rank())
            run = get_simulator(system, changed=install)
            for j, (module, function) in enumerate(processing):
                processing[j] = __import__(module).__getattribute__(function)
            targets = system.get('targets', [])
            mpi.broadcast(('complete', ), mpi.mpiroot)
            print('worker %d ready to simulate' % mpi.rank())
        elif msg == 'run':
            l = mpi.pollrecv(mpi.mpiroot)
            location = mpi.pollrecv(mpi.mpiroot)
            batchsize = mpi.pollrecv(mpi.mpiroot)
            print('worker %d run location %d' % (mpi.rank(), l))
            seeds = np.random.randint(0, maxseed, size=(batchsize, ), dtype=int)
            batch = np.array([run(s, **location) for s in seeds])
            if processing:
                batch = [p(location, targets, batch) for p in processing]
            mpi.broadcast((mpi.rank(), l, batch), mpi.mpiroot)
            print('worker %d ran location %d' % (mpi.rank(), l))
        else:
            print('msg worker!', mpi.rank(), msg)


def setup_workers(seed, system, processing):
    print('setting up workers')
    np.random.seed(seed)
    seeds = np.random.randint(0, maxseed, size=(mpi.size() - 1, ), dtype=int)
    for c, seed in zip(range(mpi.size()), seeds):
        if not c == mpi.mpiroot:
            mpi.broadcast(('seed', seed), c)
    cluster = mpi.hosts()
    for host in cluster:
        if not host == 'root':
            for j, c in enumerate(cluster[host]):
                mpi.broadcast(('setup', system, processing, j == 0), c)
                assert(mpi.pollrecv(c) == 'complete')
    print('set up workers')


def dispatch(mpirun_path, mpiout_path):
    print('start dispatch', mpi.rank(), mpirun_path)
    with open(mpirun_path, 'r') as f:
        mpirun = json.loads(f.read())
    setup_workers(mpirun['seed'], mpirun['system'], mpirun['processing'])
    idle, busy = [j for j in range(mpi.size()) if not (j == mpi.rank())], []
    if mpirun['axes']:
        locations, trajectory = walk_space(mpirun['axes'])
        data = [None] * len(trajectory)
        while None in data:
            if idle and trajectory:
                c = idle.pop(0)
                l, loc = trajectory.pop(0)
                mpi.broadcast(('run', l, loc, mpirun['batchsize']), c)
                busy.append(c)
            elif busy:
                c = mpi.passrecv()
                if c:
                    l = mpi.pollrecv(c)
                    batch = mpi.pollrecv(c)
                    data[l] = batch
                    idle.append(c)
                    busy.remove(c)
                else:
                    time.sleep(1)
    else:
        raise NotImplementedError
    mpi.broadcast(('halt', ))
    print('saving output data...')
    with open(mpiout_path, 'wb') as f:
        f.write(pickle.dumps((locations, data)))
    print('saved output data')
    print('end dispatch', mpi.rank(), mpirun_path)


def simulate(system, processing=None, batchsize=1, axes=None,
             n_workers=8):
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
    mpiargs = (n_workers, __file__, mpirun_path, mpiout_path)
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


def walk_space(axes):
    names, values = zip(*axes)
    n = np.cumprod([len(av) for av in values])[-1]
    locations = list(itertools.product(*values))
    locations = dict((a, l) for a, l in zip(names, zip(*locations)))
    trajectory = (dict((a, locations[a][l]) for a in names) for l in range(n))
    return locations, list(enumerate(trajectory))


if __name__ == '__main__':
    if mpi.root():
        mpirun_path = sys.argv[1]
        mpiout_path = sys.argv[2]
        results = dispatch(mpirun_path, mpiout_path)
    else:
        worker()

