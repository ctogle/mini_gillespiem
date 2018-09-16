"""Functions for MPI based parallelization of simulation.
Module can be run as __main__ to for MPI parallelization.

"""
import itertools
import numpy as np
import pickle
import json
import time
import sys
import mpi
from simulator import get_simulator


maxseed = 2147483647
#workpath = './.simulators'


def worker():
    """Worker function run by subordinate mpi processes."""
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
            np.random.seed(mpi.pollrecv(mpi.mpiroot))
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
    """Function to coordinate seeds and simulator compilation for
    subordinate MPI processes.

    Args:
        seed (int): Seed for superior random number generator which
            sets the seeds of subordinate MPI processes.
        system (dict): Dictionary describing the network to simulate.
        processing (seq): Sequence of module and function name pairs
            which are imported and used by subordinate MPI processes.

    """
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
    """Dispatch function for coordinating subordinate MPI processes.

    Args:
        mpirun_path (str): Path to json file describing required
            simulation and processing.
        mpiout_path (str): Path where trajectory and results are stored.

    """
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


