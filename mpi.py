from mpi4py import MPI
from collections import defaultdict
import time


mpiroot = 0


def root():
    return MPI.COMM_WORLD.rank == mpiroot


def rank():
    return MPI.COMM_WORLD.rank


def size():
    return MPI.COMM_WORLD.size


def host():
    return MPI.Get_processor_name()


def hosts():
    # return a dict of ranks organized by hosts
    hs = defaultdict(list)
    for c in range(size()):
        if c == mpiroot:
            hs['root'].append(c)
        else:
            broadcast(('host', ), c)
            hs[pollrecv(c)].append(c)
    return hs


def broadcast(m, *dests):
    if size() > 1:
        if isinstance(m, tuple):
            for sm in m:
                broadcast(sm, *dests)
        else:
            for l in (dests if dests else range(size())):
                if not l == rank():
                    MPI.COMM_WORLD.send(m, dest=l)


def pollrecv(r = None,d = 0.0000001,md = 0.001,i = 0.0001,e = 0.001):
    # effectively recv, but use polling to ease cores...
    r = MPI.ANY_SOURCE if r is None else r
    while True:
        if MPI.COMM_WORLD.Iprobe(source = r):
            return MPI.COMM_WORLD.recv(source = r)
        else:
            time.sleep(d)
            d = (d + i * (md - d)) if (d < md - e) else md


def passrecv(r=None):
    # check if recv can be immediately done, 
    #   return message if so, None otherwise
    r = MPI.ANY_SOURCE if r is None else r
    if MPI.COMM_WORLD.Iprobe(source=r):
        return MPI.COMM_WORLD.recv(source=r)


