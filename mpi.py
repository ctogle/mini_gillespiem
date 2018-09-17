"""Convenience functions for using MPI."""
from mpi4py import MPI
from collections import defaultdict
import time


mpiroot = 0
"""The process rank responsible for coordination."""


def root():
    return MPI.COMM_WORLD.rank == mpiroot


def rank():
    return MPI.COMM_WORLD.rank


def size():
    return MPI.COMM_WORLD.size


def host():
    return MPI.Get_processor_name()


def hosts():
    """Collate process ranks by host name

    Returns:
        dict: Each host is associated with a list of ranks.
        The root rank is associated with a nonexistent root host.

    """
    hs = defaultdict(list)
    for c in range(size()):
        if c == mpiroot:
            hs['root'].append(c)
        else:
            broadcast(('host', ), c)
            hs[pollrecv(c)].append(c)
    return hs


def broadcast(m, *dests):
    """Broadcast a message or messages to a set of ranks.

    Args:
        m (obj or tuple of objs): The content being sent.
        *dests : Optional sequence of ranks to which to send content.
        If none are provided, all other ranks are assumed.

    """
    if size() > 1:
        if isinstance(m, tuple):
            for sm in m:
                broadcast(sm, *dests)
        else:
            for l in (dests if dests else range(size())):
                if not l == rank():
                    MPI.COMM_WORLD.send(m, dest=l)


def pollrecv(r=None, d=0.0000001, md=0.001, i=0.0001, e=0.001):
    """Effectively recv, but uses attenuated polling to ease cores."""
    r = MPI.ANY_SOURCE if r is None else r
    while True:
        if MPI.COMM_WORLD.Iprobe(source = r):
            return MPI.COMM_WORLD.recv(source = r)
        else:
            time.sleep(d)
            d = (d + i * (md - d)) if (d < md - e) else md


def passrecv(r=None):
    """Check if anything is waiting to be received.

    Returns:
        obj: Content received if any was ready, otherwise None.

    """
    r = MPI.ANY_SOURCE if r is None else r
    if MPI.COMM_WORLD.Iprobe(source=r):
        return MPI.COMM_WORLD.recv(source=r)


if __name__ == '__main__':
    print('hello world! (%d on %s)' % (rank(), host()))


