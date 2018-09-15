from mpi4py import MPI
import time

mpiroot = 0

def root():
    return MPI.COMM_WORLD.rank == mpiroot

def rank():
    return MPI.COMM_WORLD.rank

def size():
    return MPI.COMM_WORLD.size

def comm():
    return MPI.COMM_WORLD

def host():
    return MPI.Get_processor_name()

def broadcast(m, *dests):
    if size() == 1:
        return
    if type(m) == type([]):
        for sm in m:
            broadcast(sm, *dests)
    else:
        c = comm()
        if not dests:dests = range(size())
        for l in dests:
            if not l == rank():
                c.send(m,dest = l)

# effectively recv, but use polling to ease cores...
def pollrecv(r = None,d = 0.0000001,md = 0.001,i = 0.0001,e = 0.001):
    if r is None:
        r = MPI.ANY_SOURCE
    c = comm()
    while True:
        if c.Iprobe(source = r):
            m = c.recv(source = r)
            return m
        else:
            time.sleep(d)
            d = (d + i * (md - d)) if (d < md - e) else md

# check if recv can be immediately done, 
#   return message if so, None otherwise
def passrecv(r = None):
    if r is None:
        r = MPI.ANY_SOURCE
    m = comm().Iprobe(source = r)
    if m:
        return comm().recv(source = r)

# return a dict of ranks organized by hosts
def hosts():
    hs = {'root' : mpiroot}
    for c in range(size()):
        if c == mpiroot:
            continue
        else:
            broadcast('host',c)
            h = pollrecv(c)
            if h in hs.keys():
                hs[h].append(c)
            else:
                hs[h] = [c]
    return hs


if __name__ == '__main__':
    if root():
        print('root!', rank())
    else:
        print('worker!', rank())


