from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from io import StringIO
import numpy as np
import glob
import sys
import os
import re


def write_nparray(src, name, shape, dtype='numpy.double', spacer='\n    '):
    ctype = dtype[dtype.find('.') + 1:]
    shapeslice = ','.join([':'] * len(shape))
    form = '%scdef %s [%s] %s = numpy.zeros(%s, dtype=%s)'
    elements = (spacer, ctype, shapeslice, name, str(shape), dtype)
    src.write(form % elements)


def write_carray(src, name, shape, dtype='double', spacer='\n    '):
    shapeslice = ','.join([str(s) for s in shape])
    form = '%scdef %s %s [%s]'
    elements = (spacer, dtype, name, shapeslice)
    src.write(form % elements)


def write_cython_functions(src, fs, statetargets, duration):

    def convert(substr):
        if substr in fns or substr in extfns.values():
            return '%s(state)' % substr
        elif substr in statetargets:
            return 'state[%d]' % statetargets.index(substr)
        else:
            return substr

    def read(seq,sx,oc = '(',cc = ')'):
        score = 1
        sx += 1
        while score > 0:
            sx += 1
            if seq[sx] == oc:score += 1
            elif seq[sx] == cc:
                if score > 0:score -= 1
        return sx

    def ext_func_name_gen(maxnum = 100):
        fnum = 0
        while fnum < maxnum:
            fnum += 1
            fname = 'extfunc_'+str(fnum)
            yield fname

    def write_extfunc(src,fname,fpath,dom,targets,etime,backsrc):
        if not os.path.exists(fpath):
            print('missing external signal file: \'%s\'' % fpath)
            raise ValueError
        x,y = [],[]
        with open(fpath,'r') as fh:
            extsig = fh.readlines()
            for esl in extsig:
                esl = esl.strip().split(',')
                if esl and esl[0]:
                    nx,ny = esl
                    x.append(nx)
                    y.append(ny)

        dshape = (len(x),)
        write_nparray(src,fname+'_domain',dshape,spacer = '\n')
        write_nparray(src,fname+'_codomain',dshape,spacer = '\n')
        src.write('\n')
        for j in range(len(x)):
            src.write(fname+'_domain['+str(j)+'] = '+x[j]+';')
            if j % 25 == 0 and j > 0:src.write('\n')
            if float(x[j]) > etime:break
        src.write('\n')
        for j in range(len(y)):
            src.write(fname+'_codomain['+str(j)+'] = '+y[j]+';')
            if j % 25 == 0 and j > 0:src.write('\n')
            if float(x[j]) > etime:break
        argstring = 'double['+str(len(targets))+'] state'
        src.write('\ncdef int '+fname+'_lastindex = 0')
        backsrc.write('\n    global '+fname+'_lastindex')
        backsrc.write('\n    '+fname+'_lastindex = 0')
        src.write('\ncdef inline double '+fname+'('+argstring+'):')

        src.write('\n    global '+fname+'_domain')
        src.write('\n    global '+fname+'_codomain')
        src.write('\n    global '+fname+'_lastindex')
        sdx = targets.index(dom)
        src.write('\n    cdef double xcurrent = state['+str(sdx)+']')
        src.write('\n    cdef double domvalue = '+fname+'_domain')
        src.write('['+fname+'_lastindex]')
        src.write('\n    cdef double codomvalue')
        src.write('\n    while xcurrent > domvalue:')
        src.write('\n        '+fname+'_lastindex'+' += 1')
        src.write('\n        domvalue = '+fname+'_domain')
        src.write('['+fname+'_lastindex]')
        src.write('\n    codomvalue = '+fname+'_codomain')
        src.write('['+fname+'_lastindex-1]')
        src.write('\n    return codomvalue\n')

    backsrc = StringIO()

    fng = ext_func_name_gen()
    fns = tuple(f[0] for f in fs)

    extfns = {}
    extstr = 'external_signal('
    for fn, fu in fs:
        escnt = fu.count('external_signal')

        if escnt == 1:
            nfn = next(fng)
            fid = fu.find(extstr) + len(extstr)
            espath, esdom = fu[fid:read(fu, fid, '(', ')')].split(',')
            fline = extstr+espath+','+esdom+')'
            extfns[fline] = nfn
            fu = fu.replace(fline, nfn)
            write_extfunc(src, nfn, espath, esdom, statetargets, duration, backsrc)

        elif escnt > 1:
            raise ValueError('can\'t support two external signals in a function...')

        fstrng = ''.join([convert(substr) for substr in re.split('(\W)',fu)])
        selfdex = statetargets.index(fn)
        elements = (fn, len(statetargets), fstrng, selfdex)

        src.write('''
cdef inline double %s(double [%d] state):
    cdef double val = %s
    state[%d] = val
    return val\n''' % elements)

    return backsrc


def write_propensity(rxn, stargets, funcs):
    uchecks = []
    for u in rxn[1]:
        ucnt,uspec = u
        udex = stargets.index(uspec)
        uline = ['(state['+str(udex)+']-'+str(x)+')' for x in range(ucnt)]
        uline[0] = uline[0].replace('-0','')
        if ucnt > 1:uline.append(str(1.0/ucnt))
        uchecks.append('*'.join(uline))
    try: ratestring = str(float(rxn[0]))
    except ValueError:
        found = False
        for f in funcs:
            if rxn[0] in f[0]:
                ratestring = rxn[0]+'(state)'
                found = True
                break
        if not found:
            ratestring = 'state['+str(stargets.index(rxn[0]))+']'
    uchecks.append(ratestring)
    rxnpropexprsn = '*'.join(uchecks)
    return rxnpropexprsn


def gibson_lookup(rxns,funcs):

    def depends_on(fn,fu,fs,d):
        if fu.count(fn) > 0:
            raise ValueError('cannot support recursive functions in simulation!')
        for othfn,othfu in fs:
            if fu.count(othfn) > 0:
                if depends_on(othfn,othfu,fs,d):
                    return True
        return fu.count(d) > 0

    fnames = tuple(f[0] for f in funcs)
    rcnt = len(rxns)
    alwayses = [d for d in range(rcnt) if rxns[d][0] in fnames]
    lookups = [[] for r in rxns]
    for rdx in range(rcnt):
        # enumerate the species affected by rxns[rdx]
        r = rxns[rdx]
        affected_species = []
        for p in r[2]:
            found = False
            for u in r[1]:
                if u[1] == p[1]:
                    found = True
                    if not u[0] == p[0] and not p[1] in affected_species:
                        affected_species.append(p[1])
            if not found and not p[1] in affected_species:
                affected_species.append(p[1])
        for u in r[1]:
            found = False
            for p in r[2]:
                if p[1] == u[1]:
                    found = True
                    if not p[0] == u[0] and not u[1] in affected_species:
                        affected_species.append(u[1])
            if not found and not u[1] in affected_species:
                affected_species.append(u[1])
        #print 'rxn',r[3],affected_species
        for alw in alwayses:
            fratex = fnames.index(rxns[alw][0])
            func,fuu = funcs[fratex]
            if depends_on(func,fuu,funcs,'time'):
                lookups[rdx].append(alw)
                continue
            for affs in affected_species:
                if depends_on(func,fuu,funcs,affs):
                    lookups[rdx].append(alw)
                    break
        for rdx2 in range(rcnt):
            r2 = rxns[rdx2]
            for u2 in r2[1]:
                if u2[1] in affected_species:
                    if not rdx2 in lookups[rdx]:
                        lookups[rdx].append(rdx2)
    return lookups


def write_simulator(system):
    duration = system.get('duration', 1.0)
    resolution = system.get('resolution', 0.1)
    species = system.get('species', [])
    reactions = system.get('reactions', [])
    variables = system.get('variables', [])
    functions = system.get('functions', [])
    targets = system.get('targets', [])

    ccnt = int(duration / resolution) + 1
    scnt = len(species)
    rcnt = len(reactions)
    vcnt = len(variables)
    fcnt = len(functions)
    tcnt = len(targets)

    lookup = gibson_lookup(reactions, functions)

    statetargets = ['time'] + [t[0] for t in (species + variables + functions)]

    dshape, sshape, cshape = (tcnt, ccnt), (1 + scnt + vcnt + fcnt, ), (tcnt, )

    src = StringIO()
    src.write('''
# cython:profile=False,boundscheck=False,nonecheck=False,wraparound=False,initializedcheck=False,cdivision=True
###################################
# imports:
from libc.math cimport log
from libc.math cimport sin
from libc.math cimport cos

#from libc.stdlib cimport rand
#cdef extern from "limits.h":
#   int INT_MAX

from numpy import cumprod as cumulative_product
cdef double pi = 3.14159265359
import random
import numpy
import time as timemodule
from cython.view cimport array as cvarray

cdef inline double heaviside(double value):
    if value >= 0.0:return 1.0
    else:return 0.0\n''')

    backsrc = write_cython_functions(src, functions, statetargets, duration)

    kwstring = []
    kwstring.extend([('int %s=%d' % (s, i)) for s, i in species])
    kwstring.extend([('float %s=%f' % (v, f)) for v, f in variables])
    src.write('\ncpdef simulate(int seed, %s):\n' % ', '.join(kwstring))

    src.write('%s\n' % backsrc.getvalue())

    write_nparray(src, 'data', dshape)
    write_carray(src, 'capture', cshape)
    write_carray(src, 'state', sshape)

    src.write('\n    state[0] = 0.0')

    for sx, (sp, si) in enumerate(species):
        src.write('\n    state[%d] = %s' % (sx + 1, sp))

    for vx, (vn, vv) in enumerate(variables):
        src.write('\n    state[%d] = %s' % (scnt + vx + 1, vn))

    for fx, (fn, fu) in enumerate(functions):
        src.write('\n    %s(state)' % fn)

    src.write('''
    random.seed(seed)
    cdef int totalcaptures = %d
    cdef int capturecount = 0
    cdef int rtabledex
    cdef int tdex
    cdef int cdex
    cdef double totalpropensity
    cdef double tpinv
    cdef double time = 0.0
    cdef double lasttime = 0.0
    cdef double realtime = 0.0
    cdef double del_t = 0.0
    cdef double randr
    cdef int whichrxn = 0
    cdef int rxncount = %d''' % (ccnt, rcnt))

    write_carray(src, 'reactiontable', (rcnt, ))
    write_carray(src, 'propensities', (rcnt, ))

    for j, reaction in enumerate(reactions):
        propensity = write_propensity(reaction, statetargets, functions)
        src.write('\n    propensities[%d] = %s' % (j, propensity))

    write_carray(src, 'tdexes', cshape, dtype='int')

    for j, target in enumerate(targets):
        src.write('\n    tdexes[%d] = %d' % (j, statetargets.index(target)))

    src.write('''
    while capturecount < totalcaptures:
        totalpropensity = 0.0''')
    for j, (rate, used, produced) in enumerate(reactions):
        checks = [('state[%d] >= %d' % (statetargets.index(s), c)) for c, s in used]
        checks = ('if %s:' % ' and '.join(checks)) if checks else ''
        src.write('\n        %s' % checks)
        src.write('totalpropensity += propensities[%d]' % j)
        src.write('\n        reactiontable[%d] = totalpropensity' % j)

    src.write('''
        if totalpropensity > 0.0:
            tpinv = 1.0 / totalpropensity
            del_t = -1.0 * log(random.random()) * tpinv
            randr = random.random()*totalpropensity
            for rtabledex in range(rxncount):
                if randr < reactiontable[rtabledex]:
                    whichrxn = rtabledex
                    break
        else:
            del_t = %f
            whichrxn = -1
        state[0] += del_t
        realtime = state[0]
        while lasttime < realtime and capturecount < totalcaptures:
            state[0] = lasttime
            lasttime += %f''' % (resolution, resolution))

    for j, (fn, fu) in enumerate(functions):
        src.write('\n            %s(state)' % fn)

    src.write('''
            for cdex in range(%d):
                data[cdex, capturecount] = state[tdexes[cdex]]
            capturecount += 1
        state[0] = realtime''' % (tcnt, ))

    src.write('\n        if whichrxn == -1:')
    for j, reaction in enumerate(reactions):
        propensity = write_propensity(reaction, statetargets, functions)
        src.write('\n            propensities[%d] = %s' % (j, propensity))

    for j, (rate, used, produced) in enumerate(reactions):
        src.write('\n        elif whichrxn == %d:' % j)
        for c, s in used:
            src.write('\n            state[%d] -= %d' % (statetargets.index(s), c))
        for c, s in produced:
            src.write('\n            state[%d] += %d' % (statetargets.index(s), c))
        for look in lookup[j]:
            propensity = write_propensity(reactions[look], statetargets, functions)
            src.write('\n            propensities[%d] = %s' % (look, propensity))

    src.write('\n\n    return numpy.array(data, dtype=numpy.float)\n')
    return src.getvalue()


def get_simulator(system={}, path='./.simulators', name='gillespie'):
    '''
    Generate cython source code of a gillespie simulator for a chemical network
    '''
    source = write_simulator(system)

    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(path)
    versions = glob.glob('./%s_*.pyx' % name)
    versions = [int(v[v.rfind('_') + 1:v.rfind('.pyx')]) for v in versions]
    latest = max(versions) if versions else 0
    latestname = '%s_%d' % (name, latest)
    changed = True
    if latestname in sys.modules:
        ext_path = './%s.pyx' % latestname
        if os.path.exists(ext_path):
            with open(ext_path, 'r') as fh:
                if source == fh.read():
                    print('existing cython code is identical...')
                    changed = False
        if changed:
            print('incrementing extension version...')
            for old in glob.glob('./%s_*' % name):
                os.remove(old)
            latestname = '%s_%d' % (name, latest + 1)
    print('ext module version:', latestname, latest)
    ext_path = './%s.pyx' % latestname
    if changed:
        with open(ext_path, 'w') as fh:
            fh.write(source)
        extensions = cythonize([Extension(latestname, [ext_path])])
        includes = [np.get_include()]
        setup(script_args=['clean'],
              ext_modules=extensions,
              include_dirs=includes)
        os.makedirs('./build', exist_ok=True)
        setup(script_args=['build_ext', '--inplace'],
              ext_modules=extensions,
              include_dirs=includes)
    run = __import__(latestname).simulate
    os.chdir(cwd)
    return run


