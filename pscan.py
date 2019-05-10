"""Functions for working with parameter scans."""

import itertools
import numpy as np
import pandas as pd
import plot


def pscan_view(pscan, subspace, *targets):
    """Select a subset of the output of a parameter scan.

    Args:
        pscan (pandas.DataFrame): The output of a parameter scan associated with a
            single measurement process.
        subspace (dict): Dictionary where key, value pairs designate a subspace of
            the scanned parameter space.
        targets (seq): Sequence of targets in the parameter scan output to select.

    Returns:
        pandas.DataFrame: pandas.DataFrame selected from pscan pandas.DataFrame.

    """
    subpscan = tuple(((pscan[k] == v) for k, v in subspace.items()))
    subpscan = np.bitwise_and.reduce(subpscan) if subpscan else True
    istarget = pscan['Target'].apply(lambda t: t in set(targets))
    selection = pscan.loc[istarget & subpscan]
    return selection.reset_index()


def organize_pscans(locations, data, n_processing):
    """"""
    pspace = pd.DataFrame(locations)
    if n_processing:
        pscans = []
        for j in range(n_processing):
            entries = []
            for (l, location) in pspace.iterrows():
                entries.append(data[l][j])
                #entries.extend(data[l][j])
            pscans.append(pd.DataFrame(entries))
        data = pscans
    return pspace, data


def walk_space(axes):
    """Generate a trajectory to in a parameter space defined by a set of axes.

    Args:
        axes (seq): Sequence of 2-element tuples where each tuple defines a name of
            an axis (an input parameter for the simulation) and a range of values.

    Returns:
        2-element tuple: The first element is a dictionary of axis name, axis value
        pairs. The second element is an enumeration of parameter space positions
        where each position is a dictionary which can be used as simulation input.

    """
    names, values = zip(*axes)
    n = np.cumprod([len(av) for av in values])[-1]
    locations = list(itertools.product(*values))
    locations = dict((a, l) for a, l in zip(names, zip(*locations)))
    trajectory = (dict((a, locations[a][l]) for a in names) for l in range(n))
    return locations, list(enumerate(trajectory))


