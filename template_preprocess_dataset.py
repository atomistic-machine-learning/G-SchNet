import collections
import argparse
import sys
import time
import numpy as np
import logging
from ase.db import connect
from scipy.spatial.distance import pdist, squareform
from utility_classes import ConnectivityCompressor, Molecule
from multiprocessing import Process, Queue
from pathlib import Path

# list names of collected statistics here (e.g. the number of atoms of each type)
stat_heads = ['n_atoms', 'C', 'N', 'O', 'F', 'H']
atom_types = [6, 7, 8, 9, 1]  # atom type charges in the same order as in stat_heads


def preprocess_dataset(datapath, new_db_path=None, cutoff=2.0,
                       precompute_distances=True, remove_invalid=True,
                       invalid_list=None, valence_list=None, logging_print=True):
    '''
    Pre-processes all molecules of a dataset.
    Along with a new database containing the pre-processed molecules, a
    "input_db_invalid.txt" file holding the indices of removed molecules and a
    "new_db_statistics.npz" file (containing atom count statistics for all molecules in
    the new database) are stored.

    Args:
        datapath (str): full path to dataset (ase.db database)
        new_db_path (str, optional): full path to new database where pre-processed
            molecules shall be stored (None to simply append "gen" to the name in
            datapath, default: None)
        cutoff (float, optional): cutoff value in angstrom used to determine which
            atoms in a molecule are considered as neighbors (i.e. connected, default:
            2.0)
        precompute_distances (bool, optional): if True, the pairwise distances between
            atoms in each molecule are computed and stored in the database (default:
            True)
        remove_invalid (bool, optional): if True, molecules that do not pass the
            validity or connectivity checks are removed from the new database (default:
            True)
        invalid_list (list of int, optional): precomputed list containing indices of
            molecules that are marked as invalid (default: None)
        valence_list (list, optional): the valence of atom types in the form
            [type1 valence type2 valence ...] which could be used for valence checks
            (not implemented, default: None)
        logging_print (bool, optional): set True to show output with logging.info
            instead of standard printing (default: True)
    '''
    # convert paths
    datapath = Path(datapath)
    if new_db_path is None:
        new_db_path = datapath.parent / (datapath.stem + 'gen.db')
    else:
        new_db_path = Path(new_db_path)

    def _print(x, end='\n', flush=False):
        if logging_print:
            logging.info(x)
        else:
            print(x, end=end, flush=flush)

    with connect(datapath) as db:
        n_all = db.count()
    if n_all == 0:
        _print('No molecules found in data base!')
        sys.exit(0)
    _print('\nPre-processing data...')
    if logging_print:
        _print(f'Processed:      0 / {n_all}...')
    else:
        _print(f'0.00%', end='', flush=True)

    # setup counter etc.
    count = 0  # count number of discarded (invalid etc.) molecules
    disc = []  # indices of disconnected structures
    inval = []  # indices of invalid structures
    stats = np.empty((len(stat_heads), 0))  # scaffold for statistics
    start_time = time.time()
    compressor = ConnectivityCompressor()  # used to compress connectivity matrices
    # check if list of invalid molecules was provided and cast it into a set (allows
    # for faster lookup)
    if invalid_list is not None and remove_invalid:
        invalid_list = {*invalid_list}
        n_inval = len(invalid_list)
    else:
        n_inval = 0

    # preprocess each structure in the source db and write results into target db
    with connect(datapath) as source_db:
        with connect(new_db_path) as target_db:
            for i in range(source_db.count()):

                # skip molecule if index is present in precomputed list of invalid
                # molecules and if remove_invalid is True
                if remove_invalid and invalid_list is not None:
                    if i in invalid_list:
                        continue

                # get molecule from database
                row = source_db.get(i + 1)
                # extract additional data stored with molecule
                data = row.data
                # get ase.Atoms object
                at = row.toatoms()
                # get positions and atomic numbers
                pos = at.positions
                numbers = at.numbers

                # the algorithm to sample generation traces (atom placement steps)
                # assumes that the atoms in our structures are ordered by their
                # distance to the center of mass, thus we order them in that way here:

                # center positions (using center of mass)
                pos = pos - at.get_center_of_mass()
                # order atoms by distance to center of mass
                center_dists = np.sqrt(np.maximum(np.sum(pos ** 2, axis=1), 0))
                idcs_sorted = np.argsort(center_dists)
                pos = pos[idcs_sorted]
                numbers = numbers[idcs_sorted]
                # update positions and atomic numbers accordingly in ase.Atoms object
                at.positions = pos
                at.numbers = numbers

                # retrieve connectivity matrix (and pairwise distances)
                connectivity, pairwise_distances = get_connectivity(at, cutoff)

                # check if the connectivity matrix represents a proper structure (i.e.
                # if all atoms are connected to each other via some path) as
                # disconnected structures cannot be used for training (there must be
                # an atom placement trajectory for G-SchNet)
                if is_disconnected(connectivity):
                    count += 1
                    disc += [i]
                    continue

                # you could potentially implement some valency constraint checking here
                # and remove or mark molecules that do not pass the test
                # val = [check validity e.g. with connectivity and valence list]
                # if remove_invalid:
                #     if not val:
                #         count += 1
                #         inval += [i]
                #         continue

                # update data stored in db with a compressed version of the
                # connectivity matrix (we store only indices of entries >= 1
                data.update({'con_mat': compressor.compress(connectivity)})

                # if desired, also store precomputed distances (in condensed format)
                if precompute_distances:
                    data.update({'dists': pairwise_distances})

                # write preprocessed molecule and data to target database
                target_db.write(at, data=data)

                # you can additionally gather some statistics about the training data
                # (these statistics can for example be used to filter molecules when
                # displaying them with the display_molecules.py script)
                # e.g. for QM9 we collected the atom, bond, and ring count statistics
                # when doing valency checks
                # here we simply count the number of atoms of each type
                atom_type_counts = np.bincount(numbers, minlength=10)
                # store counts [n_atoms, C, N, O, F, H] as listed in stat_heads
                statistics = np.array([len(numbers), *atom_type_counts[atom_types]])
                # update stats array with statistics of current molecule
                stats = np.hstack((stats, statistics[:, None]))

                # print progress every 1000 molecules
                if (i+1) % 1000 == 0:
                    _print(f'Processed: {i+1:6d} / {n_all}...')

    if not logging_print:
        _print('\033[K', end='\n', flush=True)
    _print(f'... successfully validated {n_all - count - n_inval} data '
           f'points!', flush=True)
    if invalid_list is not None:
        _print(f'{n_inval} structures were removed because they are on the '
               f'pre-computed list of invalid molecules!', flush=True)
        if len(disc)+len(inval) > 0:
            _print(f'CAUTION: Could not validate {len(disc)+len(inval)} additional '
                   f'molecules. You might want to increase the cutoff (currently '
                   f'{cutoff} angstrom) in order to have less disconnected structures. '
                   f'The molecules were removed and their indices are '
                   f'appended to the list of invalid molecules stored at '
                   f'{datapath.parent / (datapath.stem + f"_invalid.txt")}',
                   flush=True)
            np.savetxt(datapath.parent / (datapath.stem + f'_invalid.txt'),
                       np.append(np.sort(list(invalid_list)), np.sort(inval + disc)),
                       fmt='%d')
    elif remove_invalid:
        _print(f'Identified {len(disc)} disconnected structures, and {len(inval)} '
               f'invalid structures! You might want to increase the cutoff (currently '
               f'{cutoff} angstrom) in order to have less disconnected structures.',
               flush=True)
        np.savetxt(datapath.parent / (datapath.stem + f'_invalid.txt'),
                   np.sort(inval + disc), fmt='%d')
    _print('\nCompressing and storing statistics with numpy...')
    np.savez_compressed(new_db_path.parent/(new_db_path.stem+f'_statistics.npz'),
                        stats=stats,
                        stat_heads=stat_heads)

    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    _print(f'Done! Pre-processing needed {h:d}h{m:02d}m{s:02d}s.')


def get_connectivity(mol, cutoff=2.0):
    '''
    Write code to obtain a connectivity matrix given a molecule from your database
    here. The simple default implementation calculates pairwise distances and then
    uses a radial cutoff (e.g. 2 angstrom) to determine which atoms are labeled as
    connected. The matrix only needs to be binary as it is only used to sample
    generation traces, i.e. an order of atom placement steps for training.
    However, one could for example also use chemoinformatics tools in order to obtain
    bond order information and check the valence of provided structures on the run if
    the structures allow this (we did this for our experiments with QM9 in order to
    allow for comparison to related work, but we think that using a radial cutoff is
    actually more robust and more general as it does not depend on usually unreliable
    bond order assignment algorithms and can be used for all kinds of materials or
    molecules).
    Args:
        mol (ase.Atoms): one molecule from the database
        cutoff (float, optional): cutoff value in angstrom used to determine which
            atoms are connected

    Returns:
        numpy.ndarray: the computed connectivity matrix (n_atoms x n_atoms, float)
        numpy.ndarray: the computed pairwise distances in a condensed format
            (length is n_atoms*(n_atoms-1)/2), see scipy.spatial.distance.pdist for
            more information
    '''
    # retrieve positions
    atom_positions = mol.get_positions()
    # get pairwise distances (condensed)
    pairwise_distances = pdist(atom_positions)
    # use cutoff to obtain connectivity matrix (condensed format)
    connectivity = np.array(pairwise_distances <= cutoff, dtype=float)
    # cast to redundant square matrix format
    connectivity = squareform(connectivity)
    # set diagonal entries to zero (as we do not assume atoms to be their own neighbors)
    connectivity[np.diag_indices_from(connectivity)] = 0
    return connectivity, pairwise_distances


def is_disconnected(connectivity):
    '''
    Assess whether all atoms of a molecule are connected using a connectivity matrix

    Args:
        connectivity (numpy.ndarray): matrix (n_atoms x n_atoms) indicating bonds
            between atoms

    Returns
        bool: True if the molecule consists of at least two disconnected graphs,
            False if all atoms are connected by some path
    '''
    con_mat = connectivity
    seen, queue = {0}, collections.deque([0])  # start at node (atom) 0
    while queue:
        vertex = queue.popleft()
        # iterate over (bonded) neighbors of current node
        for node in np.argwhere(con_mat[vertex] > 0).flatten():
            # add node to queue and list of seen nodes if it has not been seen before
            if node not in seen:
                seen.add(node)
                queue.append(node)
    # if the seen nodes do not include all nodes, there are disconnected parts
    return seen != {*range(len(con_mat))}
