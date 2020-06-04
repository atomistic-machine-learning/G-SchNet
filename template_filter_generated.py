import numpy as np
import pickle
import os
import argparse
import time

from scipy.spatial.distance import pdist
from schnetpack import Properties
from utility_classes import Molecule, ConnectivityCompressor
from utility_functions import update_dict
from ase import Atoms
from ase.db import connect


def get_parser():
    """ Setup parser for command line arguments """
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('data_path',
                             help='Path to generated molecules in .mol_dict format, '
                                  'a database called "generated_molecules.db" with the '
                                  'filtered molecules along with computed statistics '
                                  '("generated_molecules_statistics.npz") will be '
                                  'stored in the same directory as the input file/s '
                                  '(if the path points to a directory, all .mol_dict '
                                  'files in the directory will be merged and filtered '
                                  'in one pass)')
    main_parser.add_argument('--valence',
                             default=[1, 1, 6, 4, 7, 3, 8, 2, 9, 1], type=int,
                             nargs='+',
                             help='the valence of atom types in the form '
                                  '[type1 valence type2 valence ...] '
                                  '(default: %(default)s)')
    main_parser.add_argument('--filters', type=str, nargs='*',
                             default=['valence', 'disconnected', 'unique'],
                             choices=['valence', 'disconnected', 'unique'],
                             help='Select the filters applied to identify '
                                  'invalid molecules (default: %(default)s)')
    main_parser.add_argument('--store', type=str, default='valid',
                             choices=['all', 'valid'],
                             help='How much information shall be stored '
                                  'after filtering: \n"all" keeps all '
                                  'generated molecules and statistics, '
                                  '\n"valid" keeps only valid molecules'
                                  '(default: %(default)s)')
    main_parser.add_argument('--print_file',
                             help='Use to limit the printing if results are '
                                  'written to a file instead of the console ('
                                  'e.g. if running on a cluster)',
                             action='store_true')
    return main_parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print_file = args.print_file
    printed_todos = False

    # read input file or fuse dictionaries if data_path is a folder
    if not os.path.isdir(args.data_path):
        if not os.path.isfile(args.data_path):
            print(f'\n\nThe specified data path ({args.data_path}) is neither a file '
                  f'nor a directory! Please specify a different data path.')
            raise FileNotFoundError
        else:
            with open(args.data_path, 'rb') as f:
                res = pickle.load(f)  # read input file
            target_db = os.path.join(os.path.dirname(args.data_path),
                                     'generated_molecules.db')
    else:
        print(f'\n\nFusing .mol_dict files in folder {args.data_path}...')
        mol_files = [f for f in os.listdir(args.data_path)
                     if f.endswith(".mol_dict")]
        if len(mol_files) == 0:
            print(f'Could not find any .mol_dict files at {args.data_path}! Please '
                  f'specify a different data path!')
            raise FileNotFoundError
        res = {}
        for file in mol_files:
            with open(os.path.join(args.data_path, file), 'rb') as f:
                cur_res = pickle.load(f)
                update_dict(res, cur_res)
        res = dict(sorted(res.items()))  # sort dictionary keys
        print(f'...done!')
        target_db = os.path.join(args.data_path, 'generated_molecules.db')

    # compute array with valence of provided atom types
    max_type = max(args.valence[::2])
    valence = np.zeros(max_type+1, dtype=int)
    valence[args.valence[::2]] = args.valence[1::2]

    # print the chosen settings
    valence_str = ''
    for i in range(max_type+1):
        if valence[i] > 0:
            valence_str += f'type {i}: {valence[i]}, '
    filters = []
    if 'valence' in args.filters:
        filters += ['valency']
    if 'disconnected' in args.filters:
        filters += ['connectedness']
    if 'unique' in args.filters:
        filters += ['uniqueness']
    if len(filters) >= 3:
        edit = ', '
    else:
        edit = ' '
    for i in range(len(filters) - 1):
        filters[i] = filters[i] + edit
    if len(filters) >= 2:
        filters = filters[:-1] + ['and '] + filters[-1:]
    string = ''.join(filters)
    print(f'\n\n1. Filtering molecules according to {string}...')
    print(f'\nTarget valence:\n{valence_str[:-2]}\n')

    # initial setup of array for statistics and some counters
    n_generated = 0
    n_valid = 0
    n_non_unique = 0
    stat_heads = ['n_atoms', 'id', 'valid', 'duplicating', 'n_duplicates',
                  'known', 'equals', 'C', 'N', 'O', 'F', 'H']
    stats = np.empty((len(stat_heads), 0))
    all_mols = []
    connectivity_compressor = ConnectivityCompressor()

    # iterate over generated molecules by length (all generated molecules with n
    # atoms are stored in one batch, so we loop over all available lengths n)
    # this is useful e.g. for finding duplicates, since we only need to compare
    # molecules of the same length (and can actually further narrow down the
    # candidates by looking at the exact atom type composition of each molecule)
    start_time = time.time()
    for n_atoms in res:
        if not isinstance(n_atoms, int) or n_atoms == 0:
            continue

        prog_str = lambda x: f'Checking {x} for molecules of length {n_atoms}'
        work_str = 'valence' if 'valence' in args.filters else 'dictionary'
        if not print_file:
            print('\033[K', end='\r', flush=True)
            print(prog_str(work_str) + ' (0.00%)', end='\r', flush=True)
        else:
            print(prog_str(work_str), flush=True)

        d = res[n_atoms]  # dictionary containing molecules of length n_atoms
        all_pos = d[Properties.R]  # n_mols x n_atoms x 3 matrix with atom positions
        all_numbers = d[Properties.Z]  # n_mols x n_atoms matrix with atom types
        n_mols = len(all_pos)
        valid = np.ones(n_mols, dtype=int)  # all molecules are valid in the beginning

        # check valency of molecules with length n
        if 'valence' in args.filters:
            if not printed_todos:
                print('Please implement a procedure to check the valence in generated '
                      'molecules! Skipping valence check...')
            # TODO
            # Implement a procedure to assess the valence of generated molecules here!
            # You can adapt and use the Molecule class in utility_classes.py,
            # but the current code is tailored towards the QM9 dataset. In fact,
            # the OpenBabel algorithm to kekulize bond orders is not very reliable
            # and we implemented some heuristics in the Molecule class to fix these
            # flaws for structures made of C, N, O, and F atoms. However, when using
            # more complex structures with a more diverse set of atom types, we think
            # that the reliability of bond assignment in OpenBabel might further
            # degrade and therefore do no recommend to use valence checks for
            # analysis unless it is very important for your use case.

        # detect molecules with disconnected parts if desired
        if 'disconnected' in args.filters:
            if not print_file:
                print('\033[K', end='\r', flush=True)
                print(prog_str("connectedness")+'...', end='\r', flush=True)
            if not printed_todos:
                print('Please implement a procedure to check the connectedness of '
                      'generated molecules! In this template script we will now remove '
                      'molecules where two atoms are closer than 0.3 angstrom as an '
                      'example processing step...')
            # TODO
            # Implement a procedure to assess the connectedness of generated
            # molecules here! You can for example use a connectivity matrix obtained
            # from kekulized bond orders (as we do in our QM9 experiments) or
            # calculate the connectivity with a simple cutoff (e.g. all atoms less
            # then 2.0 angstrom apart are connected, see get_connectivity function in
            # template_preprocess_dataset script).
            # We will remove all molecules where two atoms are closer than 0.3
            # angstrom in the following as an example filtering step

            # loop over all molecules of length n_atoms
            for i in range(len(all_pos)):
                positions = all_pos[i]  # extract atom positions
                dists = pdist(positions)  # compute pair-wise distances
                if np.any(dists) < 0.3:  # check if any two atoms are closer than 0.3 A
                    valid[i] = 0  # mark current molecule as invalid


        # identify identical molecules (e.g. using fingerprints)
        if not print_file:
            print('\033[K', end='\r', flush=True)
            print(prog_str('uniqueness')+'...', end='\r', flush=True)
        if not printed_todos:
            print('Please implement a procedure to check the uniqueness of '
                  'generated molecules! Skipping check for uniqueness...')
            printed_todos = True
        # TODO
        # Implement procedure to identify duplicate structures here.
        # This can (heuristically) be achieved in many ways but perfectly identifying
        # all duplicate structures without false positives or false negatives is
        # probably impossible (or computationally prohibitive).
        # For our QM9 experiments, we compared fingerprints and canonical smiles
        # strings of generated molecules using the Molecule class in utility_classes.py
        # that provides functions to obtain these. It would also be possible to compare
        # learned embeddings, e.g. from SchNet or G-SchNet, either as an average over
        # all atoms, over all atoms of the same type, or combined with an algorithm
        # to find the best match between atoms of two molecules considering the
        # distances between embeddings. A similar procedure could be implemented
        # using the root-mean-square deviation (RMSD) of atomic positions. Then it
        # would be required to find the best match between atoms of two structures if
        # they are rotated such that the RMSD given the match is minimal. Again,
        # the best procedure really depends on the experimental setup, e.g. the
        # goals of the experiment, used data and size of molecules in the dataset etc.

        # duplicate_count contains the number of duplicates found for each structure
        duplicate_count = np.zeros(n_mols, dtype=int)
        # duplicating contains -1 for original structures and the id of the duplicated
        # original structure for duplicates
        duplicating = -np.ones(n_mols, dtype=int)
        # remove duplicate structures from list of valid molecules if desired
        if 'unique' in args.filters:
            valid[duplicating != -1] = 0
        # count number of non-unique structures
        n_non_unique += np.sum(duplicate_count)

        # store list of valid molecules in dictionary
        d.update({'valid': valid})

        # collect statistics of generated data
        n_generated += len(valid)
        n_valid += np.sum(valid)
        # count number of atoms per type (here for C, N, O, F, and H as example)
        n_of_types = [np.sum(all_numbers == i, axis=1) for i in [6, 7, 8, 9, 1]]
        stats_new = np.stack(
            (np.ones(len(valid)) * n_atoms,     # n_atoms
             np.arange(0, len(valid)),          # id
             valid,                             # valid
             duplicating,                       # id of duplicated molecule
             duplicate_count,                   # number of duplicates
             -np.ones(len(valid)),              # known
             -np.ones(len(valid)),              # equals
             *n_of_types,                       # n_atoms per type
             ),
            axis=0)
        stats = np.hstack((stats, stats_new))

    if not print_file:
        print('\033[K', end='\r', flush=True)
    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    print(f'Needed {h:d}h{m:02d}m{s:02d}s.')

    # Update and print results
    res.update({'n_generated': n_generated,
                'n_valid': n_valid,
                'stats': stats,
                'stat_heads': stat_heads})

    print(f'Number of generated molecules: {n_generated}\n'
          f'Number of duplicate molecules: {n_non_unique}')
    if 'unique' in args.filters:
        print(f'Number of unique and valid molecules: {n_valid}')
    else:
        print(f'Number of valid molecules (including duplicates): {n_valid}')

    # Remove invalid molecules from results if desired
    if args.store != 'all':
        shrunk_res = {}
        shrunk_stats = np.empty((len(stats), 0))
        i = 0
        for key in res:
            if isinstance(key, str):
                shrunk_res[key] = res[key]
                continue
            if key == 0:
                continue
            d = res[key]
            start = i
            end = i + len(d['valid'])
            idcs = np.where(d['valid'])[0]
            if len(idcs) < 1:
                i = end
                continue
            # shrink stats
            idx_id = stat_heads.index('id')
            idx_known = stat_heads.index('known')
            new_stats = stats[:, start:end]
            new_stats = new_stats[:, idcs]
            new_stats[idx_id] = np.arange(len(new_stats[idx_id]))  # adjust ids
            shrunk_stats = np.hstack((shrunk_stats, new_stats))
            # shrink positions and atomic numbers
            shrunk_res[key] = {Properties.R: d[Properties.R][idcs],
                               Properties.Z: d[Properties.Z][idcs]}
            i = end

        shrunk_res['stats'] = shrunk_stats
        res = shrunk_res

    # transfer results to ASE db
    # get filename that is not yet taken for db
    if os.path.isfile(target_db):
        file_name, _ = os.path.splitext(target_db)
        expand = 0
        while True:
            expand += 1
            new_file_name = file_name + '_' + str(expand)
            if os.path.isfile(new_file_name + '.db'):
                continue
            else:
                target_db = new_file_name + '.db'
                break
    print(f'Transferring generated molecules to database at {target_db}...')
    # open db
    with connect(target_db) as conn:
        # store metadata
        conn.metadata = {'n_generated': int(n_generated),
                         'n_non_unique': int(n_non_unique),
                         'n_valid': int(n_valid),
                         'non_unique_removed_from_valid': 'unique' in args.filters}
        # store molecules
        for n_atoms in res:
            if isinstance(n_atoms, str) or n_atoms == 0:
                continue
            d = res[n_atoms]
            all_pos = d[Properties.R]
            all_numbers = d[Properties.Z]
            for pos, num in zip(all_pos, all_numbers):
                at = Atoms(num, positions=pos)
                conn.write(at)

    # store gathered statistics in separate file
    np.savez_compressed(os.path.splitext(target_db)[0] + f'_statistics.npz',
                        stats=res['stats'], stat_heads=res['stat_heads'])
