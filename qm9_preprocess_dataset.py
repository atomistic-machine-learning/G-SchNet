import collections
import argparse
import sys
import time
import numpy as np
import logging
from ase.db import connect
from scipy.spatial.distance import pdist
from utility_classes import ConnectivityCompressor, Molecule
from multiprocessing import Process, Queue
from pathlib import Path


def get_parser():
    """ Setup parser for command line arguments """
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('datapath', help='Full path to dataset (e.g. '
                                              '/home/qm9.db)')
    main_parser.add_argument('--valence_list',
                             default=[1, 1, 6, 4, 7, 3, 8, 2, 9, 1], type=int,
                             nargs='+',
                             help='The valence of atom types in the form '
                                  '[type1 valence type2 valence ...] '
                                  '(default: %(default)s)')
    main_parser.add_argument('--n_threads', type=int, default=16,
                             help='Number of extra threads used while '
                                  'processing the data')
    main_parser.add_argument('--n_mols_per_thread', type=int, default=100,
                             help='Number of molecules processed by each '
                                  'thread in one iteration')
    return main_parser


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


def get_count_statistics(mol=None, get_stat_heads=False):
    '''
    Collects atom, bond, and ring count statistics of a provided molecule

    Args:
        mol (utility_classes.Molecule): Molecule to be examined
        get_stat_heads (bool, optional): set True to only return the headers of
            gathered statistics (default: False)

    Returns:
        numpy.ndarray: (n_statistics x 1) array containing the gathered statistics. Use
            get_stat_heads parameter to obtain the corresponding row headers (where RX
            describes number of X-membered rings and CXC indicates the number of
            carbon-carbon bonds of order X etc.).
    '''
    stat_heads = ['n_atoms', 'C', 'N', 'O', 'F', 'H', 'H1C', 'H1N',
                  'H1O', 'C1C', 'C2C', 'C3C', 'C1N', 'C2N', 'C3N', 'C1O',
                  'C2O', 'C1F', 'N1N', 'N2N', 'N1O', 'N2O', 'N1F', 'O1O',
                  'O1F', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R>8']
    if get_stat_heads:
        return stat_heads
    if mol is None:
        return None
    key_idx_dict = dict(zip(stat_heads, range(len(stat_heads))))
    stats = np.zeros((len(stat_heads), 1))
    # process all bonds and store statistics about bond and ring counts
    bond_stats = mol.get_bond_stats()
    for key, value in bond_stats.items():
        if key in key_idx_dict:
            idx = key_idx_dict[key]
            stats[idx, 0] = value
    # store simple statistics about number of atoms
    stats[key_idx_dict['n_atoms'], 0] = mol.n_atoms
    for key in ['C', 'N', 'O', 'F', 'H']:
        idx = key_idx_dict[key]
        charge = mol.type_charges[key]
        if charge in mol._unique_numbers:
            stats[idx, 0] = np.sum(mol.numbers == charge)
    return stats


def preprocess_molecules(mol_idcs, source_db, valence,
                         precompute_distances=True, remove_invalid=True,
                         invalid_list=None, print_progress=False):
    '''
    Checks the validity of selected molecules and collects atom, bond,
    and ring count statistics for the valid structures. Molecules are classified as
    invalid if they consist of disconnected parts or fail a valence check, where the
    valency constraints of all atoms in a molecule have to be satisfied (e.g. carbon
    has four bonds, nitrogen has three bonds etc.)

    Args:
        mol_idcs (array): the indices of molecules from the source database that
            shall be examined
        source_db (str): full path to the source database (in ase.db sqlite format)
        valence (array): an array where the i-th entry contains the valency
            constraint of atoms with atomic charge i (e.g. a valency of 4 at array
            position 6 representing carbon)
        precompute_distances (bool, optional): if True, the pairwise distances between
            atoms in each molecule are computed and stored in the database (default:
            True)
        remove_invalid (bool, optional): if True, molecules that do not pass the
            valency or connectivity checks (or are on the invalid_list) are removed from
            the new database (default: True)
        invalid_list (list of int, optional): precomputed list containing indices of
            molecules that are marked as invalid (because they did not pass the
            valency or connectivity checks in earlier runs, default: None)
        print_progress (bool, optional): set True to print the progress in percent
            (default: False)

    Returns
        list of ase.Atoms: list of all valid molecules
        list of dict: list of corresponding dictionaries with data of each molecule
        numpy.ndarray: (n_statistics x n_valid_molecules) matrix with atom, bond,
            and ring count statistics
        list of int: list with indices of molecules that failed the valency check
        list of int: list with indices of molecules that consist of disconnected parts
        int: number of molecules processed
    '''
    # initial setup
    count = 0  # count the number of invalid molecules
    disc = []  # store indices of disconnected molecules
    inval = []  # store indices of invalid molecules
    data_list = []  # store data fields of molecules for new db
    mols = []  # store molecules (as ase.Atoms objects)
    compressor = ConnectivityCompressor()  # (de)compress sparse connectivity matrices
    stats = np.empty((len(get_count_statistics(get_stat_heads=True)), 0))
    n_all = len(mol_idcs)

    with connect(source_db) as source_db:
        # iterate over provided indices
        for i in mol_idcs:
            i = int(i)
            # skip molecule if present in invalid_list and remove_invalid is True
            if remove_invalid and invalid_list is not None:
                if i in invalid_list:
                    continue
            # get molecule from database
            row = source_db.get(i + 1)
            data = row.data
            at = row.toatoms()
            # get positions and atomic numbers
            pos = at.positions
            numbers = at.numbers
            # center positions (using center of mass)
            pos = pos - at.get_center_of_mass()
            # order atoms by distance to center of mass
            center_dists = np.sqrt(np.maximum(np.sum(pos ** 2, axis=1), 0))
            idcs_sorted = np.argsort(center_dists)
            pos = pos[idcs_sorted]
            numbers = numbers[idcs_sorted]
            # update positions and atomic numbers accordingly in Atoms object
            at.positions = pos
            at.numbers = numbers
            # instantiate utility_classes.Molecule object
            mol = Molecule(pos, numbers)
            # get connectivity matrix (detecting bond orders with Open Babel)
            con_mat = mol.get_connectivity()
            # stop if molecule is disconnected (and therefore invalid)
            if remove_invalid:
                if is_disconnected(con_mat):
                    count += 1
                    disc += [i]
                    continue

            # check if valency constraints of all atoms in molecule are satisfied:
            # since the detection of bond orders for the connectivity matrix with Open
            # Babel is unreliable for certain cases (e.g. some aromatic rings) we
            # try to fix it manually (with heuristics) or by reshuffling the atom
            # order (as the bond order detection of Open Babel is sensitive to the
            # order of atoms)
            nums = numbers
            random_ord = np.arange(len(numbers))
            for _ in range(10):  # try 10 times before dismissing as invalid
                if np.all(np.sum(con_mat, axis=0) == valence[nums]):
                    # valency is correct -> mark as valid and stop check
                    val = True
                    break
                else:
                    # try to fix bond orders using heuristics
                    val = False
                    con_mat = mol.get_fixed_connectivity()
                    if np.all(np.sum(con_mat, axis=0) == valence[nums]):
                        # valency is now correct -> mark as valid and stop check
                        val = True
                        break
                    # shuffle atom order before checking valency again
                    random_ord = np.random.permutation(range(len(pos)))
                    mol = Molecule(pos[random_ord], numbers[random_ord])
                    con_mat = mol.get_connectivity()
                    nums = numbers[random_ord]
            if remove_invalid:
                if not val:
                    # stop if molecule is invalid (it failed the repeated valence checks)
                    count += 1
                    inval += [i]
                    continue

            if precompute_distances:
                # calculate pairwise distances of atoms and store them in data
                dists = pdist(pos)[:, None]
                data.update({'dists': dists})

            # store compressed connectivity matrix in data
            rand_ord_rev = np.argsort(random_ord)
            con_mat = con_mat[rand_ord_rev][:, rand_ord_rev]
            data.update(
                {'con_mat': compressor.compress(con_mat)})

            # update atom, bond, and ring count statistics
            stats = np.hstack((stats, get_count_statistics(mol=mol)))

            # add results to the lists
            mols += [at]
            data_list += [data]

            # print progress if desired
            if print_progress:
                if i % 100 == 0:
                    print('\033[K', end='\r', flush=True)
                    print(f'{100 * (i + 1) / n_all:.2f}%', end='\r', flush=True)

    return mols, data_list, stats, inval, disc, count


def _processing_worker(q_in, q_out, task):
    '''
    Simple worker function that repeatedly fulfills a task using transmitted input and
    sends back the results until a stop signal is received. Can be used as target in
    a multiprocessing.Process object.

    Args:
        q_in (multiprocessing.Queue): queue to receive a list with data. The first
            entry signals whether worker can stop and the remaining entries are used as
            input arguments to the task function
        q_out (multiprocessing.Queue): queue to send results from task back
        task (callable function): function that is called using the received data
    '''
    while True:
        data = q_in.get(True)  # receive data
        if data[0]:  # stop if stop signal is received
            break
        results = task(*data[1:])  # fulfill task with received data
        q_out.put(results)  # send back results


def _submit_jobs(qs_out, count, chunk_size, n_all, working_flag,
                 n_per_thread):
    '''
    Function that submits a job to preprocess molecules to every provided worker.

    Args:
        qs_out (list of multiprocessing.Queue): queues used to send data to workers (one
            queue per worker)
        count (int): index of the earliest, not yet preprocessed molecule in the db
        chunk_size (int): number of molecules to be divided amongst workers
        n_all (int): total number of molecules in the db
        working_flag (array): flags indicating whether workers are running
        n_per_thread (int): number of molecules to be given to each thread

    Returns:
        numpy.ndarray: array with flags indicating whether workers got
            a job
        int: index of the new earliest, not yet preprocessed molecule in
            the db (after the submitted preprocessing jobs have been done)
    '''
    # calculate indices of molecules that shall be preprocessed by workers
    idcs = np.arange(count, min(n_all, count + chunk_size))
    start = 0
    for i, q in enumerate(qs_out):
        if start >= len(idcs):
            # stop if no more indices are left to submit
            break
        end = start + n_per_thread
        q.put((False, idcs[start:end]))  # submit indices (and signal to not stop)
        working_flag[i] = 1  # set flag that current worker got a job
        start = end
    new_count = count + len(idcs)
    return working_flag, new_count


def preprocess_dataset(datapath, valence_list, n_threads, n_mols_per_thread=100,
                       logging_print=True, new_db_path=None, precompute_distances=True,
                       remove_invalid=True, invalid_list=None):
    '''
    Pre-processes all molecules of a dataset using the provided valency information.
    Multi-threading is used to speed up the process.
    Along with a new database containing the pre-processed molecules, a
    "input_db_invalid.txt" file holding the indices of removed molecules (which
    do not pass the valence or connectivity checks, omitted if remove_invalid is False)
    and a "new_db_statistics.npz" file (containing atom, bond, and ring count statistics
    for all molecules in the new database) are stored.

    Args:
        datapath (str): full path to dataset (ase.db database)
        valence_list (list): the valence of atom types in the form
            [type1 valence type2 valence ...]
        n_threads (int): number of threads used (0 for no extra threads)
        n_mols_per_thread (int, optional): number of molecules processed by each
            thread at each iteration (default: 100)
        logging_print (bool, optional): set True to show output with logging.info
            instead of standard printing (default: True)
        new_db_path (str, optional): full path to new database where pre-processed
            molecules shall be stored (None to simply append "gen" to the name in
            datapath, default: None)
        precompute_distances (bool, optional): if True, the pairwise distances between
            atoms in each molecule are computed and stored in the database (default:
            True)
        remove_invalid (bool, optional): if True, molecules that do not pass the
            valency or connectivity check are removed from the new database (default:
            True)
        invalid_list (list of int, optional): precomputed list containing indices of
            molecules that are marked as invalid (because they did not pass the
            valency or connectivity checks in earlier runs, default: None)
    '''
    # convert paths
    datapath = Path(datapath)
    if new_db_path is None:
        new_db_path = datapath.parent / (datapath.stem + 'gen.db')
    else:
        new_db_path = Path(new_db_path)

    # compute array where the valency constraint of atom type i is stored at entry i
    max_type = max(valence_list[::2])
    valence = np.zeros(max_type + 1, dtype=int)
    valence[valence_list[::2]] = valence_list[1::2]

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

    # initial setup
    n_iterations = 0
    chunk_size = n_threads * n_mols_per_thread
    current = 0
    count = 0  # count number of discarded (invalid etc.) molecules
    disc = []
    inval = []
    stats = np.empty((len(get_count_statistics(get_stat_heads=True)), 0))
    working_flag = np.zeros(n_threads, dtype=bool)
    start_time = time.time()
    if invalid_list is not None and remove_invalid:
        invalid_list = {*invalid_list}
        n_inval = len(invalid_list)
    else:
        n_inval = 0

    with connect(new_db_path) as new_db:

        if n_threads >= 1:
            # set up threads and queues
            threads = []
            qs_in = []
            qs_out = []
            for i in range(n_threads):
                qs_in += [Queue(1)]
                qs_out += [Queue(1)]
                threads += \
                    [Process(target=_processing_worker,
                             name=str(i),
                             args=(qs_out[-1],
                                   qs_in[-1],
                                   lambda x:
                                   preprocess_molecules(x,
                                                        datapath,
                                                        valence,
                                                        precompute_distances,
                                                        remove_invalid,
                                                        invalid_list)))]
                threads[-1].start()

            # submit first round of jobs
            working_flag, current = \
                _submit_jobs(qs_out, current, chunk_size, n_all,
                             working_flag, n_mols_per_thread)

            while np.any(working_flag == 1):
                n_iterations += 1

                # initialize new iteration
                results = []

                # gather results
                for i, q in enumerate(qs_in):
                    if working_flag[i]:
                        results += [q.get()]
                        working_flag[i] = 0

                # submit new jobs
                working_flag, current_new = \
                    _submit_jobs(qs_out, current, chunk_size, n_all, working_flag,
                                 n_mols_per_thread)

                # store gathered results
                for res in results:
                    mols, data_list, _stats, _inval, _disc, _c = res
                    for (at, data) in zip(mols, data_list):
                        new_db.write(at, data=data)
                    stats = np.hstack((stats, _stats))
                    inval += _inval
                    disc += _disc
                    count += _c

                # print progress
                if logging_print and n_iterations % 10 == 0:
                    _print(f'Processed: {current:6d} / {n_all}...')
                elif not logging_print:
                    _print('\033[K', end='\r', flush=True)
                    _print(f'{100 * current / n_all:.2f}%', end='\r',
                           flush=True)
                current = current_new  # update current position in database

            # stop worker threads and join
            for i, q_out in enumerate(qs_out):
                q_out.put((True,))
                threads[i].join()
                threads[i].terminate()
            if logging_print:
                _print(f'Processed: {n_all} / {n_all}...')

        else:
            results = preprocess_molecules(range(n_all), datapath, valence,
                                           precompute_distances, remove_invalid,
                                           invalid_list, print_progress=True)
            mols, data_list, stats, inval, disc, count = results
            for (at, data) in zip(mols, data_list):
                new_db.write(at, data=data)

    if not logging_print:
        _print('\033[K', end='\n', flush=True)
    _print(f'... successfully validated {n_all - count - n_inval} data '
           f'points!', flush=True)
    if invalid_list is not None:
        _print(f'{n_inval} structures were removed because they are on the '
               f'pre-computed list of invalid molecules!', flush=True)
        if len(disc)+len(inval) > 0:
            _print(f'CAUTION: Could not validate {len(disc)+len(inval)} additional '
                   f'molecules. These were also removed and their indices are '
                   f'appended to the list of invalid molecules stored at '
                   f'{datapath.parent / (datapath.stem + f"_invalid.txt")}',
                   flush=True)
            np.savetxt(datapath.parent / (datapath.stem + f'_invalid.txt'),
                       np.append(np.sort(list(invalid_list)), np.sort(inval + disc)),
                       fmt='%d')
    elif remove_invalid:
        _print(f'Identified {len(disc)} disconnected structures, and {len(inval)} '
               f'structures with invalid valence!', flush=True)
        np.savetxt(datapath.parent / (datapath.stem + f'_invalid.txt'),
                   np.sort(inval + disc), fmt='%d')
    _print('\nCompressing and storing statistics with numpy...')
    np.savez_compressed(new_db_path.parent/(new_db_path.stem+f'_statistics.npz'),
                        stats=stats,
                        stat_heads=get_count_statistics(get_stat_heads=True))

    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    _print(f'Done! Pre-processing needed {h:d}h{m:02d}m{s:02d}s.')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    preprocess_dataset(**vars(args))
