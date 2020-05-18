import numpy as np
import collections
import pickle
import os
import argparse
import openbabel as ob
import pybel
import time
import json

from schnetpack import Properties
from utility_classes import Molecule, ConnectivityCompressor
from utility_functions import run_threaded, print_atom_bond_ring_stats, update_dict
from multiprocessing import Process, Queue
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
    main_parser.add_argument('--train_data_path',
                             help='Path to training data base (if provided, '
                                  'generated molecules can be compared/matched with '
                                  'those in the training data set)',
                             default=None)
    main_parser.add_argument('--model_path',
                             help='Path of directory containing the model that '
                                  'generated the molecules. It should contain a '
                                  'split.npz file with training data splits and a '
                                  'args.json file with the arguments used during '
                                  'training (if this and --train_data_path '
                                  'are provided, the generated molecules will be '
                                  'filtered for new structures which were not included '
                                  'in the training or validation data)',
                             default=None)
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
    main_parser.add_argument('--store', type=str, default='valid_connectivity',
                             choices=['all', 'valid', 'new',
                                      'valid_connectivity',
                                      'new_connectivity'],
                             help='How much information shall be stored '
                                  'after filtering: \n"all" keeps all '
                                  'generated molecules and statistics '
                                  'including the connectivity matrices,'
                                  '\n"valid" keeps only valid molecules and '
                                  'discards connectivity matrices,\n'
                                  '"new" furthermore discards all validly '
                                  'generated molecules that match training '
                                  'data (corresponds to "valid" if '
                                  'model_path is not provided), '
                                  '\n"new_connectivity" and '
                                  '"valid_connectivity" store only new or '
                                  'valid molecules and the corresponding '
                                  'connectivity matrices '
                                  '(default: %(default)s)')
    main_parser.add_argument('--print_file',
                             help='Use to limit the printing if results are '
                                  'written to a file instead of the console ('
                                  'e.g. if running on a cluster)',
                             action='store_true')
    main_parser.add_argument('--threads', type=int, default=8,
                             help='Number of threads used (set to 0 to run '
                                  'everything sequentially in the main thread,'
                                  ' default: %(default)s)')

    return main_parser


def remove_disconnected(connectivity_batch, valid=None):
    '''
    Identify structures which are actually more than one molecule (as they consist of
    disconnected structures) and mark them as invalid.

    Args:
        connectivity_batch (numpy.ndarray): batch of connectivity matrices
        valid (numpy.ndarray, optional): array of the same length as connectivity_batch
            which flags molecules as valid, if None all connectivity matrices are
            considered to correspond to valid molecules in the beginning (default:
            None)

    Returns:
        dict (str->numpy.ndarray): a dictionary containing an array which marks
            molecules as valid under the key 'valid' (identified disconnected
            structures will now be marked as invalid in contrast to the flag in input
            argument valid)
    '''
    if valid is None:
        valid = np.ones(len(connectivity_batch), dtype=bool)
    # find disconnected parts for every given connectivity matrix
    for i, con_mat in enumerate(connectivity_batch):
        # only work with molecules categorized as valid
        if not valid[i]:
            continue
        seen, queue = {0}, collections.deque([0])
        while queue:
            vertex = queue.popleft()
            for node in np.argwhere(con_mat[vertex] > 0).flatten():
                if node not in seen:
                    seen.add(node)
                    queue.append(node)
        # if the seen nodes do not include all nodes, there are disconnected
        #  parts and the molecule is invalid
        if seen != {*range(len(con_mat))}:
            valid[i] = False
    return {'valid': valid}


def filter_unique(mols, valid=None, use_bits=False):
    '''
    Identify duplicate molecules among a large amount of generated structures.
    The first found structure of each kind is kept as valid original and all following
    duplicating structures are marked as invalid (the molecular fingerprint and
    canonical smiles representation is used which means that different spatial
    conformers of the same molecular graph cannot be distinguished).

    Args:
        mols (list of utility_classes.Molecule): list of all generated molecules
        valid (numpy.ndarray, optional): array of the same length as mols which flags
            molecules as valid (invalid molecules are not considered in the comparison
            process), if None, all molecules in mols are considered as valid (default:
            None)
        use_bits (bool, optional): set True to use the list of non-zero bits instead of
            the pybel.Fingerprint object when comparing molecules (results are
            identical, default: False)

    Returns:
        numpy.ndarray: array of the same length as mols which flags molecules as
            valid (identified duplicates are now marked as invalid in contrast to the
            flag in input argument valid)
        numpy.ndarray: array of length n_mols where entry i is -1 if molecule i is
            an original structure (not a duplicate) and otherwise it is the index j of
            the original structure that molecule i duplicates (j<i)
        numpy.ndarray: array of length n_mols that is 0 for all duplicates and the
            number of identified duplicates for all original structures (therefore
            the sum over this array is the total number of identified duplicates)
    '''
    if valid is None:
        valid = np.ones(len(mols), dtype=bool)
    else:
        valid = valid.copy()
    accepted_dict = {}
    duplicating = -np.ones(len(mols), dtype=int)
    duplicate_count = np.zeros(len(mols), dtype=int)
    for i, mol1 in enumerate(mols):
        if not valid[i]:
            continue
        mol_key = _get_atoms_per_type_str(mol1)
        found = False
        if mol_key in accepted_dict:
            for j, mol2 in accepted_dict[mol_key]:
                # compare fingerprints and canonical smiles representation
                if mol1.tanimoto_similarity(mol2, use_bits=use_bits) >= 1:
                    if (mol1.get_can() == mol2.get_can()
                            or mol1.get_can() == mol2.get_mirror_can()):
                        found = True
                        valid[i] = False
                        duplicating[i] = j
                        duplicate_count[j] += 1
                        break
        if not found:
            accepted_dict = _update_dict(accepted_dict, key=mol_key, val=(i, mol1))
    return valid, duplicating, duplicate_count


def filter_unique_threaded(mols, valid=None, n_threads=16,
                           n_mols_per_thread=5, print_file=True,
                           prog_str=None):
    '''
    Identify duplicate molecules among a large amount of generated structures using
    multiple CPU-threads. The first found structure of each kind is kept as valid
    original and all following duplicating structures are marked as invalid (the
    molecular fingerprint and canonical smiles representation is used which means that
    different spatial conformers of the same molecular graph cannot be distinguished).

    Args:
        mols (list of utility_classes.Molecule): list of all generated molecules
        valid (numpy.ndarray, optional): array of the same length as mols which flags
            molecules as valid (invalid molecules are not considered in the comparison
            process), if None, all molecules in mols are considered as valid (default:
            None)
        n_threads (int, optional): number of additional threads used (default: 16)
        n_mols_per_thread (int, optional): number of molecules that are processed by
            each thread in each iteration (default: 5)
        print_file (bool, optional): set True to suppress printing of progress string
            (default: True)
        prog_str (str, optional): specify a custom progress string (if None,
            no progress will be printed, default: None)

    Returns:
        numpy.ndarray: array of the same length as mols which flags molecules as
            valid (identified duplicates are now marked as invalid in contrast to the
            flag in input argument valid)
        numpy.ndarray: array of length n_mols where entry i is -1 if molecule i is
            an original structure (not a duplicate) and otherwise it is the index j of
            the original structure that molecule i duplicates (j<i)
        numpy.ndarray: array of length n_mols that is 0 for all duplicates and the
            number of identified duplicates for all original structures (therefore
            the sum over this array is the total number of identified duplicates)
    '''
    if valid is None:
        valid = np.ones(len(mols), dtype=bool)
    else:
        valid = valid.copy()
    if len(mols) < 3*n_threads*n_mols_per_thread or n_threads == 0:
        return filter_unique(mols, valid, use_bits=True)
    current = 0
    still_valid = np.zeros_like(valid)
    working_flag = np.zeros(n_threads, dtype=bool)
    duplicating = []
    goal = n_threads*n_mols_per_thread

    # set up threads and queues
    threads = []
    qs_in = []
    qs_out = []
    for i in range(n_threads):
        qs_in += [Queue(1)]
        qs_out += [Queue(1)]
        threads += [Process(target=_filter_worker, name=str(i),
                            args=(qs_out[-1], qs_in[-1], mols))]
        threads[-1].start()

    # get first two mini-batches (workers do not need to process first one)
    new_idcs, current, dups = _filter_mini_batch(mols, valid, current, goal)
    duplicating += dups  # maintain list of which molecules are duplicated
    newly_accepted = new_idcs
    still_valid[newly_accepted] = 1  # trivially accept first batch
    newly_accepted_dict = _create_mol_dict(mols, newly_accepted)
    new_idcs, current, dups = _filter_mini_batch(mols, valid, current, goal)
    duplicating += dups

    # submit second mini batch to workers
    start = 0
    for i, q_out in enumerate(qs_out):
        if start >= len(new_idcs):
            continue
        end = start+n_mols_per_thread
        q_out.put((False, newly_accepted_dict, new_idcs[start:end]))
        working_flag[i] = 1
        start = end

    # loop while the worker threads have data to process
    k = 1
    while np.any(working_flag == 1):

        # get new mini batch
        new_idcs, current, dups = \
            _filter_mini_batch(mols, valid, current, goal)

        # gather results from workers
        newly_accepted = []
        newly_accepted_dict = {}
        for i, q_in in enumerate(qs_in):
            if working_flag[i]:
                returned = q_in.get()
                newly_accepted += returned[0]
                duplicating += returned[1]
                newly_accepted_dict = _update_dict(newly_accepted_dict,
                                                   new_dict=returned[2])
                working_flag[i] = 0

        # submit gathered results and new mini batch molecules to workers
        start = 0
        for i, q_out in enumerate(qs_out):
            if start >= len(new_idcs):
                continue
            end = start + n_mols_per_thread
            q_out.put((False, newly_accepted_dict, new_idcs[start:end]))
            working_flag[i] = 1
            start = end

        # set validity according to gathered data
        still_valid[newly_accepted] = 1
        duplicating += dups

        k += 1
        if ((k % 10) == 0 or current >= len(mols)) and not print_file \
                and prog_str is not None:
            print('\033[K', end='\r', flush=True)
            print(f'{prog_str} ({100 * min(current/len(mols), 1):.2f}%)',
                  end='\r', flush=True)

    # stop worker threads and join
    for i, q_out in enumerate(qs_out):
        q_out.put((True,))
        threads[i].join()
        threads[i].terminate()

    # fix statistics about duplicates
    duplicating, duplicate_count = _process_duplicates(duplicating, len(mols))

    return still_valid, duplicating, duplicate_count


def _get_atoms_per_type_str(mol):
    '''
    Get a string representing the atomic composition of a molecule (i.e. the number
    of atoms per type in the molecule, e.g. H2C3O1, where the order of types is
    determined by increasing nuclear charge).

    Args:
        mol (utility_classes.Molecule or numpy.ndarray: the molecule (or an array of
            its atomic numbers)

    Returns:
        str: the atomic composition of the molecule
    '''
    if isinstance(mol, Molecule):
        n_atoms_per_type = mol.get_n_atoms_per_type()
    else:
        # assume atomic numbers were provided
        n_atoms_per_type = np.bincount(mol, minlength=10)[
            np.array(list(Molecule.type_infos.keys()), dtype=int)]
    s = ''
    for t, n in zip(Molecule.type_infos.keys(), n_atoms_per_type):
        s += f'{Molecule.type_infos[t]["name"]}{int(n):d}'
    return s


def _create_mol_dict(mols, idcs=None):
    '''
    Create a dictionary holding indices of a list of molecules where the key is a
    string that represents the atomic composition (i.e. the number of atoms per type in
    the molecule, e.g. H2C3O1, where the order of types is determined by increasing
    nuclear charge). This is especially useful to speed up the comparison of molecules
    as candidate structures with the same composition of atoms can easily be accessed
    while ignoring all molecules with different compositions.

    Args:
        mols (list of utility_classes.Molecule or numpy.ndarray): the molecules or
            the atomic numbers of the molecules which are referenced in the dictionary
        idcs (list of int, optional): indices of a subset of the molecules in mols that
            shall be put into the dictionary (if None, all structures in mol will be
            referenced in the dictionary, default: None)

    Returns:
        dict (str->list of int): dictionary with the indices of molecules in mols
            ordered by their atomic composition
    '''
    if idcs is None:
        idcs = range(len(mols))
    mol_dict = {}
    for idx in idcs:
        mol = mols[idx]
        mol_key = _get_atoms_per_type_str(mol)
        mol_dict = _update_dict(mol_dict, key=mol_key, val=idx)
    return mol_dict


def _update_dict(old_dict, **kwargs):
    '''
    Update an existing dictionary (any->list of any) with new entries where the new
    values are either appended to the existing lists if the corresponding key already
    exists in the dictionary or a new list under the new key is created.

    Args:
        old_dict (dict (any->list of any)): original dictionary that shall be updated
        **kwargs: keyword arguments that can either be a dictionary of the same format
            as old_dict (new_dict=dict (any->list of any)) which will be merged into
            old_dict or a single key-value pair that shall be added (key=any, val=any)

    Returns:
        dict (any->list of any): the updated dictionary
    '''
    if 'new_dict' in kwargs:
        for key in kwargs['new_dict']:
            if key in old_dict:
                old_dict[key] += kwargs['new_dict'][key]
            else:
                old_dict[key] = kwargs['new_dict'][key]
    if 'val' in kwargs and 'key' in kwargs:
        if kwargs['key'] in old_dict:
            old_dict[kwargs['key']] += [kwargs['val']]
        else:
            old_dict[kwargs['key']] = [kwargs['val']]
    return old_dict


def _filter_mini_batch(mols, valid, start, amount):
    '''
    Prepare a mini-batch consisting of unique molecules (with respect to all molecules
    in the mini-batch) that can be divided and send to worker functions (see
    _filter_worker) to compare them to the database of all original (non-duplicate)
    molecules.

    Args:
        mols (list of utility_classes.Molecule): list of all generated molecules
        valid (numpy.ndarray): array of the same length as mols which flags molecules as
            valid (invalid molecules are not put into a mini-batch but skipped)
        start (int): index of the first molecule in mols that should be put into a
            mini-batch
        amount (int): the total amount of molecules that shall be put into the
            mini-batch (note that the mini-batch can be smaller than amount if all
            molecules in mols have been processed already).

    Returns:
        list of int: list of indices of molecules in mols that have been put into the
            mini-batch (i.e. the prepared mini-batch)
        int: index of the first molecule in mols that is not yet put into a mini-batch
        list of list of int: list of lists where the inner lists have exactly
            two integer entries: the first being the index of an identified duplicate
            molecule (skipped and not put into the mini-batch) and the second being the
            index of the corresponding original molecule (put into the mini-batch)
    '''
    count = 0
    accepted = []
    accepted_dict = {}
    duplicating = []
    max_mol = len(mols)
    while count < amount:
        if start >= max_mol:
            break
        if not valid[start]:
            start += 1
            continue
        mol1 = mols[start]
        mol_key = _get_atoms_per_type_str(mol1)
        found = False
        if mol_key in accepted_dict:
            for idx in accepted_dict[mol_key]:
                mol2 = mols[idx]
                if mol1.tanimoto_similarity(mol2, use_bits=True) >= 1:
                    if (mol1.get_can() == mol2.get_can()
                            or mol1.get_can() == mol2.get_mirror_can()):
                        found = True
                        duplicating += [[start, idx]]
                        break
        if not found:
            accepted += [start]
            accepted_dict = _update_dict(accepted_dict, key=mol_key, val=start)
            count += 1
        start += 1
    return accepted, start, duplicating


def _filter_worker(q_in, q_out, all_mols):
    '''
    Worker function for multi-threaded identification of duplicate molecules that
    iteratively receives small batches of molecules which it compares to all previously
    processed molecules that were identified as originals (non-duplicate structures).

    Args:
        q_in (multiprocessing.Queue): queue to receive a new job at each iteration
            (contains three entries: 1st a flag whether the job is done, 2nd a
            dictionary with indices of newly found original structures in the last
            iteration, and 3rd a list of indices of candidate molecules that shall be
            checked in the current iteration)
        q_out (multiprocessing.Queue): queue to send results of the current iteration
            (contains three entries: 1st a list with the indices of the candidates
            that were identified as originals, 2nd a list of lists where each inner
            list holds the index of an identified duplicate structure and the index
            of the original structure that it duplicates, and 3rd a dictionary with
            the indices of candidates that were identified as originals)
        all_mols (list of utility_classes.Molecule): list with all generated molecules
    '''
    accepted_dict = {}
    while True:
        data = q_in.get(True)
        if data[0]:
            break
        accepted_dict = _update_dict(accepted_dict, new_dict=data[1])
        mols = data[2]
        accept = []
        accept_dict = {}
        duplicating = []
        for idx1 in mols:
            found = False
            mol1 = all_mols[idx1]
            mol_key = _get_atoms_per_type_str(mol1)
            if mol_key in accepted_dict:
                for idx2 in accepted_dict[mol_key]:
                    mol2 = all_mols[idx2]
                    if mol1.tanimoto_similarity(mol2, use_bits=True) >= 1:
                        if (mol1.get_can() == mol2.get_can()
                                or mol1.get_can() == mol2.get_mirror_can()):
                            found = True
                            duplicating += [[idx1, idx2]]
                            break
            if not found:
                accept += [idx1]
                accept_dict = _update_dict(accept_dict, key=mol_key, val=idx1)
        q_out.put((accept, duplicating, accept_dict))


def _process_duplicates(dups, n_mols):
    '''
    Processes a list of duplicate molecules identified in a multi-threaded run and
    infers a proper list with the correct statistics for each molecule (how many
    duplicates of the structure are there and which is the first found structure of
    that kind)

    Args:
        dups (list of list of int): list of lists where the inner lists have exactly
            two integer entries: the first being the index of an identified duplicate
            molecule and the second being the index of the corresponding original
            molecule (which can also be a duplicate due to the applied multi-threading
            approach, hence this function is needed to identify such cases and fix
            the 'original' index to refer to the true original molecule, which is the
            first found structure of that kind)
        n_mols (int): the overall number of molecules that were examined

    Returns:
        numpy.ndarray: array of length n_mols where entry i is -1 if molecule i is
            an original structure (not a duplicate) and otherwise it is the index j of
            the original structure that molecule i duplicates (j<i)
        numpy.ndarray: array of length n_mols that is 0 for all duplicates and the
            number of identified duplicates for all original structures (therefore
            the sum over this array is the total number of identified duplicates)
    '''
    duplicating = -np.ones(n_mols, dtype=int)
    duplicate_count = np.zeros(n_mols, dtype=int)
    if len(dups) == 0:
        return duplicating, duplicate_count
    dups = np.array(dups, dtype=int)
    duplicates = dups[:, 0]
    originals = dups[:, 1]
    duplicating[duplicates] = originals
    for original in originals:
        wrongly_assigned_originals = []
        while duplicating[original] >= 0:
            wrongly_assigned_originals += [original]
            original = duplicating[original]
        duplicating[np.array(wrongly_assigned_originals, dtype=int)] = original
        duplicate_count[original] += 1
    return duplicating, duplicate_count


def check_valency(positions, numbers, valence, filter_by_valency=True,
                  print_file=True, prog_str=None, picklable_mols=False):
    '''
    Build utility_classes.Molecule objects from provided atom positions and types
    of a set of molecules and assess whether they are meeting the valency
    constraints or not (i.e. all of their atoms have the correct number of bonds).
    Note that all input molecules need to have the same number of atoms.

    Args:
        positions (list of numpy.ndarray): list of positions of atoms in euclidean
            space (n_atoms x 3) for each molecule
        numbers (numpy.ndarray): list of nuclear charges/types of atoms
            (e.g. 1 for hydrogens, 6 for carbons etc.) for each molecule
        valence (numpy.ndarray): list of valency of each atom type where the index in
            the list corresponds to the type (e.g. [0, 1, 0, 0, 0, 0, 2, 3, 4, 1] for
            qm9 molecules as H=type 1 has valency of 1, O=type 6 has valency of 2,
            N=type 7 has valency of 3 etc.)
        filter_by_valency (bool, optional): whether molecules that fail the valency
            check should be marked as invalid, else all input molecules will be
            classified as valid but the connectivity matrix is still computed and
            returned (default: True)
        print_file (bool, optional): set True to suppress printing of progress string
            (default: True)
        prog_str (str, optional): specify a custom progress string (default: None)
        picklable_mols (bool, optional): set True to remove all the information in
            the returned list of utility_classes.Molecule objects that can not be
            serialized with pickle (e.g. the underlying Open Babel ob.Mol object,
            default: False)

    Returns:
        dict (str->list/numpy.ndarray): a dictionary containing a list of
            utility_classes.Molecule ojbects under the key 'mols', a numpy.ndarray with
            the corresponding (n_atoms x n_atoms) connectivity matrices under the key
            'connectivity', and a numpy.ndarray (key 'valid') that marks whether a
            molecule has passed (entry=1) or failed (entry=0) the valency check if
            filter_by_valency is True (otherwise it will be 1 everywhere)
    '''
    n_atoms = len(numbers[0])
    n_mols = len(numbers)
    thresh = n_mols if n_mols < 30 else 30
    connectivity = np.zeros((len(positions), n_atoms, n_atoms))
    valid = np.ones(len(positions), dtype=bool)
    mols = []
    for i, (pos, num) in enumerate(zip(positions, numbers)):
        mol = Molecule(pos, num, store_positions=False)
        con_mat = mol.get_connectivity()
        random_ord = range(len(pos))
        # filter incorrect valence if desired
        if filter_by_valency:
            nums = num
            # try to fix connectivity if it isn't correct already
            for _ in range(10):
                if np.all(np.sum(con_mat, axis=0) == valence[nums]):
                    val = True
                    break
                else:
                    val = False
                    con_mat = mol.get_fixed_connectivity()
                    if np.all(
                            np.sum(con_mat, axis=0) == valence[nums]):
                        val = True
                        break
                    random_ord = np.random.permutation(range(len(pos)))
                    mol = Molecule(pos[random_ord], num[random_ord])
                    con_mat = mol.get_connectivity()
                    nums = num[random_ord]
            valid[i] = val

            if ((i + 1) % thresh == 0) and not print_file \
                and prog_str is not None:
                print('\033[K', end='\r', flush=True)
                print(f'{prog_str} ({100 * (i + 1) / n_mols:.2f}%)',
                      end='\r', flush=True)

        # reverse random order and save fixed connectivity matrix
        rand_ord_rev = np.argsort(random_ord)
        connectivity[i] = con_mat[rand_ord_rev][:, rand_ord_rev]
        if picklable_mols:
            mol.get_fp_bits()
            mol.get_can()
            mol.get_mirror_can()
            mol.remove_unpicklable_attributes(restorable=False)
        mols += [mol]
    return {'mols': mols, 'connectivity': connectivity, 'valid': valid}


def filter_new(mols, stats, stat_heads, model_path, train_data_path, print_file=False,
               n_threads=0):
    '''
    Check whether generated molecules correspond to structures in the training database
    used for either training, validation, or as test data and update statistics array of
    generated molecules accordingly.

    Args:
        mols (list of utility_classes.Molecule): generated molecules
        stats (numpy.ndarray): statistics of all generated molecules where columns
            correspond to molecules and rows correspond to available statistics
            (n_statistics x n_molecules)
        stat_heads (list of str): the names of the statistics stored in each row in
            stats (e.g. 'F' for the number of fluorine atoms or 'R5' for the number of
            rings of size 5)
        model_path (str): path to the folder containing the trained model used to
            generate the molecules
        train_data_path (str): full path to the training database
        print_file (bool, optional): set True to limit printing (e.g. if it is
            redirected to a file instead of displayed in a terminal, default: False)
        n_threads (int, optional): number of additional threads to use (default: 0)

    Returns:
        numpy.ndarray: updated statistics of all generated molecules (stats['known']
        is 0 if a generated molecule does not correspond to a structure in the
        training database, it is 1 if it corresponds to a training structure,
        2 if it corresponds to a validation structure, and 3 if it corresponds to a
        test structure, stats['equals'] is -1 if stats['known'] is 0 and otherwise
        holds the index of the corresponding training/validation/test structure in
        the database at train_data_path)
    '''
    print(f'\n\n2. Checking which molecules are new...')
    idx_known = stat_heads.index('known')

    # load training data
    dbpath = train_data_path
    if not os.path.isfile(dbpath):
        print(f'The provided training data base {dbpath} is no file, please specify '
              f'the correct path (including the filename and extension)!')
        raise FileNotFoundError
    print(f'Using data base at {dbpath}...')

    split_file = os.path.join(model_path, 'split.npz')
    if not os.path.exists(split_file):
        raise FileNotFoundError
    S = np.load(split_file)
    train_idx = S['train_idx']
    val_idx = S['val_idx']
    test_idx = S['test_idx']
    train_idx = np.append(train_idx, val_idx)
    train_idx = np.append(train_idx, test_idx)

    # check if subset was used (and restrict indices accordingly)
    train_args_path = os.path.join(model_path, f'args.json')
    with open(train_args_path) as handle:
        train_args = json.loads(handle.read())
    if 'subset_path' in train_args:
        if train_args['subset_path'] is not None:
            subset = np.load(train_args['subset_path'])
            train_idx = subset[train_idx]

    print('\nComputing fingerprints of training data...')
    start_time = time.time()
    if n_threads <= 0:
        train_fps = _get_training_fingerprints(dbpath, train_idx, print_file,
                                               use_con_mat=True)
    else:
        train_fps = {'fingerprints': [None for _ in range(len(train_idx))]}
        run_threaded(_get_training_fingerprints,
                     {'train_idx': train_idx},
                     {'dbpath': dbpath, 'use_bits': True, 'use_con_mat': True},
                     train_fps,
                     exclusive_kwargs={'print_file': print_file},
                     n_threads=n_threads)
    train_fps_dict = _get_training_fingerprints_dict(train_fps['fingerprints'])
    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    print(f'...{len(train_fps["fingerprints"])} fingerprints computed '
          f'in {h:d}h{m:02d}m{s:02d}s!')

    print('\nComparing fingerprints...')
    start_time = time.time()
    if n_threads <= 0:
        results = _compare_fingerprints(mols, train_fps_dict, train_idx,
                                        [len(val_idx), len(test_idx)],
                                        stats.T, stat_heads, print_file)
    else:
        results = {'stats': stats.T}
        run_threaded(_compare_fingerprints,
                     {'mols': mols, 'stats': stats.T},
                     {'train_idx': train_idx, 'train_fps': train_fps_dict,
                      'thresh': [len(val_idx), len(test_idx)],
                      'stat_heads': stat_heads, 'use_bits': True},
                     results,
                     exclusive_kwargs={'print_file': print_file},
                     n_threads=n_threads)
    stats = results['stats'].T
    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    print(f'... needed {h:d}h{m:02d}m{s:02d}s.')
    print(f'Number of new molecules: '
          f'{sum(stats[idx_known] == 0)+sum(stats[idx_known] == 3)}')
    print(f'Number of molecules matching training data: '
          f'{sum(stats[idx_known] == 1)}')
    print(f'Number of molecules matching validation data: '
          f'{sum(stats[idx_known] == 2)}')
    print(f'Number of molecules matching test data: '
          f'{sum(stats[idx_known] == 3)}')

    return stats


def _get_training_fingerprints(dbpath, train_idx, print_file=True,
                               use_bits=False, use_con_mat=False):
    '''
    Get the fingerprints (FP2 from Open Babel), canonical smiles representation,
    and atoms per type string of all molecules in the training database.

    Args:
        dbpath (str): path to the training database
        train_idx (list of int): list containing the indices of training, validation,
            and test molecules in the database (it is assumed
            that train_idx[0:n_train] corresponds to training data,
            train_idx[n_train:n_train+n_validation] corresponds to validation data,
            and train_idx[n_train+n_validation:] corresponds to test data)
        print_file (bool, optional): set True to suppress printing of progress string
            (default: True)
        use_bits (bool, optional): set True to return the non-zero bits in the
            fingerprint instead of the pybel.Fingerprint object (default: False)
        use_con_mat (bool, optional): set True to use pre-computed connectivity
            matrices (need to be stored in the training database in compressed format
            under the key 'con_mat', default: False)

    Returns:
        dict (str->list of tuple): dictionary with list of tuples under the key
        'fingerprints' containing the fingerprint, the canonical smiles representation,
        and the atoms per type string of each molecule listed in train_idx (preserving
        the order)
    '''
    train_fps = []
    if use_con_mat:
        compressor = ConnectivityCompressor()
    with connect(dbpath) as conn:
        if not print_file:
            print('0.00%', end='\r', flush=True)
        for i, idx in enumerate(train_idx):
            idx = int(idx)
            row = conn.get(idx + 1)
            at = row.toatoms()
            pos = at.positions
            atomic_numbers = at.numbers
            if use_con_mat:
                con_mat = compressor.decompress(row.data['con_mat'])
            else:
                con_mat = None
            train_fps += [get_fingerprint(pos, atomic_numbers,
                                          use_bits, con_mat)]
            if (i % 100 == 0 or i + 1 == len(train_idx)) and not print_file:
                print('\033[K', end='\r', flush=True)
                print(f'{100 * (i + 1) / len(train_idx):.2f}%', end='\r',
                      flush=True)
    return {'fingerprints': train_fps}


def get_fingerprint(pos, atomic_numbers, use_bits=False, con_mat=None):
    '''
    Compute the molecular fingerprint (Open Babel FP2), canonical smiles
    representation, and number of atoms per type (e.g. H2O1) of a molecule.

    Args:
        pos (numpy.ndarry): positions of the atoms (n_atoms x 3)
        atomic_numbers (numpy.ndarray): types of the atoms (n_atoms)
        use_bits (bool, optional): set True to return the non-zero bits in the
            fingerprint instead of the pybel.Fingerprint object (default: False)
        con_mat (numpy.ndarray, optional): connectivity matrix of the molecule
            containing the pairwise bond orders between all atoms (n_atoms x n_atoms)
            (can be inferred automatically if not provided, default: None)

    Returns:
        pybel.Fingerprint or set of int: the fingerprint of the molecule or a set
            containing the non-zero bits of the fingerprint if use_bits=True
        str: the canonical smiles representation of the molecule
        str: the atom types contained in the molecule followed by number of
            atoms per type, e.g. H2C3O1, ordered by increasing atom type (nuclear
            charge)
    '''
    if con_mat is not None:
        mol = Molecule(pos, atomic_numbers, con_mat)
        idc_lists = np.where(con_mat != 0)
        mol._update_bond_orders(idc_lists)
        mol = pybel.Molecule(mol.get_obmol())
    else:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for p, n in zip(pos, atomic_numbers):
            obatom = obmol.NewAtom()
            obatom.SetAtomicNum(int(n))
            obatom.SetVector(*p.tolist())
        # infer bonds and bond order
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()
        obmol.EndModify()
        mol = pybel.Molecule(obmol)
    # use pybel to get fingerprint
    if use_bits:
        return {*mol.calcfp().bits}, mol.write('can'), \
               _get_atoms_per_type_str(atomic_numbers)
    else:
        return mol.calcfp(), mol.write('can'), \
               _get_atoms_per_type_str(atomic_numbers)


def _get_training_fingerprints_dict(fps):
    '''
    Convert a list of fingerprints into a dictionary where a string describing the
    number of types in each molecules (e.g. H2C3O1, ordered by increasing nuclear
    charge) is used as a key (allows for faster comparison of molecules as only those
    made of the same atoms can be identical).

    Args:
        fps (list of tuple): list containing tuples as returned by the get_fingerprint
            function (holding the fingerprint, canonical smiles representation, and the
            atoms per type string)

    Returns:
        dict (str->list of tuple): dictionary containing lists of tuples holding the
            molecular fingerprint, the canonical smiles representation, and the index
            of the molecule in the input list using the atoms per type string of the
            molecules as key (such that fingerprint tuples of all molecules with the
            exact same atom composition, e.g. H2C3O1, are stored together in one list)
    '''
    fp_dict = {}
    for i, fp in enumerate(fps):
        fp_dict = _update_dict(fp_dict, key=fp[-1], val=fp[:-1]+(i,))
    return fp_dict


def _compare_fingerprints(mols, train_fps, train_idx, thresh, stats,
                          stat_heads, print_file=True, use_bits=False,
                          max_heavy_atoms=9):
    '''
    Compare fingerprints of generated and training data molecules to update the
    statistics of the generated molecules (to which training/validation/test
    molecule it corresponds, if any).

    Args:
        mols (list of utility_classes.Molecule): generated molecules
        train_fps (dict (str->list of tuple)): dictionary with fingerprints of
            training/validation/test data as returned by _get_training_fingerprints_dict
        train_idx (list of int): list that maps the index of fingerprints in the
            train_fps dict to indices of the underlying training database (it is assumed
            that train_idx[0:n_train] corresponds to training data,
            train_idx[n_train:n_train+n_validation] corresponds to validation data,
            and train_idx[n_train+n_validation:] corresponds to test data)
        thresh (tuple of int): tuple containing the number of validation and test
            data molecules (n_validation, n_test)
        stats (numpy.ndarray): statistics of all generated molecules where columns
            correspond to molecules and rows correspond to available statistics
            (n_statistics x n_molecules)
        stat_heads (list of str): the names of the statistics stored in each row in
            stats (e.g. 'F' for the number of fluorine atoms or 'R5' for the number of
            rings of size 5)
        print_file (bool, optional): set True to limit printing (e.g. if it is
            redirected to a file instead of displayed in a terminal, default: True)
        use_bits (bool, optional): set True if the fingerprint is provided as a list of
            non-zero bits instead of the pybel.Fingerprint object (default: False)
        max_heavy_atoms (int, optional): the maximum number of heavy atoms in the
            training data set (i.e. 9 for qm9, default: 9)

    Returns:
        dict (str->numpy.ndarray): dictionary containing the updated statistics under
            the key 'stats'
    '''
    idx_known = stat_heads.index('known')
    idx_equals = stat_heads.index('equals')
    idx_val = stat_heads.index('valid')
    n_val_mols, n_test_mols = thresh
    # get indices of valid molecules
    idcs = np.where(stats[:, idx_val] == 1)[0]
    if not print_file:
        print(f'0.00%', end='', flush=True)
    for i, idx in enumerate(idcs):
        mol = mols[idx]
        mol_key = _get_atoms_per_type_str(mol)
        # for now the molecule is considered to be new
        stats[idx, idx_known] = 0
        if np.sum(mol.numbers != 1) > max_heavy_atoms:
            continue  # cannot be in dataset
        if mol_key in train_fps:
            for fp_train in train_fps[mol_key]:
                # compare fingerprint
                if mol.tanimoto_similarity(fp_train[0], use_bits=use_bits) >= 1:
                    # compare canonical smiles representation
                    if (mol.get_can() == fp_train[1]
                            or mol.get_mirror_can() == fp_train[1]):
                        # store index of match
                        j = fp_train[-1]
                        stats[idx, idx_equals] = train_idx[j]
                        if j >= len(train_idx) - np.sum(thresh):
                            if j > len(train_idx) - n_test_mols:
                                stats[idx, idx_known] = 3  # equals test data
                            else:
                                stats[idx, idx_known] = 2  # equals validation data
                        else:
                            stats[idx, idx_known] = 1  # equals training data
                        break
        if not print_file:
            print('\033[K', end='\r', flush=True)
            print(f'{100 * (i + 1) / len(idcs):.2f}%', end='\r',
                  flush=True)
    if not print_file:
        print('\033[K', end='', flush=True)
    return {'stats': stats}


def collect_bond_and_ring_stats(mols, stats, stat_heads):
    '''
    Compute the bond and ring counts of a list of molecules and write them to the
    provided array of statistics if it contains the corresponding fields (e.g. 'R3'
    for rings of size 3 or 'C1N' for single bonded carbon-nitrogen pairs). Note that
    only statistics of molecules marked as 'valid' in the stats array are computed and
    that only those statistics will be stored, which already have columns in the stats
    array named accordingly in stat_heads (e.g. if 'R5' for rings of size 5 is not
    included in stat_heads, the number of rings of size 5 will not be stored in the
    stats array for the provided molecules).

    Args:
        mols (list of utiltiy_classes.Molecule): list of molecules for which bond and
            ring statistics are computed
        stats (numpy.ndarray): statistics of all molecules where columns
            correspond to molecules and rows correspond to available statistics
            (n_statistics x n_molecules)
        stat_heads (list of str): the names of the statistics stored in each row in
            stats (e.g. 'F' for the number of fluorine atoms or 'R5' for the number of
            rings of size 5)

    Returns:
        dict (str->numpy.ndarray): dictionary containing the updated statistics array
            under 'stats'
    '''
    idx_val = stat_heads.index('valid')
    for i, mol in enumerate(mols):
        if stats[i, idx_val] != 1:
            continue
        bond_stats = mol.get_bond_stats()
        for key, value in bond_stats.items():
            if key not in stat_heads:
                continue
            idx = stat_heads.index(key)
            stats[i, idx] = value
    return {'stats': stats}


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print_file = args.print_file

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
                  'known', 'equals', 'C', 'N', 'O', 'F', 'H', 'H1C', 'H1N',
                  'H1O', 'C1C', 'C2C', 'C3C', 'C1N', 'C2N', 'C3N', 'C1O',
                  'C2O', 'C1F', 'N1N', 'N2N', 'N1O', 'N2O', 'N1F', 'O1O',
                  'O1F', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R>8']
    stats = np.empty((len(stat_heads), 0))
    all_mols = []
    connectivity_compressor = ConnectivityCompressor()

    # construct connectivity matrix and fingerprints for filtering
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

        d = res[n_atoms]
        all_pos = d[Properties.R]
        all_numbers = d[Properties.Z]
        n_mols = len(all_pos)

        # check valency
        if args.threads <= 0:
            results = check_valency(all_pos, all_numbers, valence,
                                    'valence' in args.filters, print_file,
                                    prog_str(work_str))
        else:
            results = {'connectivity': np.zeros((n_mols, n_atoms, n_atoms)),
                       'mols': [None for _ in range(n_mols)],
                       'valid': np.ones(n_mols, dtype=bool)}
            results = run_threaded(check_valency,
                                   {'positions': all_pos,
                                    'numbers': all_numbers},
                                   {'valence': valence,
                                    'filter_by_valency': 'valence' in args.filters,
                                    'picklable_mols': True,
                                    'prog_str': prog_str(work_str)},
                                   results,
                                   n_threads=args.threads,
                                   exclusive_kwargs={'print_file': print_file})
        connectivity = results['connectivity']
        mols = results['mols']
        valid = results['valid']

        # detect molecules with disconnected parts if desired
        if 'disconnected' in args.filters:
            if not print_file:
                print('\033[K', end='\r', flush=True)
                print(prog_str("connectedness")+'...', end='\r', flush=True)
            if args.threads <= 0:
                valid = remove_disconnected(connectivity, valid)['valid']
            else:
                results = {'valid': valid}
                run_threaded(remove_disconnected,
                             {'connectivity_batch': connectivity,
                              'valid': valid},
                             {},
                             results,
                             n_threads=args.threads)
                valid = results['valid']

        # identify molecules with identical fingerprints
        if not print_file:
            print('\033[K', end='\r', flush=True)
            print(prog_str('uniqueness')+'...', end='\r', flush=True)
        if args.threads <= 0:
            still_valid, duplicating, duplicate_count = \
                filter_unique(mols, valid, use_bits=False)
        else:
            still_valid, duplicating, duplicate_count = \
                filter_unique_threaded(mols, valid,
                                       n_threads=args.threads,
                                       n_mols_per_thread=5,
                                       print_file=print_file,
                                       prog_str=prog_str('uniqueness'))
        n_non_unique += np.sum(duplicate_count)
        if 'unique' in args.filters:
            valid = still_valid  # remove non-unique from valid if desired

        # store connectivity matrices
        d.update({'connectivity': connectivity_compressor.compress_batch(connectivity),
                  'valid': valid})

        # collect statistics of generated data
        n_generated += len(valid)
        n_valid += np.sum(valid)
        n_of_types = [np.sum(all_numbers == i, axis=1) for i in
                      [6, 7, 8, 9, 1]]
        stats_new = np.stack(
            (np.ones(len(valid)) * n_atoms,     # n_atoms
             np.arange(0, len(valid)),          # id
             valid,                             # valid
             duplicating,                       # id of duplicated molecule
             duplicate_count,                   # number of duplicates
             -np.ones(len(valid)),              # known
             -np.ones(len(valid)),              # equals
             *n_of_types,                       # n_atoms per type
             *np.zeros((19, len(valid))),       # n_bonds per type pairs
             *np.zeros((7, len(valid)))         # ring counts for 3-8 & >8
             ),
            axis=0)
        stats = np.hstack((stats, stats_new))
        all_mols += mols

    if not print_file:
        print('\033[K', end='\r', flush=True)
    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    print(f'Needed {h:d}h{m:02d}m{s:02d}s.')

    if args.threads <= 0:
        results = collect_bond_and_ring_stats(all_mols, stats.T, stat_heads)
    else:
        results = {'stats': stats.T}
        run_threaded(collect_bond_and_ring_stats,
                     {'mols': all_mols, 'stats': stats.T},
                     {'stat_heads': stat_heads},
                     results=results,
                     n_threads=args.threads)
    stats = results['stats'].T

    # store statistics
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

    # filter molecules which were seen during training
    if args.model_path is not None:
        stats = filter_new(all_mols, stats, stat_heads, args.model_path,
                           args.train_data_path, print_file=print_file,
                           n_threads=args.threads)
        res.update({'stats': stats})

    # shrink results dictionary (remove invalid attempts, known molecules and
    # connectivity matrices if desired)
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
            if 'new' in args.store and args.model_path is not None:
                idcs = idcs[np.where(new_stats[idx_known, idcs] == 0)[0]]
            new_stats = new_stats[:, idcs]
            new_stats[idx_id] = np.arange(len(new_stats[idx_id]))  # adjust ids
            shrunk_stats = np.hstack((shrunk_stats, new_stats))
            # shrink positions and atomic numbers
            shrunk_res[key] = {Properties.R: d[Properties.R][idcs],
                               Properties.Z: d[Properties.Z][idcs]}
            # store connectivity matrices if desired
            if 'connectivity' in args.store:
                shrunk_res[key].update(
                    {'connectivity': [d['connectivity'][k] for k in idcs]})
            i = end

        shrunk_res['stats'] = shrunk_stats
        res = shrunk_res

    # store results in new database
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
            all_con_mats = d['connectivity']
            for pos, num, con_mat in zip(all_pos, all_numbers, all_con_mats):
                at = Atoms(num, positions=pos)
                conn.write(at, data={'con_mat': con_mat})
    # store gathered statistics in separate file
    np.savez_compressed(os.path.splitext(target_db)[0] + f'_statistics.npz',
                        stats=res['stats'], stat_heads=res['stat_heads'])

    # print average atom, bond, and ring count statistics of generated molecules
    # stored in the database and of the training molecules
    print_atom_bond_ring_stats(target_db, args.model_path, args.train_data_path)
