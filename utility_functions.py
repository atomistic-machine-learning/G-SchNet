import torch
import os
import json
import numpy as np
import torch.nn.functional as F

from multiprocessing import Queue
from scipy.spatial.distance import pdist, squareform
from torch.autograd import Variable

from schnetpack import Properties
from utility_classes import ProcessQ, IndexProvider


def boolean_string(s):
    '''
    Allows to parse boolean strings ('true' or 'false') with argparse

    Args:
        s (str): boolean string ('true' or 'false')

    Returns:
        bool: the corresponding boolean value (True or False)
    '''
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'


def cdists(mols, grid):
    '''
    Calculates the pairwise Euclidean distances between a set of molecules and a list
    of positions on a grid (uses inplace operations to minimize memory demands).

    Args:
        mols (torch.Tensor): data set (of molecules) with shape
            (batch_size x n_atoms x n_dims)
        grid (torch.Tensor): array (of positions) with shape (n_positions x n_dims)

    Returns:
        torch.Tensor: batch of distance matrices (batch_size x n_atoms x n_positions)
    '''
    if len(mols.size()) == len(grid.size())+1:
        grid = grid.unsqueeze(0)  # add batch dimension
    return F.relu(torch.sum((mols[:, :, None, :] - grid[:, None, :, :]).pow_(2), -1),
                  inplace=True).sqrt_()


def update_dict(d, d_upd):
    '''
    Updates a dictionary of numpy.ndarray with values from another dictionary of the
    same kind. If a key is present in both dictionaries, the array of the second
    dictionary is appended to the array of the first one and saved under that key in
    the first dictionary.

    Args:
        d (dict of numpy.ndarray): dictionary to be updated
        d_upd (dict of numpy.ndarray): dictionary with new values for updating
    '''
    for key in d_upd:
        if key not in d:
            d[key] = d_upd[key]
        else:
            for k in d_upd[key]:
                d[key][k] = np.append(d[key][k], d_upd[key][k], 0)


def get_dict_count(d, max_length, skip=0):
    '''
    Counts the number of molecules in a dictionary where for each integer key i
    molecules with i atoms are stored as positions and atomic numbers. Dictionaries
    must be of the form
    {i: {'_positions': numpy.ndarray, '_atomic_numbers': numpy.ndarray}}.

    Args:
        d (dict of numpy.ndarray): dictionary with atom positions and atomic numbers of
            molecules (sorted by number of atoms per molecule)
        max_length (int): the maximum number of atoms per molecule in the
            dictionary (corresponds to the largest key in the dictionary)
        skip (int, optional): a key of the dictionary which is ignored during
            counting (e.g. to ignore molecules with 5 atoms, default: 0)

    Returns:
        int: the number of molecules in the dictionary
    '''
    n = np.zeros(max_length + 1, dtype=int)
    for key in d:
        if key == skip:
            continue
        n[key] = len(d[key][Properties.Z])
    return n


def run_threaded(target, splitable_kwargs, kwargs, results, n_threads=16,
                 exclusive_kwargs={}):
    '''
    Allows to run a target callable object in several processes simultaneously and
    join the results afterwards. This can be used to parallelize independent
    computations on objects in lists (e.g. validity checks on a set of molecules).

    Args:
        target (callable object): the function that is executed by each process
            (its return value has to be a dictionary with keys matching those of the
            dictionary passed as results parameter)
        splitable_kwargs (dict of iterables): keyword arguments for the target function
            that shall be split and equally distributed among processes
            (e.g. a list of molecules as {'molecules': [mol_1, mol_2, ..., mol_n]}).
        kwargs (dict of any): keyword arguments for the target function that are passed
            to each process (not split)
        results (dict of empty list or scaffold): dictionary where the keys correspond
            to the keys in the dictionary returned by the target function and all
            values shall either be empty lists or corresponding scaffolds (i.e. lists
            or arrays of the same length as the splitable input) that will be
            filled with the returned results
        n_threads (int): number of processes used
        exclusive_kwargs (dict of any): keyword arguments for the target function which
            are only passed to the first process

    Returns:
        dict of list: the results dictionary containing lists with the returned values
            of the target function (the order of the elements in the splitable_kwargs
            inputs is preserved in the output lists)
    '''
    # check if the number of threads is higher than the number of data points
    if len(splitable_kwargs) > 0:
        for key in splitable_kwargs:
            # extract first key of splitable keyword arguments
            first_key = key
            break
        n_data = len(splitable_kwargs[first_key])
        if n_data <= n_threads:
            # decrease number of threads
            n_threads = n_data
    else:
        return results

    # create scaffold for results
    for key in results:
        if len(results[key]) != n_data:
            results[key] = [None for _ in range(n_data)]

    # initialize list for threads and a queue for results
    threads = []
    queue = Queue(n_threads)

    # initialize and start threads
    for i in range(0, n_threads):
        thread_kwargs = {}
        for key in splitable_kwargs:
            thread_kwargs[key] = splitable_kwargs[key][i::n_threads]
        thread_kwargs.update(kwargs)  # include unsplitable kwargs
        if i == 0:
            thread_kwargs.update(exclusive_kwargs)
        threads += [ProcessQ(queue, target=target, name=str(i),
                             kwargs=thread_kwargs)]
        threads[-1].start()

    # gather returned results
    n_results = 0
    while n_results < n_threads:
        id, res = queue.get(True)
        for key in res:
            id = int(id)
            results[key][id::n_threads] = res[key]  # fill results
        n_results += 1

    # join queue
    queue.close()
    queue.join_thread()

    # join threads
    for thread in threads:
        thread.join()
        thread.terminate()

    return results


def print_equally_spaced(head, value, space=13):
    '''
    Prints a provided heading followed by a provided value with a dynamical spacing
    depending on the length of the heading and value (can be used to print small
    tables).

    Args:
        head (str): the heading that is printed
        value (str): the value that is printed
        space (int, optional): the maximum number of spaces (if heading is empty and
            value is empty)
    '''
    space = max((space - len(f'{head}:') - len(f'{value}')), 1) * ' '
    print(f'{head}:{space}{value}')


def print_accumulated_staticstics(stats, stat_heads, name='generated',
                                  fields=('H', 'C', 'N', 'O', 'F'), set=None,
                                  print_stats=('mean_percentage',),
                                  additive_fields={}):
    '''
    Accumulates and prints statistics of a set of molecules (e.g. the average number
    of carbon atoms per molecule of the set).

    Args:
        stats (numpy.ndarray): statistics of all molecules where columns
            correspond to molecules and rows correspond to available statistics
            (n_statistics x n_molecules)
        stat_heads (numpy.ndarray or list of str): the names of the statistics stored
            in each row in stats (e.g. 'F' for the number of fluorine atoms or 'R5'
            for the number of rings of size 5)
        name (str, optional): name of the set of molecules (e.g. 'generated' or
            'qm9', default: 'generated')
        fields (list of str, optional): the names of statistics for which the average
            per molecule shall be printed (e.g. ['C', 'F'] to print the average number
            of carbon atoms and the average number of fluorine atoms,
            default: ['H', 'C', 'N', 'O', 'F'])
        set (list of int, optional): indices of the molecules that shall be
            considered when accumulating the statistics (set to None to consider all
            molecules, default: None)
        print_stats (list of str, optional): additional accumulations that can be
            printed, choose from 'mean_percentage' (mean of percent of each entry in
            fields in relation to all fields, e.g. mean percent of carbon atoms of
            all carbon and fluorine atoms for fields=['C', 'F']) and 'molecules with'
            (e.g. number of molecules in the set which have at least one carbon atom
            for fields=['C']) (default: ['mean_percentage'])
        additive_fields (dict of list, optional): dictionary of lists with names of
            statistics where all statistics in a list will be summed and then
            processed as the entries in the fields parameter
            (e.g. {'triple': ['C3C', 'C3N']} will print the average number of triple
            bonds by summing the triple bonds between two carbon atoms and those
            between carbon and nitrogen atoms)
    '''
    stat_heads = list(stat_heads)
    n_mols = len(stats[0])
    if set is None:
        set = np.arange(n_mols)
    idcs = [stat_heads.index(key) for key in fields]
    np_stats = np.zeros((len(fields)+len(additive_fields), len(set)))
    names = [i for i in fields] + [key for key in additive_fields]

    print(f'\nAccumulated statistics of {name}:')
    print(f'\nMean absolute values...')
    # transfer data from fields
    for i, idx in enumerate(idcs):
        np_stats[i] = np.array(stats[idx, set])
        print_equally_spaced(stat_heads[idx], f'{np.mean(np_stats[i]):.2f}')
    # accumulate and transfer data from additive_fields
    for i, key in enumerate(additive_fields):
        _idcs = [stat_heads.index(val) for val in additive_fields[key]]
        for idx in _idcs:
            np_stats[len(fields)+i] += np.array(stats[idx, set])
        print_equally_spaced(key, f'{np.mean(np_stats[len(fields)+i]):.2f}')

    if len(np_stats) > 1:
        print_equally_spaced('All', f'{np.mean(np_stats.sum(axis=0)):.2f}')

    if 'mean_percentage' in print_stats:
        print(f'\nMean percentage...')
        for i, name in enumerate(names):
            mean_p = np.mean(np_stats[i]/np.maximum(1, np_stats.sum(axis=0)))
            print_equally_spaced(name, f'{mean_p:.2f}')

    if 'molecules_with' in print_stats:
        print(f'\nMolecules with...')
        IP = IndexProvider(stats[:, set], stat_heads)
        for i, idx in enumerate(idcs):
            selected = IP.get_selected(fields[i])
            print_equally_spaced(
                stat_heads[idx],
                f'{len(selected)}  ({len(selected)/len(set):.2f}%)',
                22)


def print_atom_bond_ring_stats(generated_data_path, model_path, train_data_path):
    '''
    Print average atom, bond, and ring count statistics of generated molecules
    in the provided database and reference training molecules.

    Args:
        generated_data_path (str): path to database with generated molecules
        model_path (str): path to directory containing the model used to generate the
            molecules (it should contain a split.npz file which is used to identify
            training, validation, and test molecules and an args.json file containing
            the arguments of the training procedure)
        train_data_path (str): path to database with training data molecules
    '''
    # load data of generated molecules
    stats_path = os.path.splitext(generated_data_path)[0] + f'_statistics.npz'
    if not os.path.isfile(stats_path):
        print(f'Statistics of generated molecules not found (expected it at '
              f'{stats_path}).\nPlease specify the correct path to the database '
              f'holding the generated molecules!')
        return
    stats_dict = np.load(stats_path)
    stats = stats_dict['stats']
    stat_heads = stats_dict['stat_heads']

    # load data of training molecules
    training_stats_path = os.path.splitext(train_data_path)[0] + f'_statistics.npz'
    if not os.path.isfile(training_stats_path):
        print(f'Statistics of training data not found (expected it at '
              f'{training_stats_path}).\nWill only print statistics of generated '
              f'molecules...')
        have_train_stats = False
    else:
        have_train_stats = True
        train_stat_dict = np.load(training_stats_path)

    # load split file to identify training, validation, and test molecules
    split_file = os.path.join(model_path, f'split.npz')
    S = np.load(split_file)
    train_idx = S['train_idx']
    # check if subset was used (and restrict indices accordingly)
    train_args_path = os.path.join(model_path, f'args.json')
    with open(train_args_path) as handle:
        train_args = json.loads(handle.read())
    if 'subset_path' in train_args:
        if train_args['subset_path'] is not None:
            subset = np.load(train_args['subset_path'])
            train_idx = subset[train_idx]

    # Atom type statistics
    descr = ' concerning atom types'
    print_accumulated_staticstics(stats, stat_heads,
                                  name='generated molecules' + descr)
    if have_train_stats:
        print_accumulated_staticstics(train_stat_dict['stats'],
                                      train_stat_dict['stat_heads'],
                                      name='training molecules' + descr,
                                      set=train_idx)

    # Atom bond statistics
    descr = ' concerning atom bonds'
    print_accumulated_staticstics(stats, stat_heads, fields=(),
                                  name='generated molecules' + descr,
                                  additive_fields={
                                      'Single': ['H1C', 'H1N', 'H1O',
                                                 'C1C', 'C1N', 'C1O', 'C1F',
                                                 'N1N', 'N1O', 'N1F',
                                                 'O1O', 'O1F'],
                                      'Double': ['C2C', 'C2N', 'C2O',
                                                 'N2N', 'N2O'],
                                      'Triple': ['C3C', 'C3N']})
    if have_train_stats:
        print_accumulated_staticstics(train_stat_dict['stats'],
                                      train_stat_dict['stat_heads'],
                                      fields=(),
                                      name='training molecules' + descr,
                                      set=train_idx,
                                      additive_fields={
                                          'Single': ['H1C', 'H1N', 'H1O',
                                                     'C1C', 'C1N', 'C1O', 'C1F',
                                                     'N1N', 'N1O', 'N1F',
                                                     'O1O', 'O1F'],
                                          'Double': ['C2C', 'C2N', 'C2O',
                                                     'N2N', 'N2O'],
                                          'Triple': ['C3C', 'C3N']})

    # Ring statistics
    descr = ' concerning ring structures'
    fields = ['R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R>8']
    print_accumulated_staticstics(stats, stat_heads, fields=fields,
                                  name='generated molecules' + descr,
                                  print_stats=['molecules_with'])
    if have_train_stats:
        print_accumulated_staticstics(train_stat_dict['stats'],
                                      train_stat_dict['stat_heads'],
                                      fields=fields,
                                      name='training molecules' + descr,
                                      set=train_idx,
                                      print_stats=['molecules_with'])


def get_random_walk(mol_dict, stop_token=10, seed=None):
    '''
    Builds a random generation trace of a training molecule. Assumes that the atoms are
    ordered by distance to the center of mass (close to far) and always starts with
    the first atom (i.e. the one closest to center of mass). At each step, one of the
    already placed atoms is randomly chosen as focus. Then the first unplaced
    neighbor (closest to center of mass) is chosen as the next addition to the
    unfinished structure. If all neighbors have already been placed, the stop token
    is chosen as the next type and the focus is marked as finished (cannot be chosen
    as focus anymore). The trace is complete after all atoms have been placed and
    marked as finished.

    The resulting trace will be saved in the dictionary passed as mol_dict argument.
    The atom positions and atomic numbers will be ordered in the sequence of
    placement in the trace and additionally there will be entries 'pred_types'
    (torch.Tensor containing the next type that shall be predicted at each step
    including stop tokens) and 'current' (torch.Tensor containing the index of the
    currently focused atom at each step).

    Args:
        mol_dict (dict of torch.Tensor): dict containing the atom positions
            ('_positions') ordered by distance to the center of mass of the molecule,
            the atomic numbers ('_atomic_numbers'), the connectivity matrix
            ('_con_mat'), and, optionally, the precomputed distances ('dists')
        stop_token (int, optional): a dummy atom type which is used as the stop token
            (default: 10)
        seed (int, optional): a seed for the random selection of the focus at each
            step (default: None)
    '''

    # set seed
    if seed is not None:
        old_state = torch.get_rng_state()
        torch.manual_seed(seed)

    # extract positions, atomic numbers, and connectivity matrix
    numbers = mol_dict[Properties.Z]
    n_atoms = len(numbers)
    con_mat = (mol_dict['_con_mat'] > 0).float()

    current = [-1]  # in the first step, none of the atoms is focused
    order = [0]  # the new ordering always starts with the first atom (closest to com)
    pred_types = [numbers[0]]  # the first predicted type is that of the first atom

    # start from first atom and traverse molecular graph (choosing the focus randomly)
    con_mat[:, 0] = 0  # mark first atom as placed by removing its bonds
    avail = torch.zeros(n_atoms).float()  # list with atoms available as focus
    avail[0] = 1.  # first atom is available
    i = 1
    while torch.sum(con_mat > 0) or (torch.sum(avail) > 0):
        # take random current focus
        cur_i = torch.multinomial(avail, 1)[0]
        current += [cur_i]
        cur = order[cur_i]

        # predict finished if no neighbors are left
        if torch.sum(con_mat[cur]) == 0:
            pred_types += [stop_token]  # predict stop token
            avail[cur_i] = 0
            continue

        # else choose neighbor which is closest to center of mass (first nonzero entry)
        next = torch.nonzero(con_mat[cur] > 0)[0][0]

        pred_types += [numbers[next]]
        order += [next]
        con_mat[:, next] = 0  # mark next atom as placed by removing its bonds
        avail[i] = 1  # mark placed atom as available for being the focus
        i += 1

    # cast to torch.Tensor
    order = torch.tensor(order)
    pred_types = torch.tensor(pred_types)
    current = torch.tensor(current)

    # update dict of molecule with re-ordered positions, numbers, focus, and next types
    mol_dict.update({'pred_types': pred_types,
                    'current': current,
                     Properties.R: mol_dict[Properties.R][order],
                     Properties.Z: mol_dict[Properties.Z][order]})

    # re-order or calculate distances (depending on whether they were precomputed)
    if 'dists' in mol_dict:
        # re-order
        mol_dict['dists'] = \
            squareform(squareform(mol_dict['dists'][:, 0])[order][:, order])[:, None]
    else:
        # compute
        mol_dict['dists'] = pdist(mol_dict[Properties.R][order])[:, None]

    # reset seed
    if seed is not None:
        torch.set_rng_state(old_state)


def get_labels(n_bins, max_size, target, width_scaling):
    '''
    Get labels for distance predictions from ground truth distances. The labels are
    either obtained by 1d Gaussian smearing or by one-hot encoding.

    The bins c_i are found by taking n_bins equally spaced points in 0 <= c_i <=
    max_size (e.g. for n_bins=3 and max_size=1 the bins are [0, 0.5, 1]).
    Let w be the distance between two neighboring bins (e.g. 0.5 in the example
    before). For one-hot encodings, ground truth distances d are sorted into a bin c_i
    if (c_i - 0.5*w) <= d < (c_i + 0.5*w). For Gaussian smearing,
    e^-((d-c_i)**2 / w*width_scaling) is calculated for each bin c_i.

    Distance values larger than max_size are always one-hot encoded into the last bin
    of the distribution.

    Args:
        n_bins (int): number of bins used to discretize (1d) space of distances
        max_size (int): maximum distance covered (i.e. the disrcetized distribution
            will cover the space 0 <= dist <= max_size)
        target (torch.Tensor): distance matrix (n_atoms x n_atoms) or batch of
            distance matrices (batch_size x n_atoms x n_atoms) holding the ground truth
        width_scaling (float): factor scaling the width in the Gaussian smearing
            (the width is the distance between two neighboring bins if
            width_scaling=1., set to 0 to use one-hot encoding instead)

    Returns:
        torch.Tensor: labels for distance predictions (discretized distributions over
            distances of the shape ((batch_size x )n_atoms x n_atoms x n_bins))
    '''
    if width_scaling > 0:
        # use 1d Gaussian smearing
        centers = torch.linspace(0, max_size, n_bins)
        centers = centers.view(*[1 for _ in target.size()], -1)
        width = max_size / (n_bins - 1) * width_scaling
        labels = torch.exp(-(1 / width) * (target.unsqueeze(-1)-centers) ** 2)
        max_dist_label = torch.zeros(n_bins)
        max_dist_label[-1] = 1.
        labels = torch.where(target.unsqueeze(-1) <= max_size, labels, max_dist_label)
        labels = labels / torch.sum(labels, -1, keepdim=True)
    else:
        # use one hot encoding
        width = max_size / (n_bins - 1)
        bins = (((target + (width / 2.)) / max_size) * (n_bins - 1)).long()
        bins = torch.clamp(bins, 0, n_bins - 1)
        label_idcs = torch.arange(n_bins).reshape(1, 1, -1)
        zero_labels = torch.zeros(*target.size(), n_bins)
        labels = torch.where(label_idcs == bins.unsqueeze(-1),
                             torch.ones_like(zero_labels), zero_labels)
    return labels


def get_padded_batch(mol_dicts):
    '''
    Builds a batch of input data and applies padding where necessary.

    Args:
        mol_dicts (list of dict of torch.Tensor): the input data for each molecule
            (positions, atomic numbers, labels etc.) in a list

    Returns:
        dict of torch.Tensor: the input data as batches in a dictionary
    '''

    properties = mol_dicts[0]

    # initialize maximum sizes
    max_size = {
        prop: np.array(val.size(), dtype=np.int)
        for prop, val in properties.items()
    }

    # get maximum sizes
    for properties in mol_dicts[1:]:
        for prop, val in properties.items():
            max_size[prop] = np.maximum(max_size[prop], np.array(val.size(),
                                                                 dtype=np.int))

    # initialize batch
    batch = {
        p: torch.zeros(len(mol_dicts),
                       *[int(ss) for ss in size]).type(mol_dicts[0][p].type())
        for p, size in max_size.items()
    }
    has_atom_mask = Properties.atom_mask in batch
    has_neighbor_mask = Properties.neighbor_mask in batch

    if not has_neighbor_mask:
        batch[Properties.neighbor_mask] =\
            torch.zeros_like(batch[Properties.neighbors]).float()
    if not has_atom_mask:
        batch[Properties.atom_mask] =\
            torch.zeros_like(batch[Properties.Z]).float()

    # build batch and pad
    for k, properties in enumerate(mol_dicts):
        for prop, val in properties.items():
            shape = val.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[prop][s] = val

        # add mask
        if not has_neighbor_mask:
            nbh = properties[Properties.neighbors]
            shape = nbh.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            mask = nbh >= 0
            batch[Properties.neighbor_mask][s] = mask
            batch[Properties.neighbors][s] = nbh * mask.long()

        if not has_atom_mask:
            z = properties[Properties.Z]
            shape = z.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[Properties.atom_mask][s] = z > 0

    # wrap everything in variables
    batch = {k: Variable(v) for k, v in batch.items()}

    return batch


def collate_atoms(mol_dicts,
                  all_types=[1, 6, 7, 8, 9, 10],
                  start_token=11,
                  n_bins=300,
                  max_dist=15.,
                  label_width_scaling=0.1,
                  draw_samples=0,
                  seed=None):
    '''
    Split each molecule into a random generation trace and then build batch of input
    data and apply padding.

    Args:
        mol_dicts (list of dict): list of dicts containing properties of a training
            molecule (positions, atomic numbers, connectivity matrix etc.)
        all_types (list of int, optional): list of all atom types in the data set in
            ascending order (including a dummy index as stop token that should be
            larger than all other types and therefore the last entry, default:
            [1, 6, 7, 8, 9, 10], which are all atomic charges in QM9 and 10 as stop
            token)
        start_token (int, optional): a dummy atom type (not included in all_types)
            which is used as the start token (default: 11)
        n_bins (int, optional): number of bins used to discretize (1d) space of
            distances (default: 300)
        max_dist (int, optional): maximum distance covered in the learned
            distributions (i.e. the disrcetized distribution will cover the
            space 0<=dist<= max_size, default: 15.)
        label_width_scaling (float, optional): factor scaling the width in the Gaussian
            smearing of distance labels (the width is the distance between two
            neighboring bins if width_scaling=1., set to 0 to use one-hot encoding
            instead, default: 0.1)
        draw_samples (int, optional): the number of steps in the generation trace
            that are randomly drawn for training (set to 0 to use the complete trace,
            default: 0)
        seed (int, optional): seed for the random sampling of a generation trace (set
            None to use no seed, default: None)

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    '''

    # store all possible types in a tensor, build one-hot encoded label vectors
    all_types_tensor = torch.tensor(all_types)
    type_labels = torch.eye(len(all_types))  # one-hot encoding of types
    # build array that converts type to correct row index in the type labels
    type_idc_converter = torch.zeros(torch.max(all_types_tensor)+1).long()
    type_idc_converter[all_types_tensor.long()] = torch.arange(len(all_types))
    # extract stop token
    stop_token = all_types[-1]
    # we use the same token as dummy type for the currently focused atom
    focus_token = stop_token

    # initialize lists
    mols_gen_steps = []  # stores partial molecules at single steps of generation traces
    next_list = []  # stores next atom type at each step
    current_list = []  # stores currently focused atom at each step

    n_tokens = 2  # we use a start (origin) token and a focus token

    # divide every molecule into a generation trace (starting from the atom closest to
    # the center of mass) to cover all steps of the generation process and save partial
    # molecules along with labels of distance distributions and next type
    for mol in mol_dicts:
        # update molecule dict with random generation trace
        get_random_walk(mol, stop_token=stop_token, seed=seed)

        # extract information about molecule (and trace)
        pos = mol[Properties.R]
        numbers = mol[Properties.Z]
        neighbors = mol[Properties.neighbors]
        pred_types = mol['pred_types'].long()
        focus = mol['current'].long() + n_tokens

        # add start and focus tokens to data (positions, distances, atomic numbers)
        # use origin as position for both tokens
        pos = torch.cat((torch.zeros(n_tokens, 3), pos), 0)
        n_atoms = len(pos)
        # store (pre-computed) pairwise distances between atoms in a distance matrix
        dists = torch.zeros(n_atoms, n_atoms)
        dists[n_tokens:, n_tokens:] = torch.tensor(squareform(mol['dists'][:, 0]))
        # compute distances of atoms to origin (tokens) and save them in distance matrix
        center_dists = torch.sqrt(F.relu(torch.sum(pos ** 2, dim=1)))  # dists to com
        dists[:n_tokens, :] = center_dists.view(1, -1)
        dists[:, :n_tokens] = center_dists.view(-1, 1)
        # add start and focus token to atomic numbers
        numbers = torch.cat((torch.tensor([focus_token, start_token]), numbers), 0)
        # adjust neighbor lists due to tokens (which are basically additional atoms)
        for i in range(n_tokens, 0, -1):
            neighbors = \
                torch.cat((neighbors, torch.ones(n_atoms-i, 1).long()*n_atoms-i), 1)
            neighbors = \
                torch.cat((neighbors, torch.arange(n_atoms-i).view(1, -1)), 0)
        cell_offset = torch.zeros(n_atoms, n_atoms-1, 3)
        # update dictionary with altered data
        mol.update({Properties.R: pos,
                    Properties.Z: numbers,
                    Properties.neighbors: neighbors,
                    Properties.cell_offset: cell_offset})

        # get distance labels
        mol['_labels'] = get_labels(n_bins, max_dist, dists, label_width_scaling)

        # remove unnecessary entries
        mol.pop('dists')
        mol.pop('pred_types')
        mol.pop('current')
        mol.pop('_con_mat')

        # DIVIDE INTO PARTIAL MOLECULAR STRUCTURES #
        # i marks the current step in the trace excluding prediction of stop tokens
        # j marks the number of times the stop token has been predicted at the current i
        if draw_samples <= 0:  # place atom by atom for the whole molecule
            j = 0
            index_list = range(n_tokens, n_atoms + 1)
            next_list += list(pred_types)
            current_list += list(focus)
        else:  # randomly draw steps of the generation trace for training
            overall_steps = len(pred_types)

            # sample a few random steps
            np.random.seed(seed)  # set seed (will have no effect if None)
            random_steps = np.random.choice(overall_steps,
                                            min(draw_samples, overall_steps),
                                            replace=False)
            np.random.seed(None)  # reset seed

            index_list = []
            j_list = []
            for random_step in random_steps:
                random_step = int(random_step)
                i = int(torch.sum(pred_types[:random_step] != stop_token))
                j = random_step - i
                index_list += [i+n_tokens]
                j_list += [j]
                next_list += [pred_types[random_step]]
                current_list += [focus[random_step]]
            if len(j_list) > 0:
                j = j_list.pop(0)

        # iterate over steps in trace
        for i in index_list:
            _i = i-n_tokens  # atom index if ignoring tokens
            while True:
                partial_mol = mol.copy()
                # get the type of the next atom
                next_type = pred_types[_i+j]
                # don't consider distance predictions at stop token prediction steps
                if next_type == stop_token:
                    dist_mask = torch.zeros(i).float()
                else:
                    dist_mask = torch.ones(i).float()
                # always consider the type prediction
                type_mask = torch.ones(i).float()
                # assemble neighborhood mask and neighbor indices
                neighbor_mask = torch.ones(i, i-1)
                neighbors = partial_mol[Properties.neighbors][:i, :i-1]
                # set position and labels of the current (focus) token (first atom)
                cur = focus[_i + j]
                pos = partial_mol[Properties.R][:i]
                label_idx = i if i < n_atoms else 0
                labels = partial_mol['_labels'][label_idx, :i]
                pos = torch.cat((pos[cur:cur+1], pos[1:]), 0)
                labels = torch.cat((labels[cur:cur+1], labels[1:]), 0)

                partial_mol.update(
                    {Properties.R: pos,
                     Properties.Z: partial_mol[Properties.Z][:i],
                     '_labels': labels,
                     '_dist_mask': dist_mask,
                     '_type_mask': type_mask,
                     Properties.neighbor_mask: neighbor_mask,
                     Properties.neighbors: neighbors,
                     Properties.cell_offset: partial_mol[Properties.cell_offset][:i, :i-1]
                     })

                # store current step in list of trace steps of mini-batch molecules
                mols_gen_steps += [partial_mol]

                if draw_samples > 0:
                    if len(j_list) > 0:
                        j = j_list.pop(0)
                    break

                if pred_types[_i+j] == stop_token:
                    j += 1  # increase stop token counter
                    if j == n_atoms-n_tokens:
                        # stop predicted for every atom -> trace finished
                        break
                    else:
                        # continue with next step in trace (without changing i)
                        continue
                else:
                    # continue with next step in trace (by getting next i)
                    break

    # build nn-input mini-batch from gathered generation steps
    batch = get_padded_batch(mols_gen_steps)

    # update with remaining indicators (the type of the next atom, all available atom
    # types and the one-hot encoded labels for the type predictions)
    next_list = torch.tensor(next_list)
    batch.update({'_next_types': Variable(next_list),
                  '_all_types': Variable(all_types_tensor.view(1, -1)),
                  '_type_labels': Variable(
                      type_labels[torch.gather(type_idc_converter, 0, next_list)])
                  })

    return batch


def get_grid(radial_limits, n_bins, max_dist):
    '''
    Get a grid with candidate atom positions. A lower and upper radial cutoff is used
    to remove positions too close to or too far from the origin of the grid (which
    should manually be centered on the current focus token at each generation step).
    The grid extends in steps of 0.05 Angstrom into x, y, and z directions.

    For the very first step (when predicting the first atom position), a grid with
    positions that extend only into one dimension is returned. There is no need to
    extend the grid into all dimensions in the first step as the predicted distribution
    of positions will be radial by design (since the focus and origin token are
    located at the same position). Furthermore, the minimum and maximum distances
    between atoms that can be taken from the training data as radial limits will
    generally not apply to the first prediction, where the distance between the center
    of mass and the closest atom is determined. Therefore, the special start grid does
    not make use of the provided radial limits.

    Args:
        radial_limits (list of float): list with lower distance limit as first entry
            and upper distance limit as second entry
        n_bins (int): number of bins used for distance predictions (will be used to
            assemble the special grid for the first generation step)
        max_dist float): maximum distance covered in the predicted distance
            distributions (will be used to assemble the special grid for the first
            generation step)

    Returns:
        grid (numpy.ndarray): 2d array of grid positions (n_grid_positions x 3) for all
            generation steps except the first
        start_grid (numpy.ndarray): 2d array of grid positions (n_grid_positions x 3)
            for the first generation step

    '''
    n_dims = 3  # make grid in 3d space
    grid_max = radial_limits[1]
    grid_steps = int(grid_max * 2 * 20) + 1  # gives steps of length 0.05
    coords = np.linspace(-grid_max, grid_max, grid_steps)
    grid = np.meshgrid(*[coords for _ in range(n_dims)])
    grid = np.stack(grid, axis=-1)  # stack to array (instead of list)
    # reshape into 2d array of positions
    shape_a0 = np.prod(grid.shape[:n_dims])
    grid = np.reshape(grid, (shape_a0, -1))
    # cut off cells that are out of the spherical limits
    grid_dists = np.sqrt(np.sum(grid**2, axis=-1))
    grid_mask = np.logical_and(grid_dists >= radial_limits[0],
                               grid_dists <= radial_limits[1])
    grid = grid[grid_mask]
    # assemble special grid extending only in one direction (x-axis) for the first step
    # (we don't need to populate a 3d grid due to rotational invariance at first step)
    start_grid = np.zeros((n_bins, n_dims))
    start_grid[:, 0] = np.linspace(0, max_dist, n_bins)  # only extend along x-axis
    return grid, start_grid


def get_default_neighbors(n_atoms):
    '''
    Get a neighborhood indices matrix where every atom is the neighbor of every other
    atom (but not of itself, e.g. [[1, 2], [0, 2], [0, 1]] for three atoms).

    Args:
        n_atoms (int): number of atoms

    Returns:
        list of list of int: the indices of the neighbors of each atom
    '''
    return [list(range(0, i)) + list(range(i + 1, n_atoms)) for i in range(0, n_atoms)]


def generate_molecules(amount,
                       model,
                       t=0.1,
                       max_length=35,
                       save_unfinished=False,
                       all_types=[1, 6, 7, 8, 9, 10],
                       start_token=11,
                       n_bins=300,
                       max_dist=15.,
                       radial_limits=[0.9, 1.7],
                       device='cuda'
                       ):
    '''
    Generate molecules using a trained G-SchNet model. The atomic numbers of all
    chemical elements in the training data and the numbers assigned to focus and
    start token need to be specified. A spherical grid that is re-centered on the
    focused atom at every generation step is used. Its minimum and maximum distance
    can be provided and should be close to the minimum and maximum distances of
    neighbored atoms observed in the training data.

    Args:
        amount (int): the amount of molecules that shall be generated
        model (schnetpack.atomistic.AtomisticModel): a trained G-SchNet model
        t (float, optional): the sampling temperature which controls randomness during
            sampling of atom positions (higher values introduce more randomness,
            default: 0.1)
        max_length (int, optional): the maximum number of atoms per molecule (if not
            all atoms have been marked as finished when this number is reached then
            generation is stopped and unfinished molecules are either disregarded or
            stored in the category of unfinished examples depending on the
            save_unfinished argument, default: 35)
        save_unfinished (bool, optional): whether molecules that are still unfinished
            after sampling 'max_length' atoms shall be stored in the returned
            dictionary of generated moleclues (the key for unfinished molecules is -1,
            default: False)
        all_types (list of int optional): list of all atom types in the training data
            set in ascending order (including a dummy index as stop token that should
            be larger than all other types and therefore the last entry, default:
            [1, 6, 7, 8, 9, 10], which are all atomic charges in QM9 and 10 as stop
            token)
        start_token (int, optional): a dummy atom type (not included in all_types)
            which is used as the start token (default: 11)
        n_bins (int, optional): number of bins used to discretize (1d) space of
            distances (default: 300)
        max_dist (int, optional): maximum distance covered in the learned
            distributions (default: 15.)
        radial_limits (list of float, optional): list with lower distance limit for
            the radial atom position grid as first entry and upper distance limit as
            second entry (default: [0.9, 1.7])
        device (str, optional): choose whether to run the model on cpu ('cpu') or
            gpu('cuda', default: 'cuda')

    Returns:
        dict[int->str->numpy.ndarray]: positions and atomic numbers of generated
            molecules where the first key is the number of atoms (i.e. all generated
            molecules with 9 atoms can be found using the key 9) and the second key
            is either '_positions' to get a (n_molecules x n_atoms x 3) array of atom
            positions or '_atomic_numbers' to get the corresponding
            (n_molecules x n_atoms) array of atomic numbers

    '''
    failed_counter = 0
    n_dims = 3
    n_tokens = 2  # token for current atom and for center of mass (start)
    start_idx = 1  # index of start_token
    model = model.to(device)  # put model on chosen device (gpu/cpu)

    # increase max_length by three to compensate for tokens and last prediction step
    max_length += n_tokens+1
    all_types = torch.tensor(all_types).long().to(device)
    stop_token = all_types[-1]
    focus_token = stop_token

    # initialize tensor that stores the indices of currently focused atoms
    current_atoms = torch.ones(amount).long().to(device)  # store current atom
    # initialize tensor for atomic numbers
    atom_numbers = torch.zeros(amount, max_length).long().to(device)
    # set first atom as current (focus) token and second as center of mass (start)
    atom_numbers[:, 0] = focus_token
    atom_numbers[:, 1] = start_token
    # initialize tensor for atom positions
    positions = torch.zeros(amount, max_length, n_dims).to(device)
    # initialize mask for molecules which are not yet finished (all in the beginning)
    unfinished = torch.ones(amount, dtype=torch.bool).to(device)
    # initialize mask to mark single atoms as finished/unfinished
    atoms_unfinished = torch.ones(amount, max_length).float().to(device)
    # molecule generation stops if all regular atoms of a molecule are marked finished
    atoms_unfinished[:, [0]] = 0  # mark focus token as finished

    # create grids (a small, linear one for the very first step and a radial one
    # for all following generation steps)
    general_grid, start_grid = get_grid(radial_limits, n_bins=n_bins, max_dist=max_dist)
    general_grid = torch.tensor(general_grid).float().to(device)  # radial grid
    start_grid = torch.tensor(start_grid).float().to(device)  # small start grid

    # create default neighborhood list
    neighbors = torch.tensor(get_default_neighbors(max_length-1)).long().to(device)

    # create dictionary in which generated molecules will be stored (where the key
    # will be the number of atoms in the respective generated molecule)
    results = {}
    # create short name for function that pulls results from gpu and removes
    #  the start and current tokens (first two entries)
    s = lambda x: x[:, n_tokens:].detach().cpu().numpy()

    # define function that builds a model input batch with current state of molecules
    def build_batch(i):
        amount = torch.sum(unfinished)  # only get predictions for unfinished molecules
        # build neighborhood and neighborhood mask
        neighbors_i = neighbors[:i, :i-1].expand(amount, -1, -1).contiguous()
        neighbor_mask = torch.ones_like(neighbors_i).float()
        # set position of focus token (first entry of positions)
        positions[unfinished, 0] = positions[unfinished, current_atoms[unfinished]]
        # center positions on currently focused atom (for localized grid)
        positions[unfinished, :i] -= \
            positions[unfinished, current_atoms[unfinished]][:, None, :]

        # build batch with data of the partial molecules
        batch = {
            Properties.R: positions[unfinished, :i],
            Properties.Z: atom_numbers[unfinished, :i],
            Properties.atom_mask: torch.zeros(amount, i, dtype=torch.float),
            Properties.neighbors: neighbors_i,
            Properties.neighbor_mask: neighbor_mask,
            Properties.cell_offset: torch.zeros(amount, i, max(i-1, 1), n_dims),
            Properties.cell: torch.zeros(amount, n_dims, n_dims),
            '_next_types': atom_numbers[unfinished, i],
            '_all_types': all_types.view(1, -1),
            '_type_mask': torch.ones(amount, i, dtype=torch.float),
        }

        # put batch into torch variables and on gpu
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        return batch

    for i in range(n_tokens, max_length):
        amount = torch.sum(unfinished)
        # stop if the generation process is finished for all molecules
        if amount == 0:
            break
        # store the global state of molecules (whether they are finished)
        global_unfinished = unfinished.clone()

        ### 1st Part ###
        # predict and sample next atom type until all unfinished molecules either have
        # a proper next type (not stop token) or are completely finished
        while torch.sum(unfinished) > 0:
            # set the marker for the current (focus) atom
            current_atoms[unfinished] = \
                torch.multinomial(atoms_unfinished[unfinished, :i], 1).squeeze()

            # get batch with updated data (changes in each iteration as unfinished and
            # current_atoms are changed)
            batch = build_batch(i)

            # predict distribution over next atom types with model
            type_pred = F.softmax(model(batch)['type_predictions'], dim=-1)
            # sample types from predictions
            next_types = torch.multinomial(type_pred, 1)
            # store sampled type in tensor with atomic numbers
            atom_numbers[unfinished, i] = all_types[next_types].view(-1)
            # get molecules that predicted no proper type but the stop token
            pred_stop = torch.eq(atom_numbers[unfinished, i], stop_token)
            # set current atom of these molecules to finished
            stop_mask = torch.zeros(len(unfinished), dtype=torch.bool).to(device)
            stop_mask[unfinished] = pred_stop
            atoms_unfinished[stop_mask, current_atoms[stop_mask]] = 0
            # get molecules that were finished in this iteration (those which were
            # unfinished before and now have all atoms marked as finished)
            finished = global_unfinished & \
                       (torch.sum(atoms_unfinished[:, :i], dim=1) == 0)

            # store molecules which are not yet completely finished but have
            # predicted the stop type in the local unfished list in order to repeat
            # the prediction procedure for these molecule (until they predict a
            # proper type for which we can sample a new position)
            unfinished[unfinished] = pred_stop & ~finished[unfinished]

        # store molecules which have been finished in this generation step (i.e. all
        # of their atoms are marked as finished)
        idx = i-n_tokens  # number of atoms in the finished molecules
        if idx > 0 and torch.sum(finished) > 0:
            # center generated molecules on origin token
            positions[finished, :i] -= positions[finished, start_idx][:, None, :]
            # store positions and atomic numbers in dictionary
            results[idx] = {Properties.R: s(positions[finished, :i]),
                            Properties.Z: s(atom_numbers[finished, :i])}

        # mark finished moleclues in global unfinished mask
        global_unfinished[global_unfinished] = ~finished[global_unfinished]
        # reset local unfinished mask to global state
        unfinished[global_unfinished] = 1

        # stop if max_length of molecules is reached or all are finished
        amount = torch.sum(unfinished)
        if i >= max_length-1 or amount == 0:
            break

        ### 2nd Part ###
        # sample new position given the type of the next atom

        # get batch with updated data
        batch = build_batch(i)
        # run model to get predictions
        logits = model(batch)
        # get normalized log probabilities
        log_p = F.log_softmax(logits['distance_predictions'], -1)
        del logits

        if i == n_tokens:
            grid = start_grid  # use grid with positions on a line to sample first atom
        else:
            grid = general_grid  # use radial 3d grid for all steps after the first

        # set up storage for log pdf over grid positions
        log_pdf = torch.zeros_like(grid[:, 0].expand(amount, -1))
        step = max_dist / (n_bins-1)  # step size between two distance bins
        # iterate over atoms in order to reduce memory demands
        for atom in range(i):
            # calculate distances between grid points and respective atom
            dists = cdists(batch[Properties.R][:, atom:atom+1, :], grid)
            # calculate indices of the corresponding distance bins
            dists += step / 2.
            dists *= (n_bins-1) / max_dist
            dists.clamp_(0., n_bins-1)
            dist_labels = dists.long().squeeze(1)
            del dists
            # look up probabilities of distance bins in output
            log_p_grid = torch.gather(log_p[:, atom], -1, dist_labels)

            # multiply predictions for individual atoms to get probability
            log_pdf += log_p_grid
            del log_p_grid
        del log_p

        # normalize distribution over grid
        log_pdf -= torch.logsumexp(log_pdf, -1, keepdim=True)
        # use temperature term on logits and normalize over grid again
        if i > n_tokens:  # not for the very first atom with special grid
            log_pdf /= t
            log_pdf -= torch.logsumexp(log_pdf, -1, keepdim=True)

        log_pdf.exp_()  # take exponential

        # remove numerically failed attempts (NaN in pdf) by marking them as finished
        # (they are not stored among the properly generated molecules, only disregarded)
        if torch.isnan(log_pdf).any():
            failed_mask = torch.isnan(log_pdf).any(dim=-1)
            unfinished[unfinished] = ~failed_mask
            log_pdf = log_pdf[~failed_mask]
            failed_counter += torch.sum(failed_mask)

        # sample positions of new atoms using the calculated pdfs over grid positions
        new_atom_idcs = torch.multinomial(log_pdf, 1).view(-1)
        del log_pdf
        # store new positions
        positions[unfinished, i, :] = grid[new_atom_idcs]

        # set start token to finished at the end of the first iteration
        if i == n_tokens:
            atoms_unfinished[:, [start_idx]] = 0

    # store unfinished molecules of max_length
    if save_unfinished:
        if torch.sum(unfinished) > 0:
            batch = build_batch(i)
            results[-1] = {Properties.R: s(batch[Properties.R]),
                           Properties.Z: s(batch[Properties.Z])}

    if failed_counter > 0:
        print(f'Failed attempts: {failed_counter}')
    return results
