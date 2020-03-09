import argparse
import sys
import os
import subprocess
import numpy as np
import tempfile

from ase.db import connect
from ase.io import write
from utility_classes import IndexProvider


def get_parser():
    """ Setup parser for command line arguments """
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('--data_path', type=str, default=None,
                             help='Path to database with filtered, generated molecules '
                                  '(.db format, needs to be provided if generated '
                                  'molecules shall be displayed, default: %(default)s)')
    main_parser.add_argument('--train_data_path', type=str,
                             help='Path to training data base (.db format, needs to be '
                                  'provided if molecules from the training data set '
                                  'shall be displayed, e.g. when using --train or '
                                  '--test, default: %(default)s)',
                             default=None)
    main_parser.add_argument('--select', type=str, nargs='*',
                             help='Selection strings that specify which molecules '
                                  'shall be shown, if None all molecules from '
                                  'data_path and/or train_data_path are shown, '
                                  'providing multiple strings'
                                  ' will open multiple windows (one per string), '
                                  '(default: %(default)s). The selection string has '
                                  'the general format "Property,OperatorTarget" (e.g. '
                                  '"C,>8"to filter for all molecules with more than '
                                  'eight carbon atoms where "C" is the statistic '
                                  'counting the number of carbon atoms in a molecule, '
                                  '">" is the operator, and "8" is the target value). '
                                  'Multiple conditions can be combined to form one '
                                  'selection string using "&" (e.g "C,>8&R5,>0" to '
                                  'get all molecules with more than 8 carbon atoms '
                                  'and at least 1 ring of size 5). Prepending '
                                  '"training" to the selection string will filter and '
                                  'display molecules from the training data base '
                                  'instead of generated molecules (e.g. "training C,>8"'
                                  '). An overview of the available properties for '
                                  'molecuels generated with G-SchNet trained on QM9 can'
                                  ' be found in the README.md.',
                             default=None)
    main_parser.add_argument('--print_indices',
                             help='For each provided selection print out the indices '
                                  'of molecules that match the respective selection '
                                  'string',
                             action='store_true')
    main_parser.add_argument('--export_to_dir', type=str,
                             help='Optionally, provide a path to an directory to which '
                                  'indices of molecules matching the corresponding '
                                  'query shall be written (one .npy-file (numpy) per '
                                  'selection string, if None is provided, the '
                                  'indices will not be exported, default: %(default)s)',
                             default=None)
    main_parser.add_argument('--train',
                             help='Display all generated molecules that match '
                                  'structures used during training and the '
                                  'corresponding molecules from the training data set.',
                             action='store_true')
    main_parser.add_argument('--test',
                             help='Display all generated molecules that match '
                                  'held out test data structures and the '
                                  'corresponding molecules from the training data set.',
                             action='store_true')
    main_parser.add_argument('--novel',
                             help='Display all generated molecules that match neither '
                                  'structures used during training nor those held out '
                                  'as test data.',
                             action='store_true')
    main_parser.add_argument('--block',
                             help='Make the call to ASE GUI blocking (such that the '
                                  'script stops until the GUI window is closed).',
                             action='store_true')

    return main_parser


def view_ase(mols, name, block=False):
    '''
    Display a list of molecules using the ASE GUI.

    Args:
        mols (list of ase.Atoms): molecules as ase.Atoms objects
        name (str): the name that shall be displayed in the windows top bar
        block (bool, optional): whether the call to ase gui shall block or not block
            the script (default: False)
    '''
    dir = tempfile.mkdtemp('', 'generated_molecules_')  # make temporary directory
    filename = os.path.join(dir, name)  # path of temporary file
    format = 'traj'  # use trajectory format for temporary file
    command = sys.executable + ' -m ase gui -b'  # command to execute ase gui viewer
    write(filename, mols, format=format)  # write molecules to temporary file
    # show molecules in ase gui and remove temporary file and directory afterwards
    if block:
        subprocess.call(command.split() + [filename])
        os.remove(filename)
        os.rmdir(dir)
    else:
        subprocess.Popen(command.split() + [filename])
        subprocess.Popen(['sleep 60; rm "{0}"'.format(filename)], shell=True)
        subprocess.Popen(['sleep 65; rmdir "{0}"'.format(dir)], shell=True)


def print_indices(idcs, name='', per_line=10):
    '''
    Prints provided indices in a clean formatting.

    Args:
        idcs (list of int): indices that shall be printed
        name (str): the selection string that was used to obtain the indices
        per_line (int, optional): the number of indices that are printed per line (
            default: 10)
    '''
    biggest = len(str(max(idcs)))
    new_line = '\n'
    format = f'>{biggest}d'
    str_idcs = [f'{j:{format}}  ' + (new_line if (i+1) % per_line == 0 else '')
                for i, j in enumerate(idcs)]
    print(f'\nAll {len(idcs)} indices for selection {name}:')
    print(''.join(str_idcs))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # make sure that at least one path was provided
    if args.data_path is None and args.train_data_path is None:
        print(f'\nPlease specify --data_path to display generated molecules or '
              f'--train_data_path to display training molecules (or both)!')
        sys.exit(0)

    # sort queries into those concerning generated structures and those concerning
    # training data molecules
    train_selections = []
    gen_selections = []
    if args.select is not None:
        for selection in args.select:
            if selection.startswith('training'):
                # put queries concerning training structures aside for later
                train_selections += [selection]
            else:
                gen_selections += [selection]

    # make sure that the required paths were provided
    if args.train or args.test:
        if args.data_path is None:
            print('\nYou need to specify --data_path (and optionally '
                  '--train_data_path) if using --train or --test!')
            sys.exit(0)
    if args.novel:
        if args.data_path is None:
            print('\nYou need to specify --data_path if you want to display novel '
                  'molecules!')
            sys.exit(0)
    if len(gen_selections) > 0:
        if args.data_path is None:
            print(f'\nYou need to specify --data_path to process the selections '
                  f'{gen_selections}!')
            sys.exit(0)
    if len(train_selections) > 0:
        if args.train_data_path is None:
            print(f'\nYou need to specify --train_data_path to process the selections '
                  f'{train_selections}!')
            sys.exit(0)

    # check if statistics files are needed
    need_gen_stats = (len(gen_selections) > 0) or args.train or args.test or args.novel
    need_train_stats = (len(train_selections) > 0) or args.train or args.test

    # check if there is a database with generated molecules at the provided path
    # and load accompanying statistics file
    if args.data_path is not None:
        if not os.path.isfile(args.data_path):
            print(f'\nThe specified data path ({args.data_path}) is not a file! Please '
                  f'specify a different data path.')
            raise FileNotFoundError
        elif need_gen_stats:
            stats_path = os.path.splitext(args.data_path)[0] + f'_statistics.npz'
            if not os.path.isfile(stats_path):
                print(f'\nCannot find statistics file belonging to {args.data_path} ('
                      f'expected it at {stats_path}. Please make sure that the file '
                      f'exists.')
                raise FileNotFoundError
            else:
                stats_dict = np.load(stats_path)
                index_provider = IndexProvider(stats_dict['stats'],
                                               stats_dict['stat_heads'])

    # check if there is a database with training molecules at the provided path
    # and load accompanying statistics file
    if args.train_data_path is not None:
        if not os.path.isfile(args.train_data_path):
            print(f'\nThe specified training data path ({args.train_data_path}) is '
                  f'not a file! Please specify --train_data_path correctly.')
            raise FileNotFoundError
        elif need_train_stats:
            stats_path = os.path.splitext(args.train_data_path)[0] + f'_statistics.npz'
            if not os.path.isfile(stats_path) and len(train_selections) > 0:
                print(f'\nCannot find statistics file belonging to '
                      f'{args.train_data_path} (expected it at {stats_path}. Please '
                      f'make sure that the file exists.')
                raise FileNotFoundError
            else:
                train_stats_dict = np.load(stats_path)
                train_index_provider = IndexProvider(train_stats_dict['stats'],
                                                     train_stats_dict['stat_heads'])

    # create folder(s) for export of indices if necessary
    if args.export_to_dir is not None:
        if not os.path.isdir(args.export_to_dir):
            print(f'\nDirectory {args.export_to_dir} does not exist, creating '
                  f'it to store indices of molecules matching the queries!')
            os.makedirs(args.export_to_dir)
        else:
            print(f'\nWill store indices of molecules matching the queries at '
                  f'{args.export_to_dir}!')

    # display all generated molecules if desired
    if (len(gen_selections) == 0) and not (args.train or args.test or args.novel) and\
            args.data_path is not None:
        with connect(args.data_path) as con:
            _ats = [con.get(int(idx) + 1).toatoms() for idx in range(con.count())]
        view_ase(_ats, 'all generated molecules', args.block)

    # display generated molecules matching selection strings
    if len(gen_selections) > 0:
        for selection in gen_selections:
            # display queries concerning generated molecules
            idcs = index_provider.get_selected(selection)
            if len(idcs) == 0:
                print(f'\nNo molecules match selection {selection}!')
                continue
            with connect(args.data_path) as con:
                _ats = [con.get(int(idx) + 1).toatoms() for idx in idcs]
            if args.print_indices:
                print_indices(idcs, selection)
            view_ase(_ats, f'generated molecules ({selection})', args.block)
            if args.export_to_dir is not None:
                np.save(os.path.join(args.export_to_dir, selection), idcs)

    # display all training molecules if desired
    if (len(train_selections) == 0) and not (args.train or args.test) and \
            args.train_data_path is not None:
        with connect(args.train_data_path) as con:
            _ats = [con.get(int(idx) + 1).toatoms() for idx in range(con.count())]
        view_ase(_ats, 'all molecules in the training data set', args.block)

    # display training molecules matching selection strings
    if len(train_selections) > 0:
        # display training molecules that match the selection strings
        for selection in train_selections:
            _selection = selection.split()[1]
            stats_queries = []
            db_queries = []
            # sort into queries handled by looking into the statistics or the db
            for _sel_str in _selection.split('&'):
                prop = _sel_str.split(',')[0]
                if prop in train_stats_dict['stat_heads']:
                    stats_queries += [_sel_str]
                elif len(prop.split('+')) > 0:
                    found = True
                    for p in prop.split('+'):
                        if p not in train_stats_dict['stat_heads']:
                            found = False
                            break
                    if found:
                        stats_queries += [_sel_str]
                    else:
                        db_queries += [_sel_str]
                else:
                    db_queries += [_sel_str]
            # process queries concerning the statistics
            if len(stats_queries) > 0:
                idcs = train_index_provider.get_selected('&'.join(stats_queries))
            else:
                idcs = range(connect(args.train_data_path).count())
            # process queries concerning the db entries
            if len(db_queries) > 0:
                with connect(args.train_data_path) as con:
                    for query in db_queries:
                        head, condition = query.split(',')
                        if head not in con.get(1).data:
                            print(f'Entry {head} not found for molecules in the '
                                  f'database, skipping query {query}.')
                            continue
                        else:
                            op = train_index_provider.rel_re.search(condition).group(0)
                            op = train_index_provider.op_dict[op]  # extract operator
                            num = float(train_index_provider.num_re.search(
                                condition).group(0))  # extract numerical value
                            remaining_idcs = []
                            for idx in idcs:
                                if op(con.get(int(idx)+1).data[head], num):
                                    remaining_idcs += [idx]
                            idcs = remaining_idcs
            # extract molecules matching the query from db and display them
            if len(idcs) == 0:
                print(f'\nNo training molecules match selection {_selection}!')
                continue
            with connect(args.train_data_path) as con:
                _ats = [con.get(int(idx)+1).toatoms() for idx in idcs]
            if args.print_indices:
                print_indices(idcs, selection)
            view_ase(_ats, f'training data set molecules ({_selection})', args.block)
            if args.export_to_dir is not None:
                np.save(os.path.join(args.export_to_dir, selection), idcs)

    # display generated molecules that match structures used for training
    if args.train:
        idcs = index_provider.get_selected('known,>=1&known,<=2')
        if len(idcs) == 0:
            print(f'\nNo generated molecules found that match structures used '
                  f'during training!')
        else:
            with connect(args.data_path) as con:
                _ats = [con.get(int(idx) + 1).toatoms() for idx in idcs]
            if args.print_indices:
                print_indices(idcs, 'generated train')
            view_ase(_ats, f'generated molecules (matching train structures)',
                     args.block)
            if args.export_to_dir is not None:
                np.save(os.path.join(args.export_to_dir, 'generated train'), idcs)
        # display corresponding training structures
        if args.train_data_path is not None:
            _row_idx = list(stats_dict['stat_heads']).index('equals')
            t_idcs = stats_dict['stats'][_row_idx, idcs].astype(int)
            with connect(args.train_data_path) as con:
                _ats = [con.get(int(idx) + 1).toatoms() for idx in t_idcs]
            if args.print_indices:
                print_indices(t_idcs, 'reference train')
            view_ase(_ats, f'training molecules (train structures)', args.block)
            if args.export_to_dir is not None:
                np.save(os.path.join(args.export_to_dir, 'reference train'), t_idcs)

    # display generated molecules that match held out test structures
    if args.test:
        idcs = index_provider.get_selected('known,==3')
        if len(idcs) == 0:
            print(f'\nNo generated molecules found that match held out test '
                  f'structures!')
        else:
            with connect(args.data_path) as con:
                _ats = [con.get(int(idx) + 1).toatoms() for idx in idcs]
            if args.print_indices:
                print_indices(idcs, 'generated test')
            view_ase(_ats, f'generated molecules (matching test structures)',
                     args.block)
            if args.export_to_dir is not None:
                np.save(os.path.join(args.export_to_dir, 'generated test'), idcs)
        # display corresponding training structures
        if args.train_data_path is not None:
            _row_idx = list(stats_dict['stat_heads']).index('equals')
            t_idcs = stats_dict['stats'][_row_idx, idcs].astype(int)
            with connect(args.train_data_path) as con:
                _ats = [con.get(int(idx) + 1).toatoms() for idx in t_idcs]
            if args.print_indices:
                print_indices(t_idcs, 'reference test')
            view_ase(_ats, f'training molecules (test structures)', args.block)
            if args.export_to_dir is not None:
                np.save(os.path.join(args.export_to_dir, 'reference test'), t_idcs)

    # display generated molecules that are novel (i.e. that do not match held out
    # test structures or structures used during training)
    if args.novel:
        idcs = index_provider.get_selected('known,==0')
        if len(idcs) == 0:
            print(f'\nNo novel molecules found!')
        else:
            with connect(args.data_path) as con:
                _ats = [con.get(int(idx) + 1).toatoms() for idx in idcs]
            if args.print_indices:
                print_indices(idcs, 'novel')
            view_ase(_ats, f'generated molecules (novel)', args.block)
            if args.export_to_dir is not None:
                np.save(os.path.join(args.export_to_dir, 'generated novel'), idcs)
