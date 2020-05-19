import argparse
import logging
import os
import pickle
import time
from shutil import copyfile, rmtree

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.sampler import RandomSampler
from ase import Atoms
import ase.visualize as asv

import schnetpack as spk
from schnetpack.utils import count_params, to_json, read_from_json
from schnetpack import Properties
from schnetpack.datasets import DownloadableAtomsData

from nn_classes import AtomwiseWithProcessing, EmbeddingMultiplication,\
    NormalizeAndAggregate, KLDivergence
from utility_functions import boolean_string, collate_atoms, generate_molecules, \
    update_dict, get_dict_count

# add your own dataset classes here:
from qm9_data import QM9gen
from template_data import TemplateData
dataset_name_to_class_mapping = {'qm9': QM9gen,
                                 'template_data': TemplateData}

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def get_parser():
    """ Setup parser for command line arguments """
    main_parser = argparse.ArgumentParser()

    ## command-specific
    cmd_parser = argparse.ArgumentParser(add_help=False)
    cmd_parser.add_argument('--cuda', help='Set flag to use GPU(s)',
                            action='store_true')
    cmd_parser.add_argument('--parallel',
                            help='Run data-parallel on all available GPUs '
                                 '(specify with environment variable'
                                 + ' CUDA_VISIBLE_DEVICES)',
                            action='store_true')
    cmd_parser.add_argument('--batch_size', type=int,
                            help='Mini-batch size for training and prediction '
                                 '(default: %(default)s)',
                            default=5)
    cmd_parser.add_argument('--draw_random_samples', type=int, default=0,
                            help='Only draw x generation steps per molecule '
                                 'in each batch (if x=0, all generation '
                                 'steps are included for each molecule,'
                                 'default: %(default)s)')
    cmd_parser.add_argument('--checkpoint', type=int, default=-1,
                            help='The checkpoint of the model that is going '
                                 'to be loaded for evaluation or generation '
                                 '(set to -1 to load the best model '
                                 'according to validation error, '
                                 'default: %(default)s)')
    cmd_parser.add_argument('--precompute_distances', type=boolean_string,
                            default='true',
                            help='Store precomputed distances in the database '
                                 'during pre-processing (caution, has no effect if '
                                 'the dataset has already been downloaded, '
                                 'pre-processed, and stored before, '
                                 'default: %(default)s)')

    ## training
    train_parser = argparse.ArgumentParser(add_help=False,
                                           parents=[cmd_parser])
    train_parser.add_argument('datapath',
                              help='Path / destination of dataset '\
                                   'directory')
    train_parser.add_argument('modelpath',
                              help='Destination for models and logs')
    train_parser.add_argument('--dataset_name', type=str, default='qm9',
                              help=f'Name of the dataset used (choose from '
                                   f'{list(dataset_name_to_class_mapping.keys())}, '
                                   f'default: %(default)s)'),
    train_parser.add_argument('--subset_path', type=str,
                              help='A path to a npy file containing indices '
                                   'of a subset of the data set at datapath '
                                   '(default: %(default)s)',
                              default=None)
    train_parser.add_argument('--seed', type=int, default=None,
                              help='Set random seed for torch and numpy.')
    train_parser.add_argument('--overwrite',
                              help='Remove previous model directory.',
                              action='store_true')
    train_parser.add_argument('--pretrained_path',
                              help='Start training from the pre-trained model at the '
                                   'provided path (reset optimizer parameters such as '
                                   'best loss and learning rate and create new split)',
                              default=None)
    train_parser.add_argument('--split_path',
                              help='Path/destination of npz with data splits',
                              default=None)
    train_parser.add_argument('--split',
                              help='Split into [train] [validation] and use '
                                   'remaining for testing',
                              type=int, nargs=2, default=[None, None])
    train_parser.add_argument('--max_epochs', type=int,
                              help='Maximum number of training epochs '
                                   '(default: %(default)s)',
                              default=500)
    train_parser.add_argument('--lr', type=float,
                              help='Initial learning rate '
                                   '(default: %(default)s)',
                              default=1e-4)
    train_parser.add_argument('--lr_patience', type=int,
                              help='Epochs without improvement before reducing'
                                   ' the learning rate (default: %(default)s)',
                              default=10)
    train_parser.add_argument('--lr_decay', type=float,
                              help='Learning rate decay '
                                   '(default: %(default)s)',
                              default=0.5)
    train_parser.add_argument('--lr_min', type=float,
                              help='Minimal learning rate '
                                   '(default: %(default)s)',
                              default=1e-6)
    train_parser.add_argument('--logger',
                              help='Choose logger for training process '
                                   '(default: %(default)s)',
                              choices=['csv', 'tensorboard'],
                              default='tensorboard')
    train_parser.add_argument('--log_every_n_epochs', type=int,
                              help='Log metrics every given number of epochs '
                                   '(default: %(default)s)',
                              default=1)
    train_parser.add_argument('--checkpoint_every_n_epochs', type=int,
                              help='Create checkpoint every given number of '
                                   'epochs'
                                   '(default: %(default)s)',
                              default=25)
    train_parser.add_argument('--label_width_factor', type=float,
                              help='A factor that is multiplied with the '
                                   'range between two distance bins in order '
                                   'to determine the width of the Gaussians '
                                   'used to obtain labels from distances '
                                   '(set to 0. to use one-hot '
                                   'encodings of distances as labels, '
                                   'default: %(default)s)',
                              default=0.1)

    ## evaluation
    eval_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    eval_parser.add_argument('datapath', help='Path of dataset directory')
    eval_parser.add_argument('modelpath', help='Path of stored model')
    eval_parser.add_argument('--split',
                             help='Evaluate trained model on given split',
                             choices=['train', 'validation', 'test'],
                             default=['test'], nargs='+')

    ## molecule generation
    gen_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    gen_parser.add_argument('modelpath', help='Path of stored model')
    gen_parser.add_argument('amount_gen', type=int,
                            help='The amount of generated molecules')
    gen_parser.add_argument('--show_gen',
                            help='Whether to open plots of generated '
                                 'molecules for visual evaluation',
                            action='store_true')
    gen_parser.add_argument('--chunk_size', type=int,
                            help='The size of mini batches during generation '
                                 '(default: %(default)s)',
                            default=1000)
    gen_parser.add_argument('--max_length', type=int,
                            help='The maximum number of atoms per molecule '
                                 '(default: %(default)s)',
                            default=35)
    gen_parser.add_argument('--file_name', type=str,
                            help='The name of the file in which generated '
                                 'molecules are stored (please note that '
                                 'increasing numbers are appended to the file name '
                                 'if it already exists and that the extension '
                                 '.mol_dict is automatically added to the chosen '
                                 'file name, default: %(default)s)',
                            default='generated')
    gen_parser.add_argument('--store_unfinished',
                            help='Store molecules which have not been '
                                 'finished after sampling max_length atoms',
                            action='store_true')
    gen_parser.add_argument('--print_file',
                            help='Use to limit the printing if results are '
                                 'written to a file instead of the console ('
                                 'e.g. if running on a cluster)',
                            action='store_true')
    gen_parser.add_argument('--temperature', type=float,
                            help='The temperature T to use for sampling '
                                 '(default: %(default)s)',
                            default=0.1)

    # model-specific parsers
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument('--aggregation_mode', type=str, default='sum',
                              choices=['sum', 'avg'],
                              help=' (default: %(default)s)')

    #######  G-SchNet  #######
    gschnet_parser = argparse.ArgumentParser(add_help=False,
                                             parents=[model_parser])
    gschnet_parser.add_argument('--features', type=int,
                                help='Size of atom-wise representation '
                                     '(default: %(default)s)',
                                default=128)
    gschnet_parser.add_argument('--interactions', type=int,
                                help='Number of regular SchNet interaction '
                                     'blocks (default: %(default)s)',
                                default=9)
    gschnet_parser.add_argument('--cutoff', type=float, default=10.,
                                help='Cutoff radius of local environment '
                                     '(default: %(default)s)')
    gschnet_parser.add_argument('--num_gaussians', type=int, default=25,
                                help='Number of Gaussians to expand distances '
                                     '(default: %(default)s)')
    gschnet_parser.add_argument('--max_distance', type=float, default=15.,
                                help='Maximum distance covered by the discrete '
                                     'distributions over distances learned by '
                                     'the model '
                                     '(default: %(default)s)')
    gschnet_parser.add_argument('--num_distance_bins', type=int, default=300,
                                help='Number of bins used in the discrete '
                                     'distributions over distances learned by '
                                     'the model(default: %(default)s)')

    ## setup subparser structure
    cmd_subparsers = main_parser.add_subparsers(dest='mode',
                                                help='Command-specific '
                                                     'arguments')
    cmd_subparsers.required = True
    subparser_train = cmd_subparsers.add_parser('train', help='Training help')
    subparser_eval = cmd_subparsers.add_parser('eval', help='Eval help')
    subparser_gen = cmd_subparsers.add_parser('generate', help='Generate help')

    train_subparsers = subparser_train.add_subparsers(dest='model',
                                                      help='Model-specific '
                                                           'arguments')
    train_subparsers.required = True
    train_subparsers.add_parser('gschnet', help='G-SchNet help',
                                parents=[train_parser, gschnet_parser])

    eval_subparsers = subparser_eval.add_subparsers(dest='model',
                                                    help='Model-specific '
                                                         'arguments')
    eval_subparsers.required = True
    eval_subparsers.add_parser('gschnet', help='G-SchNet help',
                               parents=[eval_parser, gschnet_parser])

    gen_subparsers = subparser_gen.add_subparsers(dest='model',
                                                  help='Model-specific '
                                                       'arguments')
    gen_subparsers.required = True
    gen_subparsers.add_parser('gschnet', help='G-SchNet help',
                              parents=[gen_parser, gschnet_parser])

    return main_parser


def get_model(args, parallelize=False):
    # get SchNet layers for feature extraction
    representation =\
        spk.representation.SchNet(n_atom_basis=args.features,
                                  n_filters=args.features,
                                  n_interactions=args.interactions,
                                  cutoff=args.cutoff,
                                  n_gaussians=args.num_gaussians,
                                  max_z=100)

    # get output layers for prediction of next atom type
    preprocess_type = \
        EmbeddingMultiplication(representation.embedding,
                                in_key_types='_all_types',
                                in_key_representation='representation',
                                out_key='preprocessed_representation')
    postprocess_type = NormalizeAndAggregate(normalize=True,
                                             normalization_axis=-1,
                                             normalization_mode='logsoftmax',
                                             aggregate=True,
                                             aggregation_axis=-2,
                                             aggregation_mode='sum',
                                             keepdim=False,
                                             mask='_type_mask',
                                             squeeze=True)
    out_module_type = \
        AtomwiseWithProcessing(n_in=args.features,
                               n_out=1,
                               n_layers=5,
                               preprocess_layers=preprocess_type,
                               postprocess_layers=postprocess_type,
                               out_key='type_predictions')

    # get output layers for predictions of distances
    preprocess_dist = \
        EmbeddingMultiplication(representation.embedding,
                                in_key_types='_next_types',
                                in_key_representation='representation',
                                out_key='preprocessed_representation')
    out_module_dist = \
        AtomwiseWithProcessing(n_in=args.features,
                               n_out=args.num_distance_bins,
                               n_layers=5,
                               preprocess_layers=preprocess_dist,
                               out_key='distance_predictions')

    # combine layers into an atomistic model
    model = spk.atomistic.AtomisticModel(representation,
                                         [out_module_type, out_module_dist])

    if parallelize:
        model = nn.DataParallel(model)

    logging.info("The model you built has: %d parameters" %
                 count_params(model))

    return model


def train(args, model, train_loader, val_loader, device):

    # setup hooks and logging
    hooks = [
        spk.hooks.MaxEpochHook(args.max_epochs)
    ]

    # filter for trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # setup optimizer
    optimizer = Adam(trainable_params, lr=args.lr)
    schedule = spk.hooks.ReduceLROnPlateauHook(optimizer,
                                               patience=args.lr_patience,
                                               factor=args.lr_decay,
                                               min_lr=args.lr_min,
                                               window_length=1,
                                               stop_after_min=True)
    hooks.append(schedule)

    # set up metrics to log KL divergence on distributions of types and distances
    metrics = [KLDivergence(target='_type_labels',
                            model_output='type_predictions',
                            name='KLD_types'),
               KLDivergence(target='_labels',
                            model_output='distance_predictions',
                            mask='_dist_mask',
                            name='KLD_dists')]

    if args.logger == 'csv':
        logger =\
            spk.hooks.CSVHook(os.path.join(args.modelpath, 'log'),
                              metrics,
                              every_n_epochs=args.log_every_n_epochs)
        hooks.append(logger)
    elif args.logger == 'tensorboard':
        logger =\
            spk.hooks.TensorboardHook(os.path.join(args.modelpath, 'log'),
                                      metrics,
                                      every_n_epochs=args.log_every_n_epochs)
        hooks.append(logger)

    norm_layer = nn.LogSoftmax(-1).to(device)
    loss_layer = nn.KLDivLoss(reduction='none').to(device)

    # setup loss function
    def loss(batch, result):
        # loss for type predictions (KLD)
        out_type = norm_layer(result['type_predictions'])
        loss_type = loss_layer(out_type, batch['_type_labels'])
        loss_type = torch.sum(loss_type, -1)
        loss_type = torch.mean(loss_type)

        # loss for distance predictions (KLD)
        mask_dist = batch['_dist_mask']
        N = torch.sum(mask_dist)
        out_dist = norm_layer(result['distance_predictions'])
        loss_dist = loss_layer(out_dist, batch['_labels'])
        loss_dist = torch.sum(loss_dist, -1)
        loss_dist = torch.sum(loss_dist * mask_dist) / torch.max(N, torch.ones_like(N))

        return loss_type + loss_dist

    # initialize trainer
    trainer = spk.train.Trainer(args.modelpath,
                                model,
                                loss,
                                optimizer,
                                train_loader,
                                val_loader,
                                hooks=hooks,
                                checkpoint_interval=args.checkpoint_every_n_epochs,
                                keep_n_checkpoints=10)

    # reset optimizer and hooks if starting from pre-trained model (e.g. for
    # fine-tuning)
    if args.pretrained_path is not None:
        logging.info('starting from pre-trained model...')
        # reset epoch and step
        trainer.epoch = 0
        trainer.step = 0
        trainer.best_loss = float('inf')
        # reset optimizer
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(trainable_params, lr=args.lr)
        trainer.optimizer = optimizer
        # reset scheduler
        schedule =\
            spk.hooks.ReduceLROnPlateauHook(optimizer,
                                            patience=args.lr_patience,
                                            factor=args.lr_decay,
                                            min_lr=args.lr_min,
                                            window_length=1,
                                            stop_after_min=True)
        trainer.hooks[1] = schedule
        # remove checkpoints of pre-trained model
        rmtree(os.path.join(args.modelpath, 'checkpoints'))
        os.makedirs(os.path.join(args.modelpath, 'checkpoints'))
        # store first checkpoint
        trainer.store_checkpoint()

    # start training
    trainer.train(device)


def evaluate(args, model, train_loader, val_loader, test_loader, device):
    header = ['Subset', 'distances KLD', 'types KLD']

    metrics = [KLDivergence(target='_labels',
                            model_output='distance_predictions',
                            mask='_dist_mask'),
               KLDivergence(target='_type_labels',
                            model_output='type_predictions')]

    results = []
    if 'train' in args.split:
        results.append(['training'] +
                       ['%.5f' % i for i in
                        evaluate_dataset(metrics, model,
                                         train_loader, device)])

    if 'validation' in args.split:
        results.append(['validation'] +
                       ['%.5f' % i for i in
                        evaluate_dataset(metrics, model,
                                         val_loader, device)])

    if 'test' in args.split:
        results.append(['test'] + ['%.5f' % i for i in evaluate_dataset(
            metrics, model, test_loader, device)])

    header = ','.join(header)
    results = np.array(results)

    np.savetxt(os.path.join(args.modelpath, 'evaluation.csv'), results,
               header=header, fmt='%s', delimiter=',')


def evaluate_dataset(metrics, model, loader, device):
    for metric in metrics:
        metric.reset()

    for batch in loader:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        result = model(batch)

        for metric in metrics:
            metric.add_batch(batch, result)

    results = [
        metric.aggregate() for metric in metrics
    ]
    return results


def generate(args, train_args, model, device):
    # generate molecules (in chunks) and print progress

    dataclass = dataset_name_to_class_mapping[train_args.dataset_name]
    types = sorted(dataclass.available_atom_types)  # retrieve available atom types
    all_types = types + [types[-1] + 1]  # add stop token to list (largest type + 1)
    start_token = types[-1] + 2  # define start token (largest type + 2)
    amount = args.amount_gen
    chunk_size = args.chunk_size
    if chunk_size >= amount:
        chunk_size = amount

    # set parameters for printing progress
    if int(amount / 10.) < chunk_size:
        step = chunk_size
    else:
        step = int(amount / 10.)
    increase = lambda x, y: y + step if x >= y else y
    thresh = step
    if args.print_file:
        progress = lambda x, y: print(f'Generated {x}.', flush=True) \
            if x >= y else print('', end='', flush=True)
    else:
        progress = lambda x, y: print(f'\x1b[2K\rSuccessfully generated'
                                      f' {x}', end='', flush=True)

    # generate
    generated = {}
    left = args.amount_gen
    done = 0
    start_time = time.time()
    with torch.no_grad():
        while left > 0:
            if left - chunk_size < 0:
                batch = left
            else:
                batch = chunk_size
            update_dict(generated,
                        generate_molecules(
                            batch,
                            model,
                            all_types=all_types,
                            start_token=start_token,
                            max_length=args.max_length,
                            save_unfinished=args.store_unfinished,
                            device=device,
                            max_dist=train_args.max_distance,
                            n_bins=train_args.num_distance_bins,
                            radial_limits=dataclass.radial_limits,
                            t=args.temperature)
                        )
            left -= batch
            done += batch
            n = np.sum(get_dict_count(generated, args.max_length))
            progress(n, thresh)
            thresh = increase(n, thresh)
        print('')
        end_time = time.time() - start_time
        m, s = divmod(end_time, 60)
        h, m = divmod(m, 60)
        h, m, s = int(h), int(m), int(s)
        print(f'Time consumed: {h:d}:{m:02d}:{s:02d}')

    # sort keys in resulting dictionary
    generated = dict(sorted(generated.items()))

    # show generated molecules and print some statistics if desired
    if args.show_gen:
        ats = []
        n_total_atoms = 0
        n_molecules = 0
        for key in generated:
            n = 0
            for i in range(len(generated[key][Properties.Z])):
                at = Atoms(generated[key][Properties.Z][i],
                           positions=generated[key][Properties.R][i])
                ats += [at]
                n += 1
                n_molecules += 1
            n_total_atoms += n * key
        asv.view(ats)
        print(f'Total number of atoms placed: {n_total_atoms} '
              f'(avg {n_total_atoms / n_molecules:.2f})', flush=True)

    return generated


def main(args):
    # set device (cpu or gpu)
    device = torch.device('cuda' if args.cuda else 'cpu')

    # store (or load) arguments
    argparse_dict = vars(args)
    jsonpath = os.path.join(args.modelpath, 'args.json')

    if args.mode == 'train':
        # overwrite existing model if desired
        if args.overwrite and os.path.exists(args.modelpath):
            rmtree(args.modelpath)
            logging.info('existing model will be overwritten...')

        # create model directory if it does not exist
        if not os.path.exists(args.modelpath):
            os.makedirs(args.modelpath)

        # get latest checkpoint of pre-trained model if a path was provided
        if args.pretrained_path is not None:
            model_chkpt_path = os.path.join(args.modelpath, 'checkpoints')
            pretrained_chkpt_path = os.path.join(args.pretrained_path, 'checkpoints')
            if os.path.exists(model_chkpt_path) \
                    and len(os.listdir(model_chkpt_path)) > 0:
                logging.info(f'found existing checkpoints in model directory '
                             f'({model_chkpt_path}), please use --overwrite or choose '
                             f'empty model directory to start from a pre-trained '
                             f'model...')
                logging.warning(f'will ignore pre-trained model and start from latest '
                                f'checkpoint at {model_chkpt_path}...')
                args.pretrained_path = None
            else:
                logging.info(f'fetching latest checkpoint from pre-trained model at '
                             f'{pretrained_chkpt_path}...')
                if not os.path.exists(pretrained_chkpt_path):
                    logging.warning(f'did not find checkpoints of pre-trained model, '
                                    f'will train from scratch...')
                    args.pretrained_path = None
                else:
                    chkpt_files = [f for f in os.listdir(pretrained_chkpt_path)
                                   if f.startswith("checkpoint")]
                    if len(chkpt_files) == 0:
                        logging.warning(f'did not find checkpoints of pre-trained '
                                        f'model, will train from scratch...')
                        args.pretrained_path = None
                    else:
                        epoch = max([int(f.split(".")[0].split("-")[-1])
                                     for f in chkpt_files])
                        chkpt = os.path.join(pretrained_chkpt_path,
                                             "checkpoint-" + str(epoch) + ".pth.tar")
                        if not os.path.exists(model_chkpt_path):
                            os.makedirs(model_chkpt_path)
                        copyfile(chkpt, os.path.join(model_chkpt_path,
                                                     f'checkpoint-{epoch}.pth.tar'))

        # store arguments for training in model directory
        to_json(jsonpath, argparse_dict)
        train_args = args

        # set seed
        spk.utils.set_random_seed(args.seed)
    else:
        # load arguments used for training from model directory
        train_args = read_from_json(jsonpath)

    # load data for training/evaluation
    if args.mode in ['train', 'eval']:
        # find correct data class
        assert train_args.dataset_name in dataset_name_to_class_mapping, \
            f'Could not find data class for dataset {train_args.dataset}. Please ' \
            f'specify a correct dataset name!'
        dataclass = dataset_name_to_class_mapping[train_args.dataset_name]

        # load the dataset
        logging.info(f'{train_args.dataset_name} will be loaded...')
        subset = None
        if train_args.subset_path is not None:
            logging.info(f'Using subset from {train_args.subset_path}')
            subset = np.load(train_args.subset_path)
            subset = [int(i) for i in subset]
        if issubclass(dataclass, DownloadableAtomsData):
            data = dataclass(args.datapath, subset=subset,
                             precompute_distances=args.precompute_distances,
                             download=True if args.mode == 'train' else False)
        else:
            data = dataclass(args.datapath, subset=subset,
                             precompute_distances=args.precompute_distances)

        # splits the dataset in test, val, train sets
        split_path = os.path.join(args.modelpath, 'split.npz')
        if args.mode == 'train':
            if args.split_path is not None:
                copyfile(args.split_path, split_path)

        logging.info('create splits...')
        data_train, data_val, data_test = data.create_splits(*train_args.split,
                                                             split_file=split_path)

        logging.info('load data...')
        types = sorted(dataclass.available_atom_types)
        max_type = types[-1]
        # set up collate function according to args
        collate = lambda x: \
            collate_atoms(x,
                          all_types=types + [max_type+1],
                          start_token=max_type+2,
                          draw_samples=args.draw_random_samples,
                          label_width_scaling=train_args.label_width_factor,
                          max_dist=train_args.max_distance,
                          n_bins=train_args.num_distance_bins)

        train_loader = spk.data.AtomsLoader(data_train, batch_size=args.batch_size,
                                            sampler=RandomSampler(data_train),
                                            num_workers=4, pin_memory=True,
                                            collate_fn=collate)
        val_loader = spk.data.AtomsLoader(data_val, batch_size=args.batch_size,
                                          num_workers=2, pin_memory=True,
                                          collate_fn=collate)

    # construct the model
    if args.mode == 'train' or args.checkpoint >= 0:
        model = get_model(train_args, parallelize=args.parallel)
    logging.info(f'running on {device}')

    # load model or checkpoint for evaluation or generation
    if args.mode in ['eval', 'generate']:
        if args.checkpoint < 0:  # load best model
            logging.info(f'restoring best model')
            model = torch.load(os.path.join(args.modelpath, 'best_model')).to(device)
        else:
            logging.info(f'restoring checkpoint {args.checkpoint}')
            chkpt = os.path.join(args.modelpath, 'checkpoints',
                                 'checkpoint-' + str(args.checkpoint) + '.pth.tar')
            state_dict = torch.load(chkpt)
            model.load_state_dict(state_dict['model'], strict=True)

    # execute training, evaluation, or generation
    if args.mode == 'train':
        logging.info("training...")
        train(args, model, train_loader, val_loader, device)
        logging.info("...training done!")

    elif args.mode == 'eval':
        logging.info("evaluating...")
        test_loader = spk.data.AtomsLoader(data_test,
                                           batch_size=args.batch_size,
                                           num_workers=2,
                                           pin_memory=True,
                                           collate_fn=collate)
        with torch.no_grad():
            evaluate(args, model, train_loader, val_loader, test_loader, device)
        logging.info("... done!")

    elif args.mode == 'generate':
        logging.info(f'generating {args.amount_gen} molecules...')
        generated = generate(args, train_args, model, device)
        gen_path = os.path.join(args.modelpath, 'generated/')
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        # get untaken filename and store results
        file_name = os.path.join(gen_path, args.file_name)
        if os.path.isfile(file_name + '.mol_dict'):
            expand = 0
            while True:
                expand += 1
                new_file_name = file_name + '_' + str(expand)
                if os.path.isfile(new_file_name + '.mol_dict'):
                    continue
                else:
                    file_name = new_file_name
                    break
        with open(file_name + '.mol_dict', 'wb') as f:
            pickle.dump(generated, f)
        logging.info('...done!')
    else:
        logging.info(f'Unknown mode: {args.mode}')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
