import logging
from pathlib import Path
import numpy as np
import torch
from ase.db import connect

from schnetpack import Properties
from schnetpack.datasets import AtomsData
from utility_classes import ConnectivityCompressor
from template_preprocess_dataset import preprocess_dataset


class TemplateData(AtomsData):
    """ Simple template dataset class. We assume molecules made of C, N, O, F,
        and H atoms as illustration here.

        The class basically serves as interface to a database. It initiates
        pre-processing of the data in order to prepare it for usage with G-SchNet.
        To this end, it calls the template_preprocess_dataset script which provides
        very basic pre-processing (e.g. calculation of connectivity matrices) and can
        also be adapted to the data at hand.
        Single (pre-processed) data points are read from the database in the
        get_properties method (which is called in __getitem__). The class builds upon
        the AtomsData class from SchNetPack.

        Args:
            path (str): path to directory containing database
            subset (list, optional): indices of subset, set to None for entire dataset
                (default: None).
            precompute_distances (bool, optional): if True and the pre-processed
                database does not yet exist, the pairwise distances of atoms in the
                dataset's molecules will be computed during pre-processing and stored in
                the database (increases storage demand of the dataset but decreases
                computational cost during training as otherwise the distances will be
                computed once in every epoch, default: True)
            remove_invalid (bool, optional): if True, molecules that do not pass the
                implemented validity checks will be removed from the training data (
                in the simple template_preprocess_dataset script this is only a check
                for disconnectedness, i.e. if all atoms are connected by some path as
                otherwise no proper generation trace can be sampled,
                note: only works if the pre-processed database does not yet exist,
                default: True)
    """

    ##### Adjust the following settings to fit your data: #####
    # name of the database
    db_name = 'template_data.db'
    # name of the database after pre-processing (if the same as db_name, the original
    # database will be renamed to <db_name>.bak.db)
    preprocessed_db_name = 'template_data_gschnet.db'
    # all atom types found in molecules of the dataset
    available_atom_types = [1, 6, 7, 8, 9]  # for example H, C, N, O, and F
    # valence constraints of the atom types (does not need to be provided unless a
    # valence check is implemented, but this is not the case in the template script)
    atom_types_valence = [1, 4, 3, 2, 1]
    # minimum and maximum distance between neighboring atoms in angstrom (this is
    # used to determine which atoms are considered as connected in the connectivity
    # matrix, i.e. for sampling generation traces during training, and also to restrict
    # the grid around the focused atom during generation, as the next atom will always
    # be a neighbor of the focused atom)
    radial_limits = [0.9, 1.7]

    # used to decompress connectivity matrices
    connectivity_compressor = ConnectivityCompressor()

    def __init__(self, path, subset=None, precompute_distances=True,
                 remove_invalid=True):
        self.path_to_dir = Path(path)
        self.db_path = self.path_to_dir / self.preprocessed_db_name
        self.source_db_path = self.path_to_dir / self.db_name
        self.precompute_distances = precompute_distances
        self.remove_invalid = remove_invalid

        # do pre-processing (if database is not already pre-processed)
        found_connectivity = False
        if self.db_path.is_file():
            with connect(self.db_path) as conn:
                n_mols = conn.count()
                if n_mols > 0:
                    first_row = conn.get(1)
                    found_connectivity = 'con_mat' in first_row.data
        if not found_connectivity:
            self._preprocess_data()

        super().__init__(str(self.db_path), subset=subset)

    def create_subset(self, idx):
        """
        Returns a new dataset that only consists of provided indices.

        Args:
            idx (numpy.ndarray): subset indices

        Returns:
            schnetpack.data.AtomsData: dataset with subset of original data
        """
        idx = np.array(idx)
        subidx = idx if self.subset is None or len(idx) == 0 \
            else np.array(self.subset)[idx]
        return type(self)(self.path_to_dir, subidx)

    def get_properties(self, idx):
        _idx = self._subset_index(idx)
        with connect(self.db_path) as conn:
            row = conn.get(_idx + 1)
        at = row.toatoms()

        # extract/calculate structure (atom positions, types and cell)
        properties = {}
        properties[Properties.Z] = torch.LongTensor(at.numbers.astype(np.int))
        positions = at.positions.astype(np.float32)
        positions -= at.get_center_of_mass()  # center positions
        properties[Properties.R] = torch.FloatTensor(positions)
        properties[Properties.cell] = torch.FloatTensor(at.cell.astype(np.float32))

        # recover connectivity matrix from compressed format
        con_mat = self.connectivity_compressor.decompress(row.data['con_mat'])
        # save in dictionary
        properties['_con_mat'] = torch.FloatTensor(con_mat.astype(np.float32))

        # extract pre-computed distances (if they exist)
        if 'dists' in row.data:
            properties['dists'] = row.data['dists'][:, None]

        # get atom environment
        nbh_idx, offsets = self.environment_provider.get_environment(at)
        # store neighbors, cell, and index
        properties[Properties.neighbors] = torch.LongTensor(nbh_idx.astype(np.int))
        properties[Properties.cell_offset] = torch.FloatTensor(
            offsets.astype(np.float32))
        properties["_idx"] = torch.LongTensor(np.array([idx], dtype=np.int))

        return at, properties

    def _preprocess_data(self):
        # check if pre-processing source db has different name than target db (if
        # not, rename it)
        source_db = self.path_to_dir / self.db_name
        if self.db_name == self.preprocessed_db_name:
            new_name = self.path_to_dir / (self.db_name + '.bak.db')
            source_db.rename(new_name)
            source_db = new_name
        # look for pre-computed list of invalid molecules
        invalid_list_path = self.source_db_path.parent / \
                            (self.source_db_path.stem + f'_invalid.txt')
        if invalid_list_path.is_file():
            invalid_list = np.loadtxt(invalid_list_path)
        else:
            invalid_list = None
        # initialize pre-processing (calculation and validation of connectivity
        # matrices as well as computation of pairwise distances between atoms)
        valence_list = \
            np.array([self.available_atom_types, self.atom_types_valence]).flatten('F')
        preprocess_dataset(datapath=source_db,
                           cutoff=self.radial_limits[-1],
                           valence_list=list(valence_list),
                           logging_print=True,
                           new_db_path=self.db_path,
                           precompute_distances=self.precompute_distances,
                           remove_invalid=self.remove_invalid,
                           invalid_list=invalid_list)
        return True
