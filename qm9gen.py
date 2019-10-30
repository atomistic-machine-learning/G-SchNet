import logging
import os
import re
import shutil
import tarfile
import tempfile
from urllib import request as request
from urllib.error import HTTPError, URLError
from base64 import b64encode, b64decode

import numpy as np
import torch
from ase.db import connect
from ase.io.extxyz import read_xyz
from ase.units import Debye, Bohr, Hartree, eV

from schnetpack import Properties
from schnetpack.datasets import DownloadableAtomsData
from utility_classes import ConnectivityCompressor
from preprocess_dataset import preprocess_dataset


class QM9gen(DownloadableAtomsData):
    """ QM9 benchmark dataset for organic molecules with up to nine non-hydrogen atoms
        from {C, O, N, F}.

        This class adds convenience functions to download QM9 from figshare,
        pre-process the data such that it can be used for moleculec generation with the
        G-SchNet model, and load the data into pytorch.

        Args:
            path (str): path to directory containing qm9 database
            subset (list, optional): indices of subset, set to None for entire dataset
                (default: None).
            download (bool, optional): enable downloading if qm9 database does not
                exists (default: True)
            precompute_distances (bool, optional): if True and the pre-processed
                database does not yet exist, the pairwise distances of atoms in the
                dataset's molecules will be computed during pre-processing and stored in
                the database (increases storage demand of the dataset but decreases
                computational cost during training as otherwise the distances will be
                computed once in every epoch, default: True)
            remove_invalid (bool, optional): if True QM9 molecules that do not pass the
                valency check will be removed from the training data (note 1: the
                validity is per default inferred from a pre-computed list in our
                repository but will be assessed locally if the download fails,
                note2: only works if the pre-processed database does not yet exist,
                default: True)

        References:
            .. [#qm9_1] https://ndownloader.figshare.com/files/3195404
    """

    # properties
    A = 'rotational_constant_A'
    B = 'rotational_constant_B'
    C = 'rotational_constant_C'
    mu = 'dipole_moment'
    alpha = 'isotropic_polarizability'
    homo = 'homo'
    lumo = 'lumo'
    gap = 'gap'
    r2 = 'electronic_spatial_extent'
    zpve = 'zpve'
    U0 = 'energy_U0'
    U = 'energy_U'
    H = 'enthalpy_H'
    G = 'free_energy'
    Cv = 'heat_capacity'

    properties = [
        A, B, C, mu, alpha,
        homo, lumo, gap, r2, zpve,
        U0, U, H, G, Cv
    ]

    reference = {
        zpve: 0, U0: 1, U: 2, H: 3, G: 4, Cv: 5
    }

    units = [1., 1., 1., Debye, Bohr ** 3,
             Hartree, Hartree, Hartree,
             Bohr ** 2, Hartree,
             Hartree, Hartree, Hartree,
             Hartree, 1.,
            ]

    units_dict = dict(zip(properties, units))

    connectivity_compressor = ConnectivityCompressor()

    def __init__(self, path, subset=None, download=True, precompute_distances=True,
                 remove_invalid=True):
        self.path = path
        self.dbpath = os.path.join(self.path, f'qm9gen.db')
        self.atomref_path = os.path.join(self.path, 'atomref.npz')
        self.precompute_distances = precompute_distances
        self.remove_invalid = remove_invalid

        super().__init__(self.dbpath, subset=subset,
                         available_properties=self.properties,
                         units=self.units, download=download)

    def get_reference(self, property):
        if property not in QM9gen.reference:
            atomref = None
        else:
            col = QM9gen.reference[property]
            atomref = np.load(self.atomref_path)['atom_ref'][:, col:col + 1]
        return atomref

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
        return type(self)(self.path, subidx, download=False)

    def get_properties(self, idx):
        _idx = self._subset_index(idx)
        with connect(self.dbpath) as conn:
            row = conn.get(_idx + 1)
        at = row.toatoms()

        # extract/calculate structure
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
            properties['dists'] = row.data['dists']

        # get atom environment
        nbh_idx, offsets = self.environment_provider.get_environment(at)
        # store neighbors, cell, and index
        properties[Properties.neighbors] = torch.LongTensor(nbh_idx.astype(np.int))
        properties[Properties.cell_offset] = torch.FloatTensor(
            offsets.astype(np.float32))
        properties["_idx"] = torch.LongTensor(np.array([idx], dtype=np.int))

        return at, properties

    def _download(self):
        works = True

        if not os.path.exists(self.atomref_path):
            works = works and self._load_atomrefs()

        if not os.path.exists(self.dbpath):
            qm9_path = os.path.join(self.path, f'qm9.db')
            if not os.path.exists(qm9_path):
                works = works and self._load_data()
            works = works and self._preprocess_qm9()
        return works

    def _load_atomrefs(self):
        logging.info('Downloading GDB-9 atom references...')
        at_url = 'https://ndownloader.figshare.com/files/3195395'
        tmpdir = tempfile.mkdtemp('gdb9')
        tmp_path = os.path.join(tmpdir, 'atomrefs.txt')

        try:
            request.urlretrieve(at_url, tmp_path)
            logging.info("Done.")
        except HTTPError as e:
            logging.error("HTTP Error:", e.code, at_url)
            return False
        except URLError as e:
            logging.error("URL Error:", e.reason, at_url)
            return False

        atref = np.zeros((100, 6))
        labels = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']
        with open(tmp_path) as f:
            lines = f.readlines()
            for z, l in zip([1, 6, 7, 8, 9], lines[5:10]):
                atref[z, 0] = float(l.split()[1])
                atref[z, 1] = float(l.split()[2]) * Hartree / eV
                atref[z, 2] = float(l.split()[3]) * Hartree / eV
                atref[z, 3] = float(l.split()[4]) * Hartree / eV
                atref[z, 4] = float(l.split()[5]) * Hartree / eV
                atref[z, 5] = float(l.split()[6])
        np.savez(self.atomref_path, atom_ref=atref, labels=labels)
        return True

    def _load_data(self):
        logging.info('Downloading GDB-9 data...')
        tmpdir = tempfile.mkdtemp('gdb9')
        tar_path = os.path.join(tmpdir, 'gdb9.tar.gz')
        raw_path = os.path.join(tmpdir, 'gdb9_xyz')
        url = 'https://ndownloader.figshare.com/files/3195389'

        try:
            request.urlretrieve(url, tar_path)
            logging.info('Done.')
        except HTTPError as e:
            logging.error('HTTP Error:', e.code, url)
            return False
        except URLError as e:
            logging.error('URL Error:', e.reason, url)
            return False

        logging.info('Extracting data from tar file...')
        tar = tarfile.open(tar_path)
        tar.extractall(raw_path)
        tar.close()
        logging.info('Done.')

        logging.info('Parsing xyz files...')
        with connect(os.path.join(self.path, 'qm9.db')) as con:
            ordered_files = sorted(os.listdir(raw_path),
                                   key=lambda x: (int(re.sub('\D', '', x)), x))
            for i, xyzfile in enumerate(ordered_files):
                xyzfile = os.path.join(raw_path, xyzfile)

                if (i + 1) % 10000 == 0:
                    logging.info('Parsed: {:6d} / 133885'.format(i + 1))
                properties = {}
                tmp = os.path.join(tmpdir, 'tmp.xyz')

                with open(xyzfile, 'r') as f:
                    lines = f.readlines()
                    l = lines[1].split()[2:]
                    for pn, p in zip(self.properties, l):
                        properties[pn] = float(p) * self.units[pn]
                    with open(tmp, "wt") as fout:
                        for line in lines:
                            fout.write(line.replace('*^', 'e'))

                with open(tmp, 'r') as f:
                    ats = list(read_xyz(f, 0))[0]
                con.write(ats, data=properties)
        logging.info('Done.')

        shutil.rmtree(tmpdir)

        return True

    def _preprocess_qm9(self):
        # try to download pre-computed list of invalid molecules
        logging.info('Downloading pre-computed list of invalid QM9 molecules...')
        raw_path = os.path.join(self.path, 'qm9_invalid.txt')
        url = 'https://github.com/atomistic-machine-learning/G-SchNet/blob/master/' \
              'qm9_invalid.txt?raw=true'

        try:
            request.urlretrieve(url, raw_path)
            logging.info('Done.')
            invalid_list = np.loadtxt(raw_path)
        except HTTPError as e:
            logging.error('HTTP Error:', e.code, url)
            logging.info('CAUTION: Could not download pre-computed list, will assess '
                         'validity during pre-processing.')
            invalid_list = None
        except URLError as e:
            logging.error('URL Error:', e.reason, url)
            logging.info('CAUTION: Could not download pre-computed list, will assess '
                         'validity during pre-processing.')
            invalid_list = None
        # check validity of molecules and store connectivity matrices and interatomic
        # distances in database as a preprocessing step
        qm9_db = os.path.join(self.path, f'qm9.db')
        preprocess_dataset(datapath=qm9_db, valence_list=[1, 1, 6, 4, 7, 3, 8, 2, 9, 1],
                           n_threads=8, n_mols_per_thread=125, logging_print=True,
                           new_db_path=self.dbpath,
                           precompute_distances=self.precompute_distances,
                           remove_invalid=self.remove_invalid,
                           invalid_list=invalid_list)
        return True

    def get_available_properties(self, available_properties):
        # we don't use properties other than stored connectivity matrices (and
        # distances, if they were precomputed) so we skip this part
        return available_properties
