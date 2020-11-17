import operator
import re
import numpy as np
import openbabel as ob
import pybel
from multiprocessing import Process
from rdkit import Chem
from scipy.spatial.distance import squareform


class Molecule:
    '''
    Molecule class that allows to get statistics such as the connectivity matrix,
    molecular fingerprint, canonical smiles representation, or ring count given
    positions of atoms and their atomic numbers. Currently supports molecules made of
    carbon, nitrogen, oxygen, fluorine, and hydrogen (such as in the QM9 benchmark
    dataset). Mainly relies on routines from Open Babel and RdKit.

    Args:
        pos (numpy.ndarray): positions of atoms in euclidean space (n_atoms x 3)
        atomic_numbers (numpy.ndarray): list with nuclear charge/type of each atom
            (e.g. 1 for hydrogens, 6 for carbons etc.).
        connectivity_matrix (numpy.ndarray, optional): optionally, a pre-calculated
            connectivity matrix (n_atoms x n_atoms) containing the bond order between
            atom pairs can be provided (default: None).
        store_positions (bool, optional): set True to store the positions of atoms in
            self.positions (only for convenience, not needed for computations, default:
            False).
    '''

    type_infos = {1: {'name': 'H',
                      'n_bonds': 1},
                  6: {'name': 'C',
                      'n_bonds': 4},
                  7: {'name': 'N',
                      'n_bonds': 3},
                  8: {'name': 'O',
                      'n_bonds': 2},
                  9: {'name': 'F',
                      'n_bonds': 1},
                  }
    type_charges = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    def __init__(self, pos, atomic_numbers, connectivity_matrix=None,
                 store_positions=False):
        # set comparison metrics to None (will be computed just in time)
        self._fp = None
        self._fp_bits = None
        self._can = None
        self._mirror_can = None
        self._inchi_key = None
        self._bond_stats = None
        self._fixed_connectivity = False
        self._row_indices = {}
        self._obmol = None
        self._rings = None
        self._n_atoms_per_type = None
        self._connectivity = connectivity_matrix

        # set statistics
        self.n_atoms = len(pos)
        self.numbers = atomic_numbers
        self._unique_numbers = {*self.numbers}  # set for fast query
        self.positions = pos
        if not store_positions:
            self._obmol = self.get_obmol()  # create obmol before removing pos
            self.positions = None

    def sanity_check(self):
        '''
        Check whether the sum of valence of all atoms can be divided by 2.

        Returns:
             bool: True if the test is passed, False otherwise
        '''
        count = 0
        for atom in self.numbers:
            count += self.type_infos[atom]['n_bonds']
        if count % 2 == 0:
            return True
        else:
            return False

    def get_obmol(self):
        '''
        Retrieve the underlying Open Babel OBMol object.

        Returns:
             OBMol object: Open Babel OBMol representation
        '''
        if self._obmol is None:
            if self.positions is None:
                print('Error, cannot create obmol without positions!')
                return
            if self.numbers is None:
                print('Error, cannot create obmol without atomic numbers!')
                return
            # use openbabel to infer bonds and bond order:
            obmol = ob.OBMol()
            obmol.BeginModify()

            # set positions and atomic numbers of all atoms in the molecule
            for p, n in zip(self.positions, self.numbers):
                obatom = obmol.NewAtom()
                obatom.SetAtomicNum(int(n))
                obatom.SetVector(*p.tolist())

            # infer bonds and bond order
            obmol.ConnectTheDots()
            obmol.PerceiveBondOrders()

            obmol.EndModify()
            self._obmol = obmol
        return self._obmol

    def get_fp(self):
        '''
        Retrieve the molecular fingerprint (the path-based FP2 from Open Babel is used,
        which means that paths of length up to 7 are considered).

        Returns:
             pybel.Fingerprint object: moleculer fingerprint (use "fp1 | fp2" to
                calculate the Tanimoto coefficient of two fingerprints)
        '''
        if self._fp is None:
            # calculate fingerprint
            self._fp = pybel.Molecule(self.get_obmol()).calcfp()
        return self._fp

    def get_fp_bits(self):
        '''
        Retrieve the bits set in the molecular fingerprint.

        Returns:
             Set of int: object containing the bits set in the molecular fingerprint
        '''
        if self._fp_bits is None:
            self._fp_bits = {*self.get_fp().bits}
        return self._fp_bits

    def get_can(self):
        '''
        Retrieve the canonical SMILES representation of the molecule.

        Returns:
             String: canonical SMILES string
        '''
        if self._can is None:
            # calculate canonical SMILES
            self._can = pybel.Molecule(self.get_obmol()).write('can')
        return self._can

    def get_mirror_can(self):
        '''
        Retrieve the canonical SMILES representation of the mirrored molecule (the
        z-coordinates are flipped).

        Returns:
             String: canonical SMILES string of the mirrored molecule
        '''
        if self._mirror_can is None:
            # calculate canonical SMILES of mirrored molecule
            self._flip_z()  # flip z to mirror molecule using x-y plane
            self._mirror_can = pybel.Molecule(self.get_obmol()).write('can')
            self._flip_z()  # undo mirroring
        return self._mirror_can

    def get_inchi_key(self):
        '''
        Retrieve the InChI-key of the molecule.

        Returns:
             String: InChI-key
        '''
        if self._inchi_key is None:
            # calculate inchi key
            self._inchi_key = pybel.Molecule(self.get_obmol()).\
                write('inchikey')
        return self._inchi_key

    def _flip_z(self):
        '''
        Flips the z-coordinates of atom positions (to get a mirrored version of the
        molecule).
        '''
        if self._obmol is None:
            self.get_obmol()
        for atom in ob.OBMolAtomIter(self._obmol):
            x, y, z = atom.x(), atom.y(), atom.z()
            atom.SetVector(x, y, -z)
        self._obmol.ConnectTheDots()
        self._obmol.PerceiveBondOrders()

    def get_connectivity(self):
        '''
        Retrieve the connectivity matrix of the molecule.

        Returns:
            numpy.ndarray: (n_atoms x n_atoms) array containing the pairwise bond orders
                between atoms (0 for no bond).
        '''
        if self._connectivity is None:
            # get connectivity matrix
            connectivity = np.zeros((self.n_atoms, len(self.numbers)))
            for atom in ob.OBMolAtomIter(self.get_obmol()):
                index = atom.GetIdx() - 1
                # loop over all neighbors of atom
                for neighbor in ob.OBAtomAtomIter(atom):
                    idx = neighbor.GetIdx() - 1
                    bond_order = neighbor.GetBond(atom).GetBO()
                    #print(f'{index}-{idx}: {bond_order}')
                    # do not count bonds between two hydrogen atoms
                    if (self.numbers[index] == 1 and self.numbers[idx] == 1
                            and bond_order > 0):
                        bond_order = 0
                    connectivity[index, idx] = bond_order
            self._connectivity = connectivity
        return self._connectivity

    def get_ring_counts(self):
        '''
        Retrieve a list containing the sizes of rings in the symmetric smallest set
        of smallest rings (S-SSSR from RdKit) in the molecule (e.g. [5, 6, 5] for two
        rings of size 5 and one ring of size 6).

        Returns:
             List of int: list with ring sizes
        '''
        if self._rings is None:
            # calculate symmetric SSSR with RdKit using the canonical smiles
            # representation as input
            can = self.get_can()
            mol = Chem.MolFromSmiles(can)
            if mol is not None:
                ssr = Chem.GetSymmSSSR(mol)
                self._rings = [len(ssr[i]) for i in range(len(ssr))]
            else:
                self._rings = []  # cannot count rings
        return self._rings

    def get_n_atoms_per_type(self):
        '''
        Retrieve the number of atoms in the molecule per type.

        Returns:
            numpy.ndarray: number of atoms in the molecule per type, where the order
                corresponds to the order specified in Molecule.type_infos
        '''
        if self._n_atoms_per_type is None:
            _types = np.array(list(self.type_infos.keys()), dtype=int)
            self._n_atoms_per_type =\
                np.bincount(self.numbers, minlength=np.max(_types)+1)[_types]
        return self._n_atoms_per_type

    def remove_unpicklable_attributes(self, restorable=True):
        '''
        Some attributes of the class cannot be processed by pickle. This method
        allows to remove these attributes prior to pickling.

        Args:
            restorable (bool, optional): Set True to allow restoring the deleted
                attributes later on (default: True)
        '''
        # set attributes which are not picklable (SwigPyObjects) to None
        if restorable and self.positions is None and self._obmol is not None:
            # store positions to allow restoring obmol object later on
            pos = [atom.coords for atom in pybel.Molecule(self._obmol).atoms]
            self.positions = np.array(pos)
        self._obmol = None
        self._fp = None

    def tanimoto_similarity(self, other_mol, use_bits=True):
        '''
        Get the Tanimoto (fingerprint) similarity to another molecule.

        Args:
         other_mol (Molecule or pybel.Fingerprint/list of bits set):
            representation of the second molecule (if it is not a Molecule object,
            it needs to be a pybel.Fingerprint if use_bits is False and a list of bits
            set in the fingerprint if use_bits is True).
         use_bits (bool, optional): set True to calculate Tanimoto similarity
            from bits set in the fingerprint (default: True)

        Returns:
             float: Tanimoto similarity to the other molecule
        '''
        if use_bits:
            a = self.get_fp_bits()
            b = other_mol.get_fp_bits() if isinstance(other_mol, Molecule) \
                else other_mol
            n_equal = len(a.intersection(b))
            if len(a) + len(b) == 0:  # edge case with no set bits
                return 1.
            return n_equal / (len(a)+len(b)-n_equal)
        else:
            fp_other = other_mol.get_fp() if isinstance(other_mol, Molecule)\
                else other_mol
            return self.get_fp() | fp_other

    def _update_bond_orders(self, idc_lists):
        '''
        Updates the bond orders in the underlying OBMol object.

        Args:
            idc_lists (list of list of int): nested list containing bonds, i.e. pairs
                of row indices (list1) and column indices (list2) which shall be updated
        '''
        con_mat = self.get_connectivity()
        self._obmol.BeginModify()
        for i in range(len(idc_lists[0])):
            idx1 = idc_lists[0][i]
            idx2 = idc_lists[1][i]
            obbond = self._obmol.GetBond(int(idx1+1), int(idx2+1))
            obbond.SetBO(int(con_mat[idx1, idx2]))
        self._obmol.EndModify()

        # reset fingerprints etc
        self._fp = None
        self._can = None
        self._mirror_can = None
        self._inchi_key = None

    def get_fixed_connectivity(self, recursive_call=False):
        '''
        Attempts to fix the connectivity matrix using some heuristics (as some valid
        QM9 molecules do not pass the valency check using the connectivity matrix
        obtained with Open Babel, which seems to have problems with assigning correct
        bond orders to aromatic rings containing Nitrogen).

        Args:
            recursive_call (bool, do not set True): flag that indicates a recursive
                call (used internally, do not set to True)

        Returns:
            numpy.ndarray: (n_atoms x n_atoms) array containing the pairwise bond orders
                between atoms (0 for no bond) after the attempted fix.
        '''

        # if fix has already been attempted, return the connectivity matrix
        if self._fixed_connectivity:
            return self._connectivity

        # define helpers:
        # increases bond order between two atoms in connectivity matrix
        def increase_bond(con_mat, idx1, idx2):
            con_mat[idx1, idx2] += 1
            con_mat[idx2, idx1] += 1
            return con_mat

        # decreases bond order between two atoms in connectivity matrix
        def decrease_bond(con_mat, idx1, idx2):
            con_mat[idx1, idx2] -= 1
            con_mat[idx2, idx1] -= 1
            return con_mat

        # returns only the rows of the connectivity matrix corresponding to atoms of
        # certain types (and the indices of these atoms)
        def get_typewise_connectivity(con_mat, types):
            idcs = []
            for type in types:
                idcs += list(self._get_row_idcs(type))
            return con_mat[idcs], np.array(idcs).astype(int)

        # store old connectivity matrix for later comparison
        old_mat = self.get_connectivity().copy()

        # get connectivity matrix and find indices of N and C atoms
        con_mat = self.get_connectivity()
        if 6 not in self._unique_numbers and 7 not in self._unique_numbers:
            # do not attempt fixing if there is no carbon and no nitrogen
            return con_mat
        N_mat, N_idcs = get_typewise_connectivity(con_mat, [7])
        C_mat, C_idcs = get_typewise_connectivity(con_mat, [6])
        NC_idcs = np.hstack((N_idcs, C_idcs))  # indices of all N and C atoms
        NC_valences = self._get_valences()[NC_idcs]  # array with valency constraints

        # return connectivity if valency constraints of N and C atoms are already met
        if np.all(np.sum(con_mat[NC_idcs], axis=1) == NC_valences):
            return con_mat

        # if a C or N atom is "overcharged" (total bond order too high) we decrease
        # double to single bonds between N-N or N-C until it is not overcharged anymore
        # (e.g. C=N=C -> C=N-C)
        if 7 in self._unique_numbers:  # only necessary if molecule contains N
            for cur in NC_idcs:
                type = self.numbers[cur]
                if np.sum(con_mat[cur]) <= self.type_infos[type]['n_bonds']:
                    continue
                if type == 6:  # for carbon look only at nitrogen neighbors
                    neighbors = self._get_neighbors(cur, types=[7], strength=2)
                else:
                    neighbors = self._get_neighbors(cur, types=[6, 7],
                                                    strength=2)
                for neighbor in neighbors:
                    con_mat = decrease_bond(con_mat, cur, neighbor)
                    self._connectivity = con_mat
                    if np.sum(con_mat[cur]) == \
                            self.type_infos[type]['n_bonds']:
                        break

        # get updated partial connectivity matrices for N and C
        N_mat, _ = get_typewise_connectivity(con_mat, [7])
        C_mat, _ = get_typewise_connectivity(con_mat, [6])

        # increase total number of bonds by transferring the strength of a
        # double C-N bond to two neighboring bonds, if the involved atoms
        # are not yet saturated (e.g. H2C-H2C=N-H2C -> H2C=H2C-N=H2C)
        if (np.sum(N_mat) < len(N_idcs) * 3 or np.sum(C_mat) < len(C_idcs) * 4) \
                and 7 in self._unique_numbers:
            for cur in NC_idcs:
                type = self.numbers[cur]
                if sum(con_mat[cur]) >= self.type_infos[type]['n_bonds']:
                    continue
                CN_nbors = self._get_CN_neighbors(cur)
                for nbor_1, nbor_2 in CN_nbors:
                    if con_mat[nbor_1, nbor_2] <= 1:
                        continue
                    else:
                        nbor_2_nbors = np.where(con_mat[nbor_2] == 1)[0]
                        for nbor_2_nbor in nbor_2_nbors:
                            nbor_2_nbor_type = self.numbers[nbor_2_nbor]
                            if (np.sum(con_mat[nbor_2_nbor]) <
                                    self.type_infos[nbor_2_nbor_type]['n_bonds']):
                                con_mat = increase_bond(con_mat, cur, nbor_1)
                                con_mat = increase_bond(con_mat, nbor_2, nbor_2_nbor)
                                con_mat = decrease_bond(con_mat, nbor_1, nbor_2)
                self._connectivity = con_mat

        # increase bond strength between two undercharged neighbors C-N,
        # C-C or N-N (e.g HN-CH2 -> HN=CH2, starting from those atoms with least
        # available neighbors if there are multiple undercharged neighbors)
        undercharged_pairs = True
        while (undercharged_pairs):
            NC_charges = np.sum(con_mat[NC_idcs], axis=1)
            undercharged = NC_idcs[np.where(NC_charges < NC_valences)[0]]
            partial_con_mat = con_mat[undercharged][:, undercharged]
            # if non of the undercharged atoms are neighbors, stop
            if np.sum(partial_con_mat) == 0:
                break
            # sort by number of undercharged neighbors
            n_nbors = np.sum(partial_con_mat > 0, axis=0)
            # mask indices with zero undercharged neighbors to ignore them when sorting
            n_nbors[np.where(n_nbors == 0)[0]] = 1000
            cur = np.argmin(n_nbors)
            cur_nbor = np.where(partial_con_mat[cur] > 0)[0][0]
            con_mat = increase_bond(con_mat, undercharged[cur], undercharged[cur_nbor])
        self._connectivity = con_mat

        # if the molecule still is not valid, try to flip double bonds if an atom
        # forms a double bond and has at least one other neighbor that has too few bonds
        # (e.g. C-N=C -> C=N-C) and repeat above heuristics with a recursive call of
        # this function
        if not recursive_call and \
                not np.all(np.sum(con_mat[NC_idcs], axis=1) == NC_valences):
            changed = False
            candidates = np.where(np.any(con_mat[NC_idcs][:, NC_idcs] == 2, axis=0))[0]
            for cand in NC_idcs[candidates]:
                if np.sum(con_mat[cand, NC_idcs] == 2) == 0:
                    continue
                NC_charges = np.sum(con_mat[NC_idcs], axis=1)
                undercharged = NC_charges < NC_valences
                uc_neighbors = np.logical_and(con_mat[cand, NC_idcs] == 1, undercharged)
                if np.any(uc_neighbors):
                    uc_neighbor = NC_idcs[np.where(uc_neighbors)[0][0]]
                    oc_neighbor = NC_idcs[
                        np.where(con_mat[cand, NC_idcs] == 2)[0][0]]
                    con_mat = increase_bond(con_mat, cand, uc_neighbor)
                    con_mat = decrease_bond(con_mat, cand, oc_neighbor)
                    self._connectivity = con_mat
                    changed = True
            if changed:
                self._connectivity = self.get_fixed_connectivity(
                    recursive_call=True)

        # store that fixing the connectivity matrix has already been attempted
        if not recursive_call:
            self._fixed_connectivity = True
            if np.any(old_mat != self._connectivity):
                # update bond orders in underlying OBMol object (where they changed)
                self._update_bond_orders(np.where(old_mat != self._connectivity))

        return self._connectivity

    def _get_valences(self):
        '''
        Retrieve the valency constraints of all atoms in the molecule.

        Returns:
             numpy.ndarray: valency constraints (one per atom)
        '''
        valence = []
        for atom in self.numbers:
            valence += [self.type_infos[atom]['n_bonds']]
        return np.array(valence)

    def _get_CN_neighbors(self, idx):
        '''
        For a focus atom of type K returns indices of atoms C (carbon) and N (nitrogen)
        on two-step paths of the form K-C-N (and K-C-C only for K=N since one atom
        needs to be nitrogen).

        Args:
            idx (int): the index of the focus atom from which paths are examined

        Returns:
             list of lists: list1[i] contains an index of a direct neighbor of the
                focus atom and list2[i] contains the index of a second neighbor on the
                i-th identified two-step path
        '''
        con_mat = self.get_connectivity()
        nbors = con_mat[idx] > 0
        C_nbors = np.where(np.logical_and(self.numbers == 6, nbors))[0]
        type = self.numbers[idx]
        # mask types to exclude idx from neighborhood
        _numbers = self.numbers.copy()
        _numbers[idx] = 0
        CN_nbors = np.where(np.logical_and(_numbers == 7, con_mat[C_nbors] > 0))
        CN_nbors = [(C_nbors[CN_nbors[0][i]], CN_nbors[1][i])
                    for i in range(len(CN_nbors[0]))]
        if type == 7:  # for N atoms, also add C-C neighbors
            CC_nbors = np.where(np.logical_and(
                _numbers == 6, con_mat[C_nbors] > 0))
            CC_nbors = [
                (C_nbors[CC_nbors[0][i]], CC_nbors[1][i])
                for i in range(len(CC_nbors[0]))]
            CN_nbors += CC_nbors
        return CN_nbors

    def _get_neighbors(self, idx, types=None, strength=1):
        '''
        Retrieve the indices of neighbors of an atom.

        Args:
            idx (int): index of the atom
            types (list of int, optional): restrict the returned neighbors to
                contain only atoms of the specified types (set None to apply no type
                filter, default: None)
            strength (int, optional): restrict the returned neighbors to contain
                only atoms with a certain minimal bond order to the atom at idx
                (default: 1)

        Returns:
             list of int: indices of all neighbors that meet the requirements
        '''
        con_mat = self.get_connectivity()
        neighbors = con_mat[idx] >= strength
        if types is not None:
            type_arr = np.zeros(len(neighbors)).astype(bool)
            for type in types:
                type_arr = np.logical_or(type_arr, self.numbers == type)
        return np.where(np.logical_and(neighbors, type_arr))[0]

    def get_bond_stats(self):
        '''
        Retrieve the bond and ring count of the molecule. The bond count is
        calculated for every pair of types (e.g. C1N are all single bonds between
        carbon and nitrogen atoms in the molecule, C2N are all double bonds between
        such atoms etc.). The ring count is provided for rings from size 3 to 8 (R3,
        R4, ..., R8) and for rings greater than size eight (R>8).

        Returns:
             dict (str->int): bond and ring counts
        '''
        if self._bond_stats is None:

            # 1st analyze bonds
            unique_types = np.sort(list(self._unique_numbers))
            # get connectivity and read bonds from matrix
            con_mat = self.get_connectivity()
            d = {}
            for i, type1 in enumerate(unique_types):
                row_idcs = self._get_row_idcs(type1)
                n_bonds1 = self.type_infos[type1]['n_bonds']
                for type2 in unique_types[i:]:
                    col_idcs = self._get_row_idcs(type2)
                    n_bonds2 = self.type_infos[type2]['n_bonds']
                    max_bond_strength = min(n_bonds1, n_bonds2)
                    if n_bonds1 == n_bonds2:  # exclude small trivial molecules
                        max_bond_strength -= 1
                    for n in range(1, max_bond_strength + 1):
                        id = self.type_infos[type1]['name'] + str(n) + \
                            self.type_infos[type2]['name']
                        d[id] = np.sum(con_mat[row_idcs][:, col_idcs] == n)
                        if type1 == type2:
                            d[id] = int(d[id]/2)  # remove twice counted bonds

            # 2nd analyze rings
            ring_counts = self.get_ring_counts()
            if len(ring_counts) > 0:
                ring_counts = np.bincount(np.array(ring_counts))
                n_bigger_8 = 0
                for i in np.nonzero(ring_counts)[0]:
                    if i < 9:
                        d[f'R{i}'] = ring_counts[i]
                    else:
                        n_bigger_8 += ring_counts[i]
                if n_bigger_8 > 0:
                    d[f'R>8'] = n_bigger_8
            self._bond_stats = d

        return self._bond_stats

    def _get_row_idcs(self, type):
        '''
        Retrieve the indices of all atoms in the molecule corresponding to a selected
        type.

        Args:
            type (int): the atom type (atomic number, e.g. 6 for carbon)

        Returns:
             list of int: indices of all atoms with the selected type
        '''
        if type not in self._row_indices:
            self._row_indices[type] = np.where(self.numbers == type)[0]
        return self._row_indices[type]


class ConnectivityCompressor():
    '''
    Utility class that provides methods to compress and decompress connectivity
    matrices.
    '''

    def __init__(self):
        pass

    def compress(self, connectivity_matrix):
        '''
        Compresses a single connectivity matrix.

        Args:
            connectivity_matrix (numpy.ndarray): array (n_atoms x n_atoms)
                containing the bond orders of bonds between atoms of a molecule

        Returns:
            dict (str/int->int): the length of the non-redundant connectivity
            matrix (list with upper triangular part) and the indices of that list for
            bond orders > 0
        '''
        smaller = squareform(connectivity_matrix)  # get list of upper triangular part
        d = {'n_entries': len(smaller)}  # store length of list
        for i in np.unique(smaller).astype(int):  # store indices per bond order > 0
            if i > 0:
                d[int(i)] = np.where(smaller == i)[0]
        return d

    def decompress(self, idcs_dict):
        '''
        Retrieve the full (n_atoms x n_atoms) connectivity matrix from compressed
        format.

        Args:
            idcs_dict (dict str/int->int): compressed connectivity matrix
                (obtained with the compress method)

        Returns:
            numpy.ndarray: full connectivity matrix as an array of shape (n_atoms x
                n_atoms)
        '''
        n_entries = idcs_dict['n_entries']
        con_mat = np.zeros(n_entries)
        for i in idcs_dict:
            if isinstance(i, int) or i.isdigit():
                con_mat[idcs_dict[i]] = int(i)
        return squareform(con_mat)

    def compress_batch(self, connectivity_batch):
        '''
        Compress a batch of connectivity matrices.

        Args:
            connectivity_batch (list of numpy.ndarray): list of connectivity matrices

        Returns:
            list of dict: batch of compressed connectivity matrices (see compress)
        '''
        dict_list = []
        for matrix in connectivity_batch:
            dict_list += [self.compress(matrix)]
        return dict_list

    def decompress_batch(self, idcs_dict_batch):
        '''
        Retrieve a list of full connectivity matrices from a batch of compressed
        connectivity matrices.

        Args:
            idcs_dict_batch (list of dict): list with compressed connectivity
                matrices

        Return:
            list numpy.ndarray: batch of full connectivity matrices (see decompress)
        '''
        matrix_list = []
        for idcs_dict in idcs_dict_batch:
            matrix_list += [self.decompress(idcs_dict)]
        return matrix_list


class IndexProvider():
    '''
    Class which allows to filter a large set of molecules for desired structures
    according to provided statistics. The filtering is done using a selection string
    of the general format 'Statistics_nameDelimiterOperatorTarget_value'
    (e.g. 'C,>8' to filter for all molecules with more than eight carbon atoms where
    'C' is the statistic counting the number of carbon atoms in a molecule, ',' is the
    delimiter, '>' is the operator, and '8' is the target value).

    Args:
        statistics (numpy.ndarray):
            statistics of all molecules where columns correspond to molecules and rows
            correspond to available statistics (n_statistics x n_molecules)
        row_headlines (numpy.ndarray):
            the names of the statistics stored in each row (e.g. 'F' for the number of
            fluorine atoms or 'R5' for the number of rings of size 5)
        default_filter (str, optional):
            the default behaviour of the filter if no operator and target value are
            given (e.g. filtering for 'F' will give all molecules with at least 1
            fluorine atom if default_filter='>0' or all molecules with exactly 2
            fluorine atoms if default_filter='==2', default: '>0')
        delimiter (str, optional):
            the delimiter used to separate names of statistics from the operator and
            target value in the selection strings (default: ',')
    '''

    # dictionary mapping strings of available operators to corresponding function:
    op_dict = {'<': operator.lt,
               '<=': operator.le,
               '==': operator.eq,
               '=': operator.eq,
               '!=': operator.ne,
               '>': operator.gt,
               '>=': operator.ge}

    rel_re = re.compile('<=|<|={1,2}|!=|>=|>')  # regular expression for operators
    num_re = re.compile('[\-]*[0-9]+[.]*[0-9]*')  # regular expression for target values

    def __init__(self, statistics, row_headlines, default_filter='>0', delimiter=','):
        self.statistics = np.array(statistics)
        self.headlines = list(row_headlines)
        self.default_relation = self.rel_re.search(default_filter).group(0)
        self.default_number = float(self.num_re.search(default_filter).group(0))
        self.delimiter = delimiter

    def get_selected(self, selection_str, idcs=None):
        '''
        Retrieve the indices of all molecules which fulfill the selection criteria.
        The selection string is of the general format
        'Statistics_nameDelimiterOperatorTarget_value' (e.g. 'C,>8' to filter for all
        molecules with more than eight carbon atoms where 'C' is the statistic counting
        the number of carbon atoms in a molecule, ',' is the delimiter, '>' is the
        operator, and '8' is the target value).

        The following operators are available:
        '<'
        '<='
        '=='
        '!='
        '>='
        '>'

        The target value can be any positive or negative integer or float value.

        Multiple statistics can be summed using '+' (e.g. 'F+N,=0' gives all
        molecules that have no fluorine and no nitrogen atoms).

        Multiple filters can be concatenated using '&' (e.g. 'H,>8&C,=5' gives all
        molecules that have more than 8 hydrogen atoms and exactly 5 carbon atoms).

        Args:
            selection_str (str): string describing the criterion(s) for filtering (build
                as described above)
            idcs (numpy.ndarray, optional): if provided, only this subset of all
                molecules is filtered for structures fulfilling the selection criteria

        Returns:
            list of int: indices of all the molecules in the dataset that fulfill the
                selection criterion(s)
        '''

        delimiter = self.delimiter
        if idcs is None:
            idcs = np.arange(len(self.statistics[0]))  # take all to begin with
        criterions = selection_str.split('&')  # split criteria
        for criterion in criterions:
            rel_strs = criterion.split(delimiter)

            # add multiple statistics if specified
            heads = rel_strs[0].split('+')
            statistics = self.statistics[self.headlines.index(heads[0])][idcs]
            for head in heads[1:]:
                statistics += self.statistics[self.headlines.index(head)][idcs]

            if len(rel_strs) == 1:
                relation = self.op_dict[self.default_relation](
                    statistics, self.default_number)
            elif len(rel_strs) == 2:
                rel = self.rel_re.search(rel_strs[1]).group(0)
                num = float(self.num_re.search(rel_strs[1]).group(0))
                relation = self.op_dict[rel](statistics, num)
            new_idcs = np.where(relation)[0]
            idcs = idcs[new_idcs]

        return idcs


class ProcessQ(Process):
    '''
    Multiprocessing.Process class that runs a provided function using provided
    (keyword) arguments and puts the result into a provided Multiprocessing.Queue
    object (such that the result of the function can easily be obtained by the host
    process).

    Args:
        queue (Multiprocessing.Queue): the queue into which the results of running
            the target function will be put (the object in the queue will be a tuple
            containing the provided name as first entry and the function return as
            second entry).
        name (str): name of the object (is returned as first value in the tuple put
            into the queue.
        target (callable object): the function that is executed in the process's run
            method
        args (list of any): sequential arguments target is called with
        kwargs (dict (str->any)): keyword arguments target is called with
    '''

    def __init__(self, queue, name=None, target=None, args=(), kwargs={}):
        super(ProcessQ, self).__init__(None, target, name, args, kwargs)
        self._name = name
        self._q = queue
        self._target = target
        self._args = args
        self._kwargs = kwargs

    def run(self):
        '''
        Method representing the process's activity.

        Invokes the callable object passed as the target argument, if any, with
        sequential and keyword arguments taken from the args and kwargs arguments,
        respectively. Puts the string passed as name argument and the returned result
        of the callable object into the queue as (name, result).
        '''
        if self._target is not None:
            res = (self.name, self._target(*self._args, **self._kwargs))
            self._q.put(res)
