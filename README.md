# G-SchNet
Implementation of G-SchNet - a generative model for 3d molecular structures - accompanying the [paper published at NeurIPS 2019](http://papers.nips.cc/paper/8974-symmetry-adapted-generation-of-3d-point-sets-for-the-targeted-discovery-of-molecules). 

G-SchNet generates molecules in an autoregressive fashion, placing one atom after another in 3d euclidean space. The model can be trained on data sets with molecules of variable size and composition. It only uses the positions and types of atoms in a molecule, needing no bond-based information such as molecular graphs.

The code provided in this repository allows to train G-SchNet on the QM9 data set which consists of approximately 130k small molecules with up to nine heavy atoms from fluorine, oxygen, nitrogen, and carbon.

### Requirements
- schnetpack 0.3
- pytorch >= 1.2
- python >= 3.7
- Open Babel >= 2.41
- rdkit >= 2019.03.4.0

# Getting started
Clone the repository into your folder of choice:

    git clone https://github.com/atomistic-machine-learning/G-SchNet.git

### Training a model
A model with the same settings as described in the paper can be trained by running gschnet_qm9_script.py with standard parameters:

    python ./G-SchNet/gschnet_qm9_script.py train gschnet ./data/ ./models/gschnet/

The training data (QM9) is automatically downloaded and preprocessed if not present in ./data/ and the model will be stored in ./models/gschnet/.
We recommend to train on a GPU (add --cuda to the call). If your GPU has less than 16GB VRAM, you need to decrease the number of features (e.g. --features 64) or the depth of the network (e.g. --interactions 6).

### Generating molecules
Running the script with the following arguments will generate 1000 molecules using the trained model at ./model/geschnet/ and store them in ./model/gschnet/generated/generated.mol_dict:

    python ./G-SchNet/gschnet_qm9_script.py generate gschnet ./models/gschnet/ 1000

Add --cuda to the call to run on the gpu and add --show_gen to display the molecules with ASE after generation.

### Filtering and analysis of generated molecules
After generation, the generated molecules can be filtered for invalid and duplicate structures by running filter_generated.py:

    python ./G-SchNet/filter_generated.py ./models/gschnet/generated/generated.mol_dict --train_data_path ./data/qm9gen.db --model_path ./models/gschnet
    
The script will print its progress and the gathered results. To store them in a file, please redirect the console output to a file (e.g. ./results.txt) and use the --print_file argument when calling the script:

    python ./G-SchNet/filter_generated.py ./models/gschnet/generated/generated.mol_dict --print_file --train_data_path ./data/qm9gen.db --model_path ./models/gschnet >> ./results.txt
    
The script checks the valency constraints (e.g. every hydrogen atom should have exactly one bond), the connectedness (i.e. all atoms in a molecules should be connected to each other via a path over bonds), and removes duplicates*. The remaining valid structures are stored in an sqlite database with ASE (at ./models/gschnet/generated/generated_molecules.db) along with an .npz-file that records certain statistics (e.g. the number of rings of certain sizes, the number of single, double, and triple bonds, the index of the matching training/test data molecule etc. for each molecule).

*_Please note that, as described in the paper, we use molecular fingerprints and canonical smiles representations to identify duplicates which means that different spatial conformers corresponding to the same molecular graph are usually tagged as duplicates and removed in the process. Add '--filters valence disconnected' to the call in order to not remove but keep identified duplicates in the created database._

# Citation
If you are using G-SchNet in your research, please cite the corresponding paper:

N. Gebauer, M. Gastegger, and K. Schütt. Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules. In H. Wallach, H. Larochelle, A. Beygelz-
imer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, _Advances in Neural Information Processing Systems 32_, pages 7564–7576. Curran Associates, Inc., 2019.

    @incollection{NIPS2019_8974,
    title = {Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules},
    author = {Gebauer, Niklas and Gastegger, Michael and Sch\"{u}tt, Kristof},
    booktitle = {Advances in Neural Information Processing Systems 32},
    editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
    pages = {7564--7576},
    year = {2019},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/8974-symmetry-adapted-generation-of-3d-point-sets-for-the-targeted-discovery-of-molecules.pdf}
    }


# Notes
We recently had to adapt some code due to changes in schnetpack. It is currently tested but should be running fine.
The repository will be updated shortly, we will add a pre-trained model and a proper tutorial on how to use the code base.
