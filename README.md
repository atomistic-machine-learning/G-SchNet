# G-SchNet
Implementation of G-SchNet, a generative model for 3d molecular structures.

# Requirements
- schnetpack 0.3
- pytorch >= 1.2
- python >= 2.7
- Open Babel >= 2.41
- rdkit >= 2019.03.4.0

# Usage
Clone the repository into your folder of choice:

    git clone https://github.com/atomistic-machine-learning/G-SchNet.git
    
A model with the same settings as described in the paper can be trained by running the gschnet_qm9_script.py with standard parameters:

    python ./G-SchNet/gschnet_qm9_script.py train gschnet ./data/ ./models/gschnet/

The training data (QM9) is automatically downloaded and preprocessed if not present in ./data/ and the model will be stored in ./models/gschnet/.
We recommend to train on a GPU (add --cuda to the call). If your GPU has less than 16GB VRAM, you need to decrease the number of features (e.g. --features 64) or the depth of the network (e.g. --interactions 6).

Running the script with the following arguments will generate 100 molecules using the trained model at ./model/geschnet/ and store them in ./model/gschnet/generated/generated.p:

    python ./G-SchNet/gschnet_qm9_script.py generate gschnet ./models/gschnet/ 100

Add --cuda to the call to run on the gpu and add --show_gen to display the molecules with ASE after generation.

# Notes
We recently had to adapt some code due to changes in schnetpack. It is currently tested but should be running fine.
The repository will be updated shortly with more code for analysis and filtering of generated molecules. Furthermore, we will add a pre-trained model and a proper tutorial on how to use the code base.
