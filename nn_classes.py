import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import Iterable

import schnetpack as spk
from schnetpack.nn import MLP
from schnetpack.metrics import Metric


### OUTPUT MODULE ###
class AtomwiseWithProcessing(nn.Module):
    r"""
    Atom-wise dense layers that allow to use additional pre- and post-processing layers.

    Args:
        n_in (int): input dimension of representation (default: 128)
        n_out (int): output dimension (default: 1)
        n_layers (int): number of atom-wise dense layers in output network (default: 5)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, interpolate linearly between n_in and n_out.
        activation (function): activation function for hidden layers
            (default: spk.nn.activations.shifted_softplus).
        preprocess_layers (nn.Module): a torch.nn.Module or list of Modules for
            preprocessing the representation given by the first part of the network
            (default: None).
        postprocess_layers (nn.Module): a torch.nn.Module or list of Modules for
            postprocessing the output given by the second part of the network
            (default: None).
        in_key (str): keyword to access the representation in the inputs dictionary,
            it is automatically inferred from the preprocessing layers, if at least one
            is given (default: 'representation').
        out_key (str): a string as key to the output dictionary (if set to 'None', the
            output will not be wrapped into a dictionary, default: 'y')

    Returns:
        result: dictionary with predictions stored in result[out_key]
    """

    def __init__(self, n_in=128, n_out=1, n_layers=5, n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus,
                 preprocess_layers=None, postprocess_layers=None,
                 in_key='representation', out_key='y'):

        super(AtomwiseWithProcessing, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = n_layers
        self.in_key = in_key
        self.out_key = out_key

        if isinstance(preprocess_layers, Iterable):
            self.preprocess_layers = nn.ModuleList(preprocess_layers)
            self.in_key = self.preprocess_layers[-1].out_key
        elif preprocess_layers is not None:
            self.preprocess_layers = preprocess_layers
            self.in_key = self.preprocess_layers.out_key
        else:
            self.preprocess_layers = None

        if isinstance(postprocess_layers, Iterable):
            self.postprocess_layers = nn.ModuleList(postprocess_layers)
        else:
            self.postprocess_layers = postprocess_layers

        if n_neurons is None:
            # linearly interpolate between n_in and n_out
            n_neurons = list(np.linspace(n_in, n_out, n_layers + 1).astype(int)[1:-1])
        self.out_net = MLP(n_in, n_out, n_neurons, n_layers, activation)

        self.derivative = None  # don't compute derivative w.r.t. inputs

    def forward(self, inputs):
        """
        Compute layer output and apply pre-/postprocessing if specified.

        Args:
            inputs (dict of torch.Tensor): batch of input values.
        Returns:
            torch.Tensor: layer output.
        """
        # apply pre-processing layers
        if self.preprocess_layers is not None:
            if isinstance(self.preprocess_layers, Iterable):
                for pre_layer in self.preprocess_layers:
                    inputs = pre_layer(inputs)
            else:
                inputs = self.preprocess_layers(inputs)

        # get (pre-processed) representation
        if isinstance(inputs[self.in_key], tuple):
            repr = inputs[self.in_key][0]
        else:
            repr = inputs[self.in_key]

        # apply output network
        result = self.out_net(repr)

        # apply post-processing layers
        if self.postprocess_layers is not None:
            if isinstance(self.postprocess_layers, Iterable):
                for post_layer in self.postprocess_layers:
                    result = post_layer(inputs, result)
            else:
                result = self.postprocess_layers(inputs, result)

        # use provided key to store result
        if self.out_key is not None:
            result = {self.out_key: result}

        return result


### METRICS ###
class KLDivergence(Metric):
    r"""
    Metric for mean KL-Divergence.

    Args:
        target (str): name of target property
        model_output ([int], [str]): indices or keys to unpack the desired output
            from the model in case of multiple outputs, e.g. ['x', 'y'] to get
            output['x']['y'] (default: 'y').
        name (str): name used in logging for this metric. If set to `None`,
            `KLD_[target]` will be used (default: None).
        mask (str): key for a mask in the examined batch which hides irrelevant output
            values. If 'None' is provided, no mask will be applied (default: None).
        inverse_mask (bool): whether the mask needs to be inverted prior to application
            (default: False).
    """

    def __init__(self, target='_labels', model_output='y', name=None,
                 mask=None, inverse_mask=False):
        name = 'KLD_' + target if name is None else name
        super(KLDivergence, self).__init__(name)
        self.target = target
        self.model_output = model_output
        self.loss = 0.
        self.n_entries = 0.
        self.mask_str = mask
        self.inverse_mask = inverse_mask

    def reset(self):
        self.loss = 0.
        self.n_entries = 0.

    def add_batch(self, batch, result):
        # extract true labels
        y = batch[self.target]

        # extract predictions
        yp = result
        if self.model_output is not None:
            if isinstance(self.model_output, list):
                for key in self.model_output:
                    yp = yp[key]
            else:
                yp = yp[self.model_output]

        # normalize output
        log_yp = F.log_softmax(yp, -1)

        # apply KL divergence formula entry-wise
        loss = F.kl_div(log_yp, y, reduction='none')

        # sum over last dimension to get KL divergence per distribution
        loss = torch.sum(loss, -1)

        # apply mask to filter padded dimensions
        if self.mask_str is not None:
            atom_mask = batch[self.mask_str]
            if self.inverse_mask:
                atom_mask = 1.-atom_mask
            loss = torch.where(atom_mask > 0, loss, torch.zeros_like(loss))
            n_entries = torch.sum(atom_mask > 0)
        else:
            n_entries = torch.prod(torch.tensor(loss.size()))

        # calculate loss and n_entries
        self.n_entries += n_entries.detach().cpu().data.numpy()
        self.loss += torch.sum(loss).detach().cpu().data.numpy()

    def aggregate(self):
        return self.loss / max(self.n_entries, 1.)


### PRE- AND POST-PROCESSING LAYERS ###
class EmbeddingMultiplication(nn.Module):
    r"""
    Layer that multiplies embeddings of given types with the representation.

    Args:
        embedding (torch.nn.Embedding instance): the embedding layer used to embed atom
            types.
        in_key_types (str): the keyword to obtain types for embedding from inputs.
        in_key_representation (str): the keyword to obtain the representation from
            inputs.
        out_key (str): the keyword used to store the calculated product in the inputs
            dictionary.
    """

    def __init__(self, embedding, in_key_types='_next_types',
                 in_key_representation='representation',
                 out_key='preprocessed_representation'):
        super(EmbeddingMultiplication, self).__init__()
        self.embedding = embedding
        self.in_key_types = in_key_types
        self.in_key_representation = in_key_representation
        self.out_key = out_key

    def forward(self, inputs):
        """
        Compute layer output.

        Args:
            inputs (dict of torch.Tensor): batch of input values containing the atomic
                numbers for embedding as well as the representation.
        Returns:
            torch.Tensor: layer output.
        """
        # get types to embed from inputs
        types = inputs[self.in_key_types]
        st = types.size()

        # embed types
        if len(st) == 1:
            emb = self.embedding(types.view(st[0], 1))
        elif len(st) == 2:
            emb = self.embedding(types.view(*st[:-1], 1, st[-1]))

        # get representation
        if isinstance(inputs[self.in_key_representation], tuple):
            repr = inputs[self.in_key_representation][0]
        else:
            repr = inputs[self.in_key_representation]
        if len(st) == 2:
            # if multiple types are provided per molecule, expand
            # dimensionality of representation
            repr = repr.view(*repr.size()[:-1], 1, repr.size()[-1])

        # multiply embedded types with representation
        features = repr * emb

        # store result in input dictionary
        inputs.update({self.out_key: features})

        return inputs


class NormalizeAndAggregate(nn.Module):
    r"""
    Layer that normalizes and aggregates given input along specifiable axes.

    Args:
        normalize (bool): set True to normalize the input (default: True).
        normalization_axis (int): axis along which normalization is applied
            (default: -1).
        normalization_mode (str): which normalization to apply (currently only
            'logsoftmax' is supported, default: 'logsoftmax').
        aggregate (bool): set True to aggregate the input (default: True).
        aggregation_axis (int): axis along which aggregation is applied
            (default: -1).
        aggregation_mode (str): which aggregation to apply (currently 'sum' and
            'mean' are supported, default: 'sum').
        keepdim (bool): set True to keep the number of dimensions after aggregation
            (default: True).
        in_key_mask (str): key to extract a mask from the inputs dictionary,
            which hides values during aggregation (default: None).
        squeeze (bool): whether to squeeze the input before applying normalization
            (default: False).

    Returns:
        torch.Tensor: input after normalization and aggregation along specified axes.
    """

    def __init__(self, normalize=True, normalization_axis=-1,
                 normalization_mode='logsoftmax', aggregate=True,
                 aggregation_axis=-1, aggregation_mode='sum', keepdim=True,
                 mask=None, squeeze=False):

        super(NormalizeAndAggregate, self).__init__()

        if normalize:
            if normalization_mode.lower() == 'logsoftmax':
                self.normalization = nn.LogSoftmax(normalization_axis)
        else:
            self.normalization = None

        if aggregate:
            if aggregation_mode.lower() == 'sum':
                self.aggregation =\
                    spk.nn.base.Aggregate(aggregation_axis, mean=False,
                                          keepdim=keepdim)
            elif aggregation_mode.lower() == 'mean':
                self.aggregation =\
                    spk.nn.base.Aggregate(aggregation_axis, mean=True,
                                          keepdim=keepdim)
        else:
            self.aggregation = None

        self.mask = mask
        self.squeeze = squeeze

    def forward(self, inputs, result):
        """
        Compute layer output.

        Args:
            inputs (dict of torch.Tensor): batch of input values containing the mask
            result (torch.Tensor): batch of result values to which normalization and
                aggregation is applied
        Returns:
            torch.Tensor: normalized and aggregated result.
        """

        res = result

        if self.squeeze:
            res = torch.squeeze(res)

        if self.normalization is not None:
            res = self.normalization(res)

        if self.aggregation is not None:
            if self.mask is not None:
                mask = inputs[self.mask]
            else:
                mask = None
            res = self.aggregation(res, mask)

        return res
