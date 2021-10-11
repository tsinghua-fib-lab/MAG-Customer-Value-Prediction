import torch
from torch_scatter import scatter_max, scatter_add

import sys

def my_softmax(src, index, num_nodes):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    Z = torch.zeros(src.size()).view(-1).to('cuda')
    Z.require_grad = True
    nodes = torch.zeros(num_nodes).to('cuda')

    # out = src.exp().view(-1)
    # out = out / (scatter_add(out, index, dim=0))
    tmp = src.exp().view(-1)
    for i,idx in enumerate(index):
        nodes[idx] += tmp[i]
    for i,idx in enumerate(index):
        Z[i] = nodes[idx]
    out = tmp/Z

    # out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    # out = out.exp()
    # out = out / (
    #     scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out.view(src.size())


def scatter_softmax(src, index, dim=-1, eps=1e-6):
    r"""
    Softmax operation over all values in :attr:`src` tensor that share indices
    specified in the :attr:`index` tensor along a given axis :attr:`dim`.

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = {\textrm{softmax}(\mathrm{src})}_i =
        \frac{\exp(\mathrm{src}_i)}{\sum_j \exp(\mathrm{src}_j)}

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index.
            (default: :obj:`-1`)
        eps (float, optional): Small value to ensure numerical stability.
            (default: :obj:`1e-12`)

    :rtype: :class:`Tensor`
    """
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_softmax` can only be computed over tensors '
                         'with floating point data types.')

    heads = src.size()[0]
    max_per_src_element = src.max(dim=1).values
    # max_value_per_index, _ = scatter_max(src, index, dim=dim)
    # max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element.view(-1,1)

    recentered_scores_exp = recentered_scores.exp()

    sum_per_index = scatter_add(recentered_scores_exp, index, dim=dim)

    index = torch.cat([index.view(1,-1)]*heads, dim=0)

    normalizing_constants = (sum_per_index + eps).gather(dim, index)

    res = recentered_scores_exp / normalizing_constants
    # if (res==0).any():
    #     print('!!!!! res=0', res[res==0])
    #     print('!!!!! res=0', res.size())
    #     print('!!!!! recentered_scores_exp', recentered_scores_exp[res==0])
    #     print('!!!!! recentered_scores_exp', recentered_scores_exp.size())
    #     # print('!!!!! max_per_src_element', max_per_src_element[res==0])
    #     # print('!!!!! max_per_src_element', max_per_src_element.size())
    #     print('!!!!! normalizing_constants', normalizing_constants[res==0])
    #     print('!!!!! normalizing_constants', normalizing_constants.size())

    return res