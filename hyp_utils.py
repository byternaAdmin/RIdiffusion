import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import manifolds
from torch.nn.modules.module import Module
import torch.nn.init as init
import math


def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args['act']:
        act = lambda x: x
    else:
        act = getattr(F, args['act'])
    acts = [act] * (args['num_layers'] - 1)
    dims = [args['feat_dim']] + ([args['dim']] * (args['num_layers'] - 1))
    if args['task'] in ['lp', 'rec']:
        dims += [args['dim']]
        acts += [act]
        n_curvatures = args['num_layers']
    else:
        n_curvatures = args['num_layers'] - 1
    if args['c'] is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args['c']]) for _ in range(n_curvatures)]
        if not args['cuda'] == -1:
            curvatures = [curv.to(args['device']) for curv in curvatures]
    return dims, acts, curvatures


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x):
        if self.encode_graph:
            input = x
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout 
        self.use_bias = use_bias 
        self.bias = nn.Parameter(torch.Tensor(out_features)).to('cuda')
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features)).to('cuda')
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight.to('cuda'), x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )

class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

class HNNLayer(nn.Module): 
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias): 
        super(HNNLayer, self).__init__() 
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias) 
        self.hyp_act = HypAct(manifold, c, c, act) 

    def forward(self, x): 
        h = self.linear.forward(x) 
        h = self.hyp_act.forward(h) 
        return h 



class HNN(Encoder):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, c, config):
        super(HNN, self).__init__(c)
        self.manifold = getattr(manifolds, config['manifold'])()
        assert config['num_layers'] > 1
        dims, acts, _ = get_dim_act_curv(config)
        hnn_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(
                HNNLayer(
                    self.manifold, in_dim, out_dim, self.c, config['dropout'], act, config['bias'])
            )
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False

    def encode(self, x):
        # x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        a = self.manifold.proj_tan0(x, self.c)
        b = self.manifold.expmap0(a, c=self.c)
        x_hyp = self.manifold.proj(b, c=self.c)
        return super(HNN, self).encode(x_hyp)

