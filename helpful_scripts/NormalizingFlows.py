import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.distributions.transforms as transform
from torch.distributions import constraints


class Flow(transform.Transform, nn.Module):

    def _inverse(self, y):
        pass

    def log_abs_det_jacobian(self, x, y):
        pass

    def _call(self, x):
        pass

    def __init__(self, event_dim = 1):
        transform.Transform.__init__(self)
        nn.Module.__init__(self)
        self._event_dim = event_dim

    # Init all parameters
    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)

    # Hacky hash bypass
    def __hash__(self):
        return nn.Module.__hash__(self)

    @property
    def event_dim(self):
        return self._event_dim

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)


class PlanarFlow(Flow):

    def _inverse(self, y):
        pass

    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.init_parameters()


    def _call(self, x):
        f_x = F.linear(x, self.weight, self.bias)
        return x + self.scale * torch.tanh(f_x)

    def log_abs_det_jacobian(self, x, y = None):
        f_x = F.linear(x, self.weight, self.bias)
        psi = (1 - torch.tanh(f_x) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return torch.log(det_grad.abs() + 1e-9)


# Main class for normalizing flow
class NormalizingFlow(nn.Module):

    def __init__(self, dim, blocks, flow_length, density):
        super().__init__()
        biject = []
        for f in range(flow_length):
            for b_flow in blocks:
                biject.append(b_flow(dim))
        self.transforms = transform.ComposeTransform(biject)
        self.bijectors = nn.ModuleList(biject)
        self.base_density = density
        self.final_density = distrib.TransformedDistribution(density, self.transforms)
        self.log_det = []

    def forward(self, z):
        self.log_det = []
        # Applies series of flows
        for b in range(len(self.bijectors)):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(z))
            z = self.bijectors[b](z)
        return z, self.log_det

    def sample(self):
        with torch.no_grad():
            base_dens_samples = self.base_density.sample()
            out_samples, _ = self.forward(base_dens_samples)
        return out_samples

    def log_prob(self, y):
        log_det_reversed_order = []
        for b in reversed(range(len(self.bijectors))):
            log_det_reversed_order.append(self.bijectors[b].log_abs_det_jacobian(y))
            y = self.bijectors[b]._inverse(y)
            print(y)
        log_prob_base = self.base_density.log_prob(y)
        jacobian_part = torch.sum(torch.stack(log_det_reversed_order))
        return log_prob_base - jacobian_part


class MaskedCouplingFlow(Flow):
    def __init__(self, dim, mask=None, n_hidden=64, n_layers=2, activation=nn.ReLU):
        super(MaskedCouplingFlow, self).__init__()
        self.k = dim // 2
        self.g_mu = self.transform_net(dim, dim, n_hidden, n_layers, activation)
        self.g_sig = self.transform_net(dim, dim, n_hidden, n_layers, activation)
        if mask is None:
            front_back = torch.randint(2, (1,))
            if front_back < 0.5:
                self.mask = torch.cat((torch.ones(self.k), torch.zeros(self.k))).detach()
            else:
                self.mask = torch.cat((torch.zeros(self.k), torch.ones(self.k))).detach()
        else:
            self.mask = mask
        self.init_parameters()
        self.bijective = True

    def transform_net(self, nin, nout, nhidden, nlayer, activation):
        net = nn.ModuleList()
        for l in range(nlayer):
            module = nn.Linear(l == 0 and nin or nhidden, l == nlayer - 1 and nout or nhidden)
            module.weight.data.uniform_(-1, 1)
            net.append(module)
            net.append(activation())
        return nn.Sequential(*net)

    def _call(self, x):
        x_k = (self.mask * x)
        xp_D = x * torch.exp(self.g_sig(x_k)) + self.g_mu(x_k)
        return x_k + (1 - self.mask) * xp_D

    def _inverse(self, y):
        yp_k = (self.mask * y)
        y_D = (((1 - self.mask) * y) - (1 - self.mask) * (self.g_mu(yp_k)) / torch.exp(self.g_sig(yp_k)))
        return yp_k + y_D

    def log_abs_det_jacobian(self, x, y = None):
        return -torch.sum(torch.abs(self.g_sig(x * self.mask)))


