import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.distributions import MultivariateNormal as MVN


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"    
    def __init__(self, alpha=.25, gamma=2):
            super(WeightedFocalLoss, self).__init__()        
            self.alpha = torch.tensor([alpha, 1-alpha]).cuda()        
            self.gamma = gamma
            
    def forward(self, inputs, targets):
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')        
            targets = targets.type(torch.long)        
            at = self.alpha.gather(0, targets.data.view(-1))        
            pt = torch.exp(-BCE_loss)        
            F_loss = at*(1-pt)**self.gamma * BCE_loss        
            return F_loss.mean()


class ReweightL2(_Loss):
    def __init__(self, train_dist, reweight='inverse'):
        super(ReweightL2, self).__init__()
        self.reweight = reweight
        self.train_dist = train_dist

    def forward(self, pred, target):
        reweight = self.reweight
        prob = self.train_dist.log_prob(target).exp().squeeze(-1)
        if reweight == 'inverse':
            inv_prob = prob.pow(-1)
        elif reweight == 'sqrt_inv':
            inv_prob = prob.pow(-0.5)
        else:
            raise NotImplementedError
        inv_prob = inv_prob / inv_prob.sum()
        loss = F.mse_loss(pred, target, reduction='none').sum(-1) * inv_prob
        loss = loss.sum()
        return loss


class GAILossMD(_Loss):
    """
    Multi-Dimension version GAI, compatible with 1-D GAI
    """

    def __init__(self, init_noise_sigma, gmm):
        super(GAILossMD, self).__init__()
        self.gmm = gmm
        self.gmm = {k: torch.tensor(self.gmm[k]) for k in self.gmm}
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = gai_loss_md(pred, target, self.gmm, noise_var)
        return loss


def gai_loss_md(pred, target, gmm, noise_var):
    I = torch.eye(pred.shape[-1])
    mse_term = -MVN(pred, noise_var*I).log_prob(target)
    balancing_term = MVN(gmm['means'], gmm['variances']+noise_var*I).log_prob(pred.unsqueeze(1)) + gmm['weights'].log()
    balancing_term = torch.logsumexp(balancing_term, dim=1)
    loss = mse_term + balancing_term
    loss = loss * (2 * noise_var).detach()
    return loss.mean()


class BMCLossMD(_Loss):
    """
    Multi-Dimension version BMC, compatible with 1-D BMC
    """

    def __init__(self, init_noise_sigma):
        super(BMCLossMD, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss_md(pred, target, noise_var)
        return loss


def bmc_loss_md(pred, target, noise_var):
    I = torch.eye(pred.shape[-1]).to('cuda')
    logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0))
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to('cuda'))
    loss = loss * (2 * noise_var).detach()
    return loss


# https://github.com/YyzHarry/imbalanced-regression
def weighted_mse_loss(inputs, targets, weights=None):
    loss = F.mse_loss(inputs, targets, reduce=False)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


# https://github.com/YyzHarry/imbalanced-regression
def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduce=False)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


# https://github.com/YyzHarry/imbalanced-regression
def weighted_huber_loss(inputs, targets, weights=None, beta=0.5):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


# https://github.com/YyzHarry/imbalanced-regression
def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=20., gamma=1):
    loss = F.mse_loss(inputs, targets, reduce=False)
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


# https://github.com/YyzHarry/imbalanced-regression
def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=20., gamma=1):
    loss = F.l1_loss(inputs, targets, reduce=False)
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss
