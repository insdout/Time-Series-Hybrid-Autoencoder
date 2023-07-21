import torch.nn as nn
import torch
import math
import numpy as np

    
class MSE:
    """_summary_
    """
    def __call__(self, x, x_hat):
        batch_size = x.shape[0]
        loss = nn.MSELoss(reduction='none')(x, x_hat)
        loss = loss.view(batch_size, -1).sum(axis=1)
        return loss


class TotalLoss:
    """
    generative process:
    p_theta(x, y, z) = p_theta(x|z) p_theta(z|y) p(y)
    y ~ Cat(y|1/k)
    z|y ~ N(z|mu_z_theta(y), sigma^2*z_theta(y))
    x|z ~ B(x|mu_x_theta(z))

    The goal of GMVAE is to estimate the posterior
    distribution p(z, y|x),
    which is usually difficult to compute directly.
    Instead, a factorized posterior,
    known as the inference model,
    is commonly used as an approximation:

    q_phi(z, y|x) = q_phi(z|x, y) q_phi(y|x)
    y|x ~ Cat(y|pi_phi(x))
    z|x, y ~ N(z|mu_z_phi(x, y), sigma^2z_phi(x, y))

    ELBO = -KL(q_phi(z|x, y) || p_theta(z|y))
            - KL(q_phi(y|x) || p(y)) + Eq_phi(z|x,y) [log p_theta(x|z)]

    """
    def __init__(self, 
                 k, 
                 recon_loss=MSE(), 
                 reg_loss=nn.MSELoss(),  
                 recon_weight=1.0, 
                 reg_weight=10.0,
                 eps=1e-8
                 ):
        self.k = k
        self.recon_loss = recon_loss
        self.recon_weight = recon_weight
        self.reg_loss = reg_loss
        self.reg_weight = reg_weight
        self.eps = eps

    def negative_entropy_from_logit(self, qy, qy_logit):
        """
        Computes:
        ???
        ++++++++++++++++++++++++++++++++++++++++++++
        H(q, q) = - ∑q*log q
        H(q, q_logit) = - ∑q*log p(q_logit)
        p(q_logit) = softmax(qy_logit)
        H(q, q_logit) = - ∑q*log softmax(qy_logit)
        ++++++++++++++++++++++++++++++++++++++++++++
        """
        nent = torch.sum(qy * torch.nn.LogSoftmax(1)(qy_logit), 1)
        return nent

    def log_normal(self, x, mu, var, eps=0., axis=-1):
        """
        ADD DOCSTRING!!!
        """
        if eps > 0.0:
            var = torch.add(var, eps)
        return -0.5 * torch.sum(np.log(2 * math.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, axis)

    def _loss_per_class(self, x, x_hat, z, zm, zv, zm_prior, zv_prior, rul, rul_hat):
        loss_px_i = self.recon_loss(x, x_hat)*self.recon_weight
        loss_px_i += self.log_normal(z, zm, zv, eps=self.eps) - self.log_normal(z, zm_prior, zv_prior, eps=self.eps)
        loss_px_i += self.reg_loss(rul, rul_hat)*self.reg_weight
        return loss_px_i - np.log(1/self.k)

    def __call__(self, x, rul, output_dict):
        qy = output_dict["qy"]
        qy_logit = output_dict["qy_logit"]
        px = output_dict["px"]
        z = output_dict["z"]
        zm = output_dict["zm"]
        zv = output_dict["zv"]
        zm_prior = output_dict["zm_prior"]
        zv_prior = output_dict["zv_prior"]
        rul_hat = output_dict["rul_hat_per_class"]
        loss_qy = self.negative_entropy_from_logit(qy, qy_logit)
        losses_i = []
        for i in range(self.k):
            losses_i.append(
                self._loss_per_class(
                    x, px[i], z[i], zm[i], torch.exp(zv[i]), zm_prior[i], torch.exp(zv_prior[i]), rul, rul_hat[i])
                )
        loss = torch.stack([loss_qy] + [qy[:, i] * losses_i[i] for i in range(self.k)]).sum(0)
        # Alternative way to calculate loss:
        # torch.sum(torch.mul(torch.stack(losses_i), torch.transpose(qy, 1, 0)), dim=0)
        out_dict = {"cond_entropy": loss_qy.sum(), "total_loss": loss.sum()}
        return out_dict

