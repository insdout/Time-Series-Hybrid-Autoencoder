import torch.nn as nn
import torch
import torch.nn.functional as F

from hydra.utils import instantiate
import hydra


class KLLoss:
    def __init__(self, weight, z_dims):
        self.name = "KLLoss"
        self.weight = weight
        self.z_dims = z_dims

    def __call__(self, mean, log_var):
        if self.z_dims:
            mean = mean[:, self.z_dims[0]: self.z_dims[1]]
            log_var = log_var[:, self.z_dims[0]: self.z_dims[1]]
        loss = (-0.5 * (1 + log_var - mean ** 2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        return loss


class RegLoss:
    def __init__(self, weight, z_dims):
        self.name = "RegLoss"
        self.weight = weight
        self.z_dims = z_dims
        self.criterion = nn.MSELoss()

    def __call__(self, y, y_hat):
        # loss = F.mse_loss(y, y_hat, reduction='none')
        # loss = torch.mean(loss)
        return self.criterion(y, y_hat)


class ReconLoss:
    def __init__(self, weight, z_dims):
        self.name = "ReconLoss"
        self.weight = weight
        self.z_dims = z_dims

    def __call__(self, x, x_hat):
        batch_size = x.shape[0]
        loss = F.mse_loss(x, x_hat, reduction='none')
        loss = loss.view(batch_size, -1).sum(axis=1)
        loss = loss.mean()
        return loss


class TripletLoss:
    def __init__(self,  weight, z_dims, margin, p):
        self.name = "TripletLoss"
        self.weight = weight
        self.z_dims = z_dims
        self.criterion = nn.TripletMarginLoss(margin=margin, p=p)
    
    def __call__(self, z, z_pos, z_neg):
        if self.z_dims:
            z = z[:, self.z_dims[0]: self.z_dims[1]]
            z_pos = z_pos[:, self.z_dims[0]: self.z_dims[1]]
            z_neg = z_neg[:, self.z_dims[0]: self.z_dims[1]]
        return self.criterion(z, z_pos, z_neg)


class TotalLoss:
    def __init__(self, conf_file):
        self.losses = [instantiate(conf_file.loss[loss_name]) for loss_name in conf_file.loss.total_loss]
        self.weights = [loss.weight for loss in self.losses]
    
    def __call__(self, mean=None, log_var=None, y=None, y_hat=None, x=None, x_hat=None, z=None, z_pos=None, z_neg=None):
        losses_dict = {"TotalLoss": 0}
        for loss in self.losses:
            name = loss.name
            if name == "KLLoss":
                losses_dict[name] = loss(mean, log_var)*loss.weight
                losses_dict["TotalLoss"] += losses_dict[name]
            elif name == "RegLoss":
                losses_dict[name] = loss(y, y_hat)*loss.weight
                losses_dict["TotalLoss"] += losses_dict[name]
            elif name == "ReconLoss":
                losses_dict[name] = loss(x, x_hat)*loss.weight
                losses_dict["TotalLoss"] += losses_dict[name]
            elif name == "TripletLoss":
                losses_dict[name] = loss(z, z_pos, z_neg)*loss.weight
                losses_dict["TotalLoss"] += losses_dict[name]
            else:
                raise Exception(f"No such loss: {name}")
        return losses_dict


class FineTuneTotalLoss:
    def __init__(self, conf_file):
        self.losses = [instantiate(conf_file[loss_name]) for loss_name in conf_file.total_loss]
        self.weights = [loss.weight for loss in self.losses]
        self.only_healthy_rul = conf_file["only_healthy"]
        self.healthy_rul_threshold = conf_file["healthy_rul_threshold"]
    
    def __call__(self, mean=None, log_var=None, y=None, y_hat=None, x=None, x_hat=None, z=None, z_pos=None, z_neg=None):
        losses_dict = {"TotalLoss": 0}
        for loss in self.losses:
            name = loss.name
            if name == "KLLoss":
                losses_dict[name] = loss(mean, log_var)*loss.weight
                losses_dict["TotalLoss"] += losses_dict[name]
            elif name == "RegLoss":
                losses_dict[name] = loss(y, y_hat)*loss.weight
                losses_dict["TotalLoss"] += losses_dict[name]
            elif name == "ReconLoss":
                losses_dict[name] = loss(x, x_hat)*loss.weight
                losses_dict["TotalLoss"] += losses_dict[name]
            elif name == "TripletLoss":

                # Apply Triplet Loss only for healthy RUL:
                if self.only_healthy_rul:
                    loss_mask = y < self.healthy_rul_threshold
                    loss_mask = loss_mask.squeeze()
                   
                    if sum(loss_mask) > 0:
                       
                        losses_dict[name] = loss(z[loss_mask, :], z_pos[loss_mask, :], z_neg[loss_mask, :])*loss.weight
                        losses_dict["TotalLoss"] += losses_dict[name]
                    else:
                        losses_dict[name] = torch.FloatTensor([0])
                        
                else:
                    losses_dict[name] = loss(z, z_pos, z_neg)*loss.weight
                    losses_dict["TotalLoss"] += losses_dict[name]
            else:
                raise Exception(f"No such loss: {name}")
        return losses_dict
