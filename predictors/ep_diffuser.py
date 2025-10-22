from itertools import compress
from pathlib import Path
from typing import Optional
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from metrics import minADE
from metrics import minFDE
from modules import EPEncoder
from modules import EPDenoiser
from utils import WarmupCosineLR
from utils import basis_function_b, basis_function_m, transform_m_to_b, transform_b_to_m


class EPDiffuser(pl.LightningModule):
    def __init__(self,
                 pred_deg: int,
                 hidden_dim: int,
                 num_future_steps: int,
                 num_encoder_layers: int,
                 num_denoiser_layers: int,
                 num_freq_bands: int,
                 num_heads: int,
                 head_dim: int,
                 space_dim: int,
                 dropout: float,
                 pl2pl_radius: float,
                 pl2a_radius: float,
                 a2a_radius: float,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 homogenizing: bool,
                 **kwargs) -> None:
        super(EPDiffuser, self).__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.space_dim = space_dim
        self.num_future_steps = num_future_steps
        self.num_encoder_layers = num_encoder_layers
        self.num_denoiser_layers = num_denoiser_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.pl2pl_radius = pl2pl_radius
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.homogenizing = homogenizing
        
        self.pred_deg = pred_deg
        
        self.tau_pred = np.linspace(0.0, 1.0, self.num_future_steps+1)
        self.Phi_B_pred = torch.Tensor(basis_function_b(tau=self.tau_pred,
                                                        n=self.pred_deg,
                                                        delta_t = 6.0)) # note this include the current timestep
        
        self.encoder = EPEncoder(
                                hidden_dim=hidden_dim,
                                pl2pl_radius=pl2pl_radius,
                                pl2a_radius=pl2a_radius,
                                a2a_radius=a2a_radius,
                                num_layers=num_encoder_layers,
                                num_heads=num_heads,
                                head_dim=head_dim,
                                dropout=dropout,
                                homogenizing=homogenizing,
                                )
        
        self.denoiser = EPDenoiser(timesteps=1000,
                                   hidden_dim=hidden_dim,
                                   num_layers=num_denoiser_layers,
                                   num_heads=num_heads,
                                   head_dim=head_dim,
                                   dropout=dropout,
                                   pred_deg= pred_deg,
                                   space_dim = space_dim) 
        

        self.minADE_1 = minADE(max_guesses=1)
        self.minFDE_1 = minFDE(max_guesses=1)

    def forward(self, data: HeteroData):
        scene_enc = self.encoder(data)
        pred = self.denoiser(data=data,
                             scene_enc=scene_enc)

        return pred
                             
                             
    def sample(self, 
               data: HeteroData,
               num_samples: int,
               num_denoising_steps=10,
               method: str = 'ddim'):
                             
        scene_enc =self.encoder(data)
        denoised_cp_samples = self.denoiser.sample(data, 
                                                   scene_enc,
                                                   num_samples=num_samples,
                                                   num_denoising_steps=num_denoising_steps,
                                                   method = method) # [num_samples, A, (pred_deg+1)*spacedim]
                  
        Phi_B_pred_kron = torch.kron(self.Phi_B_pred[1:].contiguous().to(self.device), torch.eye(self.space_dim, device=self.device))   
        traj_samples = (denoised_cp_samples @ Phi_B_pred_kron.mT).view(num_samples, data['agent']['num_nodes'], -1, self.space_dim)
                             
        return traj_samples, denoised_cp_samples
    
    
    def training_step(self,
                      data,
                      batch_idx):
        mask = data['agent']['timestep_x_mask'][:, :].contiguous() #[A,50]
        mask_a = torch.logical_and(~torch.any(~data['agent']['timestep_y_mask'], dim=-1), mask[:, -1]) # consider all current observed and future fully observed agents
        reg_mask = data['agent']['timestep_y_mask'][mask_a] #[A, 60]
        
        pred = self(data) #[S,A,60,2]
        loss_dn = F.mse_loss(pred['pred_noise_cum'][0, mask_a], pred['target_noise_cum'][0, mask_a])
        
        pred_x0 = pred['pred_x0'] #[A, 14] 

        Phi_B_pred_kron = torch.kron(self.Phi_B_pred[1:].contiguous().to(self.device), torch.eye(self.space_dim, device=self.device))   
        pred_trajs = (pred_x0 @ Phi_B_pred_kron.mT).view(data['agent']['num_nodes'], -1, self.space_dim)[mask_a] #[A,T,2]
        gt = data['agent']['target'][mask_a, :, :self.space_dim] #[A,T,2]
        
        loss_reg = (torch.norm(gt - pred_trajs, p=2, dim=-1) * reg_mask).sum()
        
        loss_reg = loss_reg / reg_mask.sum().clamp_(min=1)

        self.log('train_dn_loss', loss_dn, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss', loss_reg, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        loss = loss_dn #+ 0.2*loss_reg
        return loss
    
    @torch.no_grad()
    def validation_step(self,
                        data,
                        batch_idx,
                        eval_len = 60):        
        gt = data['agent']['target'][:, :eval_len, :self.space_dim]
        reg_mask = data['agent']['timestep_y_mask'][:, :eval_len]
        
        scene_enc = self.encoder(data)
        pred = self.denoiser(data=data,
                             scene_enc=scene_enc,
                             timesteps = self.denoiser.timesteps)
        
        num_samples, A, _ = pred['pred_x0'].shape        
        eval_mask = data['agent']['track_category'] == 3
        
        gt_eval = gt[eval_mask]
        Phi_B_pred_kron = torch.kron(self.Phi_B_pred[1:].contiguous().to(self.device), torch.eye(self.space_dim, device=self.device))   
        pred_trajs = (pred['pred_x0'] @ Phi_B_pred_kron.mT).view(num_samples, A, self.num_future_steps, self.space_dim).transpose(0,1)
        
        traj_eval = pred_trajs[eval_mask]
        valid_mask_eval = reg_mask[eval_mask]
        pi_eval = torch.ones((traj_eval.shape[0], 1)).to(gt.device) # probabilities are not considered and are set to 1.
        
        self.minADE_1.update(pred=traj_eval[..., :self.space_dim], target=gt_eval[..., :self.space_dim], prob=pi_eval,
                            valid_mask=valid_mask_eval)

        self.minFDE_1.update(pred=traj_eval[..., :self.space_dim], target=gt_eval[..., :self.space_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)

        
        self.log('val_minADE_1', self.minADE_1, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFDE_1', self.minFDE_1, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))


    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        scheduler = WarmupCosineLR(optimizer=optimizer, min_lr=1e-8, max_lr=self.lr, warmup_epochs=10, total_epochs=self.T_max)
        return [optimizer], [scheduler]
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('EPDiffuser')
        parser.add_argument('--pred_deg', type=int, default=6)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--space_dim', type=int, default=2)
        parser.add_argument('--num_future_steps', type=int, default=60)
        parser.add_argument('--num_encoder_layers', type=int, default=1)
        parser.add_argument('--num_denoiser_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2pl_radius', type=float, default = 150)
        parser.add_argument('--pl2a_radius', type=float, default=150)
        parser.add_argument('--a2a_radius', type=float, default=150)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--homogenizing', type=bool, default=True)

        return parent_parser