'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from utils import wrap_angle, basis_function_b 
import numpy as np

NUM_SAMPLE_POINTS = 9
    
Phi_B = torch.tensor(basis_function_b(tau= np.linspace(0,1,NUM_SAMPLE_POINTS), # Bernstein basis function for map elements
                                      n=3,
                                      d=2, 
                                      k=0,
                                      delta_t=1.))

Phi_Bp = torch.tensor(basis_function_b(tau= np.linspace(0,1,NUM_SAMPLE_POINTS), # Bernstein basis function (prime) for map elements
                                      n=3,
                                      d=2, 
                                      k=1,
                                      delta_t=1.))

class TargetBuilder(BaseTransform):

    def __init__(self,
                 num_historical_steps: int,
                 num_future_steps: int) -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps

    def __call__(self, data: HeteroData) -> HeteroData:
        origin = data['agent']['x'][:, -1, :2]
        theta = data['agent']['x'][:, -1, 2]
        cos, sin = theta.cos(), theta.sin()

        rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        
        if 'y' in data['agent'].keys():
            data['agent']['target'] = origin.new_zeros(data['agent']['num_nodes'], self.num_future_steps, 3)
            data['agent']['target'][..., :2] = torch.bmm(data['agent']['y'][:, :, :2] -
                                                         origin[:, :2].unsqueeze(1), rot_mat)
            data['agent']['target'][..., 2] = wrap_angle(data['agent']['y'][:, :, 2] - theta.unsqueeze(-1))
        
        data['agent']['agent_R'] = rot_mat
        data['agent']['agent_origin'] = origin
        
        data['agent']['cps_mean_transform'] = origin.new_zeros(data['agent']['cps_mean'].shape)
        data['agent']['cps_mean_transform'] = torch.bmm(data['agent']['cps_mean'][:, :, :2] -
                                                     origin[:, :2].unsqueeze(1), rot_mat)
        
        if 'cps_mean_fut' in data['agent'].keys():
            data['agent']['cps_mean_fut_transform'] = origin.new_zeros(data['agent']['cps_mean_fut'].shape)
            data['agent']['cps_mean_fut_transform'] = torch.bmm(data['agent']['cps_mean_fut'][:, :, :2] -
                                                         origin[:, :2].unsqueeze(1), rot_mat)

        lane_cl_cps = data['map']['mapel_cps'][:, -1] #[M,4,2] (-1) means the lane center not boundaries

        reference_pos = Phi_B[None, :, :].to(lane_cl_cps.device) @ lane_cl_cps
        reference_head_vec = Phi_Bp[None, :, :].to(lane_cl_cps.device) @ lane_cl_cps
        reference_heading = torch.atan2(reference_head_vec[..., 1], reference_head_vec[..., 0])
        data['map']['reference_pos'] = reference_pos[:, 4] # middle point as reference
        data['map']['reference_heading'] = reference_heading[:, 4] # middle point as reference
        return data