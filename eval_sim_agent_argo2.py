'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

from argparse import ArgumentParser
import pickle
from tqdm.auto import tqdm

import torch
import tensorflow as tf
import numpy as np
from waymo_open_dataset.utils.sim_agents import submission_specs

from datamodules import EPDataModule
from predictors import EPDiffuser
from utils import basis_function_b
from utils import sim_agent as sim_agent_utils
from utils.geometry import get_angle_from_2d_rotation_matrix

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='data/argo2/')
    parser.add_argument('--dataset', type=str, default='argoverse2')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--val_subsample_path', type=str, required=True)
    EPDiffuser.add_model_specific_args(parser)
    args = parser.parse_args()

    model = EPDiffuser.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=False)
    model = model.to('cuda')
    model = model.eval()
    
    datamodule = EPDataModule(root = args.root,
                              dataset= args.dataset,
                              val_processed_dir='data/argo2/val/processed/',
                              train_batch_size=1,
                              val_batch_size=1,
                              test_batch_size=1,
                              val_subsample_path = args.val_subsample_path,
                              )
    
    datamodule.setup()
    
    results = []
    submission_specs.CURRENT_TIME_INDEX = 49
    submission_specs.N_SIMULATION_STEPS = 60
    submission_specs.N_FULL_SCENARIO_STEPS = 110
    print('CURRENT_TIME_INDEX {}; N_SIMULATION_STEPS {}; N_FULL_SCENARIO_STEPS {}'.format(submission_specs.CURRENT_TIME_INDEX, 
                                                                                          submission_specs.N_SIMULATION_STEPS,    
                                                                                          submission_specs.N_FULL_SCENARIO_STEPS))
 
    for i, data in enumerate(tqdm(datamodule.val_dataloader())):
        scenario_id = data['scenario_id'][0]

        src_file = 'data/argo2/val/sim_agent/' + scenario_id+'.pkl'
        with open(src_file, 'rb') as handle: scenario = pickle.load(handle)

        data=data.to(model.device)

        pred_trajs = torch.zeros(32, data['agent']['num_nodes'], 60, 2, device=model.device)
        denoised_cps = torch.zeros(32, data['agent']['num_nodes'], 14, device=model.device)
        for i in range(4):
            pred_trajs_dummy, denoised_cps_dummy = model.sample(data=data,
                                                                num_samples=8,
                                                                num_denoising_steps = 10,
                                                                method='ddim')

            pred_trajs[i*8:(i+1)*8] = pred_trajs_dummy
            denoised_cps[i*8:(i+1)*8] = denoised_cps_dummy
        
        
        '''
        If you encounter a CUDA Out-Of-Memory (OOM) issue, consider the following workaround:
        - Run the prediction step (upper part) in a separate script and save the outputs.
        - Then, load the saved predictions in a new script to run the evaluation (lower part) and compute metrics.
        '''
        ############################# post-processing #####################################
        agt_origin = data['agent']['agent_origin']
        agt_R = data['agent']['agent_R']
        agt_theta = get_angle_from_2d_rotation_matrix(agt_R.cpu().numpy())
        gl_origin = data['global_origin'][0]
        gl_R = data['global_R'][0]
        gl_theta = get_angle_from_2d_rotation_matrix(gl_R)                
                        
        phi_b_p = torch.Tensor(basis_function_b(tau=model.tau_pred[1:],
                                                n=model.pred_deg,
                                                k=1,
                                                return_kron=True,
                                                delta_t = 6.0)).to(denoised_cps.device)

        pred_v_vec= (denoised_cps@phi_b_p.mT).reshape(denoised_cps.shape[0], denoised_cps.shape[1], -1, 2) # [K, A, 60, 2]
        pred_v_norm= torch.norm(pred_v_vec, p=2, dim=-1) # [K, A, 60]
        pred_heading = torch.atan2(pred_v_vec[..., 1], pred_v_vec[..., 0]) # [K, A, 60]

        dist_agent = torch.norm(denoised_cps[:, :, :2] - denoised_cps[:, :, -2:], p=2, dim=-1) # [A], distance between start and end position
        mask_stop = torch.where(dist_agent <=1.0)
        pred_trajs[mask_stop[0], mask_stop[1]] = pred_trajs.new_zeros(pred_trajs[mask_stop[0], mask_stop[1]].shape)

        pred_trajs = ((pred_trajs@ agt_R[None, :].mT) + agt_origin[None, :, None]).cpu().numpy()@gl_R.T + gl_origin # coordinate transformation
        agt_gl_heading = np.zeros((pred_trajs.shape[0], pred_trajs.shape[1], pred_trajs.shape[2]+1), dtype=np.float32)
        agt_gl_heading[:, :, 1:] = pred_heading.cpu().numpy() + agt_theta[None, :, None] + gl_theta
        agt_gl_heading[:, :, 0] = data['agent']['x'][:, -1, 2].cpu().numpy() + gl_theta # last measured heading
        x = pred_trajs[..., 0]
        y = pred_trajs[..., 1]
        z = np.zeros(x.shape)      
        heading = agt_gl_heading # pred_trajs[:,:, 1:] - pred_trajs[:,:, :-1] used for benchmark (sequence) models


        for t in range(pred_trajs.shape[2]):
            mask_t_stop = torch.where(pred_v_norm[:, :, t] < 1.0)
            heading[mask_t_stop[0].cpu(), mask_t_stop[1].cpu(), t+1] = heading[mask_t_stop[0].cpu(), mask_t_stop[1].cpu(), t]

        heading = heading[:, :, 1:]
        dummy_heading = data['agent']['x'][mask_stop[1], -1, 2]
        heading[mask_stop[0].cpu(), mask_stop[1].cpu(), :] = dummy_heading[:, None].repeat(1, 60).cpu().numpy() + gl_theta # assign all stoped agents with last measured heading
        ###################################################################################
                        
        sim_mask = sim_agent_utils.get_sim_mask_argo2(scenario)
        simulated_states = tf.concat([x[:, sim_mask, :, None], y[:, sim_mask, :, None], z[:, sim_mask, :, None], heading[:, sim_mask, :, None]], axis=-1)

        results.append(sim_agent_utils.get_result(scenario, simulated_states))

                        
                        
    ################### output results #########################  
    sim_agent_metrics = {'metametric': [],
                         'kinematic_metrics': [],
                         'interactive_metrics': [],
                         'map_based_metrics': [], 
                         'average_displacement_error': [],
                         'linear_speed_likelihood': [],
                         'linear_acceleration_likelihood': [],
                         'angular_speed_likelihood': [],
                         'angular_acceleration_likelihood': [],
                         'distance_to_nearest_object_likelihood': [],
                         'collision_indication_likelihood': [], 
                         'time_to_collision_likelihood': [],
                         'distance_to_road_edge_likelihood': [],
                         'offroad_indication_likelihood': [],
                         'min_average_displacement_error': [],
                         'simulated_collision_rate': [],
                         'simulated_offroad_rate': []}
    
    for result in results:
        for key in sim_agent_metrics.keys():
            sim_agent_metrics[key].append(result[key]) 

    for key in sim_agent_metrics.keys():
        print('{}: {}'.format(key, np.mean(sim_agent_metrics[key])))