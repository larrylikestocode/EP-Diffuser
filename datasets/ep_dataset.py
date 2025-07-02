'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

import os
import pickle
import json
from pathlib import Path
import glob
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from urllib import request
import warnings
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
from av2.datasets.motion_forecasting import scenario_serialization

from utils.preprocess_utils import map_utils_waymo, tracking_utils
from utils.preprocess_utils.map_utils_argo2 import AV2MapInterpreter
from utils.preprocess_utils.map_utils_waymo import WOMapInterpreter
from utils.preprocess_utils.tracker import Polynomial_Tracker
from utils.preprocess_utils import data_types 
from utils.preprocess_utils import converter_utils
from utils.transform import parquet_to_protobuf

class EPDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 dataset: str, 
                 num_workers: int = 8,
                 num_historical_steps: int = 50,
                 hist_deg = 5,
                 fut_deg = 6,
                 mapel_deg = 3,
                 space_dim=2,
                 raw_dir: Optional[str] = None,
                 processed_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 subsample_file_path: Optional[str] = None,
                 use_cache: bool = False,) -> None:

        root = os.path.expanduser(os.path.normpath(root))
        if not os.path.isdir(root):
            os.makedirs(root)
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'{split} is not a valid split')
        self.split = split
        if dataset not in ('argoverse2', 'waymo'):
            raise ValueError(f'{dataset} is not a valid dataset')
        self.dataset = dataset
        self.num_workers = num_workers 
        
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = 60 if dataset == 'argoverse2' else 41
        self.num_steps = num_historical_steps + self.num_future_steps
        self.hist_timescale = (self.num_historical_steps-1) / 10
        self.fut_timescale = 6.0
        self.hist_deg = hist_deg
        self.fut_deg = fut_deg
        self.mapel_deg = mapel_deg
        self.space_dim = space_dim
        self.bool_fit_fut = (split in ('train', 'val'))
        self.subsample_file_path = subsample_file_path
        
        
        self.load_priors(dataset)
        self.tracker = Polynomial_Tracker(timescale=self.hist_timescale, 
                                          degree=self.hist_deg, 
                                          space_dim=self.space_dim, 
                                          hist_len = self.num_historical_steps)
        
        self.sim_agent_data_dir = os.path.join(root, self.split, 'sim_agent')
        if not os.path.isdir(self.sim_agent_data_dir):
            os.makedirs(self.sim_agent_data_dir)
    
        if raw_dir is None:
            raw_dir = os.path.join(root, split, 'raw')
            self._raw_dir = raw_dir
            if os.path.isdir(self._raw_dir):
                if dataset == 'argoverse2':
                    self._raw_file_names = [os.path.join(self._raw_dir, name) for name in os.listdir(self._raw_dir) if
                                            os.path.isdir(os.path.join(self._raw_dir, name))]
                else: # waymo
                    self._raw_file_names = list(glob.glob(os.path.join(self._raw_dir, '*tfrecord*')))
            else:
                self._raw_file_names = []
        else:
            raw_dir = os.path.expanduser(os.path.normpath(raw_dir))
            self._raw_dir = raw_dir
            if os.path.isdir(self._raw_dir):
                if dataset == 'argoverse2':
                    self._raw_file_names = [os.path.join(self._raw_dir, name) for name in os.listdir(self._raw_dir) if
                                            os.path.isdir(os.path.join(self._raw_dir, name))]
                else: # waymo
                    self._raw_file_names = list(glob.glob(os.path.join(self._raw_dir, '*tfrecord*')))
            else:
                self._raw_file_names = []

        if processed_dir is None:
            processed_dir = os.path.join(root, split, 'processed')
            self._processed_dir = processed_dir
            if os.path.isdir(self._processed_dir):
                self._processed_file_names = []

                for root, _, files in tqdm(os.walk(os.path.abspath(self._processed_dir))):
                    files.sort()

                    for file in files:
                        if 'pkl' in file:
                            self._processed_file_names.append(os.path.join(root, file))         
            else:
                self._processed_file_names = []
        else:
            processed_dir = os.path.expanduser(os.path.normpath(processed_dir))
            self._processed_dir = processed_dir
            if os.path.isdir(self._processed_dir):
                self._processed_file_names = []

                for root, _, files in tqdm(os.walk(os.path.abspath(self._processed_dir))):
                    files.sort()

                    for file in files:
                        if 'pkl' in file:
                            self._processed_file_names.append(os.path.join(root, file))         
  

            else:
                self._processed_file_names = []
            
  

        
        if self.subsample_file_path:
            with open(self.subsample_file_path, 'rb') as handle: interested_scenario_ids = pickle.load(handle)
            self._processed_file_names = [processed_file_name for processed_file_name in self._processed_file_names if processed_file_name.split('/')[-1][:-4] in interested_scenario_ids]
            print('Use subsamples for {} {} with {} samples'.format(self.dataset, self.split, len(self._processed_file_names)))
        
        self._num_samples = len(self._processed_file_names)
        super(EPDataset, self).__init__(root=root, transform=transform, pre_transform=None, pre_filter=None)
    
    
    def process(self) -> None:
        error_files = []
        cached_map_path = os.path.join(Path(self._processed_dir).parent.as_posix(), 'processed_map.pkl')
        if os.path.isfile(cached_map_path):
            print('Found cached map: {}'.format(cached_map_path))
            with open(cached_map_path, 'rb') as f:
                self.global_map = pickle.load(f)
        else:
            print('No cached map found')
            self.global_map = {}
        
        print('Processing ' + self.dataset + ' ' + self.split)
        print(
                "Note: This preprocessing step may take considerable time, especially initially. "
                "It involves tracking trajectories for individual agents and fitting the map geometry. "
                "The process will be slower at the beginning but will accelerate as more maps are processed and cached."
             )
        
        
        for raw_file_name in tqdm(self.raw_file_names):
            try:
                if self.dataset == 'argoverse2':
                    scenario_file = glob.glob(os.path.join(raw_file_name, '*.parquet*'))[0]
                    map_file = glob.glob(os.path.join(raw_file_name, '*.json*'))[0]
                    
                    scenario_data = scenario_serialization.load_argoverse_scenario_parquet(scenario_file)
                    
                    output_data = {'scenario_id': scenario_data.scenario_id,
                                    'hist_deg': self.hist_deg,
                                    'mapel_deg': self.mapel_deg, 
                                    'fut_deg': self.fut_deg}

                    output_data['agent'], global_origin, global_R = self.get_argo2_track_features(scenario_data, scenario_file)

                    output_data['global_origin'] = global_origin
                    output_data['global_R'] = global_R

                    output_data['map'] = self.get_argo2_map_features(map_file, origin=global_origin, R=global_R)

                    with open(os.path.join(self.processed_dir, str(output_data['scenario_id']) +'.pkl'), 'wb') as handle:
                        pickle.dump(output_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
                    ########### transform av2 parquet data to wo protobuf -> required for calculating sim agents metrics #########
                    scenario_protobuf = parquet_to_protobuf(raw_file_name)
                    
                    with open(os.path.join(self.sim_agent_data_dir, str(output_data['scenario_id']) +'.pkl'), 'wb') as handle:
                        pickle.dump(scenario_protobuf, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
                else: # waymo
                    dataset = tf.data.TFRecordDataset(raw_file_name, compression_type='')
                    p = Path(raw_file_name)
                    file_name = p.name
                    for cnt, scenario_data in tqdm(enumerate(dataset)):
                        scenario = scenario_pb2.Scenario()
                        scenario.ParseFromString(bytearray(scenario_data.numpy()))
                        
                        output_data = {'scenario_id': scenario.scenario_id,
                                       'file_name': file_name,
                                       'current_time_index': scenario.current_time_index,
                                       'sdc_track_index': scenario.sdc_track_index,
                                       'objects_of_interest': [obj for obj in scenario.objects_of_interest],
                                       'hist_deg': self.hist_deg,
                                       'mapel_deg': self.mapel_deg, 
                                       'fut_deg': self.fut_deg}
                        
                        # convert to dataframe 
                        df_dict, find_focal_agt = converter_utils.waymo_protobuf_to_dataframe(scenario, hist_len = self.num_historical_steps, min_obs_len = 1)
                        if not find_focal_agt:
                            print('no focal in scenario {}, file {}'.format(scenario.scenario_id, file_name))
                            error_files[file_name].append(scenario.scenario_id)
                            continue
                        
                        output_data['agent'], global_origin, global_R = self.get_waymo_track_features(scenario, df_dict)
                        
                        output_data['global_origin'] = global_origin
                        output_data['global_R'] = global_R
                        
                        output_data['map'] = self.get_waymo_map_features(scenario, origin=global_origin, R=global_R)
                        
                        with open(os.path.join(self.processed_dir, str(output_data['scenario_id']) +'.pkl'), 'wb') as handle:
                            pickle.dump(output_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
                        ###### save the scenario protobuf data #####
                        with open(os.path.join(self.sim_agent_data_dir, str(output_data['scenario_id']) +'.pkl'), 'wb') as handle:
                            pickle.dump(scenario, handle, protocol=pickle.HIGHEST_PROTOCOL)

            except:
                print(f'Error with {self.dataset}: '+ raw_file_name)
                error_files.append(raw_file_name)
                
        with open(os.path.join(Path(self._processed_dir).parent.as_posix(), 'processed_map.pkl'), 'wb') as f:
            pickle.dump(self.global_map, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(Path(self._processed_dir).parent.as_posix(), 'error_files_' + self.split +'.pkl'), 'wb') as f:
            pickle.dump(error_files, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    ######### preprocess argoverse 2 ###################
    def get_argo2_track_features(self, scenario_data, scenario_file_path):
        timestamps = scenario_data.timestamps_ns/ 1e9
        timestamps = timestamps - timestamps[0] # shape (110)

        df = pd.read_parquet(scenario_file_path)
        historical_df = df[df['timestep'] < self.num_historical_steps]
        future_df = df[df['timestep'] >= self.num_historical_steps-1] # including the current timestep
        timesteps = list(np.sort(df['timestep'].unique()))

        actor_ids = list(historical_df['track_id'].unique())
        actor_ids = list(filter(lambda actor_id: np.sum(historical_df[historical_df['track_id'] == actor_id]['observed'])>=1, actor_ids)) # at least observed once in the history
        historical_df = historical_df[historical_df['track_id'].isin(actor_ids)]
        df = df[df['track_id'].isin(actor_ids)]

        # DataFrame for AV and target agent
        av_df = df[df['track_id'] == 'AV'].iloc
        av_index = actor_ids.index(av_df[0]['track_id'])
        agt_df = df[df['track_id'] == scenario_data.focal_track_id].iloc
        agent_index = actor_ids.index(agt_df[0]['track_id'])


        num_actors = len(actor_ids)
        timestep_mask = np.zeros((num_actors, 110), dtype=bool) # booleans indicate if object is observed at each timestamp
        time_window = np.zeros((num_actors, 2), dtype=float) # start and end timestamps for the control points
        objects_type = np.zeros((num_actors), dtype=int)
        tracks_category = np.zeros((num_actors), dtype=int)
        x = np.zeros((num_actors, 110, 5), dtype=float) # [x, y, heading, vx, vy]
        x_origin = np.zeros((num_actors, 110, 5), dtype=float) # [x, y, heading, vx, vy]
        x_mean = np.zeros((num_actors, (self.hist_deg+1)* self.space_dim), dtype=float) 
        x_cov = np.zeros((num_actors, (self.hist_deg+1) * self.space_dim, (self.hist_deg+1) * self.space_dim), dtype=float)
        x_mean_fut = np.zeros((num_actors, (self.fut_deg+1)* self.space_dim), dtype=float) 
        x_cov_fut = np.zeros((num_actors, (self.fut_deg+1) * self.space_dim, (self.fut_deg+1) * self.space_dim), dtype=float)
        agent_id = [None] * num_actors

        # make the scene centered at target agent
        origin = np.array([agt_df[self.num_historical_steps-1]['position_x'], agt_df[self.num_historical_steps-1]['position_y']])
        theta = np.array(agt_df[self.num_historical_steps-1]['heading'])
        rotate_mat = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
        R_mat = np.kron(np.eye(self.hist_deg+1), rotate_mat)
        R_mat_fut = np.kron(np.eye(self.fut_deg+1), rotate_mat)

        ego_positions = np.array([av_df[:]['position_x'].values, av_df[:]['position_y'].values]).T
        ego_headings = np.array(av_df[:]['heading'].values)
        ego_velocities = np.array([av_df[:]['velocity_x'].values, av_df[:]['velocity_y'].values]).T

        ego_traj = np.concatenate([ego_positions, ego_headings[:, None], ego_velocities], axis=1) # This is raw data

        obj_trajs = [None] * num_actors
        priors_data, history_priors = [None] * num_actors, np.zeros((num_actors, (self.hist_deg+1)*self.space_dim, (self.hist_deg+1)*self.space_dim))


        av_last_fit_error = None
        agent_last_fit_error = None

        for actor_id, actor_df in df.groupby('track_id'):
            actor_idx = actor_ids.index(actor_id)
            agent_id[actor_idx] = actor_id
            actor_hist_steps = [timesteps.index(timestep) for timestep in historical_df[historical_df['track_id']==actor_id]['timestep']]
            actor_fut_steps = [timesteps.index(timestep) for timestep in future_df[future_df['track_id']==actor_id]['timestep']]
            actor_steps = [timesteps.index(timestep) for timestep in df[df['track_id'] == actor_id]['timestep']]
            timestep_mask[actor_idx, actor_steps] = True
            fut_timestep_mask = timestep_mask[actor_idx, self.num_historical_steps-1:]

            objects_type[actor_idx] = data_types.OBJECT_TYPES.index(actor_df['object_type'].unique()[0])
            tracks_category[actor_idx] = actor_df['object_category'].unique()[0]

            positions = np.array([actor_df[:]['position_x'].values, actor_df[:]['position_y'].values]).T
            headings = np.array(actor_df[:]['heading'].values)
            velocities = np.array([actor_df['velocity_x'].values, actor_df['velocity_y'].values]).T

            obj_traj = np.concatenate([positions, headings[:, None], velocities], axis=1) # This is raw data
            obj_trajs[actor_idx] = (obj_traj)

            x_origin[actor_idx, actor_steps, :2] = positions
            x_origin[actor_idx, actor_steps, 2] = headings
            x_origin[actor_idx, actor_steps, 3:5] = velocities

            x[actor_idx, actor_steps, :2] = (positions - origin) @ rotate_mat
            x[actor_idx, actor_steps, 2] = headings - theta
            x[actor_idx, actor_steps, 3:5] = velocities @ rotate_mat

            T_hist = timestamps[actor_hist_steps]
            T_fut = timestamps[actor_fut_steps]
            time_window[actor_idx] = np.array([np.min(T_hist), np.max(T_hist)])


            if actor_id == 'AV':
                priors_data[actor_idx] = self.hist_prior_data_ego
                history_priors[actor_idx] = self.hist_bernstein_prior_ego
                
                if self.bool_fit_fut:
                    cps_mean_fut, cps_cov_fut = tracking_utils.bayesian_regression_ego(trajectory = obj_traj[np.where(np.array(actor_steps) >= self.num_historical_steps-1)], 
                                                                                       timestamps = T_fut-timestamps[self.num_historical_steps-1],
                                                                                       prior_data = self.fut_prior_data_ego,
                                                                                       timescale = self.fut_timescale,
                                                                                       degree = self.fut_deg)
                    

                    x_mean_fut[actor_idx] = np.reshape(cps_mean_fut, (-1))
                    x_cov_fut[actor_idx] = cps_cov_fut
            else:
                prior_data, hist_prior = None, None
                object_type = objects_type[actor_idx]
                if object_type == 0: # vehicle
                    prior_data, hist_prior = self.hist_prior_data_vehicle, self.hist_bernstein_prior_vehicle
                    fut_prior_data = self.fut_prior_data_vehicle
                elif object_type == 1: # pedestrian
                    prior_data, hist_prior = self.hist_prior_data_pedestrian, self.hist_bernstein_prior_pedestrian
                    fut_prior_data = self.fut_prior_data_pedestrian
                elif object_type == 3: # cyclist
                    prior_data, hist_prior = self.hist_prior_data_cyclist, self.hist_bernstein_prior_cyclist
                    fut_prior_data = self.fut_prior_data_cyclist
                elif object_type == 2: # motor_cyclist
                    prior_data, hist_prior = self.hist_prior_data_cyclist, self.hist_bernstein_prior_cyclist
                    fut_prior_data = self.fut_prior_data_cyclist
                elif object_type == 8: # riderless_bicycle
                    prior_data, hist_prior = self.hist_prior_data_cyclist, self.hist_bernstein_prior_cyclist
                    fut_prior_data = self.fut_prior_data_cyclist
                else: # TODO: what is the prior for unknown objects?
                    prior_data, hist_prior = self.hist_prior_data_vehicle, self.hist_bernstein_prior_vehicle
                    fut_prior_data = self.fut_prior_data_vehicle

                priors_data[actor_idx] = prior_data
                history_priors[actor_idx] = hist_prior
                
                if np.any(fut_timestep_mask) and self.bool_fit_fut:
                    cps_mean_fut, cps_cov_fut = tracking_utils.bayesian_regression_agt(obj_traj[np.where(np.array(actor_steps) >= self.num_historical_steps-1)], 
                                                                                       ego_traj[actor_fut_steps],  
                                                                                       timestamps = T_fut-timestamps[self.num_historical_steps-1], 
                                                                                       prior_data = fut_prior_data,
                                                                                       timescale = self.fut_timescale,
                                                                                       degree = self.fut_deg)

                    x_mean_fut[actor_idx] = np.reshape(cps_mean_fut, (-1))
                    x_cov_fut[actor_idx] = cps_cov_fut

        x_mean_fut = (np.reshape(x_mean_fut, (x_mean_fut.shape[0], self.fut_deg+1, self.space_dim)) - origin) @ rotate_mat
        x_cov_fut = R_mat_fut.T @ x_cov_fut @ R_mat_fut

        ## Start Tracking History ##
        self.tracker.track(x=x_origin, 
                      timestep_mask=timestep_mask, 
                      timestamps=timestamps, 
                      time_window=time_window, 
                      priors=priors_data, 
                      hist_priors=history_priors, 
                      av_index=av_index, 
                      agent_index = agent_index,
                      x_mean=x_mean, 
                      x_cov=x_cov)


        x_mean = (np.reshape(x_mean, (x_mean.shape[0], self.hist_deg+1, self.space_dim)) - origin) @ rotate_mat
        x_cov = R_mat.T @ x_cov @ R_mat


        track_data = {
                    'object_type': torch.tensor(objects_type, dtype=torch.long), # [A]
                    'track_category': torch.tensor(tracks_category, dtype=torch.long), # [A]
                    'timestamps_seconds': torch.tensor(timestamps, dtype=torch.float32), # [110]
                    'x': torch.tensor(x[:, :self.num_historical_steps], dtype=torch.float32), # [A, 50, 5]
                    'y': None if self.split=='test' else torch.tensor(x[:, self.num_historical_steps:], dtype=torch.float32), # [A, 60, 5]
                    'cps_mean': torch.tensor(x_mean, dtype=torch.float32), # [A, 6, 2]
                    'cps_mean_fut': None if self.split=='test' else torch.tensor(x_mean_fut, dtype=torch.float32), # [A, 7, 2]
                    'timestep_x_mask': torch.tensor(timestep_mask[:, :self.num_historical_steps], dtype=bool), #[A, 50]
                    'timestep_y_mask': torch.tensor(timestep_mask[:, self.num_historical_steps:], dtype=bool), #[A, 60]
                    'time_window': torch.tensor(time_window, dtype=torch.float32), # [A, 2]
                    'av_index': torch.tensor(av_index, dtype=torch.long),
                    'agent_index': torch.tensor(agent_index, dtype=torch.long),
                    'agent_ids': agent_id, # [A]
                    'num_nodes': len(actor_ids),
                   }

        return track_data, origin, rotate_mat
    
    
    def get_argo2_map_features(self, map_file_path, origin, R):
        av2_map_interpreter = AV2MapInterpreter(map_file_path = map_file_path,
                                                mapel_deg = self. mapel_deg)
        
        return av2_map_interpreter.get_map_features(origin=origin, R=R, city_map = self.global_map)
        
        
        
    
    ######### preprocess waymo ###################
    def get_waymo_track_features(self, scenario, df_dict):
        df = df_dict['df']
        av_df = df_dict['av_df']
        agt_df = df_dict['agt_df']
        historical_df = df_dict['historical_df']
        actor_ids = df_dict['actor_ids']
        av_id = df_dict['av_id']
        focal_track_id = df_dict['focal_track_id']
        scored_track_id = df_dict['scored_track_id']
        av_index = df_dict['av_index']
        agent_index = df_dict['agent_index']
        
        num_actors = len(actor_ids)
        timestamps = np.array(scenario.timestamps_seconds) # shape (91)
        timestep_mask = np.zeros((num_actors, 91), dtype=bool) # booleans indicate if object is observed at each timestamp
        time_window = np.zeros((num_actors, 2), dtype=float) # start and end timestamps for the control points
        objects_type = np.zeros((num_actors), dtype=int)
        tracks_category = np.zeros((num_actors), dtype=int)
        x = np.zeros((num_actors, 91, 5), dtype=float) # [x, y, heading, vx, vy]
        x_origin = np.zeros((num_actors, 91, 5), dtype=float) # [x, y, heading, vx, vy]
        x_mean = np.zeros((num_actors, (self.hist_deg+1) * self.space_dim), dtype=float) 
        x_cov = np.zeros((num_actors, (self.hist_deg+1) * self.space_dim, (self.hist_deg+1) * self.space_dim), dtype=float)
        x_mean_fut = np.zeros((num_actors, (self.fut_deg+1)* self.space_dim), dtype=float) 
        x_cov_fut = np.zeros((num_actors, (self.fut_deg+1) * self.space_dim, (self.fut_deg+1) * self.space_dim), dtype=float)
        agent_id = [None] * num_actors
        
        # make the scene centered at focal agent
        origin = np.array([agt_df[self.num_historical_steps-1]['position_x'], agt_df[self.num_historical_steps-1]['position_y']])
        theta = np.array(agt_df[self.num_historical_steps-1]['heading'])
        rotate_mat = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
        R_mat = np.kron(np.eye(self.hist_deg+1), rotate_mat)
        R_mat_fut = np.kron(np.eye(self.fut_deg+1), rotate_mat)

        ego_positions = np.array([[x, y] for (x,y) in zip(av_df[:]['position_x'].values, av_df[:]['position_y'].values)])
        ego_headings = np.array([v for v in av_df[:]['heading'].values])
        ego_velocities = np.array([[vx, vy] for (vx, vy) in zip(av_df[:]['velocity_x'].values, av_df[:]['velocity_y'].values)])
        
        ego_traj = np.concatenate([ego_positions, ego_headings[:, None], ego_velocities], axis=1) # This is raw data

        obj_trajs = [None] * num_actors
        priors_data, history_priors = [None] * num_actors, np.zeros((num_actors, (self.hist_deg+1)*self.space_dim, (self.hist_deg+1)*self.space_dim))
        
        for actor_id, actor_df in df.groupby('track_id'):
            actor_idx = actor_ids.index(actor_id)
            agent_id[actor_idx] = actor_id
            actor_time_mask = actor_df['valid']  # [91]
            actor_steps = np.where(actor_time_mask==True)[0]
            actor_hist_steps = np.where(actor_time_mask[:self.num_historical_steps]==True)[0]
            actor_fut_steps = np.where(actor_time_mask[self.num_historical_steps-1:]==True)[0]+ (self.num_historical_steps-1)# including the current timestep
            timestep_mask[actor_idx, actor_time_mask] = True
            fut_timestep_mask = timestep_mask[actor_idx, self.num_historical_steps-1:]

            objects_type[actor_idx] = converter_utils.waymo_object_type_converter(actor_df['object_type'].unique()[0])

            if actor_id == focal_track_id: # focal track
                tracks_category[actor_idx] = 3
            elif actor_id in scored_track_id: # scored track
                tracks_category[actor_idx] = 2
            elif np.sum(actor_time_mask[:self.num_historical_steps]) < self.num_historical_steps: # track_fragment
                tracks_category[actor_idx] = 0
            else: # unscored track
                tracks_category[actor_idx] = 1

            positions = np.array([[x, y] for (x,y) in zip(actor_df['position_x'].values, actor_df['position_y'].values)])
            headings = np.array([v for v in actor_df['heading'].values])
            velocities = np.array([[vx, vy] for (vx,vy) in zip(actor_df['velocity_x'].values, actor_df['velocity_y'].values)])

            positions = positions[actor_steps]
            headings = headings[actor_steps]
            velocities = velocities[actor_steps]

            obj_traj = np.concatenate([positions, headings[:, None], velocities], axis=1) # This is raw data

            x_origin[actor_idx, actor_steps, :2] = positions
            x_origin[actor_idx, actor_steps, 2] = headings
            x_origin[actor_idx, actor_steps, 3:5] = velocities

            x[actor_idx, actor_steps, :2] = (positions - origin) @ rotate_mat
            x[actor_idx, actor_steps, 2] = headings - theta
            x[actor_idx, actor_steps, 3:5] = velocities @ rotate_mat

            T_hist = timestamps[actor_hist_steps]
            T_fut = timestamps[actor_fut_steps]
            
            time_window[actor_idx] = np.array([np.min(T_hist), np.max(T_hist)])


            if actor_id == av_id:
                priors_data[actor_idx] = self.hist_prior_data_ego
                history_priors[actor_idx] = self.hist_bernstein_prior_ego
                
                if self.bool_fit_fut:
                    cps_mean_fut, cps_cov_fut = tracking_utils.bayesian_regression_ego(trajectory = obj_traj[np.where(np.array(actor_steps) >= self.num_historical_steps-1)], 
                                                                                       timestamps = T_fut-timestamps[self.num_historical_steps-1],
                                                                                       prior_data = self.fut_prior_data_ego,
                                                                                       timescale = self.fut_timescale,
                                                                                       degree = self.fut_deg)

                    x_mean_fut[actor_idx] = np.reshape(cps_mean_fut, (-1))
                    x_cov_fut[actor_idx] = cps_cov_fut

            else:
                prior_data, hist_prior = None, None
                object_type = objects_type[actor_idx]
                if object_type == 0: # vehicle
                    prior_data, hist_prior = self.hist_prior_data_vehicle, self.hist_bernstein_prior_vehicle
                    fut_prior_data = self.fut_prior_data_vehicle
                elif object_type == 1: # pedestrian
                    prior_data, hist_prior = self.hist_prior_data_pedestrian, self.hist_bernstein_prior_pedestrian
                    fut_prior_data = self.fut_prior_data_pedestrian
                elif object_type == 3: # cyclist
                    prior_data, hist_prior = self.hist_prior_data_cyclist, self.hist_bernstein_prior_cyclist
                    fut_prior_data = self.fut_prior_data_cyclist
                elif object_type == 2: # motor_cyclist
                    prior_data, hist_prior = self.hist_prior_data_cyclist, self.hist_bernstein_prior_cyclist
                    fut_prior_data = self.fut_prior_data_cyclist
                elif object_type == 8: # riderless_bicycle
                    prior_data, hist_prior = self.hist_prior_data_cyclist, self.hist_bernstein_prior_cyclist
                    fut_prior_data = self.fut_prior_data_cyclist
                else: # TODO: what is the prior for unknown objects?
                    prior_data, hist_prior = self.hist_prior_data_vehicle, self.hist_bernstein_prior_vehicle
                    fut_prior_data = self.fut_prior_data_vehicle

                priors_data[actor_idx] = prior_data
                history_priors[actor_idx] = hist_prior
                
                if np.any(fut_timestep_mask) and self.bool_fit_fut:
                    cps_mean_fut, cps_cov_fut = tracking_utils.bayesian_regression_agt(obj_traj[np.where(np.array(actor_steps) >= self.num_historical_steps-1)], 
                                                                                       ego_traj[actor_fut_steps],  
                                                                                       timestamps = T_fut-timestamps[self.num_historical_steps-1], 
                                                                                       prior_data = fut_prior_data,
                                                                                       timescale = self.fut_timescale,
                                                                                       degree = self.fut_deg)
                          
                    x_mean_fut[actor_idx] = np.reshape(cps_mean_fut, (-1))
                    x_cov_fut[actor_idx] = cps_cov_fut

        
        x_mean_fut = (np.reshape(x_mean_fut, (x_mean_fut.shape[0], self.fut_deg+1, self.space_dim)) - origin) @ rotate_mat
        x_cov_fut = R_mat_fut.T @ x_cov_fut @ R_mat_fut

        
        self.tracker.track(x=x_origin, 
                          timestep_mask=timestep_mask, 
                          timestamps=timestamps, 
                          time_window=time_window, 
                          priors=priors_data, 
                          hist_priors=history_priors, 
                          av_index=av_index, 
                          agent_index = agent_index,
                          x_mean=x_mean, 
                          x_cov=x_cov)
        
        x_mean = (np.reshape(x_mean, (x_mean.shape[0], self.hist_deg+1, self.space_dim)) - origin) @ rotate_mat
        x_cov = R_mat.T @ x_cov @ R_mat
        
        
        track_data = {
                    'object_type': torch.tensor(objects_type, dtype=torch.long), # [A]
                    'track_category': torch.tensor(tracks_category, dtype=torch.long), # [A]
                    'timestamps_seconds': torch.tensor(timestamps, dtype=torch.float32), # [91]
                    'x': torch.tensor(x[:, :self.num_historical_steps], dtype=torch.float32), # [A, 50, 5]
                    'y': None if self.split=='test' else torch.tensor(x[:, self.num_historical_steps:], dtype=torch.float32), # [A, 41, 5]
                    'cps_mean': torch.tensor(x_mean, dtype=torch.float32), # [A, 6, 2]
                    'cps_mean_fut': None if self.split=='test' else torch.tensor(x_mean_fut, dtype=torch.float32), # [A, 7, 2]
                    'timestep_x_mask': torch.tensor(timestep_mask[:, :self.num_historical_steps], dtype=bool), #[A, 50]
                    'timestep_y_mask': torch.tensor(timestep_mask[:, self.num_historical_steps:], dtype=bool), #[A, 41]
                    'time_window': torch.tensor(time_window, dtype=torch.float32), # [A, 2]
                    'av_index': torch.tensor(av_index, dtype=torch.long),
                    'agent_index': torch.tensor(agent_index, dtype=torch.long),
                    'agent_ids': agent_id, # [A]
                    'num_nodes': len(actor_ids),
                   }
        
        return track_data, origin, rotate_mat
    
    
    def get_waymo_map_features(self, scenario, origin, R):
        wo_map_interpreter = WOMapInterpreter(scenario = scenario,
                                              mapel_deg = self.mapel_deg)
        
        return wo_map_interpreter.get_map_features(origin=origin, R=R, city_map = self.global_map)
    
    
    @property
    def raw_dir(self) -> str:
        return self._raw_dir

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @staticmethod
    def get_scenario_id(df: pd.DataFrame) -> str:
        return df['scenario_id'].values[0]

    @staticmethod
    def get_city(df: pd.DataFrame) -> str:
        return df['city'].values[0]

    def len(self) -> int:
        return self._num_samples
    
        
    def get(self, idx: int) -> HeteroData:
        with open(self._processed_file_names[idx], 'rb') as f:
            return HeteroData(pickle.load(f))

    
    def load_priors(self, dataset):
        if dataset not in ['argoverse2', 'waymo']:
            warning_msg = 'Priors for ' + dataset + ' not available, use argoverse2 priors instead'
            warnings.warn(warning_msg)
            dataset = 'argoverse2'
            
        with open('priors/' + dataset + '/vehicle/vehicle_5s.json', "r") as read_file:
            self.hist_prior_data_vehicle = json.load(read_file)
            self.hist_bernstein_prior_vehicle = tracking_utils.get_bernstein_prior(degree=self.hist_deg, timescale=self.hist_timescale, prior_data=self.hist_prior_data_vehicle)[1]


        with open('priors/' + dataset + '/cyclist/cyclist_5s.json', "r") as read_file:
            self.hist_prior_data_cyclist = json.load(read_file)
            self.hist_bernstein_prior_cyclist = tracking_utils.get_bernstein_prior(degree=self.hist_deg, timescale=self.hist_timescale, prior_data=self.hist_prior_data_cyclist)[1]


        with open('priors/' + dataset + '/pedestrian/pedestrian_5s.json', "r") as read_file:
            self.hist_prior_data_pedestrian = json.load(read_file)
            self.hist_bernstein_prior_pedestrian = tracking_utils.get_bernstein_prior(degree=self.hist_deg, timescale=self.hist_timescale, prior_data=self.hist_prior_data_pedestrian)[1]


        with open('priors/' + dataset + '/ego/ego_5s.json', "r") as read_file:
            self.hist_prior_data_ego = json.load(read_file)
            self.hist_bernstein_prior_ego = tracking_utils.get_bernstein_prior(degree=self.hist_deg, timescale=self.hist_timescale, prior_data=self.hist_prior_data_ego)[1]

            
        # Load prior parameters for future trajectory
        with open('priors/' + dataset + '/vehicle/vehicle_6s.json', "r") as read_file:
            self.fut_prior_data_vehicle = json.load(read_file)
 
        with open('priors/' + dataset + '/cyclist/cyclist_6s.json', "r") as read_file:
            self.fut_prior_data_cyclist = json.load(read_file)

        with open('priors/' + dataset + '/pedestrian/pedestrian_6s.json', "r") as read_file:
            self.fut_prior_data_pedestrian = json.load(read_file)

        with open('priors/' + dataset + '/ego/ego_6s.json', "r") as read_file:
            self.fut_prior_data_ego = json.load(read_file)